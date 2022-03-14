#!/home/suikasxt/anaconda3/envs/ros_noetic/bin/python
import os
import gym
import torch
import rospy
import numpy as np
from torch import nn
from env import EnvManager, EnvManagerDiscrete
from torch.utils.tensorboard import SummaryWriter

from torch.distributions import Independent, Normal
from tianshou.policy import DQNPolicy, DDPGPolicy, PPOPolicy
from tianshou.exploration import GaussianNoise
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from tianshou.utils.net.continuous import ActorProb, Critic


model_name = rospy.get_param("model_name", "joint_controller")
pth_name = os.path.join(os.path.dirname(__file__), 'pth', '%s.pth'%model_name)
log_path = os.path.join(os.path.dirname(__file__), 'log', model_name)
LOADPTH = False
TRAIN = rospy.get_param("train", False)
SEED = rospy.get_param("seed", 1)
EPS_TEST = rospy.get_param("eps_test", 0.001)
EPS_TRAIN = rospy.get_param("eps_train", 0.001)
EXPLORATION_NOISE = rospy.get_param("exploration_noise", 0.001)
BUFFER_SIZE = rospy.get_param("buffer_size", 1024)
LEARNING_RATE = rospy.get_param("learning_rate", 1e-4)
GAMMA = rospy.get_param("gamma", 0.99)
LAMBDA = rospy.get_param("lambda", 0.95)
TAU = rospy.get_param("tau", 0.005)
MAX_GRAD_NORM = rospy.get_param("max_grad_norm", 0.5)
VF_COEF = rospy.get_param("vf_COEF", 0.25)
ENT_COEF = rospy.get_param("ent_COEF", 0)
EPS_CLIP = rospy.get_param("eps_clip", 0.2)
VALUE_CLIP = rospy.get_param("value_clip", 0)
DUAL_CLIP = rospy.get_param("dual_clip", None)
N_STEP = rospy.get_param("n_step", 2)
REW_NORM = rospy.get_param("rew_norm", True)
NORM_ADV = rospy.get_param("norm_adv", False)
RECOMPUTE_ADV = rospy.get_param("recompute_adv", True)
TARGET_UPDATE_FREQ = rospy.get_param("target_update_freq", 320)
EPOCH = rospy.get_param("epoch", 1000)
STEP_PER_EPOCH = rospy.get_param("step_per_epoch", 600)
STEP_PER_COLLECT = rospy.get_param("step_per_collect", 10)
COLLECT_PER_STEP = rospy.get_param("collect_per_step", 10)
REPEAT_PER_COLLECT = rospy.get_param("repeat_per_collect", 2)
BATCH_SIZE = rospy.get_param("batch_size", 128)
ALPHA = rospy.get_param("alpha", 0.6)
BETA = rospy.get_param("beta", 0.4)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'







class MyActor:
    def __init__(self, env):
        print(DEVICE)
        self.env = env
        self.envs = DummyVectorEnv(
            [lambda: self.env])
        self.state_shape = env.observation_space.shape or env.observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n
        self.max_action = env.action_space.high[0]
        self.hidden_sizes = [64, 64, 64]
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        
        # model
        net_a = Net(
            self.state_shape,
            hidden_sizes=self.hidden_sizes,
            activation=nn.Tanh,
            device=DEVICE
        )
        actor = ActorProb(
            net_a,
            self.action_shape,
            max_action=self.max_action,
            unbounded=True,
            device=DEVICE
        ).to(DEVICE)
        net_c = Net(
            self.state_shape,
            hidden_sizes=self.hidden_sizes,
            activation=nn.Tanh,
            device=DEVICE
        )
        critic = Critic(net_c, device=DEVICE).to(DEVICE)
        torch.nn.init.constant_(actor.sigma_param, -0.5)
        for m in list(actor.modules()) + list(critic.modules()):
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

        optim = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE
        )

        lr_scheduler = None

        def dist(*logits):
            return Independent(Normal(*logits), 1)

        self.policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=GAMMA,
            gae_lambda=LAMBDA,
            max_grad_norm=MAX_GRAD_NORM,
            vf_coef=VF_COEF,
            ent_coef=ENT_COEF,
            reward_normalization=REW_NORM,
            action_scaling=True,
            action_bound_method='clip',
            lr_scheduler=lr_scheduler,
            action_space=env.action_space,
            eps_clip=EPS_CLIP,
            value_clip=VALUE_CLIP,
            dual_clip=DUAL_CLIP,
            advantage_normalization=NORM_ADV,
            recompute_advantage=RECOMPUTE_ADV
        )

        if LOADPTH:
            self.policy.load_state_dict(torch.load(pth_name))
        self.buf = ReplayBuffer(BUFFER_SIZE)

    def train(self):
        # collector
        train_collector = Collector(self.policy, self.envs, self.buf, exploration_noise=True)
        train_collector.collect(n_step=(BATCH_SIZE-1)//self.env.Step_Per_Round+1)
        test_collector = Collector(self.policy, self.envs, exploration_noise=False)
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        def save_fn(policy):
            torch.save(policy.state_dict(), pth_name)

        def stop_fn(mean_rewards):
            return False

        def train_fn(epoch, env_step):
            if env_step % 100 == 0:
                logger.write("train/env_step", env_step, {"train/eps": EPS_TRAIN})

        # trainer
        result = onpolicy_trainer(
            self.policy,
            train_collector,
            test_collector,
            EPOCH,
            STEP_PER_EPOCH,
            REPEAT_PER_COLLECT,
            1,
            BATCH_SIZE,
            step_per_collect=STEP_PER_COLLECT,
            save_fn=save_fn,
            logger=logger,
            test_in_train=True
        )

    def test(self):
        self.policy.set_eps(EPS_TEST)

        collector = Collector(self.policy, self.envs)
        collector.collect(n_episode=1)

    def run(self):
        if TRAIN:
            self.train()
        else:
            self.test()

        
 
if __name__ == "__main__":
    actor = MyActor(EnvManager())
    actor.run()