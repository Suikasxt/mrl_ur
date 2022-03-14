#!/home/suikasxt/anaconda3/envs/ros_noetic/bin/python
import os
import gym
import torch
import rospy
import numpy as np
from env import EnvManager, EnvManagerDiscrete, EnvMoveSingle
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy, DDPGPolicy
from tianshou.exploration import GaussianNoise
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from tianshou.utils.net.continuous import Actor, Critic


model_name = rospy.get_param("model_name", "move_single")
pth_name = os.path.join(os.path.dirname(__file__), 'pth', '%s.pth'%model_name)
log_path = os.path.join(os.path.dirname(__file__), 'log', model_name)
TRAIN = rospy.get_param("train", False)
LOADPTH = False if TRAIN else True
SEED = rospy.get_param("seed", 1)
EPS_TEST = rospy.get_param("eps_test", 0.001)
EPS_TRAIN = rospy.get_param("eps_train", 0.001)
EXPLORATION_NOISE = rospy.get_param("exploration_noise", 0.05)
BUFFER_SIZE = rospy.get_param("buffer_size", 20000)
LEARNING_RATE = rospy.get_param("learning_rate", 1e-3)
LEARNING_RATE_CRITIC = rospy.get_param("learning_rate_critic", 1e-3)
GAMMA = rospy.get_param("gamma", 0.99)
TAU = rospy.get_param("tau", 0.01)
N_STEP = rospy.get_param("n_step", 2)
EPOCH = rospy.get_param("epoch", 1000)
STEP_PER_EPOCH = rospy.get_param("step_per_epoch", 2000)
COLLECT_PER_STEP = rospy.get_param("collect_per_step", 10)
BATCH_SIZE = rospy.get_param("batch_size", 256)
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
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        net_a = Net(self.state_shape, hidden_sizes=[64, 64, 64], device=DEVICE)
        actor = Actor(
            net_a, self.action_shape, max_action=self.max_action, device=DEVICE
        ).to(DEVICE)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
        net_c = Net(
            self.state_shape,
            self.action_shape,
            hidden_sizes=[64, 64, 64],
            concat=True,
            device=DEVICE
        )
        critic = Critic(net_c, device=DEVICE).to(DEVICE)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC)
        self.policy = DDPGPolicy(
            actor,
            actor_optim,
            critic,
            critic_optim,
            tau=TAU,
            gamma=GAMMA,
            exploration_noise=GaussianNoise(sigma=EXPLORATION_NOISE),
            estimation_step=N_STEP,
            action_space=env.action_space
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
        result = offpolicy_trainer(
            self.policy, train_collector, test_collector, EPOCH,
            STEP_PER_EPOCH, COLLECT_PER_STEP, 1,
            BATCH_SIZE, train_fn=train_fn, stop_fn=stop_fn,
            save_fn=save_fn, logger=logger)

    def test(self):
        collector = Collector(self.policy, self.envs)
        collector.collect(n_episode=1)

    def run(self):
        if TRAIN:
            self.train()
        else:
            self.test()

        
 
if __name__ == "__main__":
    actor = MyActor(EnvMoveSingle())
    actor.run()