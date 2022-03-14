#!/home/suikasxt/anaconda3/envs/ros_noetic/bin/python
#from asyncio.log import logger
import os
#from statistics import mean
import gym
import torch
import rospy
import numpy as np
from env import EnvManagerDiscrete
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer


model_name = rospy.get_param("model_name", "joint_controller")
pth_name = os.path.join(os.path.dirname(__file__), 'pth', '%s.pth'%model_name)
log_path = os.path.join(os.path.dirname(__file__), 'log', model_name)
LOADPTH = False
TRAIN = rospy.get_param("train", False)
SEED = rospy.get_param("seed", 1)
EPS_TEST = rospy.get_param("eps_test", 0.001)
EPS_TRAIN = rospy.get_param("eps_train", 0.001)
BUFFER_SIZE = rospy.get_param("buffer_size", 2000)
LEARNING_RATE = rospy.get_param("learning_rate", 1e-3)
GAMMA = rospy.get_param("gamma", 0.99)
N_STEP = rospy.get_param("n_step", 2)
TARGET_UPDATE_FREQ = rospy.get_param("target_update_freq", 320)
EPOCH = rospy.get_param("epoch", 1000)
STEP_PER_EPOCH = rospy.get_param("step_per_epoch", 300)
COLLECT_PER_STEP = rospy.get_param("collect_per_step", 10)
BATCH_SIZE = rospy.get_param("batch_size", 256)
ALPHA = rospy.get_param("alpha", 0.6)
BETA = rospy.get_param("beta", 0.4)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'







class MyActor:
    def __init__(self, env):
        self.env = env
        self.envs = DummyVectorEnv(
            [lambda: env()])
        self.state_shape = env.observation_space.shape or env.observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        self.net = Net(self.state_shape, self.action_shape,
			hidden_sizes=[256, 256, 256], device=DEVICE,
            ).to(DEVICE)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.policy = DQNPolicy(
            self.net, self.optim, GAMMA, N_STEP,
            target_update_freq=TARGET_UPDATE_FREQ)
        if LOADPTH:
            self.policy.load_state_dict(torch.load(pth_name))
        self.buf = ReplayBuffer(BUFFER_SIZE)

    def train(self):
        # collector
        train_collector = Collector(self.policy, self.envs, self.buf)
        train_collector.collect(n_step=BATCH_SIZE)
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        def save_fn(policy):
            torch.save(policy.state_dict(), pth_name)

        def stop_fn(mean_rewards):
            print('mean_rewards', mean_rewards)
            return False

        def train_fn(epoch, env_step):
            self.policy.set_eps(EPS_TRAIN)
            if env_step % 100 == 0:
                logger.write("train/env_step", env_step, {"train/eps": EPS_TRAIN})

        # trainer
        result = offpolicy_trainer(
            self.policy, train_collector, None, EPOCH,
            STEP_PER_EPOCH, COLLECT_PER_STEP, 0,
            BATCH_SIZE, train_fn=train_fn, stop_fn=stop_fn,
            save_fn=save_fn, logger=logger)

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
    actor = MyActor(EnvManagerDiscrete)
    actor.run()