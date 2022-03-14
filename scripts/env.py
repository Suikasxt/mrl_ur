#!/home/suikasxt/anaconda3/envs/ros_noetic/bin/python
from socket import MsgFlag
import gym
import rospy
import threading
import numpy as np
from std_msgs.msg import MultiArrayDimension, MultiArrayLayout, Float64MultiArray
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import LinkStates, LinkState, ModelStates, ModelState
from geometry_msgs.msg import Pose, Twist, Point
from sensor_msgs.msg import JointState

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetLinkState, SetModelState
from joint_controller_pid import MyActor as PID_Actor


class EnvManager (threading.Thread):
    JOINT_NUM = 2
    spec = gym.envs.registration.EnvSpec("JointController-v1", reward_threshold = None)
    observation_space = gym.spaces.box.Box( np.array( [-np.pi]*(JOINT_NUM*2) ), np.array( [np.pi]*(JOINT_NUM*2) ) )
    action_space = gym.spaces.box.Box( np.array( [-1]*JOINT_NUM ), np.array( [1]*JOINT_NUM ) )
    JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    INIT_JOINT_STATE = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
    MAX_CLOCK = rospy.Duration(secs = 20)
    Step_Per_Round = 200

    def __init__(self, node_name = 'joint_controller'):
        threading.Thread.__init__(self)
        self.node_name = node_name
        self.joint_goal = self.INIT_JOINT_STATE
        self.joint_state = {'pos': self.INIT_JOINT_STATE, 'vel': np.zeros(6)}
        self.clock = rospy.Time()
        self.last_clock = rospy.Time()
        self.dist = None

        rospy.init_node(self.node_name)
        rospy.Subscriber('/joint_states', JointState, self.update_joint_state)
        rospy.Subscriber('/env/joint_goal', Float64MultiArray, self.update_goal)
        rospy.Subscriber('/clock', Clock, self.update_clock)
        self.step_rate = rospy.Rate(10)
        '''
        self.pause_client = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause_client = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_client = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.reset_link_state_client = rospy.ServiceProxy("/gazebo/set_link_state", SetLinkState)
        '''
        self.step_pub = rospy.Publisher('/' + rospy.get_param("velocity_controller_name", "velocity_controllers") + '/command', Float64MultiArray, queue_size=1)
        self.monitor_pub = rospy.Publisher('/env/monitor', Float64MultiArray, queue_size=1)

        self.start()
        '''
        self.init_link_states = None
        while self.init_link_states is None or len(self.init_link_states.name) == 0:
            self.init_link_states = rospy.wait_for_message("/gazebo/link_states", LinkStates)
        '''

    def get_state(self):
        return np.concatenate((self.joint_state['pos'][:self.JOINT_NUM], self.joint_goal[:self.JOINT_NUM]))
    
    def get_reward(self):
        dist = np.sum(np.abs(self.joint_goal - self.joint_state['pos'])[:self.JOINT_NUM])
        if (self.dist is None):
            self.dist = dist
        rew = self.dist - dist
        self.dist = dist
        return rew - dist/10
    

    def step(self, action, sleep = True):
        dim = [MultiArrayDimension(size = 6)]
        action = np.concatenate((action, [0]*(6-action.shape[0])))
        self.step_pub.publish(Float64MultiArray(layout = MultiArrayLayout(dim = dim), data = action))
        if sleep:
            self.step_rate.sleep()
        state = self.get_state()
        rew = self.get_reward()
        self.monitor_pub.publish(Float64MultiArray(layout = MultiArrayLayout(dim = [MultiArrayDimension(size = self.JOINT_NUM*3+1)]), data = np.concatenate((state, action[:self.JOINT_NUM], [rew]))))
        #print(state, action, rew, (self.clock - self.last_clock).nsecs)
        done = self.clock - self.last_clock >= self.MAX_CLOCK
        if rospy.get_param("train", False) == False:
            done = False
        return state, rew, done, {}
    
    def reset(self):
        '''
        self.pause_client.wait_for_service()
        self.pause_client.call()
        for i in range(len(self.init_link_states.name)):
            self.reset_link_state_client.wait_for_service()
            self.reset_link_state_client.call(LinkState(self.init_link_states.name[i], self.init_link_states.pose[i], self.init_link_states.twist[i], 'world'))
        self.unpause_client.call()
        
        #self.reset_client.wait_for_service()
        #self.reset_client.call()'''
        self.joint_goal = self.INIT_JOINT_STATE
        PID_Actor(self).run(3)

        if rospy.get_param("train", False) == True:
            self.joint_goal = self.INIT_JOINT_STATE + (np.random.random(6) - 0.5) * np.pi
        #self.joint_goal = np.array([1, -1, 0.5, 0, 0, 0])
        self.step_rate._reset = True
        self.last_clock = self.clock
        self.dist = None
        state = self.get_state()
        return state
    
    def update_link_state(self, msg):
        pass
    
    def update_clock(self, msg):
        self.clock = msg.clock

    def update_joint_state(self, msg):
        state = {}
        for i in range(len(msg.name)):
            state[msg.name[i]] = {'pos': msg.position[i], 'vel': msg.velocity[i]}
        position = []
        velocity = []
        for name in self.JOINT_NAMES:
            position += [ state[name]['pos'] ]
            velocity += [ state[name]['vel'] ]
        self.joint_state['pos'] = np.array(position)
        self.joint_state['vel'] = np.array(velocity)

    def update_goal(self, msg):
        self.joint_goal = np.array(msg.data)

    def seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def run(self):
        rospy.spin()
    
    def render(self):
        pass
    
    def close(self):
        pass

class EnvManagerDiscrete(EnvManager):
    action_space = gym.spaces.Discrete(9)
    def step(self, actionD, sleep = True):
        if np.shape(actionD) == (6,):
            return super().step(actionD, sleep)
        action = (np.array([actionD%3, actionD//3%3, actionD//9%3, actionD//27%3, actionD//81%3, actionD//243%3], dtype=np.float64) - 1)
        action[2:] = 0
        return super().step(action, sleep)

class EnvMoveSingle(EnvManager):
    spec = gym.envs.registration.EnvSpec("MoveSingle-v1", reward_threshold = None)
    observation_space = gym.spaces.box.Box( np.concatenate(( [-np.pi]*(EnvManager.JOINT_NUM*2), [-1]*6 )), np.concatenate(( [np.pi]*(EnvManager.JOINT_NUM*2), [1]*6 )) )
    action_space = gym.spaces.box.Box( np.array( [-1]*EnvManager.JOINT_NUM ), np.array( [1]*EnvManager.JOINT_NUM ) )
    MAX_CLOCK = rospy.Duration(secs = 40)
    INIT_BOX_POS = np.array([0.3, 0.3, 0.05])
    BOX_GOAL = np.array([0.3, -0.3, 0.05])
    def __init__(self, node_name='move_single'):
        super().__init__(node_name)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.update_model_state)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.update_link_state)
        self.set_model_client = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.link_pos = {}
        self.box_pos = self.INIT_BOX_POS
        self.box_goal = self.BOX_GOAL

    def get_state(self):
        return np.concatenate((
            self.joint_state['pos'][:self.JOINT_NUM],
            self.joint_goal[:self.JOINT_NUM],
            self.box_pos,
            self.box_goal
            ))
    
    def get_reward(self):
        dist = np.linalg.norm(self.box_goal - self.box_pos)
        if self.link_pos.get("robot::wrist_3_link") is not None:
            dist += np.linalg.norm(self.link_pos["robot::wrist_3_link"] - self.box_pos) / 10
        if (self.dist is None):
            self.dist = dist
        rew = self.dist - dist
        self.dist = dist
        return rew - dist/10

    def update_link_state(self, msg):
        for i in range(len(msg.name)):
            self.link_pos[msg.name[i]] = np.array([msg.pose[i].position.x, msg.pose[i].position.y, msg.pose[i].position.z])

    def update_model_state(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == "test_box":
                self.box_pos = np.array([msg.pose[i].position.x, msg.pose[i].position.y, msg.pose[i].position.z])
                break
        
        self.box_pos[self.box_pos > 1] = 1
        self.box_pos[self.box_pos <- 1] = -1


    def reset(self):
        self.set_model_client.wait_for_service()
        self.set_model_client.call(ModelState("test_box", Pose(position=Point(*self.INIT_BOX_POS)), Twist(), "world"))
        self.box_goal = self.BOX_GOAL
        return super().reset()
    

if __name__ == "__main__":
    env = EnvManager()
    env.join()