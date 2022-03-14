#!/home/suikasxt/anaconda3/envs/ros_noetic/bin/python
import rospy
import threading
import numpy as np

class MyActor:
    def __init__(self, env):
        threading.Thread.__init__(self)
        self.env = env
        self.p = 0.5
        self.i = 0
        self.d = 0
        self.delta_count = np.zeros(6)

    def run(self, time = -1):
        rate_hz = 10
        if time >= 0:
            max_count = time * rate_hz
        else:
            max_count = -1
        rate = rospy.Rate(rate_hz)
        count = 0
        while not rospy.is_shutdown() and (max_count == -1 or max_count > count):
            if self.env.joint_state['pos'] is not None:
                delta = self.env.joint_goal - self.env.joint_state['pos']
                self.delta_count += delta
                vel = self.p * delta + self.i * self.delta_count
                self.env.step(vel, False)
                #rospy.loginfo('goal' + str(self.env.joint_goal))
                #rospy.loginfo('vel' + str(vel))
            rate.sleep()
            count += 1
        
        
 
if __name__ == "__main__":
    from env import EnvManager
    env = EnvManager()
    actor = MyActor(env)
    actor.run()