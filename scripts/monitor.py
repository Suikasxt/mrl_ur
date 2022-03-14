from cProfile import label
import rospy
import matplotlib.pyplot as plt
from env import EnvManager

mem = 300
if __name__ == "__main__":
    env = EnvManager('monitor')
    clock = []
    joint_position = [[] for i in range(env.JOINT_NUM)]
    joint_goal = [[] for i in range(env.JOINT_NUM)]
    plt.ion()
    while not rospy.is_shutdown():
        clock.append(env.clock.to_sec())
        if len(clock) > mem:
            del clock[:-mem]
        for i in range(env.JOINT_NUM):
            joint_position[i].append(env.joint_state['pos'][i])
            joint_goal[i].append(env.joint_goal[i])
            if len(joint_goal[i]) > mem:
                del joint_goal[i][:-mem]
                del joint_position[i][:-mem]
            plt.subplot(1, 2, i+1)
            plt.title("Joint " + str(i))
            plt.plot(clock, joint_position[i], color='b', label='Goal Position')
            plt.plot(clock, joint_goal[i], color='r', label='Real Position')
            plt.xlabel("time/s")
            plt.ylabel("joint/rad")
        plt.pause(0.2)
