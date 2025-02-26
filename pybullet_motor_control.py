import pybullet as p
import pybullet_data
from pybullet_inverted_pendulum import get_state, send_action, reset_joint_swingup
import time
import threading
import numpy as np
import math
import matplotlib.pyplot as plt
import os

class pybulletMeasurementhread(threading.Thread):
    def __init__(self, robotId, stop_event):
        super().__init__()
        self.robotId = robotId
        self.stop_event = stop_event

        self.angle0_array = []
        self.angle1_array = []

        self.vel0_array = []
        self.vel1_array = []

    def run(self):
        while not self.stop_event.is_set():
            state = get_state(self.robotId)
            self.angle0_array.append(state[0])
            self.angle1_array.append(state[2])

            self.vel0_array.append(state[1])
            self.vel1_array.append(state[3])

            time.sleep(0.005)


if __name__ == "__main__":

    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    friction = 0.0047
    randomize_val = 0.05
    planeId = p.loadURDF("plane.urdf", [0, 0, 0])
    robotId_path = os.path.join("urdf and meshes/with smaller weight/rlr_urdf.urdf")

    robotId = p.loadURDF(robotId_path, [0, 0, 0])

    base_pos, _ = p.getBasePositionAndOrientation(robotId)
    p.resetDebugVisualizerCamera(0.5, 0, -40, base_pos)

    p.setRealTimeSimulation(1)

    reset_joint_swingup(robotId, friction=friction)
    stop_event = threading.Event()
    measurement_thread = pybulletMeasurementhread(robotId, stop_event)
    measurement_thread.start()


    for i in range(5):
        send_action(robotId, 0)
        time.sleep(0.05)

        # send_action(robotId, 0)
        # time.sleep(0.05)

    send_action(robotId, -1)

    stop_event.set()
    measurement_thread.join()


    p.disconnect()


    angle0_array = np.array(measurement_thread.angle0_array)
    angle1_array = np.array(measurement_thread.angle1_array)
    print(f"Max angle = {angle0_array.max()}")
    print(f"Min angle= {angle0_array.min()}")

    # Plot the angle array
    plt.plot(angle0_array, label = "Joint 0")
    plt.plot(angle1_array, label = "Joint 1")
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Angle (radians)')
    plt.title('Angle over steps')
    plt.show()

    vel0_array = np.array(measurement_thread.vel0_array)
    vel1_array = np.array(measurement_thread.vel1_array)
    plt.plot(vel0_array, label = "Vel 0")
    plt.plot(vel1_array, label = "Vel 1")
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Velocity (rad/s)')
    plt.title('Velocity over steps')
    plt.show()
