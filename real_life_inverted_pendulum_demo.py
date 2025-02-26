import threading
import time
import numpy as np
import math
import smbus2
import motoron
import matplotlib.pyplot as plt
from pybullet_inverted_pendulum import pybulletDQN
import os
import torch
from itertools import count


class MeasurementThread(threading.Thread):
    def __init__(self, stop_event, bus_num = 1):
        super().__init__()
        self.stop_event = stop_event

        self.bus = smbus2.SMBus(bus_num)
        self.TCA9548A_ADDR = 0x70

        # AS5600 device address
        self.AS5600_ADDR = 0x36

        # Register address
        self.ANGLE_REG = 0x0E

        self.overFlowCount0 = 0
        self.previousAngle0 = 0
        self.trueAngle0 = 0

        self.state = state

        self.angle0_array = []
        self.angle1_array = []

        self.vel0_array = []
        self.vel1_array = []

    def run(self):
        prev_time = time.time()

        prev_angle0 = self.read_angle(0)
        prev_angle1 = self.read_angle(1)

        while not self.stop_event.is_set():
            current_angle0 = self.read_angle(0)
            current_angle1 = self.read_angle(1)

            self.angle0_array.append(current_angle0)
            self.angle1_array.append(current_angle1)

            current_time = time.time()
            dt = current_time- prev_time

            delta_angle0 = current_angle0 - prev_angle0
            vel_joint0 = delta_angle0/dt

            delta_angle1 = (current_angle1 - prev_angle1 + math.pi) % (2 * math.pi) - math.pi
            vel_joint1 = delta_angle1/dt

            prev_angle0 = current_angle0
            prev_angle1 = current_angle1

            prev_time = current_time

            self.vel0_array.append(vel_joint0)
            self.vel1_array.append(vel_joint1)

            self.state = np.array((current_angle0, vel_joint0, current_angle1, vel_joint1))

            time.sleep(0.003)

    def select_channel(self, channel):
        """Selects the specified channel on the TCA9548A multiplexer."""
        try:
            self.bus.write_byte_data(self.TCA9548A_ADDR, 0x04, 1 << channel)
        except OSError as e:
            print(f"Error selecting channel on TCA9548A: {e}")

    def read_angle(self, channel):
        """Reads the angle value from the AS5600 sensor on the specified channel."""
        try:
            # Select the channel on the multiplexer
            self.select_channel(channel)

            # Read two bytes from the sensor, starting at ANGLE_REG
            result = self.bus.read_i2c_block_data(self.AS5600_ADDR, self.ANGLE_REG, 2)

            # Combine bytes and mask to 12 bits
            angle = ((result[0] << 8) | result[1]) & 0x0FFF

            if channel == 1:
                angle = (angle - 741) * 2 * math.pi / 4096
                angle = angle % (2 * math.pi)

            
            elif channel == 0:
                # print(f"Raw angle: {angle}")
                # if angle < 500 and self.previousAngle0 > 3000:
                #     self.overFlowCount0 += 1

                # elif angle > 3000 and self.previousAngle0 < 500:
                #     self.overFlowCount0 -= 1
                # print(f"Overflow counter: {self.overFlowCount0}")
                # self.previousAngle0 = angle

                # true_angle = angle + (self.overFlowCount0 * 4096)
                # # angle = true_angle
                # angle = true_angle * math.pi / 4096

                # angle = angle-1.34
                # angle = math.fmod(angle, math.pi)
                angle = (2521-angle) * 2 * math.pi / 4096
                angle = math.fmod(angle, math.pi)

            return angle
            # return true_angle
        except OSError as e:
            print(f"Error reading from AS5600 on channel {channel}: {e}")
            return None

def send_action_motor(mc, action):
    if action == 0:
        mc.set_speed(1,-1000)

    elif action == 1:
        mc.set_speed(1,1000)

    elif action == -1:
        mc.set_speed(1,0)
        

if __name__ == "__main__":

    max_steps = 500
    episode_score = 0
    model_path = os.path.join("saved_models/run24", "model_8000.pth")
    if model_path:
        model = torch.load(model_path)
    
    agent = pybulletDQN(None, model, None, None, None, None)

    motor_bus_num = 7
    mc = motoron.MotoronI2C(bus=7)
    mc.reinitialize()  # Bytes: 0x96 0x74
    mc.disable_crc()   # Bytes: 0x8B 0x04 0x7B 0x43

    mc.clear_reset_flag()  # Bytes: 0xA9 0x00 0x04

    mc.set_max_acceleration(1, 1000)
    mc.set_max_deceleration(1, 2000)

    state = np.array([0,0,0,0])
    stop_event = threading.Event()
    measurement_thread = MeasurementThread(stop_event)
    measurement_thread.start()

    state =  measurement_thread.state
    prev_state = state.copy()
    terminal = False
    truncated = False

    try:
        for step in count():

            state = measurement_thread.state
            # print(f"State: {state}")
            # if state[2] > 0.7 and state[2] < 5.58:
            #     reward = 0
            #     terminal = True
            #     print("Terminal due to joint 2")
            if (prev_state[0] > 2.57 and state[0] < 0) or (prev_state[0] < -2.57 and state[0] > 0):
                reward = -100
                terminal = True
                print("Terminal due to joint 1")
            else:
                reward = math.cos(state[2]) - 0.002 * (state[3]) ** 2 + 0.001 * math.cos(state[0])

            prev_state = state.copy()

            if step > max_steps:
                truncated = True
                print("Truncated")
            
            if step > 0:
                episode_score += reward

            if terminal or truncated:
                action = -1
                mc.set_speed(1,0)
                print(f"Score: {episode_score}")
                break

            action = agent.choose_action_greedy(state)
            # action = np.random.randint(2)
            if action == 0:
                mc.set_speed(1,-1000)

            elif action == 1:
                mc.set_speed(1,1000)

            elif action == -1:
                mc.set_speed(1,0)

            # print(f"Reward: {reward}, action: {action}")
            time.sleep(0.02)

        mc.set_speed(1,0)
        stop_event.set()
        measurement_thread.join()

    except:
        mc.set_speed(1,0)