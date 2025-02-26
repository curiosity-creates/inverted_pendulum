import threading
import time
import numpy as np
import math
import smbus2
import motoron
import matplotlib.pyplot as plt
import os
import torch
from itertools import count

bus_num = 1
bus = smbus2.SMBus(bus_num)
TCA9548A_ADDR = 0x70

# AS5600 device address
AS5600_ADDR = 0x36

# Register address
ANGLE_REG = 0x0E

overFlowCount0 = 0
previousAngle0 = 0

state = np.array((0,0,0,0))

data_lock = threading.Lock()

motor_bus_num = 7
mc = motoron.MotoronI2C(bus=7)

def get_state(stop_event):

    global state

    prev_time = time.time()

    prev_angle0 = read_angle(0)
    prev_angle1 = read_angle(1)

    while not stop_event.is_set():
        current_angle0 = read_angle(0)
        current_angle1 = read_angle(1)

        current_time = time.time()
        dt = current_time- prev_time

        delta_angle0 = current_angle0 - prev_angle0
        vel_joint0 = delta_angle0/dt

        delta_angle1 = (current_angle1 - prev_angle1 + math.pi) % (2 * math.pi) - math.pi
        vel_joint1 = delta_angle1/dt

        prev_angle0 = current_angle0
        prev_angle1 = current_angle1

        with data_lock:
            state = np.array((current_angle0, vel_joint0, current_angle1, vel_joint1))

        time.sleep(0.005)


def select_channel(channel):
        """Selects the specified channel on the TCA9548A multiplexer."""
        try:
            bus.write_byte_data(TCA9548A_ADDR, 0x04, 1 << channel)
        except OSError as e:
            print(f"Error selecting channel on TCA9548A: {e}")

def read_angle(channel):
    """Reads the angle value from the AS5600 sensor on the specified channel."""
    global AS5600_ADDR, ANGLE_REG, bus, previousAngle0, overFlowCount0

    try:
        # Select the channel on the multiplexer
        select_channel(channel)

        # Read two bytes from the sensor, starting at ANGLE_REG
        result = bus.read_i2c_block_data(AS5600_ADDR, ANGLE_REG, 2)

        # Combine bytes and mask to 12 bits
        angle = ((result[0] << 8) | result[1]) & 0x0FFF

        if channel == 1:

            angle = (angle - 638) * 2 * math.pi / 4096
            angle = angle % (2 * math.pi)

        
        elif channel == 0:
        # print(f"Raw angle: {angle}")
            if angle < 500 and previousAngle0 > 3000:
                overFlowCount0 += 1

            elif angle > 3000 and previousAngle0 < 500:
                overFlowCount0 -= 1
            # print(f"Overflow counter: {overFlowCount0}")
            previousAngle0 = angle

            true_angle = angle + (overFlowCount0 * 4096)
            angle = true_angle * math.pi / 17000

            angle = (angle - 0.28)
            angle = math.fmod(angle, math.pi)

        return angle
        # return true_angle
    except OSError as e:
        print(f"Error reading from AS5600 on channel {channel}: {e}")
        return None

def real_life_reset_joint():
    global data_lock, state, mc

    gain = 100

    with data_lock:
        current_position = state[0]
    
    error = -current_position

    while abs(error) > 0.05:
        with data_lock:
            current_position = state[0]
    
        error = -current_position

        speed = gain * error
        speed = int(speed)
        # if speed < 200:
        #     speed = 200
        # elif speed > -200:
        #     speed = -200
        speed = np.clip(speed, -500, 500)
        print(f"Speed: {speed}")
        mc.set_speed(1, speed)
    
    mc.set_speed(1,0)

    

if __name__ == "__main__":

    mc.reinitialize()  # Bytes: 0x96 0x74
    mc.disable_crc()   # Bytes: 0x8B 0x04 0x7B 0x43

    mc.clear_reset_flag()  # Bytes: 0xA9 0x00 0x04

    mc.set_max_acceleration(1, 600)
    mc.set_max_deceleration(1, 300)

    stop_event = threading.Event()
    measurement_thread = threading.Thread(target=get_state, args= (stop_event, ))
    measurement_thread.daemon = True

    measurement_thread.start()

    mc.set_speed(1,200)
    time.sleep(0.2)
    mc.set_speed(1,0)

    real_life_reset_joint()
    # while True:
    #     mc.set_speed()
    #     time.sleep(0.1)

    stop_event.set()
    measurement_thread.join()