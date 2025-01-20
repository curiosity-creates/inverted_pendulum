from pybullet_inverted_pendulum import pybulletDQN, send_action, get_state, reset_joint
import os
import torch
from itertools import count
import pybullet as p
import pybullet_data
import time

if __name__ == "__main__":
    
    max_steps = 500
    model_path = os.path.join("saved_models/run13", "model_5000.pth")
    if model_path:
        model = torch.load(model_path)
    
    agent = pybulletDQN(None, model, None, None, None, None)

    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    friction = 0.0047
    randomize_val = 0.05

    planeId = p.loadURDF("plane.urdf", [0, 0, 0])
    robotId = p.loadURDF("rlr_urdf.urdf", [0, 0, 0])


    base_pos, _ = p.getBasePositionAndOrientation(robotId)
    p.resetDebugVisualizerCamera(0.8, 0, -40, base_pos)

    p.setRealTimeSimulation(1)

    reset_joint(robotId, randomize_val= randomize_val, friction=friction)
    state = get_state(robotId)
    episode_score = 0

    terminal = False
    truncated = False
    logId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "model_5000.mp4")


    while True:
        for step in count():

            state = get_state(robotId)
            # print(f"State: {state}")
            if state[2] > 0.7 and state[2] < 5.58:
                reward = 0
                terminal = True
                # print("Terminal due to joint 2")
            elif state[0] > 3.14 or state[0] < -3.14:
                reward = -100
                terminal = True
                # print("Terminal due to joint 1")
            else:
                reward = 1

            if step > max_steps:
                truncated = True
                # print("Truncated")
            
            if step > 0:
                episode_score += reward

            if terminal or truncated:
                action = -1
                send_action(robotId, action)
                print(f"Score: {episode_score}")
                break

            action = agent.choose_action_greedy(state)
            send_action(robotId, action)

            time.sleep(0.02)

        time.sleep(0.02)

    p.stopStateLogging(logId)
    p.disconnect()
