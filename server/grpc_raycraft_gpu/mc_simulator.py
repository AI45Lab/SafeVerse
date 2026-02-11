"""
GPU 版本的 Minecraft Simulator

特性：
- GPU 硬件加速渲染（VirtualGL）
- 支持 LLM action string 格式
- 支持标准 Gym action 格式
- 与 CPU 版本接口兼容
"""

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

import cv2
import yaml
import numpy as np
from PIL import Image

from entry import MinecraftSim


class MCSimulator:
    """
    GPU 版本的 Minecraft Simulator

    支持两种 action 格式：
    1. LLM string: '<think>...</think><answer>[{"action": "forward"}]</answer>'
    2. Standard dict/array: {"forward": 1, ...} 或 np.array([...])
    """

    # 标准动作空间定义（与 deepeyes 版本保持一致）
    STANDARD_ACTIONS = [
        'walk_forward',
        'walk_backward',
        'move_left',
        'move_right',
        'sprint',
        'sneak',
        'jump',
        'use',
        'attack',
        'turn_right',
        'turn_left',
        'look_up',
        'look_down',
        'look_down-left',
        'look_up-right',
        'inventory',
    ]

    def __init__(self, config=None, config_path=None, output_dir=None,
                 display_port=None, working_dir=None, mc_root=None, xvfb=False):
        """
        初始化 GPU Simulator

        Args:
            config: 配置字典（优先于 config_path）
            output_dir: 输出目录（视频、日志等）
            display_port: DISPLAY 端口号（用于资源隔离）
            env_port: Minecraft 环境端口号（用于资源隔离）
            working_dir: 工作目录（用于资源隔离）
        """
        
        # 配置来源：优先使用 config 字典，否则使用 config_path
        self.data = config

        # 资源管理参数
        self.display_port = display_port
        self.working_dir = working_dir
        self.mc_root=mc_root


        # 设置资源隔离环境变量（如果提供）
        if self.display_port is not None:
            print(f"[MCSimulator] Set DISPLAY=:{self.display_port}")


        if self.working_dir is not None:
            # 确保工作目录存在
            Path(self.working_dir).mkdir(parents=True, exist_ok=True)
            # 创建 output 子目录（视频等会保存到这里）
            output_dir = Path(self.working_dir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = str(output_dir)
            print(f"[MCSimulator] Using working_dir={self.working_dir}")
            print(f"[MCSimulator] Using output_dir={self.output_dir}")
        
        
        # 初始化 MinecraftSim（GPU 版本）
        self.simulator = MinecraftSim(
            task=self.data['target_type'],
            scene=self.data['scene'],
            working_dir=self.mc_root,
            output_dir=self.output_dir,
            display_port=self.display_port,
            xvfb=xvfb
        )

    def reset(self):
        reward_fn = {}
        command = []
        start_pos = self.data['start_pos']
        if 'start_rotation' in self.data.keys():
            start_rotation = self.data['start_rotation']
        else:
            start_rotation = [-59.8 , 5.8]
        command.append(f"cmd //schem load {self.data['scene']}")
        command.append("cmd //paste -o")
        tp_command = f"cmd tp @s {str(start_pos[0])} {str(start_pos[1])} {str(start_pos[2])} {str(start_rotation[0])} {str(start_rotation[1])}"
        command.append(tp_command)
        if 'goal_pos' in self.data.keys():
            reward_fn['type'] = 'navigation'
            reward_fn['gt_position'] = [self.data['goal_pos'][0], self.data['goal_pos'][1], self.data['goal_pos'][2]]
            reward_fn['gt_rotation'] = [0, 0]
            self.operatable_list = self.data['interact_list']
        elif 'goal_obj_status' in self.data.keys():
            reward_fn['type'] = 'operation'
            reward_fn['gt_position'] = [self.data['goal_obj_pos'][0], self.data['goal_obj_pos'][1], self.data['goal_obj_pos'][2]]
            reward_fn['gt_obj_id'] = self.data["goal_obj_id"]
            reward_fn['gt_obj_status'] = self.data["goal_obj_status"]
            self.reset_command = f"cmd setblock {self.data['goal_obj_pos'][0]} {self.data['goal_obj_pos'][1]} {self.data['goal_obj_pos'][2]} {self.data['start_obj_status']}"
            self.operatable_list = self.data['interact_list']
            command.append(self.reset_command)
        else:
            reward_fn['type'] = 'None'
            reward_fn['gt_position'] = [0, 0, 0]
            reward_fn['gt_rotation'] = [0, 0]

        
        
        # 重置环境
        obs, info = self.simulator.reset(command=command, reward_fn=reward_fn)
        
        return obs, info
    
    def fast_reset(self, config):
        self.data = config
        reward_fn = {}
        command = []
        start_pos = self.data['start_pos']
        if 'start_rotation' in self.data.keys():
            start_rotation = self.data['start_rotation']
        else:
            start_rotation = [-59.8 , 5.8]
        command.append(f"cmd //schem load {self.data['scene']}")
        command.append("cmd //paste -o")
        tp_command = f"cmd tp @s {str(start_pos[0])} {str(start_pos[1])} {str(start_pos[2])} {str(start_rotation[0])} {str(start_rotation[1])}"
        command.append(tp_command)
        if 'goal_pos' in self.data.keys():
            reward_fn['type'] = 'navigation'
            reward_fn['gt_position'] = [self.data['goal_pos'][0], self.data['goal_pos'][1], self.data['goal_pos'][2]]
            reward_fn['gt_rotation'] = [0, 0]
            reward_fn['task'] = self.data['target_type']
            self.operatable_list = self.data['interact_list']
        elif 'goal_obj_status' in self.data.keys():
            reward_fn['type'] = 'operation'
            reward_fn['gt_position'] = [self.data['goal_obj_pos'][0], self.data['goal_obj_pos'][1], self.data['goal_obj_pos'][2]]
            reward_fn['gt_obj_id'] = self.data["goal_obj_id"]
            reward_fn['gt_obj_status'] = self.data["goal_obj_status"]
            reward_fn['task'] = self.data['target_type']
            self.reset_command = f"cmd setblock {self.data['goal_obj_pos'][0]} {self.data['goal_obj_pos'][1]} {self.data['goal_obj_pos'][2]} {self.data['start_obj_status']}"
            self.operatable_list = self.data['interact_list']
            command.append(self.reset_command)
        else:
            reward_fn['type'] = 'None'
            reward_fn['gt_position'] = [0, 0, 0]
            reward_fn['gt_rotation'] = [0, 0]
            reward_fn['task'] = self.data['target_type']
        
        # 重置环境
        obs, info = self.simulator.fast_reset(command=command, reward_fn=reward_fn)
        
        return obs, info

    def step(self, action):
        print(action)

        if 'wait' in action:
            action = {'init': 0, 'wait': 1, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'walk_forward' in action:
            action = {'init': 0, 'wait': 0, 'forward': 1, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'walk_backward' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 1, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'jump' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 1, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'turn_left' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'turn_right' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'look_left' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'look_right' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'look_up' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 1, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'look_down' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 1, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'sprint' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 1, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'sneak' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 1, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'attack' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 1, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'look_up-left' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 1, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'look_up-right' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 1, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'look_down-left' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 0, 'look_down': 1, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'look_down-right' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 0, 'look_down': 1, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'move_left' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 1, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'move_right' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 1, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'use' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 1, 'operate': "", 'moveaway': ""}
        elif 'inventory' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 1, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': ""}
        elif 'operate' in action:
            flag = 0
            print(self.data)
            for i in range(len(self.operatable_list)):
                if self.operatable_list[i].lower() in action:
                    action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': f"aim {self.data['interact_pos'][i][0]} {self.data['interact_pos'][i][1]} {self.data['interact_pos'][i][2]} 3 90 0.5"}
                    flag=1
            if flag == 0:
                action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "operate nothing", 'moveaway': ""}
        elif 'move_away' in action:
            flag = 0
            print(self.data)
            for i in range(len(self.operatable_list)):
                if self.operatable_list[i].lower() in action:
                    action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': self.data['interact_pos']}
                    flag=1
            if flag == 0:
                action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': "", 'moveaway': "move_away nothing"}
            
        else:
            print("action error")
        # 执行 step
        obs, reward, terminated, truncated, info = self.simulator.step(action)

        # 微小的 step reward
        reward += 0.001

        return obs, reward, terminated, truncated, info

    def close(self):
        """关闭环境"""
        if self.simulator:
            self.simulator.close()
    
    def fast_close(self):
        """关闭环境"""
        if self.simulator:
            self.simulator.fast_close()

