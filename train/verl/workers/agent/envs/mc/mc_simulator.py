# 添加您的路径
import sys
import os
import ast

# 在导入后添加检查
import verl

from verl.workers.agent.tool_envs import ToolBase
from .utils.client_utils import *

import grpc
from .utils import minecraft_pb2 as pb
from .utils import minecraft_pb2_grpc as pb_grpc

import json
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image
from rich import print, console
import numpy as np
import logging
import time
import uuid
import re
import shortuuid
import cv2



import requests

from concurrent.futures import ThreadPoolExecutor

SERVER_IP = os.environ.get("SERVER_IP")
SERVER_PORT = os.environ.get("SERVER_PORT")

# if SERVER_URL is None:
#     print("环境变量 SERVER_URL 未设置！")

def load_screen(uuid, path: str) -> np.ndarray:
    """
    返回 RGB 顺序、uint8、形状 (H, W, 3) 的 numpy 数组
    同时将图片保存到 debug 目录
    """
    # 1. 修正源路径
    fixed_path = fix_storage_path(path)
    
    # 2. 打开图片
    img = Image.open(os.path.join('../server',fixed_path))

    # 返回 numpy 数组 (Shape 变为: 360, 540, 3)
    return np.asarray(img)
    



def fix_storage_path(path: str) -> str:
    """
    将路径中的 steai_share 修正为 steai-share
    """
    if "steai_share" in path:
        return path.replace("/steai_share", "/steai-share")
    
    return path

def stub_channel():
    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),   # 50 MiB
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
    ]
    channel = grpc.insecure_channel(f'{SERVER_IP}:{SERVER_PORT}', options=options)
    return pb_grpc.MinecraftEnvStub(channel)



class MCSimulator(ToolBase):
    name = "MCSimulator"
    
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.data_idx = None
        self.empty_action = 'wait'

    def get_data_idx(self):
        return self.data_idx

    def get_uuid(self):
        return self.uuid

    def send_action(self, uuid, action):

        max_try_time = 128000000
        for i in range(max_try_time):
            try:
                # --- 修改开始：记录发送的 Action ---
                if hasattr(self, 'log_file_path'):
                    with open(self.log_file_path, "a") as f:
                        f.write(f"[{datetime.now()}] Send Action: {action}\n")
                    # print(f"[DEBUG-VERIFY] Logged action '{action}' to file")
                # --- 修改结束 ---

                stub = stub_channel()
                stub.SubmitStep(pb.StepReq(env_id=self.uuid, action_json=json.dumps(action)))
                while True:
                    resp = stub.GetStepResult(pb.EnvIdReq(env_id=self.uuid))
                    if resp.status == "success":
                        break
                    time.sleep(0.01)

                ### Question: step之后如何获得obs和info？
                ### obs需要作为字典，关键词是image
                obs_path = json.loads(resp.observation_json)["pov"]
                obs = {'image': load_screen(self.uuid, obs_path)}
                reward = resp.reward
                terminated = resp.terminated
                truncated = False
                info = json.loads(resp.info_json)
                break
            except:
                time.sleep(1.0)
                print(f'[DEBUG] when send action, out of time, retry={i+1}')
       
        return obs, reward, terminated, truncated, info

    def execute(self, action_string, config, **kwargs) -> tuple:
        ## 这里的action_string通过<action>...</action>包裹住
        start_time = time.perf_counter()
        
        # done的逻辑有问题，一步就 (保留你的原始注释)
        try:
            # 打印原始字符串以供调试
            print(f'[DEBUG] Raw action_string received: {action_string}')

            # TODO: 使用正则表达式提取出
            # pattern解释: 
            # <action>  : 匹配字面量开始标签
            # (.*?)     : 非贪婪匹配任意字符（捕获组1），即我们要的内容
            # </action> : 匹配字面量结束标签
            # re.DOTALL : 允许 . 匹配换行符（防止action内容包含换行导致匹配失败）
            pattern = r"<action>(.*?)</action>"
            match = re.search(pattern, action_string, re.DOTALL)
            
            if match:
                action = match.group(1).strip() # 获取捕获组内容并去除首尾空格
                print(f'[DEBUG] Success! Extracted action: {action}')
            else:
                # 如果没有匹配到，主动抛出异常，以便进入下方的 except 块处理 empty_action
                raise ValueError(f"No <action> tags found in: {action_string}")

            if action in ["wait", "walk_forward", "walk_backward", "jump",
                "look_left", "look_right", "look_up", "look_down"]:
                obs, reward, terminated, _, info = self.send_action(self.uuid, action)
                reward += 0.01
            elif "move_away" in action or 'operate' in action:
                obs, reward, terminated, _, info = self.send_action(self.uuid, action)
                reward += 0.01
            else:
                obs, reward, terminated, _, info = self.send_action(self.uuid, "wait")
            truncated = False
            

        except Exception as e:
            # 这里捕获所有异常（包括上面手动抛出的ValueError和send_action可能的报错）
            try:
                # 打印具体的报错原因，有助于debug
                print(f'[DEBUG] Error occurred: {e}')
                print(f'[DEBUG] error action string is {action_string=}')
            except:
                print(f'[DEBUG] cannot print error action string!')
            
            # 执行空动作兜底
            obs, reward, terminated, _, info = self.send_action(self.uuid, self.empty_action)
            truncated = False  
    
        done = (terminated or truncated) # 当返回reward立马停止
        
        elapsed_ms = (time.perf_counter() - start_time)
        print(f'execute action {elapsed_ms=}')

        return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, uuid, data_idx):
        start_time = time.perf_counter()
        try:
            self.data_idx = data_idx
            # 提取 uuid 内容
            self.uuid = uuid
            
            # --- 修改开始：创建日志文件 ---
            log_dir = "logs/env_logs"
            os.makedirs(log_dir, exist_ok=True)
            self.log_file_path = os.path.join(log_dir, f"{self.uuid}.log")
            try:
                with open(self.log_file_path, "w") as f:
                    f.write(f"[{datetime.now()}] Start Reset for UUID: {self.uuid}\n")
                print(f"[DEBUG-VERIFY] Log file created: {self.log_file_path}")
            except Exception as e:
                print(f"[WARNING] Failed to create log file: {e}")
            # --- 修改结束 ---

            self.step_idx = 0
            

            # 在你的方法内部（比如 reset 或 step 中）
            max_retries = 12800000
            retry_count = 0
            obs, info = None, None

            while retry_count < max_retries:
                try:
                    print(f'[DEBUG] {self.uuid}, {retry_count}-th try to get the reset result begins!')
                    start_t = time.time()
                    # 注意：get_reset_result() 应该返回一个 Ray 对象引用（ObjectRef）
                    stub = stub_channel()
                    req = pb.GetResetResultReq(env_id=self.uuid)
                    resp = stub.GetResetResult(req)
                    obs_path = json.loads(resp.observation_json)["pov"]
                    obs = {'image': load_screen(self.uuid, obs_path)}
                    info = json.loads(resp.info_json)
                    end_t = time.time()

                    # --- 修改开始：记录 Reset 成功 ---
                    if hasattr(self, 'log_file_path'):
                        with open(self.log_file_path, "a") as f:
                            f.write(f"[{datetime.now()}] Reset Status: Success\n")
                    print(f"[DEBUG-VERIFY] Logged reset success to {self.log_file_path}")
                    # --- 修改结束 ---

                    print(f"[INFO] get_reset_result succeeded after {retry_count + 1} attempts, took {end_t - start_t:.2f}s")
                    break  # 成功则跳出循环
                except Exception as e:  # <--- 修改这里：捕获异常赋值给 e
                    retry_count += 1
                    # 现在可以安全地打印错误类型和详细信息了
                    print(f"[WARNING] {self.uuid}, Attempt {retry_count} failed. Error Type: {type(e).__name__}, Detail: {e}")
                    
                    print(f"[ERROR] Failed to get reset result after {retry_count} retries.")
                    time.sleep(1.0)


            return (obs, info)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            
    def close(self):
        # 采用轮询防止超时。
        max_try_time = 10100
        
        for i in range(max_try_time):
            try:
                # print(f'[DEBUG] begin to close the env!')
                # stub = stub_channel()
                # resp = stub.CloseEnv(pb.EnvIdReq(env_id=self.uuid))
                # print(f'[DEBUG] close the env with success!')
                break
            except:
                time.sleep(1.0)
                print(f'[DEBUG] out of time, retry={i+1}')
        return





