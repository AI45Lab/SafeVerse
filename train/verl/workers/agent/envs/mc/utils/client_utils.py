#!/usr/bin/env python3
"""
轻量测试脚本：针对单实例 http_server_single_mp，批量校验多实例部署。

假设已有多个 http_server_single_mp 进程跑在递增端口（默认 8000 起）。
脚本将：
 1) 为每个 server 创建 1 个环境
 2) 等待 reset_result 就绪
 3) 对每个环境做若干 step_async 并轮询结果
 4) 关闭环境
"""

import argparse
import json
import time
import multiprocessing as mp
from typing import List, Optional, Dict, Any, Tuple

import requests

# 默认参数（可用命令行覆盖）
HOST = "127.0.0.1"
BASE_PORT = 8000
SERVER_COUNT = 2          # 要测试的实例数量（从 base_port 递增）
ENV_KWARGS_DEFAULT: Optional[List[dict]] = None
STEPS_PER_ENV = 16
HEALTH_TIMEOUT = 30
CENTER_URL = "http://127.0.0.1:7999"  # 中心调度服务地址，可通过 --center-url 覆盖


def _fmt_ts(ts: float):
    return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1)*1000):03d}"


def _fmt_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm"""
    millis = int((seconds % 1) * 1000)
    total_seconds = int(seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"


def build_servers(host: str, base_port: int, count: int) -> List[str]:
    return [f"http://{host}:{base_port + i}" for i in range(count)]


def filter_healthy(servers: List[str], timeout: float = HEALTH_TIMEOUT, interval: float = 1.0) -> List[str]:
    ok = []
    start = time.time()
    pending = set(servers)
    while pending and (time.time() - start) < timeout:
        for s in list(pending):
            try:
                r = requests.get(f"{s}/health", timeout=2)
                if r.status_code == 200:
                    ok.append(s)
                    pending.remove(s)
            except Exception:
                pass
        if pending:
            time.sleep(interval)
    return ok


def create_env(base_url: str, env_kwargs: Optional[List[dict]] = None) -> str:
    payload = {"count": 1}
    if env_kwargs is not None:
        payload["env_kwargs"] = env_kwargs
    r = requests.post(f"{base_url}/batch/envs", json=payload, timeout=30)
    r.raise_for_status()
    env_id = r.json()["env_ids"][0]
    print(f"[create] {base_url} -> env {env_id}")
    return env_id


def wait_reset(base_url: str, env_id: str, timeout: int = 600):
    url = f"{base_url}/envs/{env_id}/reset_result"
    start = time.time()
    while True:
        r = requests.get(f"{base_url}/envs/{env_id}/reset_result")

        if r.status_code == 200:
            elapsed = time.time() - start
            print(f"[reset] {env_id[:8]} ready, elapsed={_fmt_duration(elapsed)}")
            data = r.json()
            obs = data["observation"]  # 里有 rgb 等键
            info = data["info"]        # 里包含环境初始化的附加信息
            return obs, info
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"reset timeout {env_id}")
        time.sleep(1)




def step_async(base_url: str, env_id: str, step_idx: int, action: str = "turn_right"):
    submit_payload = {"action": action, "step_idx": step_idx, "client_ts": time.time()}
    t_send = time.time()
    requests.post(f"{base_url}/envs/{env_id}/step_async", json=submit_payload, timeout=10).raise_for_status()
    while True:
        poll = requests.get(f"{base_url}/envs/{env_id}/get_step_async", params={"step_idx": step_idx}, timeout=10)
        data = poll.json()
        status = data.get("status")
        if status == "success":
            elapsed = time.time() - t_send
            obs = data.get("observation", None)
            reward = data.get("reward", 0)
            terminated = data.get("terminated", False)
            timings = data.get("info", {}).get("timings", {})
            # 将时间戳字段转为可读格式
            def _fmt_time_val(val: Any) -> Any:
                if isinstance(val, (int, float)):
                    return _fmt_ts(val)
                return val
            timings_fmt = {k: _fmt_time_val(v) for k, v in timings.items()} if isinstance(timings, dict) else timings
            print(f"[step] env={env_id[:8]} idx={step_idx} reward={reward} term={terminated} elapsed={_fmt_duration(elapsed)} timings={json.dumps(timings_fmt)}")
            # return data, elapsed
            return obs, reward, terminated, False, timings
        if status == "pending":
            time.sleep(0.5)
            continue
        raise RuntimeError(f"step error: {data}")


def close_env(base_url: str, env_id: str):
    r = requests.post(f"{base_url}/envs/{env_id}/close", timeout=10)
    if r.status_code in (200, 202, 204):
        print(f"[close] {env_id[:8]} ok")
    else:
        print(f"[close] {env_id[:8]} failed: {r.text}")




def fetch_envs_from_center(center_url: str, count: int, env_kwargs: Optional[List[dict]] = None) -> List[Tuple[str, str]]:
    """Call center server /create_envs and return [(base_url, env_id), ...]."""
    payload = {"count": count}
    if env_kwargs is not None:
        payload["env_kwargs"] = env_kwargs
    r = requests.post(f"{center_url}/create_envs", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    envs = data.get("envs", [])
    return [(item["port"], item["env_id"]) for item in envs]

