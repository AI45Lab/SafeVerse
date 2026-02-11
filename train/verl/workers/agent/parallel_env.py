import re
import io
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from verl import DataProto
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.vision_utils import process_image, process_raw_image, process_video
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.workers.agent.tool_envs import ToolBase
import uuid
from PIL import Image
import copy
import os
from collections import Counter
import random

random.seed(42)

def _strip_system_block(text: str) -> str:
    """删除 text 中第一个 <|im_start|>system ... <|im_end|> 区块"""
    pattern = r"<\|im_start\|>system.*?<\|im_end\|>"
    result = re.sub(pattern, "", text, flags=re.S)
    return result


def _concat_vllm_input(prompt_token_ids, response_token_ids, tokenizer=None):
    """拼接 prompt 和 response token ids"""
    if tokenizer is not None:
        max_token_id = max(tokenizer.get_vocab().values())
        tokenizer_size = len(tokenizer)
        max_token_id = max(max_token_id, tokenizer_size)
        valid_token_mask = torch.le(response_token_ids, max_token_id)
        response_token_ids = torch.masked_select(response_token_ids, valid_token_mask)

    if isinstance(prompt_token_ids, torch.Tensor):
        output_tensor = torch.cat([
            prompt_token_ids,
            response_token_ids.to(prompt_token_ids.device),
        ], dim=-1)
        return output_tensor.cpu().numpy().flatten().tolist()
    else:
        output_array = np.concatenate([
            prompt_token_ids,
            response_token_ids.cpu().numpy(),
        ], axis=-1)
        return output_array.flatten().tolist()


def _merge_multi_modal_inputs(mm_input, other):
    """合并两个多模态输入字典"""
    if not mm_input and not other:
        return {}
    elif len(mm_input) == 0 and len(other) > 0:
        return other
    elif len(mm_input) > 0 and len(other) == 0:
        return mm_input

    output_dict = {}
    for key in mm_input.keys():
        if key not in other.keys():
            output_dict[key] = mm_input[key]
            continue

        mm_value = mm_input[key]
        other_value = other.pop(key)
        if isinstance(mm_value, np.ndarray) and isinstance(other_value, np.ndarray):
            merged_value = np.concatenate([mm_value, other_value], axis=0)
        elif isinstance(mm_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
            merged_value = torch.cat([mm_value, other_value], dim=0)
        else:
            raise ValueError(f"Invalid {type(mm_value)=}, {type(other_value)=}")

        output_dict[key] = merged_value
    return dict(**output_dict, **other)


def _preprocess_multi_modal_inputs(prompt_str, processor, **kwargs):
    """预处理多模态输入"""
    if processor is None or "multi_modal_data" not in kwargs:
        return prompt_str, prompt_str, {}

    vllm_input_prompt = prompt_str.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    input_mm_data = kwargs.get("multi_modal_data", {"image": []})
    image_info_list = []

    for idx, img in enumerate(input_mm_data["image"]):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        buf = io.BytesIO()
        arr = np.array(img, dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        buf.close()
        img_info = {"bytes": png_bytes}
        image_info_list.append(img_info)

    # 编码成bytes的图片才可以进行process_image，input_mm_data["image"]是一个list
    input_mm_data["image"] = [process_image(img) for img in image_info_list]

    model_inputs = processor(text=[vllm_input_prompt], images=input_mm_data["image"], return_tensors="pt")
    input_ids = model_inputs.pop("input_ids")[0]
    attention_mask = model_inputs.pop("attention_mask")[0]

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    mm_inputs = dict(model_inputs)
    # vllm_input_prompt是文字，input_ids是数字ids
    return vllm_input_prompt, input_ids, mm_inputs


def call_memory_agent(previous_obs, current_obs, response_text, current_memory, processor, tokenizer):
    """
    调用 memory agent 生成新的记忆和思考内容
    
    Args:
        previous_obs: 之前观测的图片的列表
        current_obs: 当前观测（图片）
        current_memory: 当前记忆内容（字符串）
        processor: 图像处理器
        tokenizer: 分词器
    
    Returns:
        new_memory: 更新后的记忆内容
        new_think: 新的思考内容
    """
    # TODO: 这里需要你实现 memory agent 的调用逻辑
    # 示例接口：
    # new_memory, new_think = your_memory_agent(
    #     previous_obs=previous_obs,
    #     observation=current_obs,
    #     memory=current_memory,
    #     prompt=original_prompt
    # )
    if previous_obs == current_obs and 'operate' in response_text.lower():
        new_memory = (
            "The current observation appears nearly identical to the previous one (no discernible change in layout, objects, or perspective), you are likely stuck. "
            "Consider the following possibilities and take appropriate actions:\n"
            "1. You might be too far away from the object, which is beyond the interaction range. Move closer to the object.\n"
            "2. Your current perspective might be too skewed, and the target object is not visible in the current view, preventing selection. Adjust your rotation to get a better view.\n"
            "3. The object might be non-interactive, meaning it does not have the necessary properties for interaction. Check if there are other objects that can be used to achieve your goal.\n"
            "Based on the context, determine the most likely scenario and execute the corresponding action."
        )
        new_think = ""
    elif previous_obs == current_obs and 'move_away' in response_text.lower():
        new_memory = (
            "The current observation appears nearly identical to the previous one (no discernible change in layout, objects, or perspective), you are likely stuck. "
            "Consider the following possibilities and take appropriate actions:\n"
            "1. You might be too far away from the object, which is beyond the interaction range. Move closer to the object.\n"
            "2. Your current perspective might be too skewed, and the target object is not visible in the current view, preventing selection. Adjust your rotation to get a better view.\n"
            "3. The object might be non-interactive, meaning it does not have the necessary properties for interaction. Check if there are other objects that can be used to achieve your goal.\n"
            "Based on the context, determine the most likely scenario and execute the corresponding action."
        )
        new_think = ""
    else:
        new_memory = ""
        new_think = ""
    
    return new_memory, new_think


def compute_time_decay_rewards(trajectory_rewards, gamma=0.99):
    """
    计算时间衰减奖励: r_t = γ^(T-t) · R_T
    
    Args:
        trajectory_rewards: 原始奖励列表
        gamma: 衰减因子
    
    Returns:
        decayed_rewards: 衰减后的奖励列表
    """
    T = len(trajectory_rewards)
    final_reward = trajectory_rewards[-1]  # R_T
    
    decayed_rewards = []
    for t in range(T):
        r_t = (gamma ** (T - t - 1)) * final_reward
        decayed_rewards.append(r_t)
    
    return decayed_rewards


def agent_rollout_loop(worker_rank, config, vllm_engine, vllm_inputs, prompts, multi_modal_inputs, sampling_params):
    """主要的 rollout 循环，支持 memory agent"""
    from vllm.distributed import parallel_state as vllm_ps

    agent_sampling_params = sampling_params.clone()
    agent_sampling_params.detokenize = True
    agent_sampling_params.skip_special_tokens = False
    agent_sampling_params.spaces_between_special_tokens = False
    agent_sampling_params.n = 1
    agent_sampling_params.include_stop_str_in_output = True
    max_generated_tokens = min(config.agent.single_response_max_tokens, config.response_length)
    agent_sampling_params.max_tokens = max_generated_tokens

    custom_stop = list(config.agent.custom_stop)
    if custom_stop:
        prev_stop = sampling_params.stop if sampling_params.stop else []
        agent_sampling_params.stop = prev_stop + custom_stop

    tokenizer = hf_tokenizer(config.agent.vl_model_path)
    processor = hf_processor(config.agent.vl_model_path)

    if multi_modal_inputs is not None:
        multi_modal_inputs = multi_modal_inputs.tolist()
    else:
        multi_modal_inputs = [{}] * len(vllm_inputs)

    batch_size = len(vllm_inputs)
    
    # ===== Memory Agent 相关状态 =====
    memory_contents = []  # 每个样本的记忆内容
    think_contents = []   # 每个样本的思考内容
    steps_since_memory_call = []  # 距离上次调用 memory agent 的步数
    original_prompt_ids = []  # 原始任务指令
    original_prompt_mask = []
    original_vllm_input_list = []
    original_reward_tensor_list = []
    original_mm_input_list = []

    # 分段数据收集
    segment_data = []  # 存储每段数据: [{samples: [...], final_reward: ...}, ...]
    segment_counts = []  # 记录每个样本已经产生的segment数量
    # =====================================
    
    vllm_input_list = []
    running_states = []
    running_action_masks = []
    running_attn_masks = []
    reward_tensor_list = []
    active_mask = []
    mm_input_list = []
    tool_call_cnt_list = []

    # 初始化
    print(f"[DEBUG] {batch_size=}")
    for i in range(batch_size):
        for j in range(sampling_params.n):
            segment_counts.append(0)
            vllm_input_list.append(deepcopy(vllm_inputs[i])) # 带占位符的ids，不是真实图片的ids

            prompt_ids = prompts.batch['input_ids'][i, :].clone() # prompt_ids是带有了图片ids的
            running_states.append(prompt_ids)
            prompt_mask = prompts.batch['attention_mask'][i, :].clone()

            running_action_masks.append(prompt_mask)
            running_attn_masks.append(prompt_mask)

            reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
            reward_tensor_list.append(reward_tensor)

            active_mask.append(True)
            mm_input_list.append(deepcopy(multi_modal_inputs[i]))
            tool_call_cnt_list.append(0)
            
            # 初始化 memory agent 状态
            memory_contents.append("")
            think_contents.append("")
            steps_since_memory_call.append(0)

            # 保存原始 prompt
            original_prompt_ids.append(prompt_ids)
            original_vllm_input_list.append(deepcopy(vllm_inputs[i]))
            original_prompt_mask.append(prompt_mask)
            original_reward_tensor_list.append(reward_tensor)
            original_mm_input_list.append(deepcopy(multi_modal_inputs[i]))
           

    pg = vllm_ps.get_tp_group()
    max_total_length = config.prompt_length + config.response_length
    
    # 初始化环境
    env = ParallelEnv(config.agent, tokenizer, processor)
    obs_results = env.reset(prompts, vllm_inputs, n=sampling_params.n)


    observations, rewards, dones, info = obs_results
    active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]
    active_vllm_inputs = [vinput for vinput, is_active in zip(vllm_input_list, active_mask) if is_active]
    actions = [None] * len(active_vllm_inputs)
    
    # 假设一开始的时候需要调用memory agent，如果不用，直接注释掉就好
    for idx in active_indices:
        obs = observations[idx]

        # 直接复用后面should_call_memory分支的代码，唯一不同的所有original_vllm_input_list使用了不带original_vllm前缀的，因为现在就是original的情况
        current_obs_image = obs.get('multi_modal_data', {}).get('image', [None])[0]
        if current_obs_image is not None:
            new_memory, new_think = "", ""
            # new_memory, new_think = call_memory_agent(
            #     previous_obs=None,
            #     current_obs=current_obs_image,
            #     current_memory=memory_contents[idx],
            #     processor=processor,
            #     tokenizer=tokenizer
            # )
            memory_contents[idx] = new_memory
            think_contents[idx] = new_think
            steps_since_memory_call[idx] = 0

            # 使用新的上下文了，需要重新处理prompt并把文字和图片进行编码
            if 'multi_modal_data' in vllm_input_list[idx]:
                if "image" not in vllm_input_list[idx]['multi_modal_data']:
                    vllm_input_list[idx]['multi_modal_data'] = {"image": []}
                vllm_input_list[idx]['multi_modal_data']['image'].append(current_obs_image)

            # 在末尾添加vllm_input_list[idx] 的输入文本
            prompt_str =  "<|im_start|>user\n" + new_memory + new_think + "Your observation: <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n" + "<|im_start|>assistant\n"

            new_vllm_input_prompt, new_input_ids, new_mm_inputs = _preprocess_multi_modal_inputs(
                prompt_str, processor, 
                multi_modal_data={"image": [current_obs_image]}  # 正确格式
            )
            # 这里是占位符的
            new_vllm_input_prompt_ids = tokenizer.encode(new_vllm_input_prompt, add_special_tokens=False, return_tensors='pt')[0]


            vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                vllm_input_list[idx]['prompt_token_ids'], 
                new_vllm_input_prompt_ids, 
                tokenizer=tokenizer
            )

            running_states[idx] = torch.cat([original_prompt_ids[idx], new_input_ids.to(running_states[idx].device)])

            # prompt给出的环境观测是不能当做模型的回答的，不能参与到策略梯度的更新的
            new_action_mask = torch.zeros_like(new_input_ids, dtype=torch.int64, device=running_action_masks[idx].device)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], new_action_mask])

            new_attn_mask = torch.ones_like(new_input_ids, dtype=torch.int64, device=running_attn_masks[idx].device)
            running_attn_masks[idx] = torch.cat([running_attn_masks[idx], new_attn_mask])

            # 新添加的部分的reward_tensor部分是 0，用原本的reward_tensor_list concat 上新的部分，就能实现更新
            new_reward_tensor = torch.zeros_like(new_input_ids, dtype=torch.float, device=running_action_masks[idx].device)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], new_reward_tensor])

            mm_input_list[idx] = _merge_multi_modal_inputs(mm_input_list[idx], new_mm_inputs)

            print(f"[DEBUG Memory] Sample {idx}: Initial memory and think generated")

    print(f"[DEBUG mc env] initialization with success!")

    for step in range(config.agent.max_turns):
        print(f'[DEBUG] Step {step}: total={batch_size}, n={sampling_params.n}, num_active={sum(active_mask)}')
        
        if sum(active_mask) == 0:
            break

        active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]
        active_vllm_inputs = [deepcopy(vllm_input_list[idx]) for idx in active_indices]
        prompt_txt_0 = tokenizer.decode(active_vllm_inputs[0].get("prompt_token_ids", None), add_special_tokens=False)
        print(f'[DEBUG] at step {step}, {prompt_txt_0=}') # 我想确认一下它的image token是不是没有

        # 生成下一步动作        
        actions = vllm_engine.generate(
            prompts=active_vllm_inputs,
            sampling_params=agent_sampling_params,
            use_tqdm=False
        )
        
        obs_results = env.step(active_indices, actions, vllm_input_list)
        observations, rewards, dones, info = obs_results # 这里的obs是新的obs，action 导致了 obs, 如果action是think， obs不是导致think的图像，是think之后的图像


        # 把上一轮的模型输出整合到上下文中
        # 并且把动作导致的图片整合到inputs中，用于下一次输入
        for idx, obs, act, rew, done in zip(active_indices, observations, actions, rewards, dones):
            # =================整合上一轮模型输出======================
            print(f'[DEBUG] Processing sample {idx}')
            should_call_memory = False
            
            # 把模型上一轮的输出文字拼接到上下文中
            response_token_ids = torch.tensor(
                act.outputs[0].token_ids,
                dtype=torch.int64,
                device=running_states[idx].device
            )
            response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
            print(f'[DEBUG] {idx=}, {step=}, {response_text=}')


            # 这部分是把response拼接到历史中，可能加上response就超长了，此时不应该把response加进去
            # TODO: 需要检查是否结束，如果结束，则需要把当前的内容拼接到上下文，然后收集为一个数据
            if running_states[idx].shape[-1] + len(response_token_ids) >= max_total_length or len(vllm_input_list[idx]['prompt_token_ids']) + len(response_token_ids) >= max_total_length:
                print(f"[DEBUG done] {idx=} inactive because exceeds max_total_length.")
                print(f'[DEBUG] {len(running_states[idx])=}, {len({response_token_ids})=}, {max_total_length=}')
                print(f'[DEBUG] The exceeded response is {response_text}.') # 我比较好奇超长的东西是什么。
                active_mask[idx] = False
                # env.close_game(idx)
                
                continue

            running_states[idx] = torch.cat([running_states[idx], response_token_ids])

            # vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
            #     vllm_input_list[idx]['prompt_token_ids'],
            #     response_token_ids,
            #     tokenizer=tokenizer,
            # )

            # 标记出大模型输出的tokens是哪些
            action_mask = torch.ones_like(response_token_ids, dtype=torch.int64, device=running_action_masks[idx].device)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])

            # 标记出非空白的tokens是哪些
            running_attn_masks[idx] = torch.cat([running_attn_masks[idx], action_mask])

            # 设置一个和模型回复一样长的全0 tensor，然后在末尾加上本轮环境的reward
            action_reward = torch.zeros_like(response_token_ids, dtype=torch.float, device=reward_tensor_list[idx].device)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], action_reward])
            if rew >= 1:
                print(f'[REWARD] {idx=}, {step=}, {rew=}')
            reward_tensor_list[idx][-1] += rew # rew放在action的位置而不是图片的位置了，都一样其实

            if done or step == config.agent.max_turns - 1:
                print(f"[DEBUG done] {idx=}, {step=}, {done=}")
                active_mask[idx] = False
                # env.close_game(idx)
                continue

            # ==================================================

                        
            # ===== 检测是否需要调用 memory agent =====
            # if '<think>' in response_text:
            #     print(f"[DEBUG Memory] Sample {idx}: Detected <think> in response")
            #     should_call_memory = True
            # elif steps_since_memory_call[idx] >= config.agent.segment_len - 1: 
            #     print(f"[DEBUG Memory] Sample {idx}: {config.agent.segment_len} steps reached, forcing memory call")
            #     should_call_memory = True
            if steps_since_memory_call[idx] >= config.agent.segment_len - 1: 
                print(f"[DEBUG Memory] Sample {idx}: {config.agent.segment_len} steps reached, forcing memory call")
                should_call_memory = True
                        
            else:
                steps_since_memory_call[idx] += 1
            

            # 如果调用memory agent就让memory agent还原初始设置，然后把当前的think和图片加到上下文中
            # 如果没调用就正常把环境反馈的文字和图片加到上下文中
            if should_call_memory:
                # 本质上就是运行了一次清空上下文，重置环境
                # 结束当前段，保存整段数据到 segment_data
                

                
                segment_data.append({
                    'segment_idx': segment_counts[idx],
                    'prompt_ids_idx': env.get_data_idx(idx),
                    'rollout_idx': idx,
                    'states': running_states[idx].clone(),
                    'action_mask': running_action_masks[idx].clone(),
                    'attn_mask': running_attn_masks[idx].clone(),
                    'rewards': reward_tensor_list[idx].clone(),
                    'mm_inputs': deepcopy(mm_input_list[idx]), # mm_inputs主要是图片的位置编码
                })

                segment_counts[idx] += 1

                prompt_and_response_str = tokenizer.decode(running_states[idx], add_special_tokens=False, skip_special_tokens=False)
                
                # 调用 memory agent
                current_obs_image = obs.get('multi_modal_data', {}).get('image', [None])[0]
                if current_obs_image is not None:
                    # 调用memory agent应该把过去所有的图片输入给agent，不仅仅是当前的
                    # 需要完成初始化加上添加两部分动作
                    new_memory, new_think = call_memory_agent(
                        previous_obs=vllm_input_list[idx]['multi_modal_data']['image'],
                        current_obs=current_obs_image,
                        response_text=response_text,
                        current_memory=memory_contents[idx],
                        processor=processor,
                        tokenizer=tokenizer
                    )

                    memory_contents[idx] = new_memory
                    think_contents[idx] = new_think
                    steps_since_memory_call[idx] = 0

                    vllm_input_list[idx] = deepcopy(original_vllm_input_list[idx]) # prompt_ids是原本输入的文本和图片, 这里需要deepcopy，因为原来的是一个字典，直接幅值无法刷新image list

                    # 在末尾添加vllm_input_list[idx] 的输入文本
                    # 这部分不算入模型的action
                    prompt_str =  "<|im_start|>user\n" + new_memory + new_think + "Your observation: <|vision_start|><|image_pad|><|vision_end|>\n"+"<|im_end|>\n<|im_start|>assistant\n"

                    # 因为这里不用obs返回的文本+图片，因此需要重新算一次mm_inputs
                    new_vllm_input_prompt, new_input_ids, new_mm_inputs = _preprocess_multi_modal_inputs(
                        prompt_str, processor, 
                        multi_modal_data={"image": [current_obs_image]}  # 正确格式
                    )
                    # 这里是占位符的
                    new_vllm_input_prompt_ids = tokenizer.encode(new_vllm_input_prompt, add_special_tokens=False, return_tensors='pt')[0]
                    vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                        vllm_input_list[idx]['prompt_token_ids'], 
                        new_vllm_input_prompt_ids, 
                        tokenizer=tokenizer
                    )

                   # 使用新的上下文了，需要重新处理prompt并把文字和图片进行编码
                    if 'multi_modal_data' in vllm_input_list[idx]:
                        if "image" not in vllm_input_list[idx]['multi_modal_data']:
                            vllm_input_list[idx]['multi_modal_data'] = {"image": []}
                        vllm_input_list[idx]['multi_modal_data']['image'].append(current_obs_image)
                        # 我需要在这里检测为什么会超过image_list的长度，理论上这里会因为重置，不会超过20张
                        print(f"[DEBUG] {len(vllm_input_list[idx]['multi_modal_data']['image'])=}")


                    # 这里是否有重复拼接？
                    mm_input_list[idx] = deepcopy(original_mm_input_list[idx]) # 同vllm_input_list，是一个字典，需要deepcopy
                    mm_input_list[idx] = _merge_multi_modal_inputs(mm_input_list[idx], new_mm_inputs)


                    # 这里最好也有一个检查，防止超过max_token_len

                    running_states[idx] = torch.cat([original_prompt_ids[idx], new_input_ids.to(running_states[idx].device)])


                    # prompt给出的环境观测是不能当做模型的回答的，不能参与到策略梯度的更新的
                    new_action_mask = torch.zeros_like(new_input_ids, dtype=torch.int64, device=running_action_masks[idx].device)
                    running_action_masks[idx] = torch.cat([original_prompt_mask[idx], new_action_mask])

                    new_attn_mask = torch.ones_like(new_input_ids, dtype=torch.int64, device=running_action_masks[idx].device)
                    running_attn_masks[idx] = torch.cat([original_prompt_mask[idx], new_attn_mask]) # 图片的tokens不能被mask

                    # 新添加的部分的reward_tensor部分是 0，用原本的reward_tensor_list concat 上新的部分，就能实现更新
                    new_reward_tensor = torch.zeros_like(new_input_ids, dtype=torch.float, device=running_action_masks[idx].device)
                    reward_tensor_list[idx] = torch.cat([original_reward_tensor_list[idx], new_reward_tensor])

                    
                        
            else:                
                tool_call_cnt_list[idx] += 1

                # 处理观测, 如果结束了，obs就没有这些关键词，就进不到这个if分支
                if 'prompt_token_ids_vllm' in obs.keys() and 'prompt_token_ids_model' in obs.keys():
                    obs_token_ids_vllm = obs['prompt_token_ids_vllm']
                    obs_token_ids_model = obs['prompt_token_ids_model'].to(running_states[idx].device)
                    # print(f'[DEBUG] {obs_token_ids_vllm=}, {obs_token_ids_model=}')

                    # 这里判断太早了，要这个obs和接下来的回答都不超过再加才合理。
                    if len(vllm_input_list[idx]['prompt_token_ids']) + len(obs_token_ids_vllm)  >= max_total_length:
                        active_mask[idx] = False
                        # env.close_game(idx)
                        continue
                        
                    if running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length:
                        active_mask[idx] = False
                        # env.close_game(idx)
                        continue

                    # 没有结束才把更新输入内容
                    running_states[idx] = torch.cat([running_states[idx], obs_token_ids_model])

                    # obs的tokens不是大模型的输出，action_masks设置为0，但是是非空的，attn_masks设置为1
                    obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64, device=running_action_masks[idx].device)
                    running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
                    
                    attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64, device=running_attn_masks[idx].device)
                    running_attn_masks[idx] = torch.cat([running_attn_masks[idx], attn_mask])

                    # 观测部分不设置reward，应该设置为全0
                    # 我不确定如果env结束了，发过来的obs还有图片吗
                    obs_reward = torch.zeros(len(obs_token_ids_model), dtype=torch.float, device=reward_tensor_list[idx].device)
                    reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward], dim=-1)

                    
                    # vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                    #     vllm_input_list[idx]['prompt_token_ids'], 
                    #     obs_token_ids_vllm,
                    #     tokenizer=tokenizer,
                    # )

                    mm_data = obs.get('multi_modal_data', {}) # 但是这里还是延续上一轮的images，并不是新的images
                    if 'image' in mm_data.keys():
                        if 'multi_modal_data' not in vllm_input_list[idx].keys() or "image" not in vllm_input_list[idx]['multi_modal_data']:
                            vllm_input_list[idx]['multi_modal_data'] = {"image": []}
                        # vllm_input_list[idx]['multi_modal_data']['image'] += mm_data['image']
                        vllm_input_list[idx]['multi_modal_data']['image'] = mm_data['image']

                    mm_input = obs.get('multi_modal_inputs', {})
                    if mm_input:
                        # vllm_input_list[idx]["multi_modal_inputs"] = mm_input
                        mm_input_list[idx] = _merge_multi_modal_inputs(mm_input_list[idx], mm_input) # 这里不能贸然换掉，因为是deepcopy
                            


    # ===== 收集最终段数据 =====
    # 将最后一轮还在运行的样本加入 segment_data
    for idx in range(len(running_states)):

        
        segment_data.append({
            'segment_idx': segment_counts[idx],
            'prompt_ids_idx': env.get_data_idx(idx), # prompt_ids_idx用于区分不同的独立样本
            'rollout_idx': idx, # rollout_idx 用于区分不同的rollout，同一个轨迹所用的rollout_idx想通
            'states': running_states[idx].clone(),
            'action_mask': running_action_masks[idx].clone(),
            'attn_mask': running_attn_masks[idx].clone(),
            'rewards': reward_tensor_list[idx].clone(),
            'mm_inputs': deepcopy(mm_input_list[idx]),
        })

        segment_counts[idx] += 1

    # **新增：填充法，计算需要填充的空白segments数量**
    expected_segments = config.agent.max_turns // config.agent.segment_len

    for idx in range(len(segment_counts)):
        current_segments = segment_counts[idx]
        segments_to_fill = expected_segments - current_segments

        print(f"[DEBUG FILL] {idx=} needs {segments_to_fill} empty segments (current={current_segments}, expected={expected_segments})")

        for _ in range(segments_to_fill):
            segment_data.append({
                'segment_idx': segment_counts[idx] - 1, # 这里减一是为了还原上一个segment的idx
                'prompt_ids_idx': env.get_data_idx(idx),
                'rollout_idx': idx,
                'states': running_states[idx].clone(),  # 使用超长之前的状态
                'action_mask': running_action_masks[idx].clone(),
                'attn_mask': running_attn_masks[idx].clone(),
                'rewards': reward_tensor_list[idx].clone(),
                'mm_inputs': deepcopy(mm_input_list[idx]),
            })

    # ===== 对 segment_data 按 idx 排序 =====
    segment_data.sort(key=lambda x: x['prompt_ids_idx'])

    # print(f'[DEBUG] Rollout loop finished, close the envs')
    # env.close()
    

    # 首先，按rollout_idx分组segment, 并收集每个segment是属于哪个prompt
    rollout_groups = {}
    prompt_ids_idx_list = []
    for segment in segment_data:
        prompt_ids_idx_list.append(segment['prompt_ids_idx'])
        rollout_idx = segment['rollout_idx']
        if rollout_idx not in rollout_groups:
            rollout_groups[rollout_idx] = []
        rollout_groups[rollout_idx].append(segment)

    # 打印每个 rollout 有多少个 segment
    # 需要确保轨迹得到的segment不一样的时候是否还能通过pad让dp正确运行
    for rollout_idx, segments in rollout_groups.items():
        print(f"Rollout {rollout_idx}: {len(segments)} segments")

    for rollout_idx, segments in rollout_groups.items():
        if not segments:
            print(f"[WARN] Rollout {rollout_idx} has no segments")
            continue

        max_reward = -float('inf')
        terminate_idx = None
        for segment in segments:
            segment_max = segment['rewards'].max().item()
            if segment_max > max_reward:
                max_reward = segment_max
                terminate_idx = segment['segment_idx']

        if max_reward >= 1.0 and terminate_idx is not None:
            for segment in segments:
                if segment['segment_idx'] == terminate_idx:
                    print(f"[DEBUG Reward] End segment (idx={terminate_idx}) of successful rollout {rollout_idx}, max reward: {max_reward:.4f}")
                else:
                    delta = terminate_idx - segment['segment_idx']
                    if delta < 0:
                        # 可选：跳过或警告
                        print(f"[WARN] Segment idx {segment['segment_idx']} > terminate_idx {terminate_idx} in rollout {rollout_idx}")
                        continue
                    bonus = (0.99 ** delta) * max_reward
                    segment['rewards'][-1] += bonus
                    print(f"[DEBUG Reward] {env.get_uuid(rollout_idx)}, Added bonus {bonus:.4f} to segment {segment['segment_idx']} end (rollout {rollout_idx}), new end reward: {segment['rewards'][-1].item():.4f}")
        else:
            print(f"[DEBUG Reward] Rollout {rollout_idx} did not complete (max_reward={max_reward:.4f})")
    print(f"[DEBUG Segment] Total segments collected: {len(segment_data)}")



    # ===== 将 segment_data 转换为 batch 格式 =====
    target_device = prompts.batch['input_ids'].device
    
    image_token_id = 151655  # Qwen2VL image token ID


    # 我需要知道为什么第一个样本的image tokens为什么到怎么后面
    # 因此我需要打印出每个samples的state_ids的前50个内容（需要进行解码，形成字符串的形式）
    # 我还需要想办法确定max_total_length为多少长比较合适
    # 现在这个版本预期得到的裁剪是正常的
    # for i, seg in enumerate(segment_data):
    #     state_ids = seg['states']
    #     mm_input = seg['mm_inputs']

    #     # ---- NEW: decode 前 50 个 token ----
    #     ### NEW: decode token ids -> text
    #     try:
    #         decoded_prefix = tokenizer.decode(state_ids[:50], add_special_tokens=False)
    #     except:
    #         decoded_prefix = "<decode error>"
    #     print(f"    decoded_prefix(前50): {decoded_prefix}")

    #     # ---- NEW: 打印整个序列总长度，用于评估 max_total_length ----
    #     seq_len = len(state_ids)
    #     print(f"    sequence_length:      {seq_len}")

    #     # 原有检查 image token 逻辑
    #     state_tensor_ids = torch.tensor(state_ids)
    #     image_positions = (state_tensor_ids == image_token_id).nonzero(as_tuple=True)[0].tolist()

    #     contains_image_token = len(image_positions) > 0
    #     image_token_count = len(image_positions)

    #     # 检查 mm_inputs 是否包含 image grid
    #     has_image_grid = 'image_grid_thw' in mm_input and mm_input['image_grid_thw'] is not None

    #     print(f"─ in parallel env, Sample {i}:")
    #     print(f"    contains_image_token: {contains_image_token}")
    #     print(f"    image_token_count:    {image_token_count}")
    #     print(f"    positions:            {image_positions[:20]}{'...' if len(image_positions) > 20 else ''}")
    #     print(f"    has_image_grid:       {has_image_grid}")
        # if has_image_grid:
            # print(f"    image_grid_thw:       {mm_input['image_grid_thw']}")


    # 适当裁剪看能不能跑通
    
    # 长度变成原来的四分之一
    # segment_data = segment_data[len(segment_data) // 2:]
    # segment_data = segment_data[len(segment_data) // 2:]
    # prompt_ids_idx_list = prompt_ids_idx_list[len(prompt_ids_idx_list) // 2:]
    # prompt_ids_idx_list = prompt_ids_idx_list[len(prompt_ids_idx_list) // 2:]
    # 从 segment_data 中提取各个字段
    segment_states = [seg['states'][:max_total_length] for seg in segment_data]
    segment_action_masks = [seg['action_mask'][:max_total_length] for seg in segment_data]
    segment_attn_masks = [seg['attn_mask'][:max_total_length] for seg in segment_data]
    segment_rewards = [seg['rewards'][:max_total_length] for seg in segment_data]
    segment_mm_inputs = [seg['mm_inputs'] for seg in segment_data]
    
    # 转换为tensor
    state_tensor = pad_2d_list_to_length(segment_states, tokenizer.pad_token_id, max_total_length).to(target_device)
    action_mask_tensor = pad_2d_list_to_length(segment_action_masks, 0, max_total_length).to(target_device)
    attn_mask_tensor = pad_2d_list_to_length(segment_attn_masks, 0, max_total_length).to(target_device)
    reward_tensor = pad_2d_list_to_length(segment_rewards, 0.0, max_total_length).to(target_device)
    
    # 计算 position_ids
    if processor is not None and processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
        position_ids_list = [
            get_rope_index(
                processor,
                input_ids=state_tensor[i, :],
                image_grid_thw=segment_mm_inputs[i].get("image_grid_thw", None),
                video_grid_thw=segment_mm_inputs[i].get("video_grid_thw", None),
                second_per_grid_ts=segment_mm_inputs[i].get("second_per_grid_ts", None),
                attention_mask=attn_mask_tensor[i, :],
            ) for i in range(len(segment_data))
        ]
        position_ids_tensor = torch.stack(position_ids_list, dim=0)
    else:
        position_ids_tensor = compute_position_id_with_mask(attn_mask_tensor)
    
    print(f"[DEBUG segment batch] Total segments in batch: {len(segment_data)}")
    print(f"[DEBUG segment batch] state_tensor shape: {state_tensor.shape}")
    
    # 计算 response 部分（从后往前取 response_length）
    response_tensor = state_tensor[:, -config.response_length:]
    env_reward_tensor = reward_tensor[:, -config.response_length:]
    
    print(f"[DEBUG reward] Final reward statistics:")
    reward_sums = env_reward_tensor.sum(dim=1)
    nonzero_mask = reward_sums != 0
    print(f"[DEBUG reward] {nonzero_mask.sum().item()} segments with non-zero rewards")
    print(f"[DEBUG reward] Non-zero indices: {nonzero_mask.nonzero(as_tuple=True)[0].tolist()}")

    # 这里打印出来的最终成功的片段的reward不一定是在末尾，因为前面sort的时候大家的prompt_ids_idx一样，最终成功的片段可能排到前面去了
    # 打印出来不同segment的奖励不同是正常的，代表着模型的空动作数量不同。
    # print(f"[DEBUG reward] Reward sums: {reward_sums[nonzero_mask]}")
    
    return DataProto.from_dict(
        tensors={
            "response": response_tensor, # 这里可能没有把image的占位token加进去
            "action_mask": action_mask_tensor,
            "attention_mask": attn_mask_tensor,
            "position_ids": position_ids_tensor,
            "env_reward": env_reward_tensor,
        },
        non_tensors={
            "multi_modal_inputs": segment_mm_inputs,
            "prompt_ids_idx": prompt_ids_idx_list # [0， 0， 0， 1， 1， 1]
        } if processor is not None else None,
    )


# ===== 保留原有的辅助函数 =====
def execute_tool_call(sample, vllm_input_subidx, config=None, tokenizer=None, processor=None, pbar=None):
    """执行工具调用"""
    action_string = sample.get('action', '')
    tool = sample.get('tool', None)

    if action_string == '' or tool is None:
        return {}, 0.0, True, {}

    obs, reward, done, info = tool.execute(action_string, config)

    # 更新image_list
    # img_list = vllm_input_subidx['multi_modal_data']['image']
    # img_list.append(obs['image'])


    # 保存图像的代码（如果需要）
    # from datetime import datetime
    # import os
    # output_dir = f"/your/path/{tool.uuid}"
    # os.makedirs(output_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    # if isinstance(obs['image'], np.ndarray):
    #     img_to_save = Image.fromarray(obs['image'])
    #     save_path = os.path.join(output_dir, f"{timestamp}.png")
    #     img_to_save.save(save_path)
    #     print(f"[INFO] Saved observation image to {save_path}")

    tool_result = {
        "prompt": "<|im_start|>user\n"+ "Your observation: <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n" + "<|im_start|>assistant\n",
        "multi_modal_data": {"image": [obs['image']]}
    }

    if not tool_result:
        tool_result_info = {}
    elif isinstance(tool_result, str):
        obs_token_ids = tokenizer.encode(tool_result, add_special_tokens=False)
        tool_result_info = {
            "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
            "prompt_token_ids_model": torch.tensor(obs_token_ids),
        }
    elif isinstance(tool_result, list) and isinstance(tool_result[0], dict):
        obs_token_ids = tokenizer.apply_chat_template(tool_result, add_generation_prompt=True, return_tensors='pt')[0]
        eos_start_idx = torch.nonzero(obs_token_ids == tokenizer.eos_token_id)
        if eos_start_idx.shape[0] > 0:
            eos_start_idx = eos_start_idx[0].item()
            obs_token_ids = obs_token_ids[eos_start_idx + 1 : ]
        else:
            raise ValueError(f"tool [{tool.name}] returned type List[str] output must be in openai/qwen format : {tool_result}")
        tool_result_info = {
            "prompt_token_ids_vllm": obs_token_ids,
            "prompt_token_ids_model": obs_token_ids,
        }
    elif isinstance(tool_result, dict):
        # 有文本，有图片是这种情况，在实际rollout的过程中也就是进入这个分支
        prompt_str = tool_result.pop("prompt", "")
        chat_list = tool_result.pop("chat", [])

        # prompt在输入之前需要把文字和图片一起输入preprocess_multi_model_inputs这个函数中转化为真正可以使用的Input token ids
        prompt_str_vllm, obs_token_ids_model, mm_inputs = _preprocess_multi_modal_inputs(prompt_str, processor, **tool_result)

        obs_token_ids_vllm = tokenizer.encode(prompt_str_vllm, add_special_tokens=False, return_tensors='pt')[0]
        tool_result_info = {
            "prompt_token_ids_vllm": obs_token_ids_vllm,
            "prompt_token_ids_model": obs_token_ids_model,
            **tool_result
        }
        if mm_inputs:
            tool_result_info["multi_modal_inputs"] = mm_inputs
    else:
        raise ValueError(f"Invalid tool_result type: {type(tool_result)=} -- {tool_result}")

    if pbar is not None:
        pbar.update(1)

    return tool_result_info, reward, done, info


class ParallelEnv:
    """并行环境接口"""
    def __init__(self, env_config, tokenizer, processor, **kwargs):
        self.config = env_config
        self.tokenizer = tokenizer
        self.processor = processor
        self.tools = []
        self.success_time = 0
        self.max_succ_reward = 1
        self.min_succ_reward = 1

    def step(self, active_indices, actions, vllm_input_list):
        """执行环境交互步骤"""
        obs_list = [{}] * len(actions)
        reward_list = [0.0] * len(actions)
        done_list = []
        valid_indices = []
        real_indices = []
        valid_actions = []

        for i, (idx, act) in enumerate(zip(active_indices, actions)):
            if act.outputs[0].finish_reason == 'length':
                done_list.append(True)
                continue

            if len(act.outputs[0].token_ids) == 0:
                done_list.append(True)
                continue

            done_list.append(False)
            real_indices.append(i)
            valid_indices.append(idx)
            valid_actions.append(act.outputs[0].text)

        agent_inputs = []
        for i, idx, action in zip(real_indices, valid_indices, valid_actions):
            agent_inputs.append(dict(
                idx=i,
                valid_idx=idx,
                action=action,
                tool=self.tools[idx],
            ))

        num_workers = min(self.config.concurrent_workers, len(valid_actions))
        pbar = tqdm(total=len(valid_actions), desc=f'Tool calling on {num_workers} workers') if self.config.show_tqdm else None
        
        if num_workers <= 1:
            for agi in agent_inputs:
                subidx = agi['idx']
                valid_idx = agi['valid_idx']
                obs, reward, done, info = execute_tool_call(agi, vllm_input_list[valid_idx], self.config, self.tokenizer, self.processor, pbar=pbar)
                if reward >= 1: 
                    # 最早完成任务的可以获得最大的奖励，要求环境是静态的，任务完成进度仅仅取决于模型
                    reward = self.min_succ_reward + (self.max_succ_reward - self.min_succ_reward) * 0.5 ** self.success_time
                    self.success_time += 1
                    print(f'[DEBUG] {self.success_time=}, {reward=}')
                obs_list[subidx] = obs 
                reward_list[subidx] = reward
                done_list[subidx] |= done
        else:
            # 为并行执行准备参数
            parallel_inputs = []
            for agi in agent_inputs:
                valid_idx = agi['valid_idx']
                parallel_inputs.append((agi, vllm_input_list[valid_idx]))
            
            partial_tool_func = partial(execute_tool_call, config=self.config, tokenizer=self.tokenizer, processor=self.processor, pbar=pbar)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 只传递第一个参数给 partial function
                raw_outputs = list(executor.map(lambda x: partial_tool_func(x[0], x[1]), parallel_inputs))
            
            for agi, raw in zip(agent_inputs, raw_outputs):
                obs, reward, done = raw[0], raw[1], raw[2]
                if reward >= 1: 
                    # 最早完成任务的可以获得最大的奖励，要求环境是静态的，任务完成进度仅仅取决于模型
                    reward = self.min_succ_reward + (self.max_succ_reward - self.min_succ_reward) * 0.5 ** self.success_time
                    self.success_time += 1
                    print(f'[DEBUG] {self.success_time=}, {reward=}')
                subidx = agi['idx']
                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done

        return obs_list, reward_list, done_list, {}

    def get_data_idx(self, env_idx):
        """获取该id的环境是由哪一个原始的input_ids生成的"""
        return self.tools[env_idx].get_data_idx()

    def get_uuid(self, env_idx):
        return self.tools[env_idx].get_uuid()


    def reset(self, prompts, vllm_inputs, n=1, **kwargs):
        """重置环境"""
        self.tools = []
        obs_list = []
        reward_list = []
        done_list = []
        assert len(prompts) == len(vllm_inputs), f"{len(prompts)=}, {len(vllm_inputs)=}"

        num_agent, num_non_agent = 0, 0

        for i in range(len(prompts)):
            data_item = prompts[i] 
            tool_name = data_item.non_tensor_batch.pop('env_name', '')
            if tool_name != "MCSimulator":
                print(f"[DEBUG mc tool] {tool_name=}. Change it to MCSimulator!")
                tool_name = "MCSimulator"
            
            raw_prompt = data_item.non_tensor_batch.pop('raw_prompt', None)
            client_list = data_item.non_tensor_batch.pop('client_list', None)
            vllm_input_item = vllm_inputs[i]
            multi_modal_data = vllm_input_item.get("multi_modal_data", None)
            origin_multi_modal_data = data_item.non_tensor_batch.pop("origin_multi_modal_data", None)
            
            

            for j in range(n):
                if tool_name:
                    tool_fns = ToolBase.create(tool_name)
                    obs, info = tool_fns.reset(
                        raw_prompt=raw_prompt, 
                        multi_modal_data=deepcopy(multi_modal_data),
                        origin_multi_modal_data=deepcopy(origin_multi_modal_data),        
                        uuid=client_list[j],
                        data_idx=i,
                    )
        
                    self.tools.append(tool_fns)
                    obs = {
                        "prompt": "<|im_start|>user\n" + "Your observation: <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"+"<|im_start|>assistant\n",
                        "multi_modal_data": {"image": [obs['image']]}
                    }

                    if not obs:
                        obs_processed = {}
                    elif isinstance(obs, dict):
                        prompt_str = obs.pop("prompt", "")
                        chat_list = obs.pop("chat", [])

                        if len(prompt_str) == 0 and len(chat_list) == 0:
                            raise ValueError("Both prompt_str and chat_list are invalid")
                        elif len(prompt_str) == 0 and len(chat_list) > 0:
                            prompt_str = self.tokenizer.apply_chat_template(chat_list, add_generation_prompt=True, tokenize=False)
                            prompt_str = _strip_system_block(prompt_str)

                        prompt_str_vllm, obs_token_ids_model, mm_inputs = _preprocess_multi_modal_inputs(prompt_str, self.processor, **obs)
                        obs_token_ids_vllm = self.tokenizer.encode(prompt_str_vllm, add_special_tokens=False, return_tensors='pt')[0]
                        obs_processed = {
                            "prompt_token_ids_vllm": obs_token_ids_vllm,
                            "prompt_token_ids_model": obs_token_ids_model,
                            **obs
                        }
                        if mm_inputs:
                            obs_processed["multi_modal_inputs"] = mm_inputs
                    else:
                        raise ValueError(f"Invalid tool_result type: {type(obs)=} -- {obs}")

                    obs_list.append(obs_processed)
                    reward_list.append(0.0)
                    done_list.append(False)
                    num_agent += 1
                else:
                    num_non_agent += 1
                    print(f"[DEBUG tool] No tools!")
                    # 如果没有工具，这里可以选择跳过或使用默认图片
                    obs_list.append({})
                    reward_list.append(0.0)
                    done_list.append(False)
                    self.tools.append(None)

        print(f'[DEBUG agent] {num_agent=}, {num_non_agent=}')
        return obs_list, reward_list, done_list, {}

    def close_game(self, idx):
        self.tools[idx].close()


    def close(self):
        """关闭环境"""
        for tool in self.tools:
            if tool and getattr(tool, "name", None) == "MCSimulator":
                tool.close()           
        self.tools = []
        self.success_time = 0