# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import re
from collections import defaultdict
from typing import List, Optional, Union
import yaml

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask



def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        # 这里的config是train脚本的config.data
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        env_config = example.get("env_config")
        target = env_config["target_type"]
        if "goal_obj_status" in env_config.keys():
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an embodied agent capable of performing the following discrete actions. Your objective is to select the most appropriate action to complete the specific **Task** provided below, based strictly on the visual information from your current **Observation**.
                                Note that each rotation or directional change corresponds to a **30-degree** movement:
                                - wait: Do nothing and stay still.
                                - walk_forward: Move forward one step.
                                - walk_backward: Move backward one step.
                                - look_left: Rotate your body 30 degrees to the left.
                                - look_right: Rotate your body 30 degrees to the right.
                                - look_up: Tilt your view 30 degrees upward.
                                - look_down: Tilt your view 30 degrees downward.
                                - operate <object>: interact with corresponding object, you should describe the object in <object>.
                                - move_away <object>: move the object to another place in free space around you, you should describe the object in <object>.

                                # Critical Navigation Strategies
                                1. **Approach Target**: Finding the target is not enough. You must navigate to the target's location and get close to it.
                                2. **Obstacle Avoidance**: Prioritize wide, open pathways. If the path ahead is blocked, you should prioritize `look_right` to find a spacious road. Move onto that spacious road to ensure you are on a clear path, and **then** proceed toward the target.
                                3. **Visual Recovery**: If your current observation lacks meaningful information (e.g., you are staring closely at a wall, the floor, or the ceiling, and cannot see any distinct objects or pathways), you must assume your view is obstructed. In this scenario, priority is to regain situational awareness by rotating to find a valid visual cue. You should strictly choose to `look_right` repeatedly until a meaningful object, open space, or your target comes into view.
                                4. **Operate Object**: You can only operate an object if it is directly in your field of view and you are close enough to interact with it. If the target object is not in front of you or you are not close enough, you must first navigate to its location.

                                # Output Requirements
                                1. **Decision Basis**: You must analyze the provided `Task` and inspect the `Observation`. Your decision must bridge the gap between what you see and what you need to achieve.
                                2. **Consistency**: Your chosen action must logically result from your reasoning process.
                                3. **Format**: You should think first and then choose the right action at this moment. Your output should be strictly in the following format: <think>Your step-by-step reasoning process...</think><action>selected_action</action>
                                4. **Constraint**: You should only choose the exact action name from the list above without any other words or explanations outside the tags.

                                Your Task is: """  + env_config["task_str"]
                }
            ]
        else:
            # 构造 system prompt
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an embodied agent capable of performing the following discrete actions. Your objective is to select the most appropriate action to complete the specific **Task** provided below, based strictly on the visual information from your current **Observation**.
                                Note that each rotation or directional change corresponds to a **30-degree** movement:
                                - wait: Do nothing and stay still.
                                - walk_forward: Move forward one step.
                                - walk_backward: Move backward one step.
                                - look_left: Rotate your body 30 degrees to the left.
                                - look_right: Rotate your body 30 degrees to the right.
                                - look_up: Tilt your view 30 degrees upward.
                                - look_down: Tilt your view 30 degrees downward.
                                - move_away <object>: move the object to another place in free space around you, you should describe the object in <object>.

                                # Critical Navigation Strategies
                                1. **Approach Target**: Finding the target is not enough. You must navigate to the target's location and get close to it.
                                2. **Obstacle Avoidance**: Prioritize wide, open pathways. If the path ahead is blocked, you should prioritize `look_right` to find a spacious road. Move onto that spacious road to ensure you are on a clear path, and **then** proceed toward the target.
                                3. **Visual Recovery**: If your current observation lacks meaningful information (e.g., you are staring closely at a wall, the floor, or the ceiling, and cannot see any distinct objects or pathways), you must assume your view is obstructed. In this scenario, priority is to regain situational awareness by rotating to find a valid visual cue. You should strictly choose to `look_right` repeatedly until a meaningful object, open space, or your target comes into view.
                                4. **Operate Object**: You can only operate an object if it is directly in your field of view and you are close enough to interact with it. If the target object is not in front of you or you are not close enough, you must first navigate to its location.

                                # Output Requirements
                                1. **Decision Basis**: You must analyze the provided `Task` and inspect the `Observation`. Your decision must bridge the gap between what you see and what you need to achieve.
                                2. **Consistency**: Your chosen action must logically result from your reasoning process.
                                3. **Format**: You should think first and then choose the right action at this moment. Your output should be strictly in the following format: <think>Your step-by-step reasoning process...</think><action>selected_action</action>
                                4. **Constraint**: You should only choose the exact action name from the list above without any other words or explanations outside the tags.

                                Your Task is: """  + f'find and navigate to the {target}.'
                }
            ]
        # print(f'[DEBUG] The task is "Your task is to navigate to {target}"')

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                # 修改prompt中的content
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages


    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # 注意，这里的我假设row_dict中有键为'data_config_path', 可以直接读取, 是写在了数据中，不用额外写的，这里我写只是占个坑而已
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict) # 我需要在这里读取任务的yaml文件，然后加上任务指令
        model_inputs = {}

        

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_raw_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            multi_modal_data = {}
            origin_multi_modal_data = {}


            images = None
            # if self.image_key in row_dict:
            #     # origin_images = [process_raw_image(image) for image in row_dict.get(self.image_key)]
            #     # images = [process_image(image) for image in row_dict.pop(self.image_key)]
            #     # multi_modal_data["image"] = images # 我们的任务中其实没有图片
            #     # origin_multi_modal_data["image"] = origin_images
            #     multi_modal_data["image"] = [] # 我们的任务中其实没有图片
            #     origin_multi_modal_data["image"] = []

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            # 把多模态的数据项设置成有图片的时候再来？
            row_dict['origin_multi_modal_data'] = origin_multi_modal_data
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

      

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        try:
            prompt_after_process = self.tokenizer.decode(input_ids[0], add_special_tokens=False)
            # 没错，是在这个process里进行了padding
            # print(f'[DEBUG] after process in rl_dataset.py, {prompt_after_process=}') # 我在这里需要验证是否在rl_dataset.py里面padding
        except:
            print(f'[DEBUG] cannot decode input_ids!')

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
