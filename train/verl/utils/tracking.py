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
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union
import os
import pandas as pd


class Tracking(object):
    supported_backend = ["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "rl_logging_board"]

    def __init__(self, 
        project_name: str,
        experiment_name: str,
        default_backend: Union[str, List[str]],
        trainer_config: dict,
        config=None,
    ):  
        # project_name=trainer_config.trainer.project_name
        # experiment_name=trainer_config.trainer.experiment_name
        # default_backend=trainer_config.trainer.logger

        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}
        self.default_backend = default_backend

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb
            import os
            os.environ["WANDB_MODE"] = "offline"

            # 1. 安全获取 resume_id，防止 trainer_config 里没有这个字段报错
            # 如果 trainer_config 是字典，请用 trainer_config.get("resume_id", None)
            resume_id = getattr(trainer_config.trainer, "resume_id", None)

            # 2. 统一初始化逻辑
            # 当 resume_id 为 None 时，wandb 会忽略 id 参数并新建 run
            # 当 resume_id 有值时，resume="allow" 会尝试续写
            wandb.init(
                project=project_name, 
                name=experiment_name, 
                config=config, 
                id=resume_id, 
                resume="allow"
            )
            self.logger["wandb"] = wandb

            # 3. 初始化成功后，再根据情况打印日志
            # 此时 wandb.run 已经存在，wandb.run.id 一定是正确的
            if resume_id is None:
                # 这是一个新 Run
                print(f"\n{'='*20} WANDB NEW RUN {'='*20}")
                print(f"!!! WANDB RUN ID: {wandb.run.id} !!!")
                print(f"!!! 请务必在 Checkpoint 中保存此 ID，以便程序崩溃后续训 !!!")
                print(f"{'='*56}\n")
            else:
                # 这是一个续训 Run
                print(f"\n{'='*20} WANDB RESUMING {'='*20}")
                print(f"!!! Resuming Run ID: {resume_id} !!!")
                print(f"!!! Current Run ID:  {wandb.run.id} (Should match) !!!")
                print(f"{'='*56}\n")

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "veRL", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if 'tensorboard' in default_backend:
            from verl.utils.tensorboard_utils import TensorboardLogger
            self.logger['tensorboard'] = TensorboardLogger(
                trainer_config.trainer.tensorboard_dir,
                project_name, 
                experiment_name
            )
        
        if 'rl_logging_board' in default_backend:
            from verl.utils.rl_logging_board_utils import RLLoggingBoardLogger
            self.logger['rl_logging_board'] = RLLoggingBoardLogger(
                trainer_config.trainer.rl_logging_board_dir,
                project_name, 
                experiment_name
            )

        if "console" in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

    def log(self, data, step, batch=None, backend=None, tokenizer=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                if default_backend == 'rl_logging_board':
                    if batch is not None:
                        logger_instance.log(data=data, step=step, batch=batch, tokenizer=tokenizer)
                else:
                    logger_instance.log(data=data, step=step)

    def __del__(self):
        if 'wandb' in self.logger:
            self.logger['wandb'].finish(exit_code=0)
        if 'swanlab' in self.logger:
            self.logger['swanlab'].finish()
        if 'vemlp_wandb' in self.logger:
            self.logger['vemlp_wandb'].finish(exit_code=0)


class _TensorboardAdapter:
    def __init__(self):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)



    def log_generations_to_wandb(self, samples, step, save_dir=None):
        if save_dir is None:
            save_dir = os.environ.get("WANDB_SAVE_DIR", "./wandb_offline")
        """Log samples to local CSV table (offline)"""
        os.makedirs(save_dir, exist_ok=True)

        # 构造列名
        columns = ["step"] + sum([[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], [])

        # 创建 row 数据
        row_data = [step]
        for sample in samples:
            row_data.extend(sample)

        # CSV 文件路径
        file_path = os.path.join(save_dir, "val_generations.csv")

        if not os.path.exists(file_path):
            # 文件不存在，新建
            df = pd.DataFrame([row_data], columns=columns)
            df.to_csv(file_path, index=False)
        else:
            # 文件存在，读取后追加
            df = pd.read_csv(file_path)
            # 如果已有列数量与新数据不一致，先对齐
            if df.shape[1] < len(columns):
                for col in columns[df.shape[1]:]:
                    df[col] = None
            df.loc[len(df)] = row_data
            df.to_csv(file_path, index=False)

        # 可选：打印日志
        print(f"[Step {step}] Saved {len(samples)} samples to {file_path}")

    # def log_generations_to_wandb(self, samples, step):
    #     """Log samples to wandb as a table"""
    #     import wandb

    #     # Create column names for all samples
    #     columns = ["step"] + sum(
    #         [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
    #     )

    #     if not hasattr(self, "validation_table"):
    #         # Initialize the table on first call
    #         self.validation_table = wandb.Table(columns=columns)

    #     # Create a new table with same columns and existing data
    #     # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
    #     new_table = wandb.Table(columns=columns, data=self.validation_table.data)

    #     # Add new row with all data
    #     row_data = []
    #     row_data.append(step)
    #     for sample in samples:
    #         row_data.extend(sample)

    #     new_table.add_data(*row_data)

    #     # Update reference and log
    #     wandb.log({"val/generations": new_table}, step=step)
    #     self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_text_list}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")