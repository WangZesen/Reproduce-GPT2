import os
import datetime
import subprocess
from typing import Literal, Optional
from pydantic import BaseModel, Field, computed_field
from functools import cached_property

SUPPORTED_DATASETS = Literal['openwebtext']
SUPPORTED_OPTIMIZERS = Literal['adamw']


class Logging(BaseModel):
    log_interval: int = 100
    wandb_log: bool = True
    wandb_project: str = 'gpt2-owt'
    eval_interval: int = 1000
    eval_steps: int = 200

    @computed_field
    @cached_property
    def run_name(self) -> str:
        return os.environ.get('SLURM_JOB_ID', datetime.datetime.now().strftime('%m-%d-%H-%M'))


class CosineLRSchedule(BaseModel):
    name: Literal['cosine'] = 'cosine'
    lr: float = 6e-4
    warmup_steps: int = 2000
    eta_min: float = 6e-5


class AdamWConfig(BaseModel):
    name: Literal['adamw'] = 'adamw'
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-1


class Network(BaseModel):
    @computed_field
    @cached_property
    def world_size(self) -> int:
        return int(os.environ.get('WORLD_SIZE', 1))

    @computed_field
    @cached_property
    def local_world_size(self) -> int:
        return int(os.environ.get('LOCAL_WORLD_SIZE', 1))

    @computed_field
    @cached_property
    def rank(self) -> int:
        return int(os.environ.get('RANK', 0))

    @computed_field
    @cached_property
    def gpu_model(self) -> str:
        raw = subprocess.check_output('nvidia-smi --query-gpu=name --format=csv,noheader', shell=True)
        return raw.decode().strip().split('\n')[0]

    @computed_field
    @cached_property
    def node_list(self) -> str:
        return os.environ.get('SLURM_NODELIST', 'localhost')


class Train(BaseModel):
    seed: int = 42
    global_batch_size: int = 480
    context_length: int = 1024
    n_steps: int = 600_000
    grad_norm_clip: float = 1.0
    optim: AdamWConfig = Field(default_factory=AdamWConfig,
                               discriminator='name')
    lr_schedule: CosineLRSchedule = Field(default_factory=CosineLRSchedule)
    network: Network = Field(default_factory=Network)
    compile: bool = True
    amp: bool = True

    @computed_field
    @property
    def local_batch_size(self) -> int:
        return self.global_batch_size // self.network.world_size
    
    @computed_field
    @cached_property
    def global_tokens_per_step(self) -> int:
        return (self.global_batch_size // self.network.world_size) * self.context_length


class Model(BaseModel):
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    vocab_size: int = 50304
    bias: bool = False


class Config(BaseModel):
    log_dir: str = './log'
    log: Logging = Field(default_factory=Logging)
    train: Train = Field(default_factory=Train)
    model: Model = Field(default_factory=Model)
    dataset: SUPPORTED_DATASETS = 'openwebtext'
    data_dir: str = './data/openwebtext'
    in_memory: bool = True
    checkpoint: Optional[str] = None

    @computed_field
    @cached_property
    def run_dir(self) -> str:
        return os.path.join(self.log_dir, self.log.run_name)

