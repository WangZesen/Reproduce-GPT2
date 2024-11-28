# Pre-training GPT-2 on OpenWebText with Decentralized Training

## Introduction

This project reproduces the pre-training of GPT-2 on OpenWebText. The project adopts most codebase from the open-source implementation of GPT-2, [nanoGPT](https://github.com/karpathy/nanoGPT), but re-organizes the key arguments using [Pydantic](https://docs.pydantic.dev/latest/), which provides a easy-to-adapt codebase.

This project additionally includes decentralized training.

## Reproduce Experiments

> **Note**: The instruction is subject to the systems under test (Linux system with SLURM as job scheduler). The steps may differ for other systems. To run the whole experiments, it will take ~35 hours with 16 A100 GPUs (the estimated time includes time for both training and validation).

### Setup Python Environment

1. Create a virtual environment using either `virtualenv` or `conda`, and activate the environment.
2. Install dependencies
    ```shell
    # Latest version of PyTorch
    pip install torch torchvision torchaudio
    # Other dependencies
    pip install wandb tomli_w loguru datasets tqdm tiktoken pydantic
    ```

### Download and Preprocess [Open Web Text Dataset](https://huggingface.co/datasets/Skylion007/openwebtext)

After activating the environment, run
```shell
python data/openwebtext/prepare.py
```

The whole process takes roughly 10 minutes for a 16-core CPU. The tokenized and binarinized data will be at `data/openwebtext/train.bin` (~17GB) and `data/openwebtext/val.bin` (~9MB).


### Launch Training via [SLURM](https://slurm.schedmd.com/documentation.html)

After activating the environment, run the following command to submit the job
```shell
# make a new directory for logging
mkdir log

# 1. submit a job for training GPT-2 (small) with 4x4xA100 (4xA100 per node) and AllReduce training. <proj-account> is the account used for SLURM.
sbatch -A <proj-account> scripts/16xA100.sh $(which torchrun) config/gpt2-small.toml

# 2. submit a job for training GPT-2 (small) with 4x4xA100 and decentralized training
sbatch -A <proj-account> scripts/d-16xA100.sh $(which torchrun) config/decent-gpt2-small.toml
```

> **Note**: In the default config file, the hyperparameters are for this specific hardware setting (16 workers in total, and 4 workers per node). The hyperparameters may subject to changes for a different number of workers or a different number of workers per node.

The log files and the checkpoints will be placed at `log/<slurm-job-id>/train.log` which includes the training and validation losses for every 1000 steps, and the average time taken by one step evaluated on the last 1000 steps. An example of parts of the log file is shown below
```
[11:19:22  INFO] Step 207000, Val loss: 2.971195, Train loss: 2.957595, Train time: 177.869 s (177.869 ms/step), Total train time: 36862.458 s
[11:23:09  INFO] Step 208000, Val loss: 2.969273, Train loss: 2.956899, Train time: 177.871 s (177.871 ms/step), Total train time: 37040.329 s
[11:23:11  INFO] New best model found. Saving...
[11:23:12  INFO] Saved model to ./log/3197816/best_model.pth
[11:26:58  INFO] Step 209000, Val loss: 2.970544, Train loss: 2.957326, Train time: 177.943 s (177.943 ms/step), Total train time: 37218.271 s
[11:30:45  INFO] Step 210000, Val loss: 2.971496, Train loss: 2.957171, Train time: 177.950 s (177.950 ms/step), Total train time: 37396.221 s
[11:34:32  INFO] Step 211000, Val loss: 2.968879, Train loss: 2.956959, Train time: 177.974 s (177.974 ms/step), Total train time: 37574.195 s
```

### Explanation of Configuration File

Two examples of configuration files for AllReduce training and decentralized training, respectively, are given at [`gpt2-small.toml`](./config/gpt2-small.toml) and [`decent-gpt2-small.toml`](./config/decent-gpt2-small.toml) under the `config` directory.

Take [`decent-gpt2-small.toml`](./config/decent-gpt2-small.toml) as an example, the explanation of all configurable arugments is shown below.

```toml
# logging directory
log_dir = "./log"
# dataset. "openwebtext" is the only choice
dataset = "openwebtext"
# directory for train.bin and val.bin
data_dir = "./data/openwebtext"
# set to True if loading the whole dataset into memory before the training.
# please set it as True for optimal performance when each worker has >18GB memory.
in_memory = true

[log]
# set to True to enable wandb logging
wandb_log = true
# project name in wandb
wandb_project = "gpt2-owt"
# evaluate the training and validation losses for every <eval_interval> steps
eval_interval = 1000
# <eval_steps> batches will be sampled to evaluate the loss
eval_steps = 200

[train]
# either 'decentdp' or 'pytorchddp' for decentralized training and AllReduce training, respectively.
backend = 'decentdp'
# topology for decentralized training which could be 'complete', 'exp', 'ring', or 'alternating-exp-ring'
topology = 'alternating-exp-ring'
# random seed for reproducibility.
# the seed only controls the initialization of the models and the order of training/validation data.
# however, since tensor cores are activated for efficiency, there might be slight differences in the figures for a repeated run.
seed = 42
# global batch size (sum of local batch sizes)
# a local batch size with 30 will take ~25GB GPU memory
global_batch_size = 480
# sequence packing is used, and the context length is 1024 for GPT-2
context_length = 1024
# total number of steps for training
n_steps = 600000
# clip the gradient by its norm
grad_norm_clip = 1.0
# use torch.compile to speedup the training
compile = true
# set to True to enable automatic mixed precision training
amp = true

[train.optim]
# name of the optimizer. currently, it only supports `accumadamw` for decentralized training, and `adamw` for AllReduce training
name = "accumadamw"
# hyperparameters for AccumAdamW. beta_2 is set to 0.98 for decentralized training.
betas = [
    0.9,
    0.98,
]
eps = 1e-08
weight_decay = 0.1
# the number of steps to accumulate the gradient
accum_steps = 8

[train.lr_schedule]
# the name of learning rate scheduler. only 'cosine' is suppported.
name = "cosine"
# base learning rate
lr = 6e-04
# number of steps for warmup
warmup_steps = 2000
# minimal learning rate
eta_min = 6e-05

[model]
# the definition of the architecture of GPT-2 (small), could be changed to larger values for other larger variants of GPT-2.
n_layers = 12
n_heads = 12
n_embd = 768
dropout = 0.0
vocab_size = 50304
bias = false

```

> **Note**: The hyperparameters are mostly taken from [nanoGPT](https://github.com/karpathy/nanoGPT). Minor changes were done for decentralized training.