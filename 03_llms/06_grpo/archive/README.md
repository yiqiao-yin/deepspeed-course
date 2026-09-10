# README

This markdown file walks through how to use `deepseed` to finetune. First, it is assumed you have this image `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` installed. Additionally, we assume you have `uv` installed as well. 

## Project Starter

Use `uv` to start project. 

```bash
uv init project_name
````

If you do not have `uv`, please install it.

```bash
brew install uv
```

Or alternatively, you can use `pip`.

```bash
pip install uv
```

Next, add packages or dependencies

```bash
cd project_name
uv add torch
uv add transformers
uv add accelerate
uv add datasets
uv add deepspeed
uv add bitsandbytes
uv add trl
uv add unsloth
```

Or in one line, we have

```bash
uv add torch transformers accelerate datasets deepspeed bitsandbytes trl unsloth
```

We can examine the package dependency trees.

```bash
uv tree
```

You should expect something like the following.

```bash
root@1b0c67c74d6a:/workspace/deepspeed_project# uv tree
Resolved 97 packages in 0.68ms
deepspeed-project v0.1.0
├── accelerate v1.6.0
│   ├── huggingface-hub v0.31.1
│   │   ├── filelock v3.18.0
│   │   ├── fsspec v2025.3.0
│   │   │   └── aiohttp v3.11.18 (extra: http)
│   │   │       ├── aiohappyeyeballs v2.6.1
│   │   │       ├── aiosignal v1.3.2
│   │   │       │   └── frozenlist v1.6.0
│   │   │       ├── async-timeout v5.0.1
│   │   │       ├── attrs v25.3.0
│   │   │       ├── frozenlist v1.6.0
│   │   │       ├── multidict v6.4.3
│   │   │       │   └── typing-extensions v4.13.2
│   │   │       ├── propcache v0.3.1
│   │   │       └── yarl v1.20.0
│   │   │           ├── idna v3.10
│   │   │           ├── multidict v6.4.3 (*)
│   │   │           └── propcache v0.3.1
│   │   ├── hf-xet v1.1.0
│   │   ├── packaging v25.0
│   │   ├── pyyaml v6.0.2
│   │   ├── requests v2.32.3
│   │   │   ├── certifi v2025.4.26
│   │   │   ├── charset-normalizer v3.4.2
│   │   │   ├── idna v3.10
│   │   │   └── urllib3 v2.4.0
│   │   ├── tqdm v4.67.1
│   │   └── typing-extensions v4.13.2
│   ├── numpy v2.2.5
│   ├── packaging v25.0
│   ├── psutil v7.0.0
│   ├── pyyaml v6.0.2
│   ├── safetensors v0.5.3
│   └── torch v2.7.0
│       ├── filelock v3.18.0
│       ├── fsspec v2025.3.0 (*)
│       ├── jinja2 v3.1.6
│       │   └── markupsafe v3.0.2
│       ├── networkx v3.4.2
│       ├── nvidia-cublas-cu12 v12.6.4.1
│       ├── nvidia-cuda-cupti-cu12 v12.6.80
│       ├── nvidia-cuda-nvrtc-cu12 v12.6.77
│       ├── nvidia-cuda-runtime-cu12 v12.6.77
│       ├── nvidia-cudnn-cu12 v9.5.1.17
│       │   └── nvidia-cublas-cu12 v12.6.4.1
│       ├── nvidia-cufft-cu12 v11.3.0.4
│       │   └── nvidia-nvjitlink-cu12 v12.6.85
│       ├── nvidia-cufile-cu12 v1.11.1.6
│       ├── nvidia-curand-cu12 v10.3.7.77
│       ├── nvidia-cusolver-cu12 v11.7.1.2
│       │   ├── nvidia-cublas-cu12 v12.6.4.1
│       │   ├── nvidia-cusparse-cu12 v12.5.4.2
│       │   │   └── nvidia-nvjitlink-cu12 v12.6.85
│       │   └── nvidia-nvjitlink-cu12 v12.6.85
│       ├── nvidia-cusparse-cu12 v12.5.4.2 (*)
│       ├── nvidia-cusparselt-cu12 v0.6.3
│       ├── nvidia-nccl-cu12 v2.26.2
│       ├── nvidia-nvjitlink-cu12 v12.6.85
│       ├── nvidia-nvtx-cu12 v12.6.77
│       ├── sympy v1.14.0
│       │   └── mpmath v1.3.0
│       ├── triton v3.3.0
│       │   └── setuptools v80.3.1
│       └── typing-extensions v4.13.2
├── bitsandbytes v0.45.5
│   ├── numpy v2.2.5
│   └── torch v2.7.0 (*)
├── datasets v3.6.0
│   ├── dill v0.3.8
│   ├── filelock v3.18.0
│   ├── fsspec[http] v2025.3.0 (*)
│   ├── huggingface-hub v0.31.1 (*)
│   ├── multiprocess v0.70.16
│   │   └── dill v0.3.8
│   ├── numpy v2.2.5
│   ├── packaging v25.0
│   ├── pandas v2.2.3
│   │   ├── numpy v2.2.5
│   │   ├── python-dateutil v2.9.0.post0
│   │   │   └── six v1.17.0
│   │   ├── pytz v2025.2
│   │   └── tzdata v2025.2
│   ├── pyarrow v20.0.0
```

Afterwards, you should be able to expect the following folder structure.

```bash
root@1b0c67c74d6a:/workspace/deepspeed_project# ls -l
total 448
-rw-r--r-- 1 root root      0 May  9 17:22 README.md
-rw-r--r-- 1 root root    318 May  9 17:30 ds_config_zero1.json
-rw-r--r-- 1 root root    413 May  9 17:32 main.py
-rw-r--r-- 1 root root    357 May  9 17:27 pyproject.toml
drwxr-xr-x 2 root root     10 May  9 17:39 results
-rw-r--r-- 1 root root   1418 May  9 18:10 train_ds.py
-rw-r--r-- 1 root root 439339 May  9 17:27 uv.lock
```

## Run Project

Use `uv` to run project.

```bash
uv run main.py
```