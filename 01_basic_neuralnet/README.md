# README

# Go to K8s:
kubectl -n test2 exec -it yiqiao -c yiqiao -- /bin/bash
# Connects to the 'yiqiao' container in the 'yiqiao' pod within the 'test2' namespace on Kubernetes, opening an interactive bash shell.

# Get into my pod:
ssh -i yiqiao-genai-experiments-test1-key.pem ubuntu@10.57.158.169
# SSH into an Ubuntu VM (possibly hosting a pod) using a private key for authentication.

# Get into my EC2:
docker run --gpus all -it --entrypoint /bin/bash   genai-docker-development.repo.prod.us-west-2.aws.<company_name>.com:443/deepspeed/basic_training:v1.0.1
# Starts an interactive bash shell inside a Docker container with GPU access, using the specified image from a private AWS ECR registry.

# Watch GPU:
watch -n 0.1 nvidia-smi
# Continuously monitors GPU usage and status every 0.1 seconds using 'nvidia-smi'.

# Run deep speed
deepspeed --num_gpus=4 train_ds.py
# Launches distributed training with DeepSpeed on 4 GPUs, running the 'train_ds.py' script.

# Get pass outbound traffic block
mkdir -p ~/.pip
nano ~/.pip/pip.conf
[global]
index-url = https://yiqiaoyin@<company_name>.com:xxx@repo.prod.us-west-2.aws.<company_name>.com/artifactory/api/pypi/genai-pypi/simple
# Creates a pip configuration file to use a custom PyPI index, allowing pip to install packages from a private repository (useful if outbound traffic is restricted).

# Execute
deepspeed --num_gpus=1 train_ds.py --deepspeed --deepspeed_config ds_config.json
# Runs DeepSpeed training on 1 GPU, using 'train_ds.py' and a custom DeepSpeed configuration file