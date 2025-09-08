# main3.py

import subprocess

def main():
    """
    Launches the DeepSpeed training script using 2 GPUs with a specified configuration file.
    Equivalent to running:
    deepspeed --num_gpus=2 cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json
    """
    cmd = [
        "deepspeed",
        "--num_gpus=2",
        "cifar10_deepspeed.py",
        "--deepspeed",
        "--deepspeed_config", "ds_config.json"
    ]

    # Execute the command
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
