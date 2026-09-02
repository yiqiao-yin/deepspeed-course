import subprocess

if __name__ == "__main__":
    # Define the DeepSpeed command
    command = ["deepspeed", "--num_gpus=4", "train_ds.py"]

    # Run the training script
    result = subprocess.run(command)

    # Check if the command was successful
    if result.returncode == 0:
        print("Training completed successfully.")
    else:
        print(f"Training failed with return code {result.returncode}.")

