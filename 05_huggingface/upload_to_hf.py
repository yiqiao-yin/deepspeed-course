import os
from huggingface_hub import create_repo, upload_folder

# ======== Config ==========
hf_token = "xxx"
local_path = "/workspace/my_proj/eagle0504/warren-buffett-letters-qna-r1-enhanced-1998-2024-finetuned-llama-3.2-1B-Instruct"
repo_id = "eagle0504/finetuned-warren-buffett-letter-model-llama-3.2-1B-Instruct-2024"
# ===========================

# Step 1: Create the repo (safe even if it already exists)
create_repo(repo_id, token=hf_token, repo_type="model", exist_ok=True, private=False)

# Step 2: Upload the entire folder
upload_folder(
    folder_path=local_path,
    repo_id=repo_id,
    repo_type="model",
    token=hf_token,
    ignore_patterns=["*.ipynb", "*.log"]
)

print(f"✅ Successfully pushed model and tokenizer to: https://huggingface.co/{repo_id}")


