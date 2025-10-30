import os
from huggingface_hub import create_repo, upload_folder

# ======== Config ==========
hf_token = "xxx"
local_path = "/workspace/proj/eagle0504/openai-gsm8k-enhanced-deepseek-r1-distill-qwen-1.5b"
repo_id = "eagle0504/finetuned-deepseek-r1-distill-qwen-1.5b-by-openai-gsm8k-enhanced-v2"
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



