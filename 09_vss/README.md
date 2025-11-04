# Video-Speech-to-Speech (VSS) Fine-tuning with LongCat-Flash-Omni

Fine-tune [LongCat-Flash-Omni](https://huggingface.co/meituan-longcat/LongCat-Flash-Omni) for video-speech-to-speech tasks using LoRA, DeepSpeed ZeRO-3, and optional W&B/HuggingFace Hub integration.

---

## 📋 Overview

**Model:** LongCat-Flash-Omni (560B parameters, 27B activated)
- State-of-the-art omni-modal model
- Shortcut-connected Mixture-of-Experts (MoE) architecture
- Supports up to 128K context tokens
- Real-time audio-visual interaction capabilities

**Task:** Video-Speech-to-Speech (VSS)
- **Inputs:** Video (.mp4) + Audio (.wav or .mp3)
- **Output:** Audio (.wav or .mp3)
- **Use Cases:** Video dubbing, audio replacement, multimodal speech synthesis

**Training Approach:**
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- DeepSpeed ZeRO-3 with CPU offloading for massive model support
- Optional Weights & Biases experiment tracking
- Optional HuggingFace Hub model sharing

---

## ⚠️ Important Notes

Before starting, please review these critical requirements:

### 1. Hardware Requirements
- This is a **560B parameter model** (27B activated)
- **Minimum:** 8x H100 (80GB) or 8x H200 (141GB) GPUs
- **System RAM:** 512GB+ for CPU offloading
- **Storage:** 2TB+ (model weights are ~1.1TB)

### 2. Data Structure
- Must follow exact naming: `input.mp4`, `input.wav`/`input.mp3`, `output.wav`/`output.mp3`
- Each sample in its own numbered folder: `01`, `02`, `03`, etc.
- Place in `data/training/` and optionally `data/test/`

### 3. Model Loading
- First run will download ~1.1TB from HuggingFace Hub
- Set `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads
- May require accepting terms at https://huggingface.co/meituan-longcat/LongCat-Flash-Omni

### 4. Training
- Uses LoRA (only ~200MB trainable parameters vs 560B total)
- DeepSpeed ZeRO-3 with CPU offload is essential
- Expect slow training due to model size (hours to days)

---

## ⚠️ Hardware Requirements

**Critical:** LongCat-Flash-Omni is a **560 billion parameter** model (27B activated per token). Training requires **substantial** computational resources even with LoRA + DeepSpeed ZeRO-3.

### Minimum Requirements

**For Training (with LoRA + ZeRO-3 + CPU offload):**
- **GPUs:** 8x H100 (80GB) or 8x H200 (141GB)
- **System RAM:** 512GB+ (for CPU offloading)
- **Storage:** 2TB+ NVMe SSD (model weights ~1.1TB in BF16)
- **Network:** High-speed interconnect (InfiniBand recommended)

**For Inference Only:**
- Minimum: 1x node with 8x H20 (141GB) in FP8
- Recommended: 2x nodes with 16x H800 (80GB) in BF16

### Why So Much Hardware?

Even with aggressive optimizations:
- **Base model weights:** ~1.1TB (560B params × 2 bytes for BF16)
- **LoRA adapters:** ~200MB (trainable parameters only)
- **Optimizer states (ZeRO-3):** Sharded across GPUs + CPU offload
- **Activations:** Gradient checkpointing + ZeRO-3 partitioning

**Bottom Line:** If you don't have 8+ high-end datacenter GPUs, consider:
1. Using a smaller model (e.g., Mistral-7B, Llama-2-13B)
2. Running inference only (no training)
3. Cloud services (CoreWeave, RunPod, Lambda Labs)

---

## 📁 Data Structure

Organize your data as follows:

```
data/
├── training/
│   ├── 01/
│   │   ├── input.mp4      # Video input
│   │   ├── input.wav      # Audio input (or .mp3)
│   │   └── output.wav     # Target audio output (or .mp3)
│   ├── 02/
│   │   ├── input.mp4
│   │   ├── input.wav
│   │   └── output.wav
│   ├── 03/
│   │   ├── input.mp4
│   │   ├── input.mp3      # .mp3 also supported
│   │   └── output.mp3     # .mp3 also supported
│   └── ...
└── test/
    └── (same structure)
```

**Requirements:**
- Each sample folder must contain: `input.mp4`, `input.wav` (or `.mp3`), `output.wav` (or `.mp3`)
- Folder names can be numeric (01, 02, ...) or any unique identifier
- Audio files can be `.wav` or `.mp3` (will be automatically resampled to 16kHz)
- Video files must be `.mp4` format

---

## 🚀 Quick Start

### 1. Initialize Project with `uv`

```bash
# Navigate to this folder
cd 09_vss

# Initialize uv project
uv init

# The uv tool will create pyproject.toml and .python-version files
```

### 2. Install Dependencies

```bash
# Core dependencies for training
uv add torch torchvision torchaudio transformers accelerate datasets deepspeed peft

# Additional required packages
uv add opencv-python pillow numpy

# Required: TensorBoard for training logs
uv add tensorboard

# Required: Fast model downloads from HuggingFace
uv add hf_transfer

# Optional: Install W&B for experiment tracking
uv add wandb

# Optional: Install HuggingFace Hub for model uploads
uv add huggingface_hub
```

**Complete Dependency List:**

| Package | Purpose | Required? |
|---------|---------|-----------|
| `torch` | Deep learning framework | ✅ Required |
| `torchvision` | Computer vision utilities | ✅ Required |
| `torchaudio` | Audio processing | ✅ Required |
| `transformers` | HuggingFace models | ✅ Required |
| `accelerate` | Distributed training | ✅ Required |
| `datasets` | Dataset management | ✅ Required |
| `deepspeed` | Memory optimization | ✅ Required |
| `peft` | LoRA implementation | ✅ Required |
| `opencv-python` | Video processing | ✅ Required |
| `pillow` | Image processing | ✅ Required |
| `numpy` | Numerical operations | ✅ Required |
| `tensorboard` | Training visualization | ✅ Required |
| `hf_transfer` | Fast downloads | ✅ Recommended |
| `wandb` | Experiment tracking | ⭐ Optional |
| `huggingface_hub` | Model sharing | ⭐ Optional |

### 3. Prepare Your Data

```bash
# Create data directory structure
mkdir -p data/training data/test

# Add your samples (example)
mkdir -p data/training/01
cp /path/to/video.mp4 data/training/01/input.mp4
cp /path/to/input_audio.wav data/training/01/input.wav
cp /path/to/output_audio.wav data/training/01/output.wav

# Repeat for more samples...
```

### 4. Configure Environment Variables

```bash
# Optional: Weights & Biases tracking
export WANDB_API_KEY=your_wandb_api_key
# Get key from: https://wandb.ai/authorize

# Optional: HuggingFace Hub upload
export HF_TOKEN=your_huggingface_token
# Get token from: https://huggingface.co/settings/tokens

# Optional: Set your HuggingFace username (for hub uploads)
export HF_USER=your_hf_username

# Optional: Enable fast downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Optional: Control hub upload behavior
export PUSH_TO_HUB=true  # or false to disable
```

### 5. Run Training

**Multi-GPU Training (Recommended):**

```bash
# Train with DeepSpeed on all available GPUs
uv run deepspeed --num_gpus=8 train_ds.py

# Or specify exact number of GPUs
uv run deepspeed --num_gpus=4 train_ds.py
```

**Single-GPU Training (Not Recommended for 560B Model):**

```bash
# This will likely fail due to memory constraints
uv run deepspeed --num_gpus=1 train_ds.py
```

**SLURM Cluster (Coming Soon):**

```bash
# Submit batch job (run_deepspeed.sh to be added)
sbatch run_deepspeed.sh
```

---

## 📊 Monitoring Training

### TensorBoard (Local)

```bash
# Start TensorBoard in a separate terminal
tensorboard --logdir=./tensorboard_logs/

# Open browser to: http://localhost:6006
```

### Weights & Biases (Optional)

If you set `WANDB_API_KEY`, training metrics will automatically sync to W&B:

```bash
# View your runs at:
https://wandb.ai/your-username/projects
```

**Key Metrics to Monitor:**
- Training loss (should decrease steadily)
- Learning rate (cosine schedule with warmup)
- GPU memory usage (should be stable)
- Throughput (samples/second)

---

## 🔧 Configuration

### DeepSpeed Configuration (`ds_config.json`)

Current configuration uses **ZeRO Stage 3** with aggressive memory optimization:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},  // Offload optimizer to CPU
    "offload_param": {"device": "cpu"},      // Offload params to CPU
    "stage3_max_live_parameters": 1e9,       // Max params in GPU memory
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

**Key Settings:**
- **ZeRO-3:** Shards optimizer states, gradients, and parameters across all GPUs
- **CPU Offload:** Moves optimizer and parameters to CPU RAM when not in use
- **BF16 Precision:** Better numerical stability than FP16 for large models
- **Gradient Checkpointing:** Trades computation for memory (reduces activations)

### LoRA Configuration (`train_ds.py`)

```python
lora_config = LoraConfig(
    r=32,           # LoRA rank (higher = more capacity, more memory)
    lora_alpha=64,  # Scaling factor
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",     # MLP layers
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Trainable Parameters:**
- Original model: 560B parameters (frozen ❄️)
- LoRA adapters: ~200M parameters (trainable 🔥)
- **Reduction:** 99.96% fewer trainable parameters!

### Training Hyperparameters

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=1,      # Very small due to model size
    gradient_accumulation_steps=32,     # Effective batch size = 32 × num_gpus
    learning_rate=1e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    bf16=True,
)
```

**Adjust These If:**
- **OOM Errors:** Reduce `per_device_train_batch_size` to 1 (already minimum), increase `gradient_accumulation_steps`
- **Slow Training:** Increase `per_device_train_batch_size` if you have memory headroom
- **Poor Convergence:** Increase `learning_rate` or adjust `warmup_ratio`

---

## 📂 Output Structure

After training, you'll find:

```
09_vss/
├── longcat-flash-omni-vss-lora/     # Model checkpoint directory
│   ├── adapter_config.json          # LoRA configuration
│   ├── adapter_model.safetensors    # LoRA weights (~200MB)
│   ├── training_args.bin            # Training configuration
│   └── checkpoint-*/                # Intermediate checkpoints
├── tensorboard_logs/                # TensorBoard logs
│   └── events.out.tfevents.*
└── logs/                            # SLURM logs (if using batch scripts)
```

**Model Size:**
- Base model: ~1.1TB (downloaded once from HuggingFace Hub)
- LoRA adapters: ~200MB (what you actually train and save)
- **Storage needed:** 1.5TB total (base model + checkpoints + logs)

---

## 🤝 HuggingFace Hub Integration

### Automatic Upload (During Training)

If `HF_TOKEN` is set, the model will automatically upload to HuggingFace Hub after training:

```bash
export HF_TOKEN=your_token_here
export HF_USER=your_username

# Model will be uploaded to: your_username/longcat-flash-omni-vss-lora
uv run deepspeed --num_gpus=8 train_ds.py
```

### Manual Upload (After Training)

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./longcat-flash-omni-vss-lora",
    repo_id="your-username/longcat-flash-omni-vss-lora",
    repo_type="model",
)
```

### Loading Your Fine-tuned Model

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meituan-longcat/LongCat-Flash-Omni",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Load your LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "your-username/longcat-flash-omni-vss-lora"
)

# Merge and unload (optional, for inference)
model = model.merge_and_unload()
```

---

## 🛠️ Troubleshooting

### 1. Out of Memory (OOM) Errors

**Problem:** `torch.cuda.OutOfMemoryError`

**Solutions:**
```bash
# A. Reduce batch size (already at minimum = 1)
# B. Increase gradient accumulation
# Edit train_ds.py line ~280:
gradient_accumulation_steps=64  # Increase from 32

# C. Enable more aggressive CPU offloading
# Edit ds_config.json:
"stage3_max_live_parameters": 5e8  # Reduce from 1e9
```

### 2. Model Download Fails

**Problem:** `Connection timeout` or `403 Forbidden`

**Solutions:**
```bash
# A. Use HF_TRANSFER for faster/more reliable downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
uv add hf_transfer

# B. Authenticate with HuggingFace
huggingface-cli login

# C. Check model access (may require agreement to terms)
# Visit: https://huggingface.co/meituan-longcat/LongCat-Flash-Omni
```

### 3. No Data Found

**Problem:** `ValueError: No valid samples found`

**Solutions:**
```bash
# A. Verify data structure
ls -R data/training/

# B. Check file naming (must be exact)
# Correct: input.mp4, input.wav, output.wav
# Wrong: Input.mp4, audio.wav, target.mp3

# C. Ensure at least one complete sample exists
data/training/01/input.mp4   ✅
data/training/01/input.wav   ✅
data/training/01/output.wav  ✅
```

### 4. DeepSpeed Initialization Fails

**Problem:** `RuntimeError: NCCL error`

**Solutions:**
```bash
# A. Check CUDA/NCCL versions
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.nccl.version())"

# B. Verify all GPUs are visible
nvidia-smi

# C. Set NCCL debug level
export NCCL_DEBUG=INFO
uv run deepspeed --num_gpus=8 train_ds.py

# D. Use different NCCL backend
export NCCL_IB_DISABLE=1  # Disable InfiniBand
export NCCL_P2P_DISABLE=1  # Disable peer-to-peer
```

### 5. Video/Audio Loading Errors

**Problem:** `cv2.error` or `torchaudio` errors

**Solutions:**
```bash
# A. Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libsndfile1 ffmpeg libavcodec-extra

# B. Verify file formats
file data/training/01/input.mp4
file data/training/01/input.wav

# C. Re-encode problematic files
ffmpeg -i input.mp4 -c:v libx264 -preset fast input_fixed.mp4
ffmpeg -i input.wav -ar 16000 -ac 1 input_fixed.wav
```

---

## 📚 Model Information

### LongCat-Flash-Omni

**Paper:** [LongCat-Flash-Omni: Efficient Omni-Modal Language Model](https://huggingface.co/meituan-longcat/LongCat-Flash-Omni)

**Architecture:**
- **Type:** Mixture-of-Experts (MoE) Causal Language Model
- **Total Parameters:** 560 billion
- **Activated Parameters:** 27 billion per token
- **Context Length:** 128K tokens
- **Precision:** BF16 (mixed precision training)

**Capabilities:**
- Multimodal understanding (text, image, video, audio)
- Speech generation and synthesis
- Long-context reasoning
- Real-time audio-visual interaction

**Benchmarks:**
- MMLU: 90.30% accuracy
- MATH500: 97.60% accuracy
- LibriSpeech ASR: 1.57% CER (test-clean)

**License:** MIT License (with trademark/patent restrictions)

---

## 🔬 Advanced Usage

### Custom Data Preprocessing

Edit `preprocess_function()` in `train_ds.py` to customize:

```python
def preprocess_function(examples: Dict) -> Dict:
    # Add your custom preprocessing here
    # Example: apply data augmentation, different frame sampling, etc.
    pass
```

### Custom Training Callbacks

Add custom callbacks to the Trainer:

```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Custom logic at epoch end
        pass

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[CustomCallback()],  # Add here
)
```

### Multi-Node Training

For training across multiple nodes:

```bash
# Node 0 (master)
deepspeed --num_gpus=8 --num_nodes=2 --master_addr=node0_ip --master_port=29500 train_ds.py

# Node 1
deepspeed --num_gpus=8 --num_nodes=2 --master_addr=node0_ip --master_port=29500 train_ds.py
```

---

## 📖 References

- [LongCat-Flash-Omni Model Card](https://huggingface.co/meituan-longcat/LongCat-Flash-Omni)
- [LongCat-Flash-Omni GitHub](https://github.com/meituan-longcat/LongCat-Flash-Omni)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft/)
- [Weights & Biases](https://docs.wandb.ai/)
- [HuggingFace Hub](https://huggingface.co/docs/hub/)

---

## 🤝 Contributing

This is a template implementation. Contributions welcome:

1. Better multimodal data loading
2. Custom collators for video+audio batching
3. Evaluation scripts
4. Inference examples
5. SLURM batch scripts

---

## ⚖️ License

This training code is released under MIT License.

**Note:** LongCat-Flash-Omni model is also under MIT License with restrictions on Meituan's trademarks and patents. See [model card](https://huggingface.co/meituan-longcat/LongCat-Flash-Omni) for details.

---

## 🙏 Acknowledgments

- **Meituan LongCat Team** for releasing LongCat-Flash-Omni
- **Microsoft DeepSpeed** for memory optimization
- **HuggingFace** for PEFT and model hosting
- **PyTorch** for deep learning framework

---

**Happy Training!** 🚀

If you encounter issues or have questions, please check the [troubleshooting section](#-troubleshooting) or open an issue on GitHub.
