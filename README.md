# QLoRA Fine-Tuning on Llama-3.2-3B

Fine-tune Llama-3.2-3B on instruction-following data using QLoRA (4-bit quantization + LoRA adapters). Runs on a free Colab T4 GPU in ~30 minutes.

## Requirements

- Google Colab with **T4 GPU** (Runtime → Change runtime type → T4 GPU)
- HuggingFace account with access to [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) (accept license first)
- HF token with read access

```
pip install bitsandbytes transformers peft trl accelerate datasets huggingface_hub
```

## What it does

| Step | Detail |
|------|--------|
| Model | `meta-llama/Llama-3.2-3B` (3.2B params) |
| Quantization | NF4 4-bit via `bitsandbytes` — reduces GPU memory from ~6GB → ~1.5GB |
| Adapter | LoRA injected into all attention + MLP projection layers (`r=16`, `alpha=32`) |
| Trainable params | ~24M / 3.2B — **0.75%** of total |
| Dataset | `HuggingFaceH4/ultrachat_200k` — 2,000 examples (instruction-following conversations) |
| Training | 100 steps, effective batch size 8 (2 × 4 grad accum), cosine LR schedule, `paged_adamw_8bit` |
| Runtime | ~30 min on T4 |

## Training results

Loss decreases from ~1.65 → ~1.41 over 100 steps, confirming the adapter is learning.

## Output

Only the LoRA adapter weights are saved (~63MB), not the full base model (~6GB):

```
adapter_weights/
├── adapter_config.json        # LoRA hyperparameters
├── adapter_model.safetensors  # Trained adapter weights (~47MB)
├── tokenizer.json
└── tokenizer_config.json
```

## Loading the adapter later

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base, "./adapter_weights")
```
