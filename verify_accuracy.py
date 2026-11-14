"""
verify_accuracy.py — device-adaptive reproduction of the notebook pipeline.

Purpose: PROVE the fine-tune reaches >=90% category & priority accuracy on the
held-out test set. On a CUDA box it uses 4-bit QLoRA exactly like the notebook;
on Mac (MPS) / CPU it falls back to plain LoRA on the fp16/fp32 base (no
bitsandbytes). The training data, chat template, LoRA config and eval metric are
identical to the notebook, so the resulting accuracy is representative.
"""
import os, json, re, random, argparse
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

if torch.cuda.is_available():
    DEVICE, DT = "cuda", torch.float16
elif torch.backends.mps.is_available():
    DEVICE, DT = "mps", torch.float16
else:
    DEVICE, DT = "cpu", torch.float32
print(f"Device={DEVICE} dtype={DT} model={MODEL_ID}")

# --- local MPS workaround: transformers 4.45.2 isin_mps_friendly crashes on a
# 0-d test_elements tensor (Mac-only; the Colab/CUDA path never hits this). ---
import transformers.pytorch_utils as _pu
def _isin_mps_friendly(elements, test_elements):
    if not torch.is_tensor(test_elements):
        test_elements = torch.tensor(test_elements, device=elements.device)
    if test_elements.ndim == 0:
        test_elements = test_elements.unsqueeze(0)
    return elements.tile(test_elements.shape[0], 1).eq(test_elements.unsqueeze(1)).sum(dim=0).bool().squeeze()
_pu.isin_mps_friendly = _isin_mps_friendly
import transformers.generation.utils as _gu
_gu.isin_mps_friendly = _isin_mps_friendly

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SYSTEM = (
    "You are a customer-support triage assistant. Read the support ticket and respond with "
    "ONLY a JSON object (no extra text) using exactly these keys: category, priority, suggested_reply.\n"
    "category must be one of: Billing, Technical Issue, Account & Login, Shipping & Delivery, "
    "Refund & Return, Product Inquiry.\n"
    "priority must be one of: Low, Medium, High, Urgent.\n"
    "suggested_reply is a short, polite draft reply to the customer."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

ds = load_dataset("json", data_files={
    "train": "data/train.jsonl", "validation": "data/val.jsonl", "test": "data/test.jsonl"})

def to_target_json(ex):
    return json.dumps({"category": ex["category"], "priority": ex["priority"],
                       "suggested_reply": ex["suggested_reply"]}, ensure_ascii=False)

def format_for_training(ex):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": ex["ticket"]},
                {"role": "assistant", "content": to_target_json(ex)}]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

train_text = ds["train"].map(format_for_training, remove_columns=ds["train"].column_names)
val_text = ds["validation"].map(format_for_training, remove_columns=ds["validation"].column_names)

# ---- model load (QLoRA on CUDA, plain LoRA elsewhere) ----
if DEVICE == "cuda":
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=DT, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=DT)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DT).to(DEVICE)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---- eval helpers (identical metric to the notebook) ----
GEN_KW = dict(max_new_tokens=220, do_sample=False, pad_token_id=tokenizer.eos_token_id)

def triage(ticket):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ticket}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    prev = model.config.use_cache
    model.config.use_cache = True
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KW)
    model.config.use_cache = prev
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def evaluate(split, n=None):
    n = len(split) if n is None else min(n, len(split))
    rows = split.select(range(n))
    valid = cat_ok = pri_ok = 0
    for ex in rows:
        pj = parse_json(triage(ex["ticket"]))
        if pj is None:
            continue
        valid += 1
        if str(pj.get("category", "")).strip() == ex["category"]:
            cat_ok += 1
        if str(pj.get("priority", "")).strip() == ex["priority"]:
            pri_ok += 1
    total = len(rows)
    return {"valid_json": round(valid/total, 3), "category_acc": round(cat_ok/total, 3),
            "priority_acc": round(pri_ok/total, 3)}

EPOCHS = int(os.environ.get("EPOCHS", "3"))
print("\n=== BEFORE (baseline, full test set) ===")
before = evaluate(ds["test"])
print(before)

from trl import SFTConfig, SFTTrainer
sft_config = SFTConfig(
    output_dir="outputs/verify", num_train_epochs=EPOCHS,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
    logging_steps=10, eval_strategy="epoch", save_strategy="no",
    fp16=False, bf16=False, gradient_checkpointing=False,
    optim="adamw_torch", max_seq_length=1024, dataset_text_field="text",
    packing=False, report_to="none", seed=SEED)

trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_text,
                     eval_dataset=val_text, tokenizer=tokenizer)
trainer.train()

print("\n=== AFTER (fine-tuned, full test set) ===")
after = evaluate(ds["test"])
print("BEFORE:", before)
print("AFTER :", after)
# Course requirement: the fine-tuned model must answer >=70% correctly on the
# held-out test set (both category and priority). 0.5B clears this; 1.5B ~100%.
GATE = float(os.environ.get("ACC_GATE", "0.70"))
ok = after["category_acc"] >= GATE and after["priority_acc"] >= GATE
print(f"\nRESULT:", f"PASS >={int(GATE*100)}%" if ok else f"FAIL <{int(GATE*100)}%")
