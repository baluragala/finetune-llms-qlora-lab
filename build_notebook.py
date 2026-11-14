"""
build_notebook.py — constructs Fine_Tuning_LLMs_S2_Python_Bala_Krishna_Ragala_V2.ipynb.

Run:  python3 build_notebook.py
It also compiles every Python code cell to catch syntax errors before class.
"""
import json
import re
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Embed the dataset generator source into a notebook cell so students see it inline.
with open(os.path.join(HERE, "generate_dataset.py"), encoding="utf-8") as f:
    DATAGEN = f.read()
DATAGEN = DATAGEN.replace('if __name__ == "__main__":\n    main()',
                          'main()  # generate data/train.jsonl, val.jsonl, test.jsonl now')

cells = []


def md(text):
    cells.append(("markdown", text, False))


def code(text, shell=False):
    cells.append(("code", text.rstrip("\n"), shell))


# ───────────────────────────────────────────────────────────────────────────
md(r"""# 🛠️ Fine-Tuning LLMs — Session 2 (V2): A Customer-Support Triage Assistant

*Hands-on lab · Bala Krishna Ragala*

| | |
|---|---|
| **Model** | `Qwen/Qwen2.5-0.5B-Instruct` (small, open, instruction-tuned) |
| **Method** | **QLoRA**: 4-bit frozen base + small trainable **LoRA** adapters, using `trl`'s `SFTTrainer` |
| **Hardware** | Free Google Colab **T4 GPU** (~15 GB VRAM). Local-GPU differences are flagged inline. |
| **Time** | ~5 min training · ~60–90 min for the whole lab with discussion |

---

## 📌 The business context: why this lab exists

A SaaS company's support desk receives **hundreds of tickets a day** by email and chat. Before anyone can
fix a problem, someone has to **triage** each ticket:

1. **What kind of problem is it?** (billing? login? shipping?) → decides *which team* gets it.
2. **How urgent is it?** → decides *how fast* someone must respond.
3. **What's a sensible first reply?** → buys time and reassures the customer.

Today a human agent does this by hand. It is slow and repetitive, and agents disagree with each other. A
"Low" ticket from one agent is "High" from another. Misrouted tickets bounce between teams, and urgent
ones (*"production API is down"*) sit in the queue behind password resets.

## ❓ Problem statement

> **Given the raw text of a customer support ticket, automatically produce a structured triage decision
> (category, priority, and a draft reply) in an exact, machine-readable JSON format that the
> ticketing system can consume directly, with no human reformatting.**

Why is this *hard* for an off-the-shelf LLM, even a good one?

| What we need | What a general LLM does out of the box |
|---|---|
| One of **our 6 categories**, spelled exactly: `Billing`, `Technical Issue`, `Account & Login`, `Shipping & Delivery`, `Refund & Return`, `Product Inquiry` | Invents its own labels (`"Billing and Service Issues"`, `"Payment problem"`) |
| One of **our 4 priorities**: `Low`, `Medium`, `High`, `Urgent`, applied by *our* policy | Guesses, with no idea what "Urgent" means *at this company* |
| **Only JSON**, with exactly 3 keys, every single time | Adds chatty preambles, markdown fences, extra keys, or stops mid-object |
| Small and cheap enough to run on **every** ticket | Big API models work, but cost money per ticket and send customer data off-site |

A downstream program can't use output that is *usually* right. It needs output that is
**reliably right in the exact format**.

## 💼 The use case (concretely)

**Input:** a raw ticket, written however the customer felt like writing it:
```text
Hi, I was charged $49 twice for my Pro plan this month. Can you fix this ASAP?
```

**Output:** exactly this shape, nothing more, nothing less:
```json
{"category": "Billing",
 "priority": "High",
 "suggested_reply": "Hi, I'm sorry about the duplicate charge on your Pro plan. I've flagged it to billing..."}
```

**Who consumes it:** the ticketing system. `category` routes the ticket to a team queue, `priority` sets the
SLA timer, and `suggested_reply` pre-fills the agent's response box. The human still approves; the model
does the grunt work.

## 🤔 Why *fine-tuning*, and not just a better prompt?

| Approach | Why it falls short here |
|---|---|
| **Prompt engineering** ("reply only in JSON, use these labels…") | A 0.5B model follows long instructions inconsistently. Every request also carries a long prompt, which costs latency and tokens. |
| **RAG** (retrieve docs into the prompt) | RAG adds *knowledge*. Our gap is *behaviour*: a consistent format and *our* labelling policy. There's nothing to look up. |
| **Big hosted LLM via API** | Works, but costs money per ticket, adds latency, and sends customer PII to a third party. |
| ✅ **Fine-tune a small model** | Bakes the format and policy *into the weights*. Short prompts, fast, cheap, private, consistent. |

**Rule of thumb:** *prompting/RAG change what the model **knows** at request time; fine-tuning changes how
the model **behaves** every time.* Our problem is a behaviour problem.

## 🎯 Goal of this hands-on

Take a small, general instruction-tuned model that **fails** at our triage task and, using **QLoRA on a free
T4 GPU in a few minutes**, turn it into a specialist that **reliably** emits correct, schema-valid triage JSON.
Then **prove** the improvement with numbers and side-by-side examples on tickets it has never seen.

## ✅ Learning objectives: by the end you will be able to…

1. **Explain** the difference between a *base* and an *instruct* model, and why we start from instruct.
2. **Design** a fine-tuning dataset: define the output schema *first*, then generate balanced, labelled
   examples and split them into train / validation / test.
3. **Format** data with the model's **chat template**, and choose a sequence length that never cuts off
   the answer.
4. **Load** a model in **4-bit** and attach **LoRA** adapters, and explain what each knob (`r`, `alpha`,
   `target_modules`, `dropout`) does and why it's set that way.
5. **Configure and run** `SFTTrainer` within T4 memory limits (fp16, gradient checkpointing, batch size and
   gradient accumulation).
6. **Read a training curve** and tell a healthy run from one that's overfitting or broken.
7. **Evaluate** before vs after on a held-out test set with task-specific metrics, not just loss.
8. **Save, merge, and run inference** with the fine-tuned adapter.

## 📏 How we'll know it worked (success criteria)

We measure the *same* model on the *same* held-out test tickets **before** and **after** training:

| Metric | What it checks | Before (typical) | Target after |
|---|---|---|---|
| **Valid JSON rate** | Can a program parse the output at all? | high, but not 100% | **100%** |
| **Category accuracy** | Right team? (exact match to our 6 labels) | ~30% | **≥ 90%** |
| **Priority accuracy** | Right urgency? (exact match to our 4 levels) | ~35% | **≥ 85%** |

*(The "before" numbers come from a dry run of this exact notebook. Yours will be close but not identical.)*

## 🗺️ Roadmap: what we'll do, in order

| Step | Section | Question it answers |
|---|---|---|
| Set up | §0–1 | Do we have the GPU and the exact library versions? |
| Motivate | §1c | Why start from an *instruct* model, and why isn't that enough? |
| Data | §2 | What does "good" look like, and how do we get labelled examples? |
| Prepare | §3–4 | How do we turn examples into tokens the model trains on? |
| Model | §5–6 | How do we fit training into 15 GB? (4-bit + LoRA) |
| Baseline | §7 | How bad is the model *before* we train? |
| Train | §8 | How do we train, and how do we know training is healthy? |
| Prove | §9 | Did it actually get better, on tickets it never saw? |
| Ship | §10 | How do we save, merge, and use it? |
| Go further | §11–13 | Which knobs to turn next, and how to fix common errors |

**Prerequisites:** comfortable reading Python; a basic idea of what an LLM and a token are. No prior
fine-tuning experience needed.

---

### How to read this notebook

Every section uses the same layout, so you always know where to look:

| Block | What it tells you |
|---|---|
| **📍 In one line** | The whole section in a sentence. Skim these alone and you have the story. |
| **❓ Why** | The problem this step solves, and what goes wrong if you skip it. |
| **🧩 What** | The concept or the knobs involved, in plain words. |
| **⚙️ How** | What the code cell below actually does. |
| **👀 What you should see** | The expected output, so you can tell right away whether your run is healthy. |

Instructor notes are in `> 🎙 Talking track` blockquotes. Run the cells **top to bottom**, and look for two flags:
- 🔁 **SWAP HERE**: where you'd plug in your own dataset or a different base model.
- ⚠️ **T4 NOTE**: where free-Colab hardware changes a setting.

### Key terms (quick glossary)
| Term | Plain meaning |
|---|---|
| **Fine-tuning** | Continue training a pretrained model on *your* examples so it learns *your* task. |
| **SFT** (supervised fine-tuning) | Fine-tuning on (input → ideal output) pairs. |
| **LoRA** | Train tiny "add-on" matrices instead of the whole model. Cheap, and the original weights stay untouched. |
| **QLoRA** | LoRA on top of a base model compressed to **4-bit**, so it fits on a small GPU. |
| **Adapter** | The small file of trained LoRA weights (tens of MB, vs ~1 GB for the full model). |
| **Chat template** | The exact text format (special tokens for system/user/assistant) an instruct model was trained on. |
| **VRAM** | GPU memory. The main constraint on what you can train. |
| **Loss** | How wrong the model's next-token predictions are. Lower is better. |

> **🎙 Talking track (open the session):** "We are not building a chatbot today. We are taking a
> capable generalist and giving it a single, sharp, reliable skill — like sending a smart new hire
> through a focused onboarding so triage becomes muscle memory. The base model already *can* read
> English; what it lacks is the *habit* of always answering in our exact structured format. That habit
> is what fine-tuning installs."

### The central analogy (threaded through the lab)
The base model is a **culinary-school chef**: broadly skilled, not yet *ours*. Today is the
**apprenticeship** — we drill one house dish (structured triage) until the chef produces it on reflex.""")

md(r"""## 0 · Hardware & VRAM: know your kitchen before you cook

**📍 In one line:** check that a GPU is attached and see how much memory it has, because memory decides
what we can train.

**❓ Why:** fine-tuning is limited by **GPU memory (VRAM)**, not by cleverness. A free T4 has **~15 GB**.
Storing a model's weights is the cheap part. *Training* also needs gradients, optimizer state (Adam keeps 2
extra numbers per weight) and activations, which adds up to roughly **4–5× the model size**. Without tricks,
even a 1.5B model blows past 15 GB.

**🧩 What QLoRA buys us.** The budget below is for a 1.5B model, the *harder* case. Our default 0.5B model
has even more headroom, and the same maths is what lets you scale up later.

| Item | Full fine-tune (fp16) | **QLoRA (this lab)** |
|---|---|---|
| Base weights | ~3 GB | **~1 GB (4-bit)** |
| Gradients + optimizer | ~12 GB+ | **~tens of MB (LoRA only)** |
| Activations | high | lower (gradient checkpointing) |
| **Fits a 15 GB T4?** | ❌ tight/no | ✅ comfortably (~6–8 GB) |

**⚙️ How (the four tricks we'll use, one per later section):**
1. Load the base model in **4-bit** (§5).
2. **Freeze** it, so no gradients or optimizer state for the big weights.
3. Train only small **LoRA** adapters (§6).
4. Use **gradient checkpointing**: recompute activations instead of storing them (§8).

The cell below runs `nvidia-smi`, the GPU's status report.

**👀 What you should see:** a table that names a **Tesla T4** with about **15360 MiB** total memory and
almost none of it used. If you get `command not found` or *"NVIDIA-SMI has failed"*, no GPU is attached: go to
**Runtime ▸ Change runtime type ▸ T4 GPU**.

> **🎙 Talking track:** "Why not just fully fine-tune a 1.5B model? It's small! Because *training* memory
> is dominated by the optimizer and activations, not the weights themselves. QLoRA sidesteps both: a frozen
> 4-bit base costs almost nothing to keep, and we only compute gradients for about 1–2% of the parameters."

⚠️ **T4 NOTE:** the T4 is a *Turing* GPU. It supports **fp16** but not efficient **bf16**, so we set
`compute_dtype=float16` and `fp16=True`. On an A100/L4/RTX-30xx+ (Ampere or newer), prefer **bf16** because
it is more numerically stable. That is the only dtype change between a T4 and a bigger GPU.""")

code("!nvidia-smi", shell=True)

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 1 · Environment setup: pinned versions (this is not optional)

**📍 In one line:** install an exact, tested set of library versions so every student's run behaves the
same.

**❓ Why pin?** The `transformers`/`peft`/`trl` ecosystem moves fast and makes **breaking API changes**
between minor versions (the `SFTTrainer` constructor alone has changed several times). A notebook that
"worked last month" can break tomorrow if `pip install` picks the newest versions. Pinning means
reproducibility: 30 students getting the same result instead of 30 different stack traces.

**🧩 What we install (a verified-compatible set):**
```
transformers==4.45.2   peft==0.13.2   trl==0.11.4
datasets==3.0.1        accelerate==1.0.1   bitsandbytes==0.49.2
```

⚠️ **Why bitsandbytes 0.49.2 specifically:** `bitsandbytes` ships pre-compiled GPU code, one file per CUDA
version, and it must have the one matching the PyTorch build Colab provides. Colab currently ships
**Python 3.13 + PyTorch built for CUDA 13.0**. `0.49.2` bundles CUDA 11.8 through **13.0**; older releases
stop at 12.x and fail at the 4-bit load (§5) with `🚨 CUDA VERSION MISMATCH 🚨`, or with
`No module named 'triton.ops'` on even older ones. If Colab moves to a newer CUDA and you see that error,
run `!pip install -U bitsandbytes`, then restart.

**⚙️ How:** install, then **restart the runtime**. Colab imports some libraries at boot, and a restart
guarantees the new versions are the ones you actually use.

**👀 What you should see:** a quiet install (maybe some *"dependency resolver"* warnings, which are safe to
ignore) ending with `✅ Install done`.

> **🎙 Talking track:** "Treat the version block like a recipe's oven temperature. Change it freely once
> you understand the dish, but for your first bake use exactly these numbers."

⚠️ **T4 NOTE:** `bitsandbytes` ships CUDA kernels, and `0.49.2` covers Colab's current CUDA (13.0). On a
local machine, check that your PyTorch's CUDA version (`torch.version.cuda`) is one bitsandbytes ships,
or 4-bit loading will fail to find its kernels.""")

code(r"""# NOTE: bitsandbytes must include a binary for the CUDA version Colab's PyTorch uses
# (currently 13.0 -> needs bitsandbytes >= 0.48). If §5 later fails with
# "CUDA VERSION MISMATCH", run:  !pip install -U bitsandbytes   then restart the runtime.
!pip -q install \
  transformers==4.45.2 \
  peft==0.13.2 \
  trl==0.11.4 \
  datasets==3.0.1 \
  accelerate==1.0.1 \
  bitsandbytes==0.49.2 \
  && echo "✅ Install done. Now: Runtime ▸ Restart session, then continue from the next cell." \
  || echo "❌ pip install FAILED — read the error above before continuing."
""", shell=True)

md(r"""> ⚠️ **STOP — restart the runtime now** (Runtime ▸ Restart session). Then continue below.
> You do **not** need to re-run the `pip install` after restarting; the packages persist for the session.""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 1b · Imports & reproducibility

**📍 In one line:** import the libraries, print their versions, and fix the random seed so runs are
repeatable.

**❓ Why:** printed versions turn the notebook's output into a record of *exactly* what ran. Seeding means
that if your numbers differ from your neighbour's, the cause is a real difference, not random luck.

**🧩 What each library does:**

| Library | Its job in this lab |
|---|---|
| `transformers` | The model, the tokenizer, the 4-bit loader (`BitsAndBytesConfig`), and text generation |
| `bitsandbytes` | The actual 4-bit quantization maths (used *through* transformers) |
| `peft` | LoRA adapters (PEFT stands for Parameter-Efficient Fine-Tuning) |
| `trl` | `SFTTrainer`, a ready-made training loop for supervised fine-tuning |
| `datasets` | Fast loading and transforming of our JSONL files |
| `accelerate` | Device placement and mixed-precision plumbing (works behind the scenes) |

**⚙️ How:** import, print versions, seed Python/NumPy/PyTorch with `SEED = 42`, and detect the GPU.

**👀 What you should see:** the six versions matching the pins above, then `Device: cuda`,
`GPU: Tesla T4` and `VRAM: ~15.6 GB`. If you see `Device: cpu`, stop and attach a GPU first.""")

code(r"""import os, json, re, gc, random
import numpy as np
import torch

try:
    import transformers, peft, trl, datasets, accelerate
except ImportError as e:
    # Classic symptom of skipping the restart: Colab's preinstalled transformers was
    # imported earlier in this session, then pip swapped the files underneath it
    # (e.g. "cannot import name 'is_flax_available' from 'transformers.utils'").
    raise SystemExit(
        f"Import failed: {e}\n\n"
        "FIX: Runtime ▸ Restart session, then run from THIS cell down "
        "(no need to re-run the pip install)."
    )
print("transformers ", transformers.__version__)
print("peft         ", peft.__version__)
print("trl          ", trl.__version__)
print("datasets     ", datasets.__version__)
print("accelerate   ", accelerate.__version__)

# Import bitsandbytes defensively: if it's the wrong build for Colab's CUDA, give
# a clear fix instead of crashing the whole cell. (transformers loads bnb for 4-bit.)
try:
    import bitsandbytes
    print("bitsandbytes ", bitsandbytes.__version__)
except Exception as e:
    print("bitsandbytes ⚠️ import problem:", repr(e))
    print("FIX: run  !pip install -U bitsandbytes  then Runtime ▸ Restart session, and re-run.")

# Reproducibility: seed Python / NumPy / Torch so each run is comparable.
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nDevice:", device)
if device == "cuda":
    print("GPU   :", torch.cuda.get_device_name(0))
    print("VRAM  :", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB")
else:
    print("⚠️ No GPU detected. In Colab: Runtime ▸ Change runtime type ▸ T4 GPU.")""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 1c · Base (pretrained) vs Instruct: why the starting point matters

**📍 In one line:** send the same request to a *base* model and an *instruct* model, and see why we start
fine-tuning from the instruct one.

**❓ Why:** there are two very different kinds of model, and picking the wrong starting point wastes the
whole exercise.

**🧩 What's the difference?**

| | **Base / pretrained** (`Qwen2.5-0.5B`) | **Instruction-tuned** (`Qwen2.5-0.5B-Instruct`) |
|---|---|---|
| Trained to… | predict the next token, and nothing else | answer `(instruction → response)` pairs in a chat format |
| Behaves like… | a brilliant **autocomplete** | an **assistant** |
| Given a question it… | rambles, continues the text, echoes the prompt, doesn't stop | answers, in the format asked, then stops |

Same architecture, same size, same starting weights. The *only* difference is the extra instruction tuning.

**⚙️ How:** the first code cell loads each model briefly, generates one greedy answer to the same prompt
(*"classify this ticket, reply with ONLY JSON"*), then frees it. The second cell shows the two answers side
by side.

**👀 What you should see** (typical for a 0.5B pair):
- **BASE** ignores the instruction and **repeats or continues** the text: no JSON, no stopping.
- **INSTRUCT** returns clean JSON such as `{"category": "Billing and Service Issues"}`. It *follows the
  instruction and the format*, **but the label isn't one of ours** (we need exactly `Billing`).

That last point is the motivation for the whole lab. Instruction tuning gives us format-following for free,
but it does **not** know *our* six categories, *our* priority scale or *our* exact JSON keys. **That**
is the habit we install next with QLoRA.

> **🎙 Talking track (chef analogy):** "The base model is a culinary-school graduate who can cook anything
> but ignores your order ticket — they'll just keep cooking. The instruct model is the same chef *after*
> a week on the line: hand them a ticket, they make exactly that dish and call 'service'. Today's lab goes
> one step further — we take the instruct chef and drill our **one house dish** (JSON triage) until it's
> reflex. You can't drill a house dish into someone who won't read the ticket — which is why we start from
> an instruct model, not a base one."

⚠️ **T4 NOTE:** we load each small (0.5B) model briefly, generate, then **free it** (`del` + empty cache)
so nothing is left occupying VRAM before the real QLoRA training later.

🔑 **If Colab pops up *"Notebook does not have secret access… HF_TOKEN. Grant access?"*:** this appears
when you've saved a Hugging Face token in Colab's 🔑 Secrets panel. The Qwen models are **public**, so a token
isn't needed. Either answer works; **Cancel** keeps your token private. Note that the run **pauses until you
answer**, and it may ask again when §5 loads the model.""")

code(r"""# Same prompt -> base (pretrained) model vs instruction-tuned model, side by side.
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID   = "Qwen/Qwen2.5-0.5B"            # pretrained ONLY (no instruction tuning)
INSTRUCT_ID = "Qwen/Qwen2.5-0.5B-Instruct" # same model, AFTER instruction tuning

demo_prompt = ('Classify the support ticket below into ONE category and reply with ONLY a JSON '
               'object like {"category": "..."}.\n'
               "Ticket: I was charged twice for my Pro plan this month.")

def quick_generate(model_id, prompt, use_chat_template):
    "Load a model, generate once (greedy), free it. Returns the generated text."
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt  = torch.float16 if torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt).to(device)

    if use_chat_template:   # instruct models expect their chat format
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
    else:                   # base models just continue raw text
        text = prompt

    inputs = tok(text, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=80, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    del mdl, inputs, out; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return resp

base_reply     = quick_generate(BASE_ID,     demo_prompt, use_chat_template=False)
instruct_reply = quick_generate(INSTRUCT_ID, demo_prompt, use_chat_template=True)""")

code(r"""# Render the comparison as a side-by-side card (falls back to plain text).
def _c(s):
    s = (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s

try:
    from IPython.display import display, HTML
    display(HTML(f'''
    <div style="font-family:system-ui">
      <p style="font-size:13px"><b>Prompt:</b> {_c(demo_prompt)}</p>
      <table style="border-collapse:collapse;width:100%;font-size:12px">
        <tr style="background:#34495e;color:#fff">
          <th style="padding:8px;width:50%">❌ BASE — pretrained <code>Qwen2.5-0.5B</code><br>(ignores the instruction; repeats/continues text)</th>
          <th style="padding:8px;width:50%">✅ INSTRUCT — <code>Qwen2.5-0.5B-Instruct</code><br>(follows the instruction &amp; format)</th>
        </tr>
        <tr>
          <td style="vertical-align:top;padding:10px;border:1px solid #e74c3c;white-space:pre-wrap">{_c(base_reply)}</td>
          <td style="vertical-align:top;padding:10px;border:1px solid #27ae60;white-space:pre-wrap">{_c(instruct_reply)}</td>
        </tr>
      </table>
    </div>'''))
except Exception:
    print("PROMPT:", demo_prompt)
    print("\n--- BASE (pretrained Qwen2.5-0.5B) — tends to ramble/continue ---\n", base_reply)
    print("\n--- INSTRUCT (Qwen2.5-0.5B-Instruct) — follows the instruction ---\n", instruct_reply)

print("\nTakeaway: the BASE model ignores the instruction (often repeating/continuing); the INSTRUCT")
print("model follows it — but won't know OUR exact categories/priorities/keys. That gap is what the")
print("rest of this lab closes with QLoRA/LoRA fine-tuning.")""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 2 · Data strategy: generate a synthetic triage dataset

**📍 In one line:** create ~350 labelled support tickets and split them into train, validation and test files.

**❓ Why synthetic (for the classroom):** real support logs are proprietary, contain personal data (PII), and
need heavy cleaning. A **seeded synthetic generator** gives every student the *identical* dataset in seconds,
with labels we *know* are correct. So when accuracy jumps after training, we can trust the number.

**🧩 What the data looks like** (one JSON record per ticket):
```json
{"ticket": "Hi, I was charged $49 twice...", "category": "Billing",
 "priority": "High", "suggested_reply": "Hi Alex, I'm sorry about the duplicate charge..."}
```
- **6 categories** × **4 priorities**, **balanced** across categories, so the model can't score well by
  always guessing the most common class.
- A few templates are expanded combinatorially (**product × opener**) into ~350 natural, unique tickets.

**Why three splits?**

| Split | Size | Used for | Model learns from it? |
|---|---|---|---|
| **train** | ~280 | updating the weights | ✅ yes |
| **validation** | 35 | checking for overfitting *during* training | ❌ no |
| **test** | 35 | the final before/after score, touched only for evaluation | ❌ no |

Scoring on tickets the model never trained on is what makes the "after" number honest.

**⚙️ How:** the cell below *is* the generator script (`generate_dataset.py`), embedded so you can read it.
It writes `data/train.jsonl`, `data/val.jsonl` and `data/test.jsonl`. Read the comments; the code is part of
the lesson.

**👀 What you should see:** a short summary of how many records went into each file.

🔁 **SWAP HERE (use your own data):** replace this cell with your own loader that writes the **same four keys**
(`ticket`, `category`, `priority`, `suggested_reply`) to the same three files. Everything after this point
depends only on those keys.

> **🎙 Talking track:** "Notice the discipline: we wrote the *evaluation target* (the JSON schema) before
> we wrote a line of training data. In real projects, defining 'what good looks like' first is what
> separates a fine-tune that ships from one that wanders. Also note the balancing step — an imbalanced
> dataset teaches the model to be lazy."

> **🎙 Talking track (the format choice):** "We put the suggested reply *inside* the JSON. That means the
> model learns classification AND drafting in one shot, and downstream code gets one parseable object
> instead of scraping free text."

The full generator (read the comments — this is itself a teaching artifact):""")

code(DATAGEN)

md(r"""### 2b · Load & inspect: always look at your data before training

**📍 In one line:** load the three files and print one record plus the label counts.

**❓ Why:** the #1 cause of a "failed fine-tune" is bad data nobody looked at: wrong labels, empty fields,
one class making up 80% of examples. **Trust nothing you haven't printed.**

**⚙️ How:** `load_dataset("json", ...)` reads the JSONL files into a `DatasetDict` with `train`,
`validation` and `test` splits. Then we print one raw record and count categories and priorities.

**👀 What you should see:**
- Split sizes of about **281 / 35 / 35**.
- One record with all four keys filled in.
- Category counts that are **roughly equal** (about 44–52 each).
- Priority counts that are **not** equal: about 100 each for High and Low, ~50 Medium, and only **~20
  Urgent**. That mirrors real life (true emergencies are rare), but it also means the model sees few Urgent
  examples. Remember this when priority accuracy comes out lower than category accuracy in §9.""")

code(r"""from datasets import load_dataset

ds = load_dataset("json", data_files={
    "train":      "data/train.jsonl",
    "validation": "data/val.jsonl",
    "test":       "data/test.jsonl",
})
print(ds)

print("\n--- one raw training record ---")
print(json.dumps(ds["train"][0], indent=2, ensure_ascii=False))

from collections import Counter
print("\nTrain category balance:", dict(Counter(ds["train"]["category"])))
print("Train priority balance:", dict(Counter(ds["train"]["priority"])))""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 3 · The chat template: speak the model's native dialect

**📍 In one line:** turn each (ticket, answer) record into one string in the exact chat format the model was
trained on.

**❓ Why:** instruction-tuned models were trained on a **specific chat format**, with special tokens marking
the *system*, *user* and *assistant* turns. Feed plain text that ignores this format and you fight the
model's training, and quality collapses. `tokenizer.apply_chat_template` formats messages correctly for us,
using the template stored in the tokenizer.

**🧩 What one training example looks like (Qwen2.5 uses the "ChatML" format):**
```
<|im_start|>system
You are a customer-support triage assistant...<|im_end|>
<|im_start|>user
Hi, I was charged $49 twice...<|im_end|>
<|im_start|>assistant
{"category": "Billing", "priority": "High", "suggested_reply": "..."}<|im_end|>
```

| Turn | Contains | Role in training |
|---|---|---|
| **system** | the job description: allowed categories, priority scale, "reply with ONLY JSON" | the rules |
| **user** | the raw ticket | the input |
| **assistant** | our target JSON | the answer the model learns to write |

**⚙️ How the cell works:**
1. Load the tokenizer for `MODEL_ID` and make sure it has a padding token. We pad on the **right** for
   training.
2. Define `SYSTEM` (the job description) and `to_target_json` (builds the exact answer string with
   `json.dumps`, so it is always valid JSON).
3. `format_for_training` builds the 3-message list and renders it to a single `text` string.
4. `.map(...)` applies that to every train and validation record.

For **training** we include the assistant turn. For **inference** (from §7 on) we stop after the user turn
and pass `add_generation_prompt=True`, so the model writes the assistant turn itself.

**👀 What you should see:** one fully formatted example with `<|im_start|>system`, `<|im_start|>user` and
`<|im_start|>assistant` blocks, the assistant block holding the JSON answer. This string is *exactly* what the
model trains on.

> **🎙 Talking track:** "The system message is where we install the *job description*: the allowed
> categories, the priority scale, and 'reply with ONLY JSON'. The model already knows English; we're
> handing it the company handbook and the form to fill out."

🔁 **SWAP HERE (different base model):** change `MODEL_ID`. The template auto-adapts because every
instruct model ships its own chat template — that's the beauty of `apply_chat_template`.""")

code(r"""from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"   # 🔁 SWAP HERE for another instruct model
# Why 0.5B for class: it trains in a few minutes on a free T4 and already clears
# our accuracy bar on this narrow task. Swap up for even stronger results:
#   "Qwen/Qwen2.5-1.5B-Instruct"        (proven ~100% on this lab; still T4-friendly)
#   "meta-llama/Llama-3.2-1B-Instruct"  (gated: accept license on the Hub first)
#   "microsoft/Phi-3-mini-4k-instruct"  (3.8B: heavier on a T4 during training)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Causal LMs need a PAD token to batch sequences of different lengths. Qwen2.5
# already defines one (<|endoftext|>), distinct from its end-of-turn token
# (<|im_end|>). Some models don't, so we fall back to reusing EOS as PAD.
# Caution: if PAD == EOS, the collator also masks the EOS from the loss, and the
# model may not learn to STOP. Prefer a model with its own pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Right-padding for TRAINING (left-padding is for batched generation).
tokenizer.padding_side = "right"

# The job description we install via the system role:
SYSTEM = (
    "You are a customer-support triage assistant. Read the support ticket and respond with "
    "ONLY a JSON object (no extra text) using exactly these keys: category, priority, suggested_reply.\n"
    "category must be one of: Billing, Technical Issue, Account & Login, Shipping & Delivery, "
    "Refund & Return, Product Inquiry.\n"
    "priority must be one of: Low, Medium, High, Urgent.\n"
    "suggested_reply is a short, polite draft reply to the customer."
)

def to_target_json(ex):
    # The exact string we want the model to LEARN to produce.
    return json.dumps({
        "category": ex["category"],
        "priority": ex["priority"],
        "suggested_reply": ex["suggested_reply"],
    }, ensure_ascii=False)

def format_for_training(ex):
    messages = [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": ex["ticket"]},
        {"role": "assistant", "content": to_target_json(ex)},   # the answer to learn
    ]
    # tokenize=False -> return the rendered STRING; SFTTrainer tokenizes later.
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

train_text = ds["train"].map(format_for_training, remove_columns=ds["train"].column_names)
val_text   = ds["validation"].map(format_for_training, remove_columns=ds["validation"].column_names)

print("===== ONE FULLY-FORMATTED TRAINING EXAMPLE (what the model actually sees) =====\n")
print(train_text[0]["text"])""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 4 · Tokenization: padding, truncation and labels (the part people get wrong)

**📍 In one line:** measure how long our examples are in tokens, and pick a maximum length that never cuts
off an answer.

**❓ Why:** the model never sees text, only **token IDs**. The answer (our JSON) sits at the *end* of each
example, so if a sequence is cut short, the answer is what gets lost.

**🧩 What: three choices shape training:**

| Concept | What it is | What goes wrong if you get it wrong |
|---|---|---|
| **Padding** | Batches must be rectangular, so shorter sequences are filled with PAD tokens up to the longest. PAD positions get label `-100`, meaning "ignore in the loss". | The model is "taught" to predict padding. |
| **Truncation** (`max_seq_length`) | A hard cap on sequence length; anything longer is chopped. | Too low: the answer is cut off and the model learns to start a JSON it never finishes. Too high: wasted VRAM. |
| **Labels** | For next-token training, the labels are the input tokens shifted by one. `SFTTrainer` builds them for you. | Usually nothing, because it's automatic. |

**⚙️ How:** tokenize every formatted training example, print the min / median / 95th-percentile / max
length, and set `MAX_SEQ_LEN` comfortably above the maximum.

**👀 What you should see:** lengths of a few hundred tokens (the system prompt is the bulk of it), with the
max **well below 1024**. So `MAX_SEQ_LEN = 1024` truncates nothing. If your own data has a max above your
cap, raise the cap or shorten the system prompt.

> **🎙 Talking track:** "If you remember one tokenization rule, remember this: *never let truncation eat
> your label*. A truncated answer means you're optimizing toward a target that isn't there — the model
> learns to start a JSON it never finishes."

> **🎙 Talking track (full-text vs completion-only):** "By default, SFT computes the loss over the *entire*
> sequence, prompt included. That's fine and standard. A common optimization is to mask the prompt and
> score only the assistant's answer (`DataCollatorForCompletionOnlyLM`), which can sharpen short-answer
> tasks. It's covered in §11; here we keep the robust default."

⚠️ **T4 NOTE:** sequence length is a *quadratic-ish* VRAM lever. If you OOM, the cheapest safe fix is to
lower `MAX_SEQ_LEN` (e.g., 768) before touching anything else.""")

code(r"""# Measure real token lengths of the formatted training texts.
lengths = [len(tokenizer(t["text"])["input_ids"]) for t in train_text]
lengths_np = np.array(lengths)
print(f"token length  ->  min {lengths_np.min()}  median {int(np.median(lengths_np))}  "
      f"p95 {int(np.percentile(lengths_np, 95))}  max {lengths_np.max()}")

# Pick a cap safely above the longest example so answers are never truncated.
MAX_SEQ_LEN = 1024
n_cut = int((lengths_np > MAX_SEQ_LEN).sum())
print(f"\nChosen MAX_SEQ_LEN = {MAX_SEQ_LEN}  (lower this first if you hit OOM)")
print(f"Examples that would be truncated: {n_cut} / {len(lengths_np)}",
      "✅" if n_cut == 0 else "⚠️  raise MAX_SEQ_LEN or shorten the prompt!")""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 5 · Load the base model in 4-bit (the "Q" in QLoRA)

**📍 In one line:** load the model with its weights compressed from 16 bits to 4 bits, then prepare it for
training.

**❓ Why 4-bit:** storing each weight in 4 bits instead of 16 shrinks the frozen base about **4×** (≈3 GB →
≈1 GB for a 1.5B model). That is what lets models train on a 15 GB T4 with room to spare. We never update
these weights, so compressing them costs very little quality.

**🧩 What each `BitsAndBytesConfig` knob means:**

| Knob | Value | Meaning |
|---|---|---|
| `load_in_4bit` | `True` | Quantize (compress) the weights to 4-bit as they load. |
| `bnb_4bit_quant_type` | `"nf4"` | **NF4** (NormalFloat-4): a 4-bit format whose levels follow the bell-curve shape of real weights, so it loses less than plain int4. The QLoRA paper's key trick. |
| `bnb_4bit_compute_dtype` | `float16` | *Storage* is 4-bit, but the *maths* runs in fp16. Memory savings come from storage; quality comes from compute precision. |
| `bnb_4bit_use_double_quant` | `True` | Also compresses the compression constants, for a little more VRAM saved for free. |

**⚙️ How:**
1. `from_pretrained(..., quantization_config=bnb_config)` loads the compressed model onto the GPU.
2. `use_cache=False`: the KV-cache speeds up *generation* but conflicts with gradient checkpointing during
   *training*. Our generation helper turns it back on temporarily.
3. `prepare_model_for_kbit_training` makes the 4-bit model trainable: it keeps layer norms in fp32 for
   stability, enables gradient checkpointing, and lets gradients flow to the adapters we add next.

**👀 What you should see:** `✅ Base model loaded in 4-bit` and GPU memory used of **roughly half a GB**
for the 0.5B model. (The embedding table stays in fp16, so it's not a perfect 4× shrink.)

> **🎙 Talking track:** "Analogy: NF4 is like compressing a photo with a codec tuned to how real photos
> look, instead of a generic one. Same bits, more retained detail — because weights, like photos, aren't
> uniformly random."

⚠️ **T4 NOTE:** `compute_dtype=torch.float16`. On Ampere+ switch to `torch.bfloat16`.""")

code(r"""from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

compute_dtype = torch.float16   # ⚠️ T4: float16. Ampere+ (A100/L4/RTX30xx+): torch.bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",            # place the model on the GPU automatically
    torch_dtype=compute_dtype,
)
model.config.use_cache = False    # required with gradient checkpointing during training
model.config.pretraining_tp = 1

# Make the 4-bit model ready for LoRA training (fp32 layernorms, input grads, etc.)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

print("✅ Base model loaded in 4-bit.")
print("Approx GPU memory used:", round(torch.cuda.memory_allocated()/1e9, 2), "GB")""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 6 · LoRA configuration (the "LoRA" in QLoRA): every knob explained

**📍 In one line:** attach small trainable "adapter" matrices to the frozen model. These are the only
weights we will train.

**❓ Why LoRA:** we can't train the 4-bit weights directly (you can't backpropagate into 4-bit storage),
and we don't want to: retraining everything is expensive and risks erasing what the model already knows.
Instead we **freeze the base** and add tiny trainable matrices beside its big weight matrices. We train only
those, about **1–2% of the parameters**, yet they steer behaviour strongly.

**🧩 What the maths is (intuition):** the change ΔW to a big `d × d` weight matrix can be approximated by
the product of two **skinny** matrices: `ΔW ≈ B · A`, where `A` is `r × d`, `B` is `d × r`, and **r is
tiny** (here 16). For a 896 × 896 matrix that's 2 × 16 × 896 ≈ 29K trainable numbers instead of ~800K.
`B` starts at **zero**, so at step 0 the adapter adds nothing and *the model behaves exactly like the base*.
§7 uses this to measure a free "before" baseline.

**Every knob:**

| Knob | Value | What it controls | How to choose |
|---|---|---|---|
| `r` (rank) | **16** | Adapter capacity | 8–16 for a narrow task like ours; 32–64 for broad or multi-task adaptation. Higher = more expressive, more overfit risk. |
| `lora_alpha` | **32** | Adapter strength: its effect is scaled by `alpha / r` | Common rule: **alpha = 2 × r**. Think of it as the adapter's volume knob. |
| `target_modules` | attention + MLP | *Which* weight matrices get adapters | All attention projections (`q,k,v,o_proj`) **and** the MLP (`gate,up,down_proj`) is the modern default and beats attention-only. |
| `lora_dropout` | **0.05** | Randomly drops adapter inputs during training | Light regularization against overfitting a small dataset. |
| `bias` | `"none"` | Whether to train bias terms | Negligible benefit, so keep it lean. |
| `task_type` | `"CAUSAL_LM"` | Tells PEFT this is next-token prediction | Fixed for chat or text-generation models. |

**⚙️ How:** build a `LoraConfig`, wrap the model with `get_peft_model`, and print how many parameters are
now trainable.

**👀 What you should see:** a line like `trainable params: ~8.8M || all params: ... || trainable%: ~2–3`.
Millions of trainable numbers sounds like a lot, but it's a tiny fraction of the model, and the adapter file
we save in §10 is only tens of MB. (With a 4-bit model, "all params" is under-reported because the
quantized weights are packed, so the percentage reads a bit high.)

> **🎙 Talking track:** "Sticky-note analogy: we're not rewriting the chef's brain (that's full
> fine-tuning). We clip a small stack of targeted sticky-notes onto existing skills — 'in OUR kitchen,
> always plate triage as JSON'. The base mastery is untouched, which is also why LoRA resists
> catastrophic forgetting: we literally never overwrote the original weights."

🔁 **SWAP HERE (different model family):** module names can differ. Qwen/Llama/Mistral use the names
below; some models name them `query_key_value`/`dense`. If unsure, print `model` and read the layer names,
or set `target_modules="all-linear"` to adapt every linear layer.""")

code(r"""from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                 # adapter rank (capacity). 8-16 for narrow tasks.
    lora_alpha=32,        # scaling; alpha/r = 2 here (the common 2x heuristic).
    lora_dropout=0.05,    # light regularization on a small dataset.
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[      # adapt attention AND MLP projections (modern default).
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

# The punchline of PEFT: how few parameters we actually train.
model.print_trainable_parameters()""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 7 · BEFORE training: measure the baseline (so the improvement is undeniable)

**📍 In one line:** score the *untrained* model on the 35 test tickets, so we have a "before" number to
beat.

**❓ Why:** "the loss went down" is not evidence that the model is *useful*. We measure **task metrics on the
held-out test set** *before* training, so after training we can show a concrete gain. Without a baseline,
"85% accuracy" means nothing. Was it 80% already?

**🧩 What we measure** (the same three metrics as in the success criteria at the top):

| Metric | Question it answers | Counted as correct when… |
|---|---|---|
| **valid_json** | Can a program use the output at all? | the output contains a `{...}` block that parses as JSON |
| **category_acc** | Did it route to the right team? | `category` *exactly* equals the true label |
| **priority_acc** | Did it judge urgency correctly? | `priority` *exactly* equals the true label |

Exact match is strict on purpose: `"Billing and Service Issues"` is wrong, because our ticketing system only
knows `"Billing"`.

**Why the baseline is "free":** remember LoRA's `B = 0` start. Right now the adapted model behaves
**identically to the base model**, so evaluating it now *is* evaluating the un-fine-tuned model. No need to
load a second copy.

**⚙️ How: the helper functions** (reused for the "after" score in §9):
- `triage(ticket)` builds the prompt (system + ticket), generates greedily so results are repeatable,
  and returns only the newly generated text.
- `parse_json(text)` pulls out the first `{...}` block and parses it, returning `None` if it can't.
- `predict_split(split)` runs `triage` on every ticket and records true vs predicted labels.
- `metrics_from(records)` turns those records into the three scores.

We also save 3 raw outputs to compare side by side after training.

**👀 What you should see** (this takes ~1–2 minutes): something like
`BEFORE fine-tuning: {'valid_json': ~0.95, 'category_acc': ~0.25, 'priority_acc': ~0.4}`. The model
*mostly* produces JSON, but with the wrong labels. That's exactly the gap fine-tuning closes.

> **🎙 Talking track:** "Watch the baseline closely. A good instruct model often *already* produces
> JSON-ish output sometimes — that's prompt engineering working. The question fine-tuning answers is:
> can we make it **reliable**, on **our** label set, **every** time? The before/after gap is exactly the
> value fine-tuning added over a good prompt alone."

⚠️ **T4 NOTE:** generation needs the KV-cache; our helper temporarily flips `use_cache=True`, then back.""")

code(r"""# Greedy decoding (do_sample=False) for reproducible, comparable outputs.
GEN_KW = dict(max_new_tokens=220, do_sample=False, pad_token_id=tokenizer.eos_token_id)

def triage(ticket):
    "Run the (current) model on one ticket and return its raw text output."
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user",   "content": ticket}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    prev = model.config.use_cache
    model.config.use_cache = True                 # cache ON for fast generation
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KW)
    model.config.use_cache = prev                 # restore (False) for training
    gen = out[0][inputs["input_ids"].shape[1]:]   # keep only newly generated tokens
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def parse_json(text):
    "Extract the first {...} block and parse it; return None if unparseable."
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def predict_split(split, n=None):
    "Generate on each ticket and return per-example records (reused by the charts)."
    n = len(split) if n is None else min(n, len(split))
    recs = []
    for ex in split.select(range(n)):
        raw = triage(ex["ticket"])
        pj  = parse_json(raw)
        recs.append({
            "ticket":   ex["ticket"],
            "true_cat": ex["category"], "true_pri": ex["priority"],
            "pred_cat": (str(pj.get("category", "")).strip() if pj else "(invalid)"),
            "pred_pri": (str(pj.get("priority", "")).strip() if pj else "(invalid)"),
            "valid":    pj is not None,
            "raw":      raw,
        })
    return recs

def metrics_from(recs):
    "Aggregate accuracy from per-example records."
    total  = len(recs)
    valid  = sum(r["valid"] for r in recs)
    cat_ok = sum(r["valid"] and r["pred_cat"] == r["true_cat"] for r in recs)
    pri_ok = sum(r["valid"] and r["pred_pri"] == r["true_pri"] for r in recs)
    return {"valid_json":   round(valid/total, 3),
            "category_acc": round(cat_ok/total, 3),
            "priority_acc": round(pri_ok/total, 3)}

def evaluate(split, n=None):                 # thin wrapper used throughout the lab
    return metrics_from(predict_split(split, n))

# Stash 3 baseline generations to show before/after later.
sample_tickets   = [ds["test"][i]["ticket"] for i in range(3)]
baseline_outputs = [triage(t) for t in sample_tickets]

# Full-test predictions BEFORE training (we reuse before_preds in the charts).
before_preds = predict_split(ds["test"])
before = metrics_from(before_preds)
print("BEFORE fine-tuning:", before)""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 8 · Training with `SFTTrainer`: every important knob, and why

**📍 In one line:** configure the training run (how long, how fast, how much memory) and build the trainer.

**❓ Why `SFTTrainer`:** writing a training loop by hand means tokenizing, building labels, masking padding,
mixed precision, gradient accumulation, evaluation and checkpointing. `SFTTrainer` (from `trl`) does all of
that correctly for supervised fine-tuning. We just describe *what* we want in an `SFTConfig`.

**🧩 What every knob does (this is the cell to slow down on):**

| Knob | Value | Why |
|---|---|---|
| `num_train_epochs` | 3 | Small dataset → a few passes. Too many = memorize/overfit; watch eval loss. |
| `per_device_train_batch_size` | 2 | What fits in T4 VRAM per step. Bigger = smoother but more memory. |
| `gradient_accumulation_steps` | 8 | Accumulate 8 micro-batches before an update → **effective batch = 2×8 = 16** without the memory of a batch-16 step. |
| `learning_rate` | 2e-4 | LoRA tolerates higher LRs than full FT (few params). 1e-4–3e-4 is the LoRA sweet spot. |
| `lr_scheduler_type` | cosine | Decay LR smoothly to ~0 by the end → stable convergence. |
| `warmup_ratio` | 0.03 | Ramp LR up for the first 3% of steps so early updates don't destabilize. |
| `gradient_checkpointing` | True | Recompute activations in backward instead of storing them → big VRAM save (costs ~20–30% speed). |
| `fp16` | True | ⚠️ T4 mixed precision. Use `bf16=True` instead on Ampere+. |
| `optim` | paged_adamw_8bit | 8-bit Adam (bitsandbytes) + paging → optimizer state barely uses VRAM. |
| `max_seq_length` | 1024 | The truncation cap we chose from the length distribution. |
| `packing` | False | Don't concatenate multiple examples into one sequence. Off = each ticket is its own example (clearer for short, structured tasks). |

**How the levers relate (the key mental model):**
> **effective batch size = per_device_batch × grad_accumulation × #GPUs**
> On one T4 that's `2 × 8 × 1 = 16`. Want a bigger effective batch but running out of memory? Raise
> *accumulation*, not *per-device batch*, because accumulation costs almost no memory.

**How many update steps?** ~281 examples ÷ 16 per update ≈ **17 steps per epoch**, × 3 epochs ≈ **~51
steps**. That's small, which is why training takes minutes, not hours.

**⚙️ How:** build the `SFTConfig` (the table above, as code), then create the `SFTTrainer` with the model,
the formatted train/validation text, and the tokenizer. Nothing trains yet; that happens in §8b.

**👀 What you should see:** `✅ Trainer ready. Steps/epoch: 17` (possibly preceded by harmless warnings
about deprecated arguments).

> **🎙 Talking track:** "Gradient accumulation is the single most useful T4 trick. It decouples the batch
> size your *optimizer* sees from the batch size your *GPU* has to hold. You get the stability of a large
> batch with the footprint of a tiny one — you just trade a little wall-clock time."

⚠️ **T4 NOTE:** if training OOMs, in order: (1) lower `max_seq_length`, (2) set `per_device_batch=1` and
double `grad_accumulation`, (3) keep gradient checkpointing on. We'll revisit in Troubleshooting.""")

code(r"""from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="outputs/qwen-support-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,          # effective batch = 2 * 8 = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    logging_steps=10,
    eval_strategy="epoch",                  # evaluate on val set each epoch
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,                              # ⚠️ T4. On Ampere+: fp16=False, bf16=True
    bf16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",               # memory-frugal optimizer
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",              # the column produced in Section 3
    packing=False,
    report_to="none",                       # set "wandb"/"tensorboard" to log curves
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_text,
    eval_dataset=val_text,
    tokenizer=tokenizer,                    # newer trl renames this to `processing_class`
)
print("✅ Trainer ready. Steps/epoch:",
      len(train_text) // (sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps))""")

md(r"""### 8b · Run training

**📍 In one line:** one call, `trainer.train()`, and the adapters learn our task.

**❓ Why this is the cheap part:** notice that most of the lab went into data, format and config. The
actual training is one line. Getting the setup right is the real work.

**⚙️ How:** for each step, the trainer takes 8 micro-batches of 2 examples, computes how wrong the model's
next-token predictions are (the loss) **on the full formatted text**, back-propagates into the LoRA
adapters only, and updates them. At the end of each epoch it scores the validation set (`eval_loss`).

**👀 What you should see** (about **4–10 minutes** on a T4):
- A progress bar to ~51 steps.
- `loss` logged every 10 steps, starting around **2** and falling towards **~0.1**.
- `eval_loss` after each epoch, also falling (to ~0.1 by the end).

**Overfitting check:** if `eval_loss` starts **rising** while the training `loss` keeps falling, the model
is memorizing the training tickets instead of learning the pattern. Use fewer epochs.""")

code(r"""train_result = trainer.train()
print("\n✅ Training complete.")
print("train_loss:", round(train_result.training_loss, 4))""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""### 8c · The training curve: what "healthy" looks like (vs a run that broke)

**📍 In one line:** plot this run's loss so you can see at a glance whether training worked.

**❓ Why plot it:** a falling loss curve is the *first* evidence that your fine-tune is working. Knowing the
shape of a *broken* run lets you **recognize failure instantly** instead of wasting a whole evaluation.

**🧩 What healthy vs broken looks like:**

| | Healthy run (this lab) | Broken run (a classic failure) |
|---|---|---|
| loss pattern | `~2.1 → ~0.1`, smooth, monotone-ish | `403 → 6947 …` then snaps to **`0.000000`** forever |
| what happened | the model is learning | LR too high / no chat template / training a *base* model on full text → weights diverge to **NaN**, and the reported loss collapses to 0 |

> **🎙 Talking track:** "If you ever see loss in the **hundreds or thousands**, stop. Even a model guessing
> completely at random scores only about ln(vocab size) ≈ 12, and a pretrained instruct model starts much
> lower (~2 here). And a loss that suddenly reads a perfect `0.000000` is not success; it's a **NaN
> graveyard**. The cure is almost always: use an *instruct* model with its chat template, and a sane LoRA
> learning rate (1e-4 to 3e-4)."

**⚙️ How:** read `trainer.state.log_history` (every logged loss value) and plot train loss and eval loss
against the step number.

**👀 What you should see:** a blue train-loss line falling steeply at first and then flattening near ~0.1,
with green eval-loss points following it down. No spikes, nothing in the hundreds, no sudden `0.0`.""")

code(r"""import matplotlib.pyplot as plt

hist = trainer.state.log_history
train_pts = [(h["step"], h["loss"])      for h in hist if "loss" in h]
eval_pts  = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]

fig, ax = plt.subplots(figsize=(7, 4))
if train_pts:
    xs, ys = zip(*train_pts); ax.plot(xs, ys, "o-",  color="#2980b9", label="train loss")
if eval_pts:
    xs, ys = zip(*eval_pts);  ax.plot(xs, ys, "s--", color="#27ae60", label="eval loss")
ax.set_xlabel("training step"); ax.set_ylabel("loss")
ax.set_title("Healthy fine-tuning: loss decreases smoothly and stays finite")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()

if train_pts:
    print(f"Start loss ≈ {train_pts[0][1]:.2f}   →   final loss ≈ {train_pts[-1][1]:.2f}")
    print("✅ Finite and decreasing. (A broken run would have exploded to 1e2–1e3, then read 0.000000.)")""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 9 · AFTER training: re-measure and compare

**📍 In one line:** run the *same* evaluation on the *same* 35 test tickets, now with the trained adapter,
and compare with the baseline.

**❓ Why:** this is the payoff. Same metrics, same test set, same code; the only thing that changed is the
fine-tuning. Any difference is therefore caused by the fine-tuning.

**⚙️ How:** call `predict_split` on the test set again (the model now includes the trained adapters),
compute the metrics, show a before / after / gain table, and check each metric against the targets set in
**"How we'll know it worked"** at the top.

**👀 What you should see** (two verified runs; yours will land in the same range):

| | valid_json | category_acc | priority_acc |
|---|---|---|---|
| before | 0.94–0.97 | 0.23–0.31 | 0.37–0.40 |
| after | **1.00** | **1.00** | **0.89–0.94** |

The "before" numbers wobble between runs because GPU kernels and library versions shift slightly; the
"after" numbers are stable. Expect a few harmless `UserWarning`s about `do_sample` / `temperature` above
the table. They come from the model's default generation settings, and we decode greedily on purpose.

**How to read it:**
- **valid_json → 1.0:** the format habit is installed. This is usually learned first.
- **category_acc → ~1.0:** our 6 labels are learned exactly.
- **priority_acc lags:** priority is genuinely harder. "High" vs "Urgent" is a judgement call, and
  **Urgent had only ~20 training examples** (see §2b). The fix is more and better-balanced priority data,
  not more epochs.
- If accuracy barely moved at all, suspect the **data** (too few or inconsistent examples) before blaming the
  method.""")

code(r"""# Full-test predictions AFTER training (reused by the charts in 9c).
after_preds = predict_split(ds["test"])
after = metrics_from(after_preds)

# Success targets: the same numbers as the "How we'll know it worked" table at the top.
TARGETS = {"valid_json": 1.00, "category_acc": 0.90, "priority_acc": 0.85}

print("BEFORE:", before)
print("AFTER :", after)

try:
    import pandas as pd
    delta = {k: round(after[k] - before[k], 3) for k in before}
    tbl = pd.DataFrame([before, after, delta, TARGETS],
                       index=["before", "after", "Δ (gain)", "target"])
    print()
    print(tbl)
except Exception:
    pass

print("\nTarget check:")
for k, t in TARGETS.items():
    print(f"  {k:13s} after = {after[k]:.3f}   target ≥ {t:.2f}   ",
          "✅ met" if after[k] >= t else "❌ not yet")""")

md(r"""### 9b · Qualitative side-by-side (people trust examples more than numbers)

**📍 In one line:** show 3 real test tickets with the model's answer before training, after training, and
the expected answer.

**❓ Why:** a metrics table convinces analysts. *Seeing* a messy "before" turn into clean, correct JSON
convinces everyone else, and it often reveals *how* a model is wrong, which a number can't.

**⚙️ How:** reuse the 3 "before" outputs saved in §7, generate fresh "after" outputs with the trained model,
and render a 4-column table: Ticket | ❌ BEFORE | ✅ AFTER | 🎯 EXPECTED.

**👀 What you should see:** BEFORE answers with invented labels (`"Billing and Payment"`, `"Normal"`) or extra
text. AFTER answers that match EXPECTED on category and priority, with a polite `suggested_reply` in the same
style as the training data. The reply wording won't match word for word, and it doesn't need to.""")

code(r"""# Render the 3 stashed examples as a BEFORE → AFTER → EXPECTED table.
# Falls back to plain text if not in a notebook display context.
rows_html = []
for i, (ticket, base_out) in enumerate(zip(sample_tickets, baseline_outputs)):
    after_out = triage(ticket)
    expected  = to_target_json(ds["test"][i])
    rows_html.append((ticket, base_out, after_out, expected))

def _cell(s):
    s = (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<td style="vertical-align:top;padding:8px;border:1px solid #ddd;font-size:12px;white-space:pre-wrap">{s}</td>'

try:
    from IPython.display import display, HTML
    html = ['<table style="border-collapse:collapse;font-family:monospace">',
            '<tr style="background:#34495e;color:white">'
            '<th style="padding:8px">Ticket</th><th>❌ BEFORE (base model)</th>'
            '<th>✅ AFTER (fine-tuned)</th><th>🎯 EXPECTED</th></tr>']
    for t, b, a, e in rows_html:
        html.append("<tr>" + _cell(t)
                     + _cell(b).replace('#ddd', '#e74c3c')
                     + _cell(a).replace('#ddd', '#27ae60')
                     + _cell(e) + "</tr>")
    html.append("</table>")
    display(HTML("".join(html)))
except Exception:
    for t, b, a, e in rows_html:
        print("=" * 90); print("TICKET   :", t)
        print("-" * 90); print("BEFORE   :", b)
        print("AFTER    :", a); print("EXPECTED :", e)
    print("=" * 90)""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""### 9c · Visual proof: charts the whole class can read at a glance

**📍 In one line:** three visuals of the same before/after result: bars, confusion matrices, and a
per-ticket table.

**❓ Why charts:** numbers in a table are easy to skim past. A **red bar jumping to green** and a
**confusion matrix collapsing onto the diagonal** make the improvement impossible to miss. All three views
reuse the predictions from §7 and §9, so there is no extra generation and they run instantly.

**🧩 What each chart shows, and how to read it:**
1. **Before/after bar chart:** the three metrics side by side, with each metric's **target marked**. Green
   bars at or above the marks mean we hit the goal.
2. **Category confusion matrices:** rows are the *true* category, columns are what the model *predicted*.
   Every number on the **diagonal** is a correct answer, and anything off it is a mistake. (Invalid outputs
   and invented labels have no column, so they don't appear in BEFORE at all; a sparse BEFORE matrix is
   itself a sign of failure.)
3. **Per-ticket table:** exactly which tickets flipped from ✗ to ✓.

**👀 What you should see:** short red bars and tall green bars; a scattered or sparse BEFORE matrix and a
clean diagonal AFTER; and most rows in the table going ✗ → ✓.""")

code(r"""import matplotlib.pyplot as plt
import numpy as np

# ---- (1) Before/after metric bar chart -------------------------------------
labels = ["valid_json", "category_acc", "priority_acc"]
bvals  = [before[k] for k in labels]
avals  = [after[k]  for k in labels]
x = np.arange(len(labels)); w = 0.38

fig, ax = plt.subplots(figsize=(7.5, 4.3))
b1 = ax.bar(x - w/2, bvals, w, label="before", color="#e74c3c")
b2 = ax.bar(x + w/2, avals, w, label="after",  color="#27ae60")
# Draw each metric's own target (from TARGETS in §9) as a short dashed line over its bars.
for i, k in enumerate(labels):
    ax.hlines(TARGETS[k], i - w, i + w, ls="--", color="gray")
    ax.text(i + w + 0.02, TARGETS[k], f"target {TARGETS[k]:.0%}", color="gray", fontsize=8, va="center")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.08)
ax.set_ylabel("score"); ax.set_title("Proof of improvement: BEFORE vs AFTER fine-tuning"); ax.legend()
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.02, f"{r.get_height():.2f}",
                ha="center", fontsize=9)
plt.tight_layout(); plt.show()""")

code(r"""# ---- (2) Category confusion matrices: before vs after ----------------------
CATS = ["Billing", "Technical Issue", "Account & Login",
        "Shipping & Delivery", "Refund & Return", "Product Inquiry"]

def confusion(recs, field):
    idx = {c: i for i, c in enumerate(CATS)}
    M = np.zeros((len(CATS), len(CATS)), dtype=int)
    for r in recs:
        t, p = r["true_" + field], r["pred_" + field]
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, recs, title in [(axes[0], before_preds, "BEFORE"), (axes[1], after_preds, "AFTER")]:
    M = confusion(recs, "cat")
    ax.imshow(M, cmap="Greens")
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (correct answer)")
    ax.set_title(f"Category confusion — {title}\n(bright diagonal = correct)")
    for i in range(len(CATS)):
        for j in range(len(CATS)):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > M.max()/2 else "black")
plt.tight_layout(); plt.show()
print("Read it like this: BEFORE scatters off the diagonal (wrong guesses); AFTER concentrates on it.")""")

code(r"""# ---- (3) Per-example results table: which tickets flipped ✗ -> ✓ ------------
try:
    import pandas as pd
    rows = []
    for b, a in zip(before_preds, after_preds):
        rows.append({
            "ticket": (b["ticket"][:55] + "…") if len(b["ticket"]) > 56 else b["ticket"],
            "true category": b["true_cat"],
            "before": ("✓" if b["pred_cat"] == b["true_cat"] else f"✗ {b['pred_cat']}"),
            "after":  ("✓" if a["pred_cat"] == a["true_cat"] else f"✗ {a['pred_cat']}"),
        })
    df = pd.DataFrame(rows)
    flipped = sum(1 for b, a in zip(before_preds, after_preds)
                  if b["pred_cat"] != b["true_cat"] and a["pred_cat"] == a["true_cat"])
    print(f"Category predictions that went WRONG → RIGHT after fine-tuning: {flipped}/{len(before_preds)}\n")
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string(index=False))
except Exception as e:
    print("results table skipped:", e)""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 10 · Saving, merging and (optionally) pushing to the Hub

**📍 In one line:** save what we trained, in two forms: a small adapter file and a standalone merged model.

**❓ Why:** everything trained so far lives only in GPU memory and disappears when the Colab session ends.
And there are two ways to ship a fine-tune, which suit different deployments.

**🧩 What: two things to save:**

| | **LoRA adapter** | **Merged model** |
|---|---|---|
| What it is | only the trained adapter weights | adapter folded into the base weights |
| Size | **~35 MB** for our 0.5B setup | ~1 GB (fp16, the full model) |
| To use it you need | the base model + the `peft` library | nothing extra: plain `from_pretrained` |
| Best for | many fine-tunes sharing one base; instant rollback (just detach it) | simplest deployment, one self-contained model |

**⚙️ How (three cells):**
1. `trainer.save_model(...)` writes the adapter and tokenizer, and we print the folder size.
2. To **merge**, free the 4-bit training model, reload the base in **fp16** (you can't merge into 4-bit
   weights), attach the adapter, and call `merge_and_unload()`.
3. *(Optional)* push the merged model to the Hugging Face Hub. This is off by default.

**👀 What you should see:** the adapter files (`adapter_config.json`, `adapter_model.safetensors`, tokenizer
files) and a size in the tens of MB; then `✅ Merged standalone model saved`; then `PUSH_TO_HUB is False —
skipping`.

> **🎙 Talking track:** "Adapter vs merged is a deployment decision, not a quality one — the weights are
> the same. Adapter = modular & swappable (one base, many behaviors, instant rollback). Merged = one
> self-contained model. Pick by your serving story."

⚠️ **T4 NOTE:** merging loads a *second* copy of the model in fp16 (~1 GB for 0.5B, ~3 GB for 1.5B). We
`del` the trainer and the 4-bit model and clear the cache first, so the T4 doesn't run out of memory
during the merge. **After this cell, `model` and `trainer` no longer exist**, so re-run from §5 if you want to
train again.""")

code(r"""# 1) Save just the LoRA adapter (tiny).
ADAPTER_DIR = "outputs/qwen-support-qlora-adapter"
trainer.save_model(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("Adapter files:", os.listdir(ADAPTER_DIR))
print("Adapter size:",
      round(sum(os.path.getsize(os.path.join(ADAPTER_DIR, f)) for f in os.listdir(ADAPTER_DIR))/1e6, 2),
      "MB  <-- small vs the full model (~1 GB in fp16)")""")

code(r"""# 2) Merge the adapter into fp16 base weights -> a standalone model.
# Free GPU memory from the 4-bit training model first (avoid OOM on T4).
del trainer, model
gc.collect()
torch.cuda.empty_cache()

from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto",
)
merged = PeftModel.from_pretrained(base, ADAPTER_DIR)
merged = merged.merge_and_unload()      # fold LoRA into the base weights

MERGED_DIR = "outputs/qwen-support-merged"
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print("✅ Merged standalone model saved to", MERGED_DIR)""")

code(r"""# 3) (OPTIONAL) Push to the Hugging Face Hub.
# 🔁 Set to True, run notebook_login(), and change the repo id to your username.
PUSH_TO_HUB = False
if PUSH_TO_HUB:
    from huggingface_hub import notebook_login
    notebook_login()
    REPO = "your-username/qwen2.5-0.5b-support-triage"   # 🔁 SWAP HERE
    merged.push_to_hub(REPO)
    tokenizer.push_to_hub(REPO)
    print("Pushed to", REPO)
else:
    print("PUSH_TO_HUB is False — skipping. Your model is saved locally in", MERGED_DIR)""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 11 · Optimizations tour: which knob to turn, when, and why

**📍 In one line:** a field guide for improving or adapting this lab after the first successful run.

**❓ Why:** the first run proves the pipeline works. Real projects then iterate: more accuracy, less memory,
a different GPU. Knowing *which* knob addresses *which* symptom saves hours of blind trial and error.

**How to use it:** find your symptom or goal below, **change one knob at a time**, and re-measure on the same
test set. Change two at once and you can't tell which one helped.

**Quick map: symptom → first knob to try**

| Symptom / goal | First knob to try | Details below |
|---|---|---|
| Priority (or any label) accuracy too low | more / better-balanced data for that label; then raise `r` | Rank sweeps |
| Eval loss rising while train loss falls (overfitting) | fewer epochs, lower `r`, more `lora_dropout` | Rank sweeps; LR & epochs |
| Out of memory | lower `max_seq_length`, then batch 1 + more accumulation | Batch size; §12 |
| Training too slow | turn off gradient checkpointing (if VRAM allows), `packing=True` | Gradient checkpointing; Throughput |
| Bigger GPU available | bf16; maybe plain LoRA without 4-bit | LoRA vs QLoRA; cheat sheet |""")

md(r"""### LoRA vs QLoRA vs full fine-tuning: the core trade-off
- **QLoRA (4-bit base, this lab):** lowest VRAM; fits big models on small GPUs. Costs a little speed and a
  hair of quality from quantization. **Default for a single consumer/Colab GPU.**
- **LoRA (16-bit base):** if you *have* the VRAM, skip 4-bit — slightly faster and marginally higher
  quality, since the base isn't quantized. **Use when memory isn't the bottleneck.**
- **Full fine-tuning:** all weights trained. Max capacity, max cost, highest catastrophic-forgetting risk.
  Reserve for deep domain shifts with lots of data and a real GPU budget.

### Quantization (`bnb_4bit_*`)
- `nf4` > `fp4` for quality (matches weight distribution). Keep `use_double_quant=True` for a free VRAM win.
- `compute_dtype`: fp16 on T4, **bf16 on Ampere+** (more stable, fewer overflow issues).

### Gradient checkpointing
- **On** (this lab): saves large activation memory, costs ~20–30% speed. Keep it on whenever memory is tight.
- **Off:** faster, if you have VRAM headroom (e.g., big GPU + short sequences).

### Batch size vs gradient accumulation
- Want a bigger **effective** batch (steadier gradients) but OOM? Raise `gradient_accumulation_steps`, not
  `per_device_train_batch_size`. Accumulation is ~free in memory; per-device batch is not.
- Rule: `effective_batch = per_device_batch × grad_accum × num_gpus`.

### Rank `r` / `alpha` sweeps
- Underfitting (train loss plateaus high, outputs bland)? **Raise `r`** (16 → 32) and keep `alpha = 2r`.
- Overfitting (eval loss rises, it parrots training phrasings)? **Lower `r`** (16 → 8), raise `lora_dropout`
  (0.05 → 0.1), or cut epochs.
- `alpha` is the volume: with `alpha = 2r` the effective scale stays constant as you change `r`, so you can
  sweep `r` without re-tuning the LR.

### Learning rate & epochs
- LoRA likes **1e-4 – 3e-4**. Loss exploding/NaN → halve the LR. Loss barely moving → raise it or add epochs.
- More epochs help until eval loss turns up — that inflection is your stop sign.

### Throughput
- `optim="paged_adamw_8bit"` already saves optimizer memory. For more speed on a long run, try `packing=True`
  (concatenates short examples to reduce padding waste) — but it muddies per-example boundaries, so verify
  quality on short structured tasks like ours.
- **Completion-only loss:** mask the prompt and score only the answer (`DataCollatorForCompletionOnlyLM`).
  Often sharpens short-output tasks and speeds convergence.

> **🎙 Talking track:** "Don't sweep blindly. Form a hypothesis ('outputs are bland → underfitting →
> raise r'), change ONE knob, re-measure on the SAME test set. A tuning session without a fixed eval is
> just vibes."

### Free-Colab-T4 vs a bigger local GPU — the cheat sheet
| Setting | Free T4 (this lab) | Ampere+ (A100/L4/RTX 30xx+) |
|---|---|---|
| compute dtype | `float16` | `bfloat16` |
| trainer precision | `fp16=True` | `bf16=True` |
| 4-bit base | usually needed | optional (use 16-bit LoRA if VRAM allows) |
| batch / seq len | small + accumulation | larger; less need for checkpointing |""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 12 · Troubleshooting appendix

**📍 In one line:** the errors students actually hit in this lab, what causes each, and the fix, cheapest first.

**How to use it:** find the heading that matches your error message, apply the fixes **in order**, and stop as
soon as it works.

| You see… | Jump to |
|---|---|
| `CUDA out of memory` | 🔴 OOM |
| `TypeError` / `unexpected keyword argument` in `SFTTrainer`/`SFTConfig` | 🔴 Version conflicts |
| `CUDA VERSION MISMATCH`, `triton.ops`, *"compiled without GPU support"* | 🔴 bitsandbytes |
| Progress bar crawling | 🔴 Training is very slow |
| `valid_json` below 1.0 after training | 🔴 Outputs aren't valid JSON |
| Loss in the hundreds, or exactly `0.000000` | §8c (broken-run signs) |

### 🔴 CUDA out of memory (OOM)
Apply in this order (cheapest first), re-running from the model-load cell:
1. **Lower `max_seq_length`** (1024 → 768 → 512). Sequence length hits memory hardest.
2. **`per_device_train_batch_size=1`**, and **double `gradient_accumulation_steps`** to keep the effective
   batch the same.
3. Ensure **`gradient_checkpointing=True`** and **`optim="paged_adamw_8bit"`**.
4. Confirm the base loaded **in 4-bit** (the model-load cell printed its memory).
5. Restart the runtime to clear leaked memory (`Runtime ▸ Restart`), then re-run from imports.
6. Insert `import torch, gc; gc.collect(); torch.cuda.empty_cache()` between heavy cells.

### 🔴 Version conflicts / `ImportError` / `TypeError` in `SFTTrainer`/`SFTConfig`
- The trl API changes across versions. With the **pinned set** here, the constructor uses `tokenizer=`.
  On a **newer trl**, rename `tokenizer=` → `processing_class=`, and some `SFTConfig` args may move.
- If pip resolves something newer anyway: `pip install -U --force-reinstall` the exact pins, then
  **restart the runtime** (stale modules stay imported until you do).
- `evaluation_strategy` vs `eval_strategy`: transformers renamed it; we use `eval_strategy` for 4.45.

### 🔴 `bitsandbytes`: `CUDA VERSION MISMATCH`, `triton.ops`, or `cuda1xx.so not found`
This is the most common breakage, and it's a **version mismatch with Colab's current CUDA**:
- Symptoms: `🚨 CUDA VERSION MISMATCH 🚨 … Requested CUDA version: 13.0` when §5 loads the model;
  `Could not find the bitsandbytes CUDA binary … libbitsandbytes_cudaXXX.so`; `compiled without GPU
  support`; or `ModuleNotFoundError: No module named 'triton.ops'`.
- Cause: the installed bitsandbytes has no binary for the CUDA version Colab's PyTorch uses. Colab
  upgrades CUDA every few months (it moved from 12.x to **13.0**), and older bitsandbytes only go up to 12.x.
- **Fix:** `!pip install -U bitsandbytes` (0.49.2 or newer covers CUDA 13.0), then **Runtime ▸ Restart
  session** and re-run from the imports cell.
- Check what Colab has: `import torch; print(torch.version.cuda)`.
- **Locally:** install a bitsandbytes build matching *your* CUDA toolkit, or 4-bit loading errors out.
- Verify a GPU is actually attached (`nvidia-smi`); 4-bit needs CUDA, it won't work on CPU.

### 🔴 Training is very slow
- Confirm you're on **GPU** (`device` printed `cuda`), not CPU.
- Gradient checkpointing costs ~20–30% speed — acceptable for fitting in memory; turn off only if you have
  VRAM to spare.
- Reduce `num_train_epochs` for a classroom run; 1–2 epochs already show a clear before/after.
- A T4 is modest hardware: minutes, not seconds, is normal for this size.

### 🔴 Outputs aren't valid JSON after training
- Increase `max_new_tokens` (a long `suggested_reply` may be getting cut off mid-object).
- Make sure training targets were **clean JSON** (we built them with `json.dumps`).
- Add 1–2 epochs; format adherence is usually the *first* thing the model nails, so persistent failures
  point at data or truncation, not training time.

### 🔴 Model "forgot" general ability / sounds robotic
- Classic **catastrophic forgetting** signal. Lower `r`, fewer epochs, and/or mix a few general examples
  into training. LoRA already mitigates this (frozen base), but aggressive training can still overfit.""")

# ───────────────────────────────────────────────────────────────────────────
md(r"""## 13 · Instructor talking-track — keyed recap

Use this as your closing map of the session.

| Cell / Section | The one sentence to say |
|---|---|
| **Problem statement** | "A general model *can* read tickets; it can't reliably answer in *our* labels and *our* format. That's a behaviour gap, so we fine-tune." |
| **0 · Hardware** | "Fine-tuning is bounded by VRAM; QLoRA is how a 1.5B model fits on a free T4." |
| **1 · Versions** | "Pin the stack. This ecosystem breaks across minor versions, and pins give reproducibility." |
| **1c · Base vs Instruct** | "Start from an instruct model: you can't drill a house dish into a chef who won't read the ticket." |
| **2 · Data** | "We defined the target (the JSON schema) before writing data, balanced the categories, and held out a test set." |
| **3 · Chat template** | "Speak the model's native dialect; the system message installs the job description." |
| **4 · Tokenization** | "Never let truncation eat your label. Choose the max length from the real distribution." |
| **5 · 4-bit load** | "NF4 compresses weights with a codec tuned to how weights are actually distributed." |
| **6 · LoRA** | "We clip tiny sticky-notes (~1–2% of params) onto a frozen base: a strong steer, and the original stays intact." |
| **7 · Baseline** | "Measure before training; LoRA's zero init gives us the base-model baseline for free." |
| **8 · Training** | "Effective batch = per-device × accumulation; accumulation is the T4's best friend." |
| **9 · After** | "valid_json → 1.0, category → ~100%, priority lags because Urgent is rare: the next fix is data, not epochs." |
| **10 · Save/merge** | "Adapter = modular and swappable; merged = standalone. Same weights, different deployment." |
| **11 · Optimizations** | "One hypothesis, one knob, re-measure on the same test set." |
| **12 · Troubleshooting** | "OOM? Lower the sequence length first, then batch 1 with more accumulation." |

> **🎙 Closing line:** "We turned a generalist into a reliable specialist by training **about 1–2% of its
> parameters** on a **tiny dataset** on a **free GPU**, and we can roll it back by deleting one small file.
> That asymmetry (a huge behavioural change for a tiny cost and risk) is why LoRA/QLoRA is the default way
> teams ship fine-tunes today."

### Where to go next
- Swap in your **own** JSONL (same 4 keys) and re-run — that's the whole point.
- Try `Llama-3.2-1B-Instruct` or `Phi-3-mini` and compare.
- Add **completion-only loss** and an `r` sweep; chart the before/after metrics.
- Serve the merged model behind a small API and watch for drift — then iterate on failures.""")


# ───────────────────────── build + verify ─────────────────────────────────
def to_source_list(text):
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


nb = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

def escape_tildes(md_text):
    """Escape '~' outside code so markdown never renders '~0.3 ... ~0.4' as strikethrough.

    Colab (and GitHub) treat a pair of single tildes as ~~strikethrough~~, which silently
    swallowed the tildes in phrases like 'about ~50 Medium, and only **~20 Urgent**'.
    """
    out = []
    # Split into fenced blocks / inline code (kept verbatim) and plain text (escaped).
    for part in re.split(r"(```.*?```|`[^`\n]*`)", md_text, flags=re.S):
        if part.startswith("`"):
            out.append(part)
        else:
            out.append(re.sub(r"(?<!\\)~", r"\\~", part))
    return "".join(out)


errors = 0
for idx, (kind, text, shell) in enumerate(cells):
    if kind == "markdown":
        nb["cells"].append({
            "cell_type": "markdown", "metadata": {}, "source": to_source_list(escape_tildes(text)),
        })
    else:
        if not shell:  # syntax-check pure-Python cells
            try:
                compile(text, f"<cell {idx}>", "exec")
            except SyntaxError as e:
                errors += 1
                print(f"❌ SYNTAX ERROR in cell {idx}: {e}")
        nb["cells"].append({
            "cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": to_source_list(text),
        })

out_path = os.path.join(HERE, "Fine_Tuning_LLMs_S2_Python_Bala_Krishna_Ragala_V2.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

n_md = sum(1 for c in cells if c[0] == "markdown")
n_code = sum(1 for c in cells if c[0] == "code")
print(f"Wrote {out_path}")
print(f"Cells: {len(cells)}  (markdown {n_md}, code {n_code})")
print(f"Syntax errors in Python code cells: {errors}")
