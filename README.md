# Fine-Tuning LLMs: Hands-On QLoRA Lab

Turn a small, general-purpose language model into a **reliable support-ticket triage assistant** in
about 4 minutes of training on a **free Google Colab GPU**, and prove that it got better.

[![Open the lab in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/finetune-llms-qlora-lab/blob/main/Fine_Tuning_LLMs_S2_Python_Bala_Krishna_Ragala_V2.ipynb)
&nbsp; **[▶ View the slides](https://baluragala.github.io/finetune-llms-qlora-lab/slides/)**

| | |
|---|---|
| **You will build** | A model that reads a raw ticket and returns `{"category", "priority", "suggested_reply"}` as exact JSON |
| **Model** | `Qwen/Qwen2.5-0.5B-Instruct` (small, open, no sign-up or token needed) |
| **Method** | QLoRA: a 4-bit frozen base model plus small trainable LoRA adapters, using Hugging Face `trl` |
| **You need** | A Google account (for Colab). No local GPU, no installs, no payment. |
| **Time** | 60–90 min to work through; about 15 min if you just run it |
| **Last verified** | 2026-11-14 on a free Colab T4 (Python 3.13, CUDA 13.0); every cell ran |

---

## Start here: the learning path

| Step | What to do | Why |
|---|---|---|
| 1 | **[Watch or read the slides](https://baluragala.github.io/finetune-llms-qlora-lab/slides/)** (46 slides) | *When* to fine-tune (vs prompting or RAG) and *how* LoRA/QLoRA work. Answer each **Checkpoint** before revealing it. |
| 2 | **[Open the lab in Colab](https://colab.research.google.com/github/baluragala/finetune-llms-qlora-lab/blob/main/Fine_Tuning_LLMs_S2_Python_Bala_Krishna_Ragala_V2.ipynb)** and run it (steps below) | Do it yourself: data → format → 4-bit load → LoRA → train → evaluate → save. |
| 3 | Keep [`notes.md`](notes.md) open beside the notebook | A deeper Why/What/How explanation of every cell, with analogies and a glossary. |
| 4 | Try the **"Make it yours"** exercises at the bottom | Reusing the recipe on new data is how it sticks. |

Want the full spoken narration and the answers to common questions? See
[`finetuning-llms-teaching-script.md`](finetuning-llms-teaching-script.md).

---

## Run the lab in Colab (step by step)

1. Click **Open in Colab** above. It opens a read-only copy; choose **File ▸ Save a copy in Drive** so your
   changes are kept.
2. **Runtime ▸ Change runtime type ▸ T4 GPU ▸ Save.** (The free tier is enough.)
3. Run the first two code cells: `!nvidia-smi` (it should show a **Tesla T4**) and the `pip install` cell
   (it should end with `✅ Install done`).
4. **Runtime ▸ Restart session.** This step matters; the next section explains why.
5. Click into the **1b · Imports** cell and choose **Runtime ▸ Run cell and below**.
6. Read each section's **📍 In one line** and **👀 What you should see** while it runs, and compare with
   your output.

> **Colab asks "Grant access to HF_TOKEN?"** You'll only see this if you've saved a Hugging Face token in
> Colab's 🔑 Secrets. The models are public, so **Cancel** is fine. The run pauses until you answer.

### What success looks like

On the 35 held-out test tickets (two verified runs; yours will land in this range):

| Metric | Before training | After training | Target |
|---|---|---|---|
| Valid JSON | 94–97% | **100%** | 100% |
| Category accuracy | 23–31% | **100%** | ≥ 90% |
| Priority accuracy | 37–40% | **89–94%** | ≥ 85% |

Training loss should fall from about **2.1 to 0.1**, with validation loss falling every epoch.

---

## If something goes wrong

| You see | Cause | Fix |
|---|---|---|
| `NameError` / `cannot import name 'is_flax_available'` in the imports cell | You skipped the restart after `pip install` | **Runtime ▸ Restart session**, then run from **1b · Imports** down |
| `🚨 CUDA VERSION MISMATCH 🚨` when §5 loads the model | Colab upgraded CUDA again and `bitsandbytes` is too old | Run `!pip install -U bitsandbytes`, restart, run from **1b** down. Please open an issue too. |
| `No GPU detected` / `Device: cpu` | No GPU runtime | **Runtime ▸ Change runtime type ▸ T4 GPU** |
| `CUDA out of memory` | Settings too large for the GPU | Lower `MAX_SEQ_LEN`, then use batch size 1 with more accumulation (§12 in the notebook) |
| "Too many sessions" / "Cannot connect to GPU" | Free-tier limits | Close other Colab tabs (**Runtime ▸ Manage sessions**) or try again later |

The notebook's **§12 Troubleshooting** covers more cases.

---

## Make it yours (exercises)

1. **Fix the weak spot.** Priority accuracy lags because only ~21 of 281 training tickets are *Urgent*.
   Add more Urgent templates in `generate_dataset.py` (or the dataset cell), retrain, and compare.
2. **Scale up.** Change `MODEL_ID` to `Qwen/Qwen2.5-1.5B-Instruct`. It still fits a free T4.
3. **Sweep one setting.** Try LoRA rank `r = 8` and `r = 32`. Change one thing at a time and re-measure on
   the same test set.
4. **Bring your own data.** Write JSONL files with the same four keys (`ticket`, `category`, `priority`,
   `suggested_reply`) and replace the dataset cell (marked 🔁 **SWAP HERE**).

---

## What's in this repo

| Path | What it is |
|---|---|
| `Fine_Tuning_LLMs_S2_Python_Bala_Krishna_Ragala_V2.ipynb` | **The lab.** Every section follows *In one line → Why → What → How → What you should see*. |
| `slides/index.html` | The slide deck as one self-contained page. Open it in any browser, or use the link above. |
| `notes.md` | Companion reading: a detailed Why/What/How for every cell, plus a glossary. |
| `finetuning-llms-teaching-script.md` | Full instructor narration for the 2.5-hour theory session, with checkpoint answers and Q&A. |
| `generate_dataset.py` | The synthetic ticket generator (the same code is embedded in the notebook). |
| `data/` | A pre-generated copy of the dataset. The notebook regenerates it, so you don't need it in Colab. |
| `requirements.txt` | The exact package versions the lab was verified with. |
| `verify_accuracy.py` | Optional: reproduce the result outside Colab (QLoRA on CUDA, plain LoRA on Mac or CPU). |
| `build_notebook.py` · `slides/build_slides.py` · `slides/src/` | For maintainers: the notebook and slides are **generated** from these sources. |

### Run locally (optional)

The 4-bit QLoRA path needs an NVIDIA GPU. On a Mac or CPU, use the verification script, which falls back
to plain LoRA:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 generate_dataset.py                                       # writes data/{train,val,test}.jsonl
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" python3 verify_accuracy.py  # prints BEFORE/AFTER + PASS/FAIL
```

### For maintainers

Don't hand-edit the `.ipynb` or `slides/index.html`. Edit the sources and rebuild:

```bash
python3 build_notebook.py          # regenerates the notebook and syntax-checks every code cell
python3 slides/build_slides.py     # regenerates slides/index.html from slides/src/
```

Colab's Python, CUDA and preinstalled packages change every few months. Re-run the notebook in Colab after
any change and update **Last verified** above.
