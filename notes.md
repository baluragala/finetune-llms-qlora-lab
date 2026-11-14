# 📚 Deep Conceptual Notes — Fine-Tuning LLMs (Session 2, V2)

> Companion to `Fine_Tuning_LLMs_S2_Python_Bala_Krishna_Ragala_V2.ipynb`.
> For **every cell and concept** in the notebook this file gives a **WHY** (the problem it
> solves), a **WHAT** (the precise idea), and a **HOW** (what the code actually does), each
> anchored to a **real-world analogy** so the intuition sticks.

---

## 🧠 The one mental model to carry through the whole lab

We are running an **apprenticeship for a chef**.

- A **base / pretrained model** is a *culinary-school graduate*: broadly skilled, has tasted everything,
  but follows no one's orders — hand them a ticket and they just keep cooking whatever they feel like.
- An **instruction-tuned model** is that same chef *after a week on the line*: give them an order ticket
  and they make exactly that dish and call "service."
- **Fine-tuning (our job today)** is the *house onboarding*: we drill **one dish** — structured support
  triage as JSON — until the chef plates it on reflex, in *our* exact house style.
- **LoRA/QLoRA** is *how* we run that onboarding cheaply: instead of rewiring the chef's brain, we clip a
  few **sticky notes** onto their existing skills ("in OUR kitchen, always plate triage as JSON").

Keep this image in mind; every section below is one step of that apprenticeship.

The lab also threads a **three-act story** you should be able to narrate:

1. **Act 1 (§1c):** a base model *can't follow instructions*; an instruct model *can* — but it doesn't
   know *our* labels.
2. **Act 2 (§5–§8):** we install *our* labels/format with QLoRA + LoRA.
3. **Act 3 (§9):** we *prove* it worked with before/after metrics and pictures.

---

## §0 · Hardware & VRAM — "know your kitchen before you cook" (cell 2: `nvidia-smi`)

**WHY.** Fine-tuning fails most often not because the idea is wrong but because you **run out of GPU
memory (VRAM)**. Knowing your memory budget *before* you write training code is the difference between a
lab that runs and one that dies at `CUDA out of memory`.

**WHAT.** VRAM holds four things during training: (1) the **model weights**, (2) the **gradients**, (3)
the **optimizer state** (Adam keeps *two* extra numbers per weight — momentum and variance), and (4) the
**activations** (intermediate results saved for backprop). Naïvely, full training needs **~4–5× the model
size** in memory. A free Colab **T4 has ~15 GB**.

**HOW.** `nvidia-smi` is a "look in the fridge" command — it prints which GPU you got, how much VRAM it
has, and what's already in use. Run it first so you're never guessing.

> 🔬 **Analogy — the kitchen counter.** The model weights are the *ingredients on the counter*. But to
> actually cook you also need *prep space* (activations), *a notepad of what you changed* (gradients), and
> *the recipe-in-progress with two bookmarks per step* (Adam's optimizer state). A tiny ingredient list
> can still overflow a small counter once all the prep clutter is out. QLoRA, coming up, is how we keep
> the counter clear.

---

## §1 · Environment setup — pinned versions (cells 3–5)

**WHY.** The `transformers` / `peft` / `trl` ecosystem makes **breaking changes between minor versions**.
A notebook that worked last month can break today on a default `pip install`. In a classroom that means
30 students hitting 30 different stack traces.

**WHAT.** We **pin exact versions** (`transformers==4.45.2`, `peft==0.13.2`, `trl==0.11.4`, …). Pinning =
reproducibility. We pin `bitsandbytes==0.49.2` specifically because it must ship a binary for the CUDA
version Colab's PyTorch uses. As of Nov 2026, Colab runs Python 3.13 + PyTorch on **CUDA 13.0**; older
bitsandbytes (≤ 0.47) stops at CUDA 12.x and the 4-bit load fails with `CUDA VERSION MISMATCH`.

**HOW.** Cell 4 is a `pip install` with locked versions. Cell 5 is a **STOP — restart the runtime** note:
Colab pins some libraries at boot, so a restart guarantees the freshly installed versions are the ones
actually imported. You do *not* re-run the install after restarting; packages persist for the session.

> 🔬 **Analogy — a recipe's oven temperature.** "Bake at 180 °C" is a pinned version. You can experiment
> with the temperature once you understand the dish — but for your *first* bake, use exactly the number
> on the card, or you'll get 30 differently-burnt cakes and blame the recipe.

---

## §1b · Imports & reproducibility (cells 6–7)

**WHY.** Two goals: (1) load the toolset, and (2) make the run **deterministic** so results are
*comparable* across students and across before/after.

**WHAT.** Each library is a station in the kitchen:
- `transformers` — the model, tokenizer, 4-bit loader, generation.
- `bitsandbytes` — the actual 4-bit quantization kernels (used *through* transformers).
- `peft` — LoRA adapters (Parameter-Efficient Fine-Tuning).
- `trl` — `SFTTrainer`, a correct wrapper over HF `Trainer` for supervised fine-tuning.
- `datasets` — fast, memory-mapped data loading.
- `accelerate` — device placement / mixed precision plumbing.

**Reproducibility** means seeding every source of randomness (Python, NumPy, PyTorch) with a fixed `SEED`.

**HOW.** Cell 7 imports, prints versions (so the output documents the run), defensively imports
`bitsandbytes` (clear fix message instead of a crash), seeds everything, and prints the device + VRAM.

> 🔬 **Analogy — a dealt deck with a known shuffle.** A "random" seed is like agreeing on a specific
> shuffle of the cards before the game. Everyone at every table gets the *same* hands, so when one table's
> strategy wins, you know it was the strategy — not luck.

---

## §1c · Base (pretrained) vs Instruct — why the starting point matters (cells 8–10)

**WHY.** Choosing the wrong *kind* of model wastes the whole exercise. You cannot drill a house dish into a
chef who won't read the order ticket. This section *proves*, live, that the two model types behave
completely differently — and motivates starting from an **instruct** model.

**WHAT.** Two kinds of model, same family, same size:
- **Base / pretrained** (`Qwen2.5-0.5B`): trained on a single objective — *predict the next token*. It is a
  world-class **autocomplete**, but was never taught to *follow an instruction* or to *stop*.
- **Instruction-tuned** (`Qwen2.5-0.5B-Instruct`): the same base, trained *further* on `(instruction →
  response)` pairs using a **chat template**. It learned the **habit of answering**.

**HOW.** Cell 9 defines `quick_generate(model_id, prompt, use_chat_template)`: it loads a model, generates
once with **greedy decoding** (`do_sample=False`, so the output is reproducible), and then **frees the
model** (`del` + `gc.collect()` + `empty_cache()`) so nothing squats on VRAM before real training. It runs
the **same prompt** ("classify this ticket, reply with ONLY JSON") through both models. Cell 10 renders a
**red/green side-by-side card** (with a plain-text fallback).

**The result you'll see** (real, captured):
- **BASE** ignores the instruction and *loops*: "I have a new phone and I need to upgrade to a Pro plan."
  ×4 — no JSON, never stops.
- **INSTRUCT** returns clean JSON: `{"category": "Billing and Service Issues"}`.

**The crucial subtlety (the segue to the whole lab):** the instruct model follows the *format* but invents a
label outside *our* schema ("Billing and Service Issues" ≠ our `"Billing"`). Instruction tuning gives us
format-following **for free**; it does **not** give us *our* six categories, *our* priority scale, or *our*
exact keys. **That** gap is what fine-tuning closes.

> 🔬 **Analogy — two new hires.** The base model is a brilliant new grad who, asked "file this ticket,"
> launches into a TED talk and never stops. The instruct model is a hire who's done reception before: they
> *fill out a form* — but they use their *last company's* categories. Onboarding (fine-tuning) teaches them
> *your* form.

---

## §2 · Data strategy — synthetic triage dataset (cells 11–12)

**WHY.** The #1 cause of a failed fine-tune is **bad data**. For a classroom we also need data that is
reproducible (no downloads), privacy-safe (no real PII), and shaped *exactly* like the task. Real support
logs are proprietary, messy, and slow to clean.

**WHAT.** A **seeded synthetic generator** produces one JSON record per ticket with four keys:
`ticket`, `category` (1 of 6), `priority` (1 of 4: Low/Medium/High/Urgent), `suggested_reply`. We expand a
few templates **combinatorially** (product × opener) into ~350 natural, unique tickets, then **cap every
category to the same count** so classes are balanced.

**HOW.** Cell 12 *is* the generator (`generate_dataset.py`, embedded so students read it). It fills
templates, de-duplicates, balances per category, adds a few `URGENT` outliers (so every priority class is
represented), shuffles, and writes `data/{train,val,test}.jsonl`.

**Two disciplines worth stealing:**
1. We wrote the **evaluation target** (the JSON schema) *before* writing any training data. Defining "what
   good looks like" first is what separates a fine-tune that ships from one that wanders.
2. We **balanced the classes**. An imbalanced dataset teaches the model to be lazy — always guess the
   majority class and still score "okay."

> 🔬 **Analogy — flash-cards for an exam.** You don't study from a random pile; you make a *balanced* deck
> (equal cards per topic) and you write the *answer key first* so you can grade yourself honestly. An
> unbalanced deck (90% one topic) tricks you into feeling ready when you've only learned to guess.

---

## §2b · Load & inspect — "always look at your data" (cells 13–14)

**WHY.** *Trust nothing you haven't printed.* Most "the model didn't learn" bugs are actually "the data was
broken and nobody looked."

**WHAT/HOW.** Cell 14 loads the three JSONL files with `datasets.load_dataset`, prints the split sizes, a
full sample record, and the **category/priority balance** (a `Counter`). If the balance is lopsided or a
record looks wrong, you catch it *now* — before spending GPU time.

> 🔬 **Analogy — mise en place.** Professional kitchens lay out and inspect every ingredient *before* the
> burner is lit. Looking at your data is mise en place: cheap now, catastrophic to skip.

---

## §3 · The chat template — "speak the model's native dialect" (cells 15–16)

**WHY.** Instruction-tuned models were trained with a **specific chat format** (special tokens marking
*system*, *user*, *assistant* turns). Feed plain text that ignores this format and you fight the model's
training — quality collapses.

**WHAT.** A **chat template** is the exact punctuation/markup a model expects. Qwen2.5 uses *ChatML*:
```
<|im_start|>system
You are a customer-support triage assistant...<|im_end|>
<|im_start|>user
Hi, I was charged $49 twice...<|im_end|>
<|im_start|>assistant
{"category": "Billing", "priority": "High", "suggested_reply": "..."}<|im_end|>
```
The **system message** is where we install the *job description*: the allowed categories, the priority
scale, and "reply with ONLY JSON."

**HOW.** Cell 16 sets `MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"`, ensures a **pad token** exists (decoder
models often lack one; we reuse EOS — standard practice), sets **right padding** (for *training*; left
padding is for batched *generation*), and defines:
- `to_target_json(ex)` — the exact string we want the model to *learn to produce*.
- `format_for_training(ex)` — builds the 3-message list (system → user ticket → assistant JSON) and renders
  it to one `text` string via `tokenizer.apply_chat_template(..., tokenize=False)`. For **training** we
  include the assistant turn (the answer to learn). For **inference** (later) we stop after the user turn
  and set `add_generation_prompt=True` so the model writes the answer itself.

The template **auto-adapts** if you swap `MODEL_ID` — every instruct model ships its own template.

> 🔬 **Analogy — the right envelope and letterhead.** You can write a perfect cover letter, but if you mail
> it on a napkin with no address, it never gets read. The chat template is the company letterhead and the
> properly addressed envelope: same content, but now in the format the recipient was trained to open.

---

## §4 · Tokenization — padding, truncation, labels (cells 17–18)

**WHY.** The model never sees text — it sees **token IDs** (integers). Three choices here silently make or
break training.

**WHAT.**
1. **Tokens** — text is chopped into sub-word pieces; each maps to an integer the model embeds.
2. **Padding** — batches must be rectangular, so short sequences get padded to the batch's longest. Pad
   tokens are **masked out of the loss** (label `-100`) so the model isn't "taught" to predict padding.
3. **Truncation / `max_seq_length`** — a hard length cap. Too low **chops off the answer** (you'd train on
   an input whose target isn't there!); too high wastes VRAM. Choose it from the *real* length distribution.
4. **Labels** — for causal-LM SFT, labels are the inputs shifted by one (predict the next token);
   `SFTTrainer` builds these automatically and masks pads to `-100`.

**HOW.** Cell 18 tokenizes every formatted example, prints the length distribution (min/median/p95/max),
and sets `MAX_SEQ_LEN = 1024` — safely above p95 so **no answer is ever truncated**, but not wastefully
large. Sequence length is a *quadratic-ish* VRAM lever: if you OOM, lower this first.

**The `-100` trick, concretely:** cross-entropy in PyTorch ignores any position whose label is `-100`. So
masking padding (and, optionally, the prompt) to `-100` means "don't grade the model on predicting these."

> 🔬 **Analogy — graded vs ungraded parts of an exam.** The exam booklet has a cover page (your name) and
> the actual questions. You don't get *marked* on copying your name onto the cover (`-100`), only on the
> answers. And never photocopy a test that cuts off the last question (truncation) — the student would be
> graded on an answer the paper never showed.

---

## §5 · Load the base model in 4-bit — the "Q" in QLoRA (cells 19–20)

**WHY.** Storing each weight in **4 bits instead of 16** shrinks the frozen base ~4× (≈3 GB → ≈1 GB). That
shrinkage is exactly what lets a real model train on a 15 GB T4 with room to spare.

**WHAT — quantization.** A 16-bit weight is a high-precision number. **4-bit quantization** stores a coarse
approximation plus a scale. Key settings:
- `load_in_4bit=True` — turn it on.
- `bnb_4bit_quant_type="nf4"` — **NF4** ("normal float 4") is a code designed to match how neural-net
  weights are actually distributed (bell-curved around zero), so it loses less quality than naïve `fp4`.
- `bnb_4bit_use_double_quant=True` — also quantize the *quantization constants*; a free extra VRAM win.
- `bnb_4bit_compute_dtype=float16` — math is still done in 16-bit; only **storage** is 4-bit. (On Ampere+
  GPUs prefer `bfloat16`.)

**HOW.** Cell 20 builds a `BitsAndBytesConfig`, loads the model with `device_map="auto"`, sets
`use_cache=False` (required with gradient checkpointing during training), and calls
`prepare_model_for_kbit_training(...)` — which makes a 4-bit model trainable (casts layernorms to fp32,
enables input gradients, etc.). It then prints the GPU memory actually used (~1 GB for the base).

> 🔬 **Analogy — JPEG for weights.** A RAW photo (16-bit) is gorgeous but huge. A well-tuned JPEG (NF4) is
> a fraction of the size and, to the eye, nearly identical — *because the compression is designed around
> how real images look*. We store the chef's giant cookbook as a smart JPEG so it fits on the shelf, then
> do the actual cooking in full precision.

---

## §6 · LoRA configuration — the "LoRA" in QLoRA (cells 21–22)

**WHY.** You **can't backprop into 4-bit weights** — they're frozen. So instead of editing the giant
matrices, we inject **tiny trainable matrices** beside them and train only those: **<1% of parameters**,
yet enough to strongly steer behavior.

**WHAT — the low-rank idea.** A weight update `ΔW` to a big `d×d` matrix can be *approximated* by the
product of two **skinny** matrices: `ΔW ≈ B · A`, where `A` is `r×d` and `B` is `d×r` with **r tiny**
(e.g. 16). We train `A` and `B` instead of the full `ΔW`. Crucially **`B` starts at zero**, so at step 0
the adapter adds *nothing* — the model behaves exactly like the base (we exploit this for a free baseline
in §7). The knobs:
- `r=16` — adapter **rank** = capacity. Higher = more expressive but more params / overfit risk. 8–16 is
  plenty for a narrow task; 32–64 for broad/multi-task.
- `lora_alpha=32` — a **scaling** factor; the adapter's effect is scaled by `alpha/r`. The heuristic
  `alpha = 2×r` keeps the effective scale constant as you change `r`. Think "volume knob."
- `target_modules=[q,k,v,o_proj, gate,up,down_proj]` — *which* matrices get adapters. We adapt **attention
  AND MLP** (the modern default; better than attention-only).
- `lora_dropout=0.05` — light regularization against overfitting on a small dataset.
- `bias="none"`, `task_type="CAUSAL_LM"` — keep it lean; tell PEFT it's next-token prediction.

**HOW.** Cell 22 builds the `LoraConfig`, wraps the model with `get_peft_model(...)`, and calls
`print_trainable_parameters()` — the punchline: you'll see something like "0.5% of parameters trainable."

> 🔬 **Analogy — sticky notes on a textbook.** Full fine-tuning rewrites the textbook (expensive, risky,
> and you might ruin pages you needed). LoRA leaves the textbook untouched and adds a thin stack of
> **sticky notes** in the margins: "in our context, do it *this* way." `r` is how many sticky notes; `alpha`
> is how boldly they're written; `target_modules` is which chapters get notes. Because the original pages
> are untouched, the model also resists **catastrophic forgetting**.

**QLoRA = quantized base (§5) + LoRA adapters (§6).** The "Q" shrinks the frozen book; the "LoRA" is the
sticky notes. Together: a big model, fine-tuned on a small GPU.

---

## §7 · BEFORE training — measure the baseline (cells 23–24)

**WHY.** "The loss went down" is **not** proof of usefulness. We measure **task metrics on the held-out
test set *before* training**, so the after-number has something honest to beat.

**WHAT — the metrics.** On raw test tickets:
- `valid_json` — fraction of outputs that parse as JSON with our keys (does it obey the format at all?).
- `category_acc` — fraction with the *correct* category.
- `priority_acc` — fraction with the *correct* priority.

**The baseline is "free":** thanks to LoRA's `B=0` init (§6), the adapted model right now is
*behaviorally identical to the base*. So evaluating now == evaluating the un-fine-tuned model.

**HOW.** Cell 24 defines the inference + eval toolkit:
- `triage(ticket)` — formats the ticket with the chat template, generates **greedily** (reproducible),
  temporarily flips `use_cache=True` for fast generation then restores it, and returns the raw text.
- `parse_json(text)` — extracts the first `{...}` block and parses it; returns `None` if unparseable.
- `predict_split(split)` — runs `triage` on every row and returns **per-example records** (true vs
  predicted category/priority, validity, raw output). *These records are reused by the charts later — we
  generate once, visualize many times.*
- `metrics_from(records)` — aggregates those records into the three numbers.
- It stashes 3 baseline generations for a later side-by-side, computes `before_preds` / `before`, and
  prints them.

> 🔬 **Analogy — the "before" photo.** Nobody believes a fitness transformation without the before photo
> taken on the *same scale, same lighting*. Measuring before training, on the same test set, is that photo.
> And because of LoRA's zero-init, the before photo is free — it's literally the base model.

---

## §8 · Training with `SFTTrainer` — every knob (cells 25–26)

**WHY.** `SFTTrainer` is a correct, batteries-included wrapper over HF `Trainer`: it tokenizes the `text`
field, builds causal-LM labels, masks padding, and runs the loop. The knobs are where the real learning
(yours) happens.

**WHAT — the knobs that matter (slow down here):**
- `num_train_epochs=3` — passes over the data. Too many → memorize/overfit; watch eval loss.
- `per_device_train_batch_size=2` — what fits in T4 VRAM per step.
- `gradient_accumulation_steps=8` — accumulate 8 micro-batches before one optimizer update →
  **effective batch = 2 × 8 = 16** *without* the memory of a true batch-16 step.
- `learning_rate=2e-4` — LoRA tolerates higher LRs than full FT (few params); 1e-4–3e-4 is the sweet spot.
- `lr_scheduler_type="cosine"` — decay the LR smoothly toward ~0 by the end → stable convergence.
- `warmup_ratio=0.03` — ramp the LR up over the first 3% of steps so early updates don't destabilize.
- `gradient_checkpointing=True` — recompute activations in the backward pass instead of storing them; big
  VRAM save for ~20–30% speed cost.
- `fp16=True` — T4 mixed precision (use `bf16=True` on Ampere+).
- `optim="paged_adamw_8bit"` — 8-bit Adam + paging → optimizer state barely uses VRAM.
- `max_seq_length=MAX_SEQ_LEN` — the truncation cap chosen in §4.
- `packing=False` — keep each ticket its own example (clearer for short, structured tasks).

**The key mental model:** `effective_batch = per_device_batch × grad_accum × num_gpus`. On one T4 that's
`2 × 8 × 1 = 16`. Want a bigger *effective* batch but you're OOM? **Raise accumulation, not per-device
batch** — accumulation is nearly free in memory.

**HOW.** Cell 26 builds the `SFTConfig`, constructs the `SFTTrainer` (model + config + train/val text), and
prints steps-per-epoch. (Note: newer `trl` renames `tokenizer=` to `processing_class=`.)

> 🔬 **Analogy — gradient accumulation = carrying groceries in trips.** You need to move 16 bags but can
> only hold 2 at a time. Gradient accumulation is making 8 trips and *only then* locking the front door
> (one optimizer step) — you moved all 16 bags' worth without needing 16 hands. Warmup is easing the car
> out of the driveway before flooring it; the cosine schedule is gently braking as you arrive.

---

## §8b · Run training (cells 27–28)

**WHY.** Notice we spent most of the lab on *data, format, and config* — the actual fit is **one line**.
That ratio is the real lesson: fine-tuning is 80% preparation.

**WHAT/HOW.** Cell 28 calls `trainer.train()`. On a T4 this is ~5–12 minutes for 3 epochs at this size.
Watch `loss` fall and `eval_loss` fall. If `eval_loss` starts **rising** while `train_loss` keeps falling,
you're **overfitting** — stop early / reduce epochs.

> 🔬 **Analogy — the bake.** All the work was in the prep and the recipe card. Pressing "start" on the oven
> is anticlimactic — as it should be. If the cake rises then sinks (eval loss turns up), you over-baked.

---

## §8c · The training curve — healthy vs broken (cells 29–30)

**WHY.** A falling loss curve is the **first** proof your fine-tune is working — and it lets you spot
failure instantly. This section exists because the *original* version of this lab **diverged**, and
recognizing that pattern is a core skill.

**WHAT — read two curves:**
| | Healthy run (this lab) | Broken run (classic failure) |
|---|---|---|
| pattern | `~2.1 → ~0.1`, smooth | `403 → 6947 …` then snaps to **`0.000000`** forever |
| meaning | the model is learning | weights **diverged to NaN**; the reported loss then collapses to 0 |
| usual cause | — | LR too high / no chat template / training a *base* model on full text |

A healthy cross-entropy **starts around `ln(vocab) ≈ 10`**, not in the hundreds. Loss in the
hundreds/thousands is not a valid loss; a sudden perfect `0.000000` is not success — it's a **NaN
graveyard**.

**HOW.** Cell 30 reads `trainer.state.log_history`, plots train loss and eval loss vs step with
matplotlib, and prints the start→final loss with a one-line sanity verdict.

> 🔬 **Analogy — a heart-rate monitor.** A steady, descending trace means the patient is fine. A spike to
> impossible numbers followed by a **flat line at zero** is not "perfectly calm" — it's a flat-line. Same
> for loss: `0.000000` everywhere is the monitor screaming, not the model winning.

---

## §9 · AFTER training — re-measure and compare (cells 31–32)

**WHY.** The payoff. Same metrics, same test set, now on the fine-tuned model.

**WHAT/HOW.** Cell 32 calls `predict_split(ds["test"])` again to get `after_preds`, aggregates to `after`,
and prints a **before / after / Δ-gain** table plus an explicit "Pass the 70% bar? ✅/❌". Expect
`valid_json → ~1.0` and large jumps in category/priority accuracy. (Verified: category `0.31 → 1.00`,
priority `0.37 → 0.89` on 0.5B.)

**How to read it:** if `valid_json` went from "sometimes" to "always" and accuracy jumped, the
apprenticeship worked. If accuracy barely moved, suspect the **data** (too few/inconsistent examples)
before blaming the method.

> 🔬 **Analogy — the after photo, same scale.** We re-weigh on the exact scale from §7. The *delta* is the
> only number that matters — an absolute score with no baseline is a vanity metric.

---

## §9b · Qualitative side-by-side (cells 33–34)

**WHY.** A metrics table convinces analysts; *seeing* a messy "before" become clean structured JSON
convinces everyone else.

**WHAT/HOW.** Cell 34 reuses the 3 stashed baseline generations, regenerates with the trained adapter, and
renders a **Ticket → ❌ BEFORE → ✅ AFTER → 🎯 EXPECTED** table (red/green coded HTML, with a text fallback).
HTML is escaped so ticket text can't break the layout.

> 🔬 **Analogy — courtroom exhibit.** Numbers are testimony; the side-by-side is the *photo entered into
> evidence*. Juries (and stakeholders) believe what they can see.

---

## §9c · Visual proof — charts (cells 35–38)

**WHY.** Numbers in a table get skimmed; a **red bar jumping to green** and a **confusion matrix collapsing
onto the diagonal** make the improvement impossible to miss. All views **reuse predictions already
generated** — no extra inference.

**WHAT/HOW.**
- Cell 36 — **before/after bar chart** of the three metrics with the **70% target line** drawn in.
- Cell 37 — **category confusion matrices**, before vs after. Rows = true category, columns = predicted; a
  perfect model is bright **only on the diagonal**. BEFORE scatters off-diagonal (wrong guesses); AFTER
  concentrates on it.
- Cell 38 — **per-example results table**: exactly which tickets flipped from ✗ to ✓, and a count of
  WRONG→RIGHT flips.

> 🔬 **Analogy — the confusion matrix is a seating chart.** Imagine guests (tickets) who must sit at their
> correct table (category). BEFORE: people scattered everywhere. AFTER: everyone on the diagonal — at their
> own table. One glance tells you the party is organized.

---

## §10 · Inference, saving, merging, pushing (cells 39–42)

**WHY two artifacts?**
- The **LoRA adapter** alone is tiny (a few MB). Ship it next to the shared base, load on demand — perfect
  for serving many fine-tunes from one base, and for **instant rollback** (just detach it).
- A **merged model** folds the adapter into the base weights → a standalone model servable with vanilla
  `from_pretrained`, no PEFT at inference. Bigger on disk, simplest to deploy.

**WHAT/HOW.**
- Cell 40 — `trainer.save_model(ADAPTER_DIR)` + save tokenizer; prints the adapter size (tiny vs a full
  model).
- Cell 41 — **merge**: you can't merge into 4-bit, so it frees the training model (`del` + `empty_cache`,
  to avoid OOM on T4), reloads the base in **fp16**, attaches the adapter via `PeftModel.from_pretrained`,
  and calls `merge_and_unload()` → saves a standalone model.
- Cell 42 — optional `push_to_hub` (off by default).

Adapter vs merged is a **deployment** decision, not a quality one — the weights are identical.

> 🔬 **Analogy — recipe card vs new cookbook edition.** The adapter is a *recipe card* you clip into the
> existing cookbook (swap it, remove it, hand it to a colleague). The merged model is *printing a new
> edition* with the recipe baked in. Same dish; different distribution choice.

---

## §11 · Optimizations tour (cell 43) — a field guide

**WHY.** When results aren't good enough, you need to know *which knob* to turn, and to turn **one at a
time** (otherwise you can't attribute the change).

**Key trade-offs (the headline is LoRA vs QLoRA):**
- **QLoRA** (4-bit base, this lab): lowest VRAM; fits big models on small GPUs. Costs a little speed and a
  hair of quality from quantization. **Default for one consumer/Colab GPU.**
- **LoRA** (16-bit base): if you *have* the VRAM, skip 4-bit — slightly faster, marginally higher quality.
- **Full fine-tuning**: all weights trained. Max capacity, max cost, highest forgetting risk. Reserve for
  deep domain shifts with lots of data.

**Tuning levers, with the symptom that calls for each:**
- *Underfitting* (train loss plateaus high, outputs bland) → **raise `r`** (16→32, keep `alpha=2r`), or
  add epochs / raise LR.
- *Overfitting* (eval loss rises, parrots training phrasings) → **lower `r`**, raise `lora_dropout`, cut
  epochs.
- *OOM* → lower `max_seq_length` first; then `per_device_batch=1` with double `grad_accum`; keep gradient
  checkpointing + `paged_adamw_8bit` on.
- *Quality from quantization* → `nf4` over `fp4`, `bf16` compute on Ampere+.

> 🔬 **Analogy — tuning an instrument one string at a time.** Tighten all six strings at once and you'll
> never know which one fixed (or wrecked) the chord. Form a hypothesis, turn one peg, play the same chord
> (the fixed test set), listen.

---

## §12 · Troubleshooting appendix (cell 44)

**WHY.** The same handful of errors hit everyone; knowing the *order* of fixes saves hours.

**The greatest hits:**
- 🔴 **CUDA OOM** → (1) lower `max_seq_length`, (2) `per_device_batch=1` + double `grad_accum`, (3) ensure
  gradient checkpointing + `paged_adamw_8bit`, (4) confirm 4-bit load, (5) restart runtime to clear leaks.
- 🔴 **`SFTTrainer`/`SFTConfig` TypeError** → version drift; with our pins use `tokenizer=`, on newer `trl`
  use `processing_class=`; `eval_strategy` (not `evaluation_strategy`) for transformers 4.45.
- 🔴 **`bitsandbytes`: `CUDA VERSION MISMATCH` / `triton.ops` / `cudaXXX.so not found`** → bnb too old for
  Colab's CUDA; `!pip install -U bitsandbytes`, then **restart**. (Pinning bnb *too low* causes this.)
- 🔴 **Outputs aren't valid JSON** → raise `max_new_tokens` (answer cut off), confirm training targets were
  clean `json.dumps`, add 1–2 epochs.
- 🔴 **Model "forgot" general ability** → catastrophic forgetting: lower `r`, fewer epochs, mix in a few
  general examples.

> 🔬 **Analogy — a pre-flight checklist.** Pilots don't *remember* what to do when a light blinks; they run
> the checklist *in order*. This appendix is that checklist — follow it top to bottom instead of panicking.

---

## §13 · Instructor talking-track recap (cell 45)

**WHY.** One-sentence anchors per section, so you can teach (or revise) the whole arc from memory.

**The closing line worth memorizing:** *we turned a generalist into a reliable specialist by changing
**<1% of its parameters** on a **tiny dataset** on a **free GPU** — and we can roll it back by deleting one
small file. That asymmetry — huge behavioral change, tiny cost and risk — is why LoRA/QLoRA is the default
way teams ship fine-tunes today.*

---

## 🔁 Glossary of the core concepts (quick reference)

| Term | One-line meaning | Analogy |
|---|---|---|
| **Base / pretrained model** | Trained only to predict the next token; great autocomplete, ignores instructions | New grad who won't read the order ticket |
| **Instruction-tuned model** | Further trained to follow instructions via a chat template | Hire who fills out the form |
| **Token** | Sub-word integer the model actually reads | Letters tiles in Scrabble |
| **Chat template** | Exact special-token format the model expects | Company letterhead + envelope |
| **Padding / `-100`** | Fill short sequences; don't grade the filler | Ungraded cover page of an exam |
| **Truncation / `max_seq_length`** | Hard length cap; never cut the answer | Don't photocopy a test missing its last question |
| **Quantization (4-bit, NF4)** | Store weights coarsely to save memory | JPEG for the cookbook |
| **LoRA (`r`, `alpha`)** | Train tiny low-rank adapters, freeze the base | Sticky notes in the margins |
| **QLoRA** | 4-bit base + LoRA adapters | JPEG cookbook + sticky notes |
| **Gradient accumulation** | Sum several micro-batches before one update | Carrying 16 bags in 8 trips |
| **Effective batch** | `per_device × grad_accum × #GPUs` | Total groceries moved, not per-trip |
| **Gradient checkpointing** | Recompute activations to save VRAM | Re-deriving a step instead of storing it |
| **Catastrophic forgetting** | Fine-tuning erases general skill | Cramming for one exam, forgetting the rest |
| **Adapter vs merged** | Swappable add-on vs standalone model | Recipe card vs new cookbook edition |
| **NaN divergence** | Loss explodes then reads `0.000000` | Heart monitor spike → flat line |

---

*These notes pair 1:1 with the notebook sections. Read a section here, then run the matching cell — WHY
first (so you know what problem you're solving), WHAT next (the precise idea), HOW last (the code that
does it).*
