# Curriculum Learning Pipeline — Full Analysis

Analysis of the Qwen3-VL 8B LoRA curriculum-learning SFT pipeline for the
nuScenes-based VQA dataset. Covers (1) pipeline architecture & modules and
(2) loss-side setup.

---

## Part 1 — Pipeline Architecture

### 1.1 High-level shape

Ten sequential LoRA fine-tuning stages over the 10 nuScenes-VQA categories,
ordered easy → hard:

```
OBS → IDN → AAS → SRO → TSS → RML → DRA → RWP → ESC → CHR
 ▲ perceptual ─────────────────────── reasoning ▲
```

Each stage:

1. Trains on **only that category's** SFT slice
   (`sft_train_qwen3vl_<CAT>.json`).
2. Warm-starts from the **previous stage's LoRA adapter** (stage 0 starts
   from a fresh adapter).
3. Runs its own WSD (warmup → stable → decay) LR cycle — LR returns to 0 at
   every stage boundary.
4. Logs **per-category eval loss** on every category at every eval step
   (forgetting monitor).
5. Runs a **per-category generation accuracy** sweep at stage end →
   `eval_report.json`.
6. After the last stage, an aggregator produces a stage × category matrix +
   forgetting column.

The orchestrator is **`qwen-vl-finetune/scripts/run_curriculum.sh`**,
parameterised by a single YAML config.

### 1.2 Module map

| Layer | File | Role |
|---|---|---|
| Data prep — split | `nuscenes_pipeline/postprocessing/split_sft_by_category.py` | One unified `sft_train_qwen3vl.json` → 10 `sft_train_qwen3vl_<CAT>.json` files. Same for val. Uses the `category` tag added by `prepare_sft_dataset.py`; falls back to `qa_results/` lookup for legacy data. |
| Data prep — eval subset | `nuscenes_pipeline/postprocessing/build_eval_subset.py` | Stratified per-category sample: N=200 per category, seeded *per-category* so adding/removing a category doesn't perturb others. Writes `sft_val_qwen3vl_<CAT>_subset.json` + manifest, idempotent via `.subset_built_<N>` marker. Smoke version uses N=20. |
| Config schema | `qwen-vl-finetune/configs/curriculum_v1.yaml` (+ `curriculum_smoke.yaml`) | YAML with `defaults:` (shared hyperparams) + `stages:` list (per-stage overrides). |
| Config loader | `qwen-vl-finetune/qwenvl/curriculum/config.py` | Parses YAML, merges `defaults` into each stage, rejects unknown keys. Dual-use: import as Python, or `python -m qwenvl.curriculum.config --emit_shell <i>` to emit `KEY=VALUE` bash assignments for stage `i`. Also writes `eval_dataset_paths.json` into each stage dir. |
| Orchestrator | `qwen-vl-finetune/scripts/run_curriculum.sh` | Loops stages `[START_STAGE..END_STAGE]`, `eval`s the per-stage env, launches `torchrun`, runs end-of-stage eval, then aggregates. Sets `NCCL_TIMEOUT=3600` for long eval sweeps. |
| Trainer | `train_nuscenes_qwen3vl.py` | `setup_lora()` does `PeftModel.from_pretrained(..., is_trainable=True)` when `--lora_pretrained` is set; otherwise fresh `get_peft_model`. Builds **dict-form `eval_dataset`** from `--eval_dataset_paths_json` so HF Trainer logs `eval_<cat>_loss` per key. |
| LR scheduler | `qwen-vl-finetune/qwenvl/train/wsd_scheduler.py` + `WSDTrainer` in `train_nuscenes_qwen3vl.py:481` | `LambdaLR` multiplier: linear warmup → 1.0 plateau → linear decay to 0. `WSDTrainer.create_scheduler` swaps it in when `--use_wsd_scheduler True`; falls through to upstream otherwise. CLI sets `lr_scheduler_type=constant` so HF Trainer doesn't fight the override. |
| Stage-end eval driver | `nuscenes_pipeline/scripts/run_stage_end_eval.sh` | Thin wrapper that calls `python -m nuscenes_pipeline.modules.sft_model_tester --base_model … --lora_path <stage_dir> --per_category_eval_dir <subset_dir> --resize_factor 1 --max_new_tokens 1024`. |
| Stage-end eval impl | `nuscenes_pipeline/modules/sft_model_tester.py::run_per_category_eval` | Loads base + LoRA, iterates each `sft_val_qwen3vl_<CAT>(_subset)?.json`, runs greedy generation, extracts answer, scores against GT. Writes `eval_report.json` (per-cat `n / n_correct / acc / parse_rate` + `macro_acc`) and optionally `eval_predictions.json` into the LoRA dir. |
| Aggregator | `hf_dataset_train/aggregate_curriculum_reports.py` | Walks `<output_root>/stage_*/eval_report.json`, emits `curriculum_report.{csv,md}` with stage × category matrix + forgetting column (`best_earlier_acc − final_acc`). |

### 1.3 Stage transition mechanics

The non-obvious bits, in execution order:

1. **Per-stage env emission** (`config.py:emit_shell`) writes a fresh
   `eval_dataset_paths.json` *inside the stage output dir* listing all 10
   per-cat subset paths. This is what makes per-category eval losses appear
   in TensorBoard at every eval step, not just at stage end.
2. **Warm-start** (`train_nuscenes_qwen3vl.py:452-461`): stage `i>0` passes
   `--lora_pretrained <stage_(i-1)_dir>`. `PeftModel.from_pretrained(...,
   is_trainable=True)` loads the adapter weights AND reuses the previous
   adapter's own `LoraConfig` (its r/alpha/dropout/target_modules). CLI
   `--lora_r/_alpha/_dropout` flags are silently ignored on warm-start —
   documented to "avoid silent mismatches". Practical consequence: you can't
   grow/shrink LoRA rank mid-curriculum without doing it explicitly.
3. **Independent WSD per stage**: each stage starts at LR=0, warms up to
   `peak_lr=2e-4` over `warmup_ratio*total_steps` (default 10%), holds for
   70%, decays back to 0 over the final 20%. The shape is a trapezoid:

   ```
   1.0 |        ───────────────────
       |       /                   \
   0.0 |__   _/                     \____
            warmup ── stable ── decay
   ```

   With config defaults, this gives 10 trapezoidal LR cycles back-to-back —
   each category gets a clean LR sweep on its own data.
4. **Per-stage generation eval** runs after training, on the *same* subset
   dir used for in-training loss. So the `eval_<cat>_loss` curve and the
   stage-end `eval_report.json` are over the same samples — you can compare
   loss-on-cat vs. accuracy-on-cat without confounds.

### 1.4 Configuration & key defaults (`curriculum_v1.yaml`)

- `output_root: output/curriculum_v1`,
  `base_model: ckpts/qwen3_vl_8b_instruct`
- `eval_subset_dir: sft_dataset/eval_subset_200` (200/cat for in-training
  eval)
- LoRA: r=64, α=128, dropout=0.05, all 7 linear targets
- Optim: 1 epoch/stage, peak LR 2e-4, weight_decay 0.01, max_grad_norm 1.0
- Batch: per_device=1, grad_accum=4 (effective 4/GPU),
  eval_steps/save_steps=500
- Vision: `max_pixels=1_440_208`, `model_max_length=16384`,
  `resize_factor=1` (labels in 1600×900 space — see CLAUDE.md §6)
- DeepSpeed: zero2 by default; `DEEPSPEED_CONFIG=""` disables
- WSD ratios `[0.10, 0.70, 0.20]` (warmup/stable/decay)
- `bf16: true`, `gradient_checkpointing: true`, `report_to: tensorboard`

`curriculum_smoke.yaml` caps each stage at `max_steps: 20` (2 warmup / 14
stable / 4 decay) and uses N=20 eval subsets — a 5–10 minute end-to-end
sanity check of warm-start + dict-eval + WSD + report aggregation.

### 1.5 End-to-end run, command level

```bash
# 1. Per-category SFT splits (one-off)
python -m nuscenes_pipeline.postprocessing.split_sft_by_category \
    --input sft_dataset/sft_train_qwen3vl.json sft_dataset/sft_val_qwen3vl.json \
    --output_dir sft_dataset

# 2. Stratified eval subsets (one-off)
python -m nuscenes_pipeline.postprocessing.build_eval_subset --n_per_cat 200

# 3. Drive the curriculum
bash qwen-vl-finetune/scripts/run_curriculum.sh \
    qwen-vl-finetune/configs/curriculum_v1.yaml

# Resume / partial:
START_STAGE=5 END_STAGE=9 bash …/run_curriculum.sh …yaml
SKIP_STAGE_END_EVAL=1     bash …/run_curriculum.sh …yaml  # train-only
STAGE_END_EVAL_LIMIT=50   bash …/run_curriculum.sh …yaml  # cap generation eval
```

### 1.6 Strengths

- **Clean separation**: YAML config owns hyperparameters; bash owns
  execution; Python owns training/eval — easy to swap any layer.
- **Idempotency hooks**: `START_STAGE`/`END_STAGE` make resume trivial;
  `.subset_built_<N>` marker prevents accidental eval-set churn (which would
  invalidate cross-stage comparisons).
- **Forgetting is observable, not assumed**: dict-form eval logs all 10
  `eval_<cat>_loss` curves continuously through training, so catastrophic
  forgetting shows up before the final aggregator.
- **WSD per stage** is a defensible choice for sequential SFT — each
  category gets a fresh anneal so the previous stage's "cold" decay doesn't
  poison the next stage's adaptation.

### 1.7 Pipeline-level gotchas

- **LoRA capacity is fixed at r=64 across 10 sequential stages**. Late
  stages must overwrite/share earlier-stage features in the same low-rank
  subspace — that's the central forgetting risk. The aggregator's
  "forgetting" column is exactly the right metric to watch.
- **No replay / rehearsal buffer**: each stage sees only its own category.
  If forgetting on early categories (e.g. OBS) is bad by stage CHR, the
  natural lever is to mix in a small fraction of prior categories —
  currently not in the pipeline.
- **Warm-start ignores CLI `lora_*` flags** (`setup_lora` returns early).
  Documented, but means changing LoRA hyperparams mid-curriculum requires a
  deliberate fresh-adapter stage.
- **Stage-end gen eval extracts a single answer and string-matches**
  (`sft_model_tester.run_per_category_eval`). Fine for MCQ / y/n, but
  `open_ended` style answers will score 0 unless the extractor + normaliser
  cover them — worth checking which categories rely on that answer type
  before reading accuracy as ground truth.
- **`gradient_accumulation_steps=4` × 1 GPU × 1 batch** means effective
  batch 4. On multi-GPU runs (`NPROC_PER_NODE>1`) effective batch scales
  with world size — the WSD warmup fraction is over *optimizer steps*, so
  warmup wall-clock shrinks linearly with world size. Not a bug, just
  something to factor in when tuning.

---

## Part 2 — Loss Setup

### 2.1 TL;DR

Standard causal-LM cross-entropy from HF `Trainer`, applied **only to
assistant-turn tokens**. System + user turns are masked with
`IGNORE_INDEX=-100`. No label smoothing, no token weighting, no custom
`compute_loss`, no per-category loss balancing. The "curriculum" only
changes the **data distribution per stage** — the loss objective itself is
identical at every stage and identical between train/eval.

### 2.2 Loss function

`WSDTrainer` (`train_nuscenes_qwen3vl.py:481`) subclasses HF `Trainer` and
only overrides `create_scheduler`. It does **not** override `compute_loss`:

```python
# train_nuscenes_qwen3vl.py:481-505
class WSDTrainer(Trainer):
    """Trainer that swaps in our WSD scheduler when
    `args.lr_scheduler_type == 'warmup_stable_decay'`. All other scheduler
    types fall through to the upstream HF implementation unchanged."""

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if not getattr(self.args, "use_wsd_scheduler", False):
            return super().create_scheduler(num_training_steps, optimizer)

        if self.lr_scheduler is not None:
            return self.lr_scheduler

        opt = optimizer if optimizer is not None else self.optimizer
        self.lr_scheduler = get_wsd_schedule(
            opt,
            num_training_steps=num_training_steps,
            warmup_ratio=self.args.wsd_warmup_ratio,
            decay_ratio=self.args.wsd_decay_ratio,
        )
        rank0_print(
            f"[WSDTrainer] WSD scheduler: total_steps={num_training_steps} "
            f"warmup_ratio={self.args.wsd_warmup_ratio} "
            f"decay_ratio={self.args.wsd_decay_ratio}"
        )
        return self.lr_scheduler
```

So loss = whatever `Qwen3VLForConditionalGeneration.forward(...,
labels=labels)` returns:

- Standard causal LM cross-entropy with `ignore_index=-100`
- Shift-by-one (next-token prediction)
- Mean reduction over non-ignored positions in the batch (HF transformers
  ≥4.36 default; with `gradient_accumulation_steps=4` and recent
  transformers, `num_items_in_batch` is forwarded so gradient accumulation
  is mathematically equivalent to a larger batch — i.e. no per-microbatch
  mean bias)

No flags touch this:

- `label_smoothing_factor`: not set (default 0)
- `optim: adamw_torch` (`TrainingArguments` default in
  `train_nuscenes_qwen3vl.py:118`)
- `weight_decay: 0.01`, `max_grad_norm: 1.0`, `bf16: true` — these affect
  the optimizer step, not the loss surface

### 2.3 Label masking — what loss is computed on

The whole thing lives in `preprocess_with_system_prompt`
(`train_nuscenes_qwen3vl.py:133-209`). Each conversation turn is tokenized
independently with the project's hand-rolled ChatML template:

```
<|im_start|>{role}\n{content}<|im_end|>\n
```

Then per-turn label handling:

| Turn role | `labels` written |
|---|---|
| `system` | all tokens → `IGNORE_INDEX` (prefix context only) |
| `human` (user) | all tokens → `IGNORE_INDEX` (instruction + image placeholders) |
| `gpt` (assistant) | tokens copied as labels, **except the first 3 tokens** (`<\|im_start\|>`, `assistant`, `\n`) which are masked |

The actual masking code:

```python
# train_nuscenes_qwen3vl.py:186-204
conv_msg = [{"role": role, "content": content}]
encode_id = tokenizer.apply_chat_template(conv_msg)
# transformers >=5 returns a BatchEncoding (dict-like, but not a
# subclass of dict) containing {"input_ids", "attention_mask"};
# older versions returned a plain list of token IDs. Normalize.
if not isinstance(encode_id, list):
    encode_id = encode_id["input_ids"]
input_id += encode_id

if role in ["user", "system"]:
    target += [IGNORE_INDEX] * len(encode_id)
else:
    target_mask = encode_id.copy()
    target_mask[:3] = [IGNORE_INDEX] * 3  # Mask <|im_start|>assistant\n
    target += target_mask

assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
```

So loss targets are:

```
{assistant content tokens}<|im_end|>\n
```

i.e. the assistant body **plus** the `<|im_end|>\n` closer. Masking those
first 3 tokens (the assistant-role header) is intentional — the model
should learn to *generate* the body conditioned on the prompt, not learn to
predict the role header itself.

This matches the system-prompt training note in CLAUDE.md §8: system
content is "prefix context but not loss target."

**Padding** in the collator (`NuScenesDataCollator.__call__`, line 345)
pads `labels` with `IGNORE_INDEX`, so pad positions also don't enter the
loss:

```python
# train_nuscenes_qwen3vl.py:337-354
def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
    input_ids = [inst["input_ids"].squeeze(0) for inst in instances]
    labels = [inst["labels"].squeeze(0) for inst in instances]
    position_ids = [inst["position_ids"] for inst in instances]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    position_ids = pad_and_cat(position_ids)

    # Truncate to model_max_length
    max_len = self.tokenizer.model_max_length
    input_ids = input_ids[:, :max_len]
    labels = labels[:, :max_len]
    position_ids = position_ids[:, :, :max_len]
```

Truncation to `model_max_length=16384` is by tail-slicing — if a sample
exceeded 16K tokens (extremely unlikely given the token budget in
CLAUDE.md §7) it would silently drop assistant tokens from the loss. With
the current `gpt` length distribution this never bites in practice.

### 2.4 What's actually inside the labels

The unified Qwen3-VL JSON format (`convert_to_qwen3vl_format`) means the
loss target is the *entire JSON string*. A real OBS sample's `gpt` value:

```json
{
  "reasoning": "- No cyclists are present in the scene; all agents are pedestrians or vehicles.\n- Immediate vicinity (within ~10m) includes car and car, both cars, not cyclists.\n- Since no cyclists exist, they cannot be in the immediate vicinity.",
  "grounding": [
    {
      "image_idx": 3,
      "camera": "Front-right",
      "ref": "<|object_ref_start|>car<|object_ref_end|><|box_start|>(0, ...
    }
  ],
  "answer": "B"
}
```

Concretely from `sft_dataset/sft_train_qwen3vl_OBS.json`:

- Mean GPT chars: **662**, p95: **1334**, max: **14877** (one outlier)
- The model is being asked to maximize likelihood over reasoning bullets +
  grounding bbox JSON + the short answer field jointly

Practical consequence for loss interpretation:

- A typical sample's loss is **dominated by reasoning tokens** (the longest
  field), then grounding (mid-length JSON with numeric bbox coords), then
  answer (often a single letter or a short phrase).
- The `answer` field — the thing you care about at eval time — accounts
  for a tiny fraction of loss-weighted tokens. The pipeline has **no token
  re-weighting** to compensate. This is the single biggest implicit choice
  in the loss setup.
- Outlier-long GPT responses (max 14877 chars in OBS) pull more loss
  weight per sample than short ones. Token-level mean reduction → long
  samples just contribute more terms to the average, not more per-token
  weight.

### 2.5 Train vs. eval loss

Both use `preprocess_with_system_prompt` → both apply the same mask. The
dict-form eval dataset that feeds the per-category loss curves:

```python
# train_nuscenes_qwen3vl.py:583-603
eval_dataset = None
# Dict-form eval: per-category losses logged as eval_{cat}_loss every eval step.
if data_args.eval_dataset_paths_json and os.path.exists(data_args.eval_dataset_paths_json):
    with open(data_args.eval_dataset_paths_json) as f:
        eval_paths = json.load(f)
    eval_dataset = {}
    for cat_name, cat_path in eval_paths.items():
        if not os.path.exists(cat_path):
            rank0_print(f"[warn] eval dataset for '{cat_name}' not found: {cat_path}")
            continue
        eval_dataset[cat_name] = NuScenesVQADataset(
            data_path=cat_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_pixels=data_args.max_pixels,
            min_pixels=data_args.min_pixels,
        )
    rank0_print(
        f"Loaded dict-form eval_dataset with {len(eval_dataset)} categories: "
        f"{sorted(eval_dataset)}"
    )
```

So `eval_<cat>_loss` (the per-category loss logged at every
`eval_steps=500` via the dict-form `eval_dataset`) is directly comparable
to `train_loss`:

- Same labels function
- Same `IGNORE_INDEX` masking
- Same model in eval mode (no dropout, but **LoRA dropout=0.05** applies
  only at train time, so train loss is mildly noisier than eval loss for
  that reason alone)
- Eval batches: `per_device_eval_batch_size=2`, no grad accumulation
- The per-category eval losses share the **same vision pipeline**
  (`max_pixels=1,440,208`, smart_resize) as training — no eval-only
  resolution shift

Per-category eval loss curves vs. the stage-end **generation accuracy** are
NOT measuring the same thing:

- Eval loss: average teacher-forced NLL over all assistant tokens
  (reasoning + grounding + answer)
- Stage-end acc: whether the *extracted answer field* from greedy
  generation string-equals GT

These can diverge: a model can drift on bbox formatting (raising eval loss)
while keeping answer accuracy stable, or vice versa. Worth keeping in mind
when reading the aggregator report.

### 2.6 Per-category implications

The loss is category-agnostic by construction. But the per-category
training sets differ in ways that affect what the optimizer "sees":

| Stage | Category | N samples (`train_qwen3vl_<CAT>.json`) | Loss character |
|---|---|---|---|
| 0 | OBS | 32,551 | Mostly short factual reasoning + grounding; smallest answer space → easy NLL |
| 9 | CHR | 19,446 | Hypothetical/causal — longer reasoning chains, p50 GPT chars 662 vs OBS 563 |

Two implicit biases that fall out of "raw NLL with no weighting":

1. **Categories with longer GPT outputs contribute more loss terms per
   sample.** Across a single-category stage that doesn't matter (uniform
   within stage), but it means the *gradient signal per sample* is not
   uniform across stages. Late, reasoning-heavier stages get more loss
   terms per sample, but also smaller stage sizes — these partly cancel.
2. **Imbalanced field difficulty.** The `answer` field is the
   highest-information / lowest-token portion of the target. The pipeline
   never up-weights it. If you wanted answer-accuracy-first training, you
   could either:
   - Weight tokens inside the `answer` field higher in a custom
     `compute_loss`, or
   - Mask reasoning/grounding tokens for the last K% of a stage's steps
     (curriculum-within-curriculum on output structure)

   Both are easy hooks because nothing else customizes the loss right now.

### 2.7 Gradient & optimizer wiring (loss-adjacent)

Worth flagging because they shape what the loss number actually means:

- **`gradient_accumulation_steps=4`,
  `per_device_train_batch_size=1`** → effective batch = 4 per GPU. With
  multi-GPU `NPROC_PER_NODE=N`, effective batch = 4N. The logged
  `train_loss` is averaged across the accumulation window, so it's
  directly comparable to eval loss.
- **`max_grad_norm=1.0`** clips gradients per optimizer step. With WSD
  warmup starting at lr=0, the first few steps see large parameter norms
  relative to update size — clipping prevents the warmup transient from
  blowing up loss.
- **`weight_decay=0.01`** is applied to **non-bias / non-norm** params via
  the custom `create_optimizer` in
  `qwen-vl-finetune/qwenvl/train/trainer.py:316`. The LoRA-relevant branch
  (no projector/vision-tower LRs set) takes the simple two-group path:

  ```python
  # qwen-vl-finetune/qwenvl/train/trainer.py:466-484
  else:
      optimizer_grouped_parameters = [
          {
              "params": [
                  p
                  for n, p in opt_model.named_parameters()
                  if (n in decay_parameters and p.requires_grad)
              ],
              "weight_decay": self.args.weight_decay,
          },
          {
              "params": [
                  p
                  for n, p in opt_model.named_parameters()
                  if (n not in decay_parameters and p.requires_grad)
              ],
              "weight_decay": 0.0,
          },
      ]
  ```

  For LoRA, this means decay on the `lora_A`/`lora_B` matrices (since
  they're nn.Linear weights with no "bias" in their names). Decoupled
  AdamW means this acts as an L2 regulariser *outside* the loss — it does
  NOT show up in the printed `train_loss`. If you see eval loss creep up
  but train loss flat, that's the regulariser doing its job, not a bug.
- **LoRA dropout 0.05** is on the LoRA adapters only (base model frozen).
  It's a forward-pass perturbation — small noise on train loss but doesn't
  change the loss function.
- **`use_wsd_scheduler=True`, `lr_scheduler_type="constant"`**: the WSD
  override in `WSDTrainer.create_scheduler` is what's actually active.
  Documented at `train_nuscenes_qwen3vl.py:483`.

### 2.8 What is *not* in the loss path

To make scope explicit:

- **No KL/distillation loss** — base model is fully frozen, only LoRA
  adapters train, no teacher signal.
- **No replay buffer or EWC-style regulariser** against forgetting earlier
  categories. The only forgetting safeguard is *measurement* (per-cat
  eval loss + the aggregator's forgetting column), not *prevention*.
- **No auxiliary head losses** — bbox coords are just JSON tokens. There
  is no separate IoU/regression loss on the grounding field. The model
  learns bbox formatting purely from next-token prediction over digits,
  commas, brackets.
- **No length normalization across categories** — each stage's loss is
  just an average over its tokens.
- **No special handling for hallucinated/empty grounding** — if GT
  grounding is `[]`, the loss target is the literal three tokens `[]`.
  Generating extra bboxes hurts accuracy but not necessarily eval loss
  (depending on tokenizer behavior).

### 2.9 Things to watch / consider tuning

In rough priority order:

1. **Answer-field weighting.** The answer is ~5–20 tokens out of ~150–490
   GPT tokens (CLAUDE.md §7), yet it determines stage-end accuracy. A
   simple `compute_loss` override that adds 2–3× weight on tokens inside
   the `"answer":` field would likely tighten the loss → accuracy
   correspondence.
2. **Bbox-format degradation tracking.** Track an *eval-time* metric for
   "fraction of samples with valid grounding JSON" alongside answer
   accuracy. The current loss number conflates reasoning quality,
   grounding quality, and answer correctness — the aggregator only
   reports the last.
3. **Train/eval mismatch from outlier-long samples.** With max-len 16384
   and one OBS sample at ~14877 chars (likely ~5K tokens), it might exceed
   `model_max_length` once vision tokens (5,796 at current max_pixels)
   are added. The tail-truncation silently drops loss targets at the end.
   Worth a one-line assertion or pre-filter.
4. **Loss-on-system tokens for grounding context?** Currently system +
   user are masked. The system prompt contains the JSON-output schema
   spec. The model is conditioned on it (in-context) but never penalized
   for misremembering it — which is fine, but if format drift becomes a
   problem you could either tighten the system spec or unmask a small
   number of "schema reminder" tokens.
5. **Per-category loss could be normalized by *answer-token count* not
   *all-token count* in logged metrics** so the curves are comparable
   across categories of differing reasoning verbosity. The training loss
   itself can stay as-is; this is just a logging change.
