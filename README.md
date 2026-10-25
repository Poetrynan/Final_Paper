# Cross-Modal Consistency Is Not a Cross-Generator Invariant

An attributed **negative-result** study of lightweight, frozen-backbone audio–visual deepfake
detection. This repository contains the code, raw evaluation logs, derived result tables and
figures that back every number in the accompanying thesis.

**Author:** Chris Feng (Monash University, FIT5128/FIT4444/FIT4448) · **Supervisor:** Asad Malik · **Year:** 2026

---

## What the study asks

Audio–visual consistency is widely proposed as the transferable cue for deepfake detection.
We test that hypothesis under a deliberately constrained regime: two **frozen, separately
pre-trained** backbones (CLIP ViT-B/16 for faces, wav2vec2-base-960h for audio) with light
adapters trained on a **single** supervised corpus (DeepSpeak v1.1). The result is negative,
and the paper attributes it precisely rather than reporting a leaderboard number.

## Headline numbers (single seed unless stated)

| Evaluation set | Role | AUC |
|---|---|---|
| DeepSpeak v1.1 (in-domain) | training corpus | 0.9758 |
| FakeAVCeleb (19,725 clips; 178 real / 19,547 fake) | zero-shot cross-domain | 0.6187 |
| Deepfake-Eval-2024 (728 clips: 393 real / 335 fake) | zero-shot in-the-wild | 0.5350 |
| HiFi-AVDF (3,787 clips: 1,891 / 1,896) | zero-shot, 6 commercial generators | 0.4735 |

Module ablation (single seed, identical features/caches):

| Variant | In-domain | FakeAV | DFE-2024 | HiFi-AVDF |
|---|---:|---:|---:|---:|
| Principal (full) | 0.9758 | 0.6187 | 0.5350 | 0.4735 |
| w/o LCMA | **0.7457** | **0.4909** | **0.4757** | **0.4570** |
| w/o DWF | 0.9827 | 0.6287 | 0.5145 | 0.4777 |
| LACA-corr | 0.9839 | 0.6194 | 0.5171 | 0.4870 |
| LACA-biomech | 0.9845 | 0.6229 | 0.5207 | 0.4784 |

Principal configuration, three seeds (per-seed AUC; mean ± std):

| Seed | FakeAVCeleb | DFE-2024 | HiFi-AVDF |
|---|---:|---:|---:|
| 0 | 0.6187 | 0.5350 | 0.4735 |
| 1 | 0.5816 | 0.5279 | 0.4873 |
| 2 | 0.6214 | 0.5007 | 0.4892 |
| mean ± std | 0.6072 ± 0.0222 | 0.5212 ± 0.0181 | 0.4833 ± 0.0086 |

Carrier probe (linear CKA / RV coefficient on cached features, per generator): verdict
**NOT_SEPARABLE** — only veo3.1 reaches strength ≥ 0.6 and it is sign-reversed
(226–394 usable clips per generator; raising the per-class cap fivefold reproduces the same
statistics).

## Repository layout

```
code/            training, evaluation, preprocessing, probes, ablation driver
  ├─ models.py           Sync-X adapter (LCMA / LACA / LTCA / DWF / AAM)
  ├─ train_stage1.py     Stage-1 self-supervised pre-training (VoxCeleb2)
  ├─ train_stage2.py     Stage-2 supervised training (DeepSpeak v1.1)
  ├─ evaluate.py         zero-shot cross-domain evaluation -> jsonl
  ├─ run_module_ablation.py / train_laca.py     module-ablation tooling
  ├─ run_carrier_probe_expanded.py / diag_*.py  evidence-chain probes
  └─ prep_*.py           dataset preprocessing into per-clip .pt caches
results/
  ├─ evidence_chain_crossdomain.md            engineering evidence chain (§0–§18)
  ├─ module_ablation_and_probe_results_20261024.md
  └─ logs/<exp_id>/eval/<dataset>__<variant>.jsonl   raw evaluation records
figures/         the five figures used in the paper (PNG + PDF)
paper/           ResearchPaper.pdf (Part 2) and Appendices_Part3.pdf
docs/            dataset audit and caveat notes
```

## Reproducing

Environment: PyTorch 2.8.0+cu128 with torchvision 0.23.0+cu128, one 24 GB GPU, `ffmpeg`.
All reported runs use `SYNCX_COMPILE=0` (torch.compile is disabled; it changes speed only).

```bash
# 1. preprocess each corpus into per-clip feature caches (.pt)
python code/prep_deepspeak.py --root <DeepSpeak_v1_1>/clips --out <caches>/deepspeak
python code/prep_fakeavceleb.py --root <FakeAVCeleb_v1.2> --out <caches>/fakeavceleb
python code/prep_eval2024.py    --root <Deepfake-Eval-2024> --out <caches>/dfe2024
python code/prep_hifi.py        --root <HiFi-AVDF>/prep --out <caches>/hifi

# 2. Stage-1 (VoxCeleb2, authentic clips only) then Stage-2 (DeepSpeak v1.1)
python code/train_stage1.py --real_cache <voxceleb2_cache> --val_cache <voxceleb2_val> \
    --out_ckpt ckpt/stage1.pt
python code/train_stage2.py --train_cache <deepspeak_train> --val_cache <deepspeak_val> \
    --stage1_ckpt ckpt/stage1.pt --out_ckpt ckpt/syncx_best_seed0.pt

# 3. zero-shot evaluation (one jsonl per dataset and variant)
python code/evaluate.py --ckpt ckpt/syncx_best_seed0.pt --eval_cache <eval_cache>

# 4. module ablation and expanded carrier probe
python code/run_module_ablation.py --stage all --seeds 0
python code/run_carrier_probe_expanded.py --max_per_real 3000 --max_per_fake 3000
```

> The exact cache layout, `.pt` keys (`audio_seq`, `v_clip`, `v_lip`, `label`, `identity`,
> `source_id`) and jsonl schema are documented in `paper/Appendices_Part3.pdf` (Appendices B–D).

## Data availability

Raw corpora are **third-party** and are not redistributed here. They are used under their own
licences: DeepSpeak v1.1 (Barrington et al., 2024), VoxCeleb2 (Chung et al., 2018),
FakeAVCeleb v1.2 (Khalid et al., 2021), Deepfake-Eval-2024 (Chandra et al., 2025),
HiFi-AVDF (Peng et al., 2026). What this repository provides instead is everything derived:
feature-cache specifications, evaluation logs, probe statistics and figures.

## Scope and limitations

Single training source; frozen separate backbones that are never co-trained; clip-level AUC on
three cross-domain sets; external state-of-the-art recipes (HAVIC, X-AVDT) are **not**
reproduced and no head-to-head SOTA comparison is claimed — those systems co-train joint
backbones and pre-train on multi-source data, which this regime cannot do. `abl_wo_lcma_seed0`
has evaluation logs but no per-epoch training curve (it was not archived on the cluster).

## Citation

```bibtex
@misc{feng2026syncx,
  author = {Chris Feng},
  title  = {Cross-Modal Consistency Is Not a Cross-Generator Invariant:
            An Attributed Negative-Result Study of Lightweight Frozen-Backbone
            Deepfake Detection},
  year   = {2026},
  note   = {Monash University FIT5128/FIT4444/FIT4448 thesis},
  url    = {https://github.com/Poetrynan/Final_Paper}
}
```

## Licence

Code: MIT (see `LICENSE`). Text, tables and figures: CC BY 4.0. Third-party datasets retain
their original licences. Defensive research use only.
