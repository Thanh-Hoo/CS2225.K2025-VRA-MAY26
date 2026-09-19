# IDEA — Test Results Report

Document generated from all `test_log` files found in the project:

| Log file | Dataset | Run time |
|---|---|---|
| `IDEA_RGBNT201/test_full_log_20260727_2225.txt` | RGBNT201 (person) | 2026-07-27 22:25–22:26 |
| `IDEA_RGBNT201/test_log.txt` | RGBNT201 (video visualization) | 2026-07-27 23:00 |
| `IDEA_MSVR310/test_full_log_20260727_2228.txt` | MSVR310 (vehicle) | 2026-07-27 22:28–22:29 |
| `IDEA_MSVR310/test_log.txt` | MSVR310 (video visualization) | 2026-07-27 23:00 |

---

## 1. Headline results (best checkpoint, `IDEAbest.pth`)

Main protocol = fused **`LOCAL`** feature, all modalities present (`TEST.MISS: nothing`).

| Dataset | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|---|
| **RGBNT201** | **60.1 %** | **61.7 %** | 69.1 % | 76.6 % |
| **MSVR310** | **41.7 %** | **53.5 %** | 72.4 % | 77.3 % |

---

## 2. Experimental setup

### Common
- Model: `IDEA`, backbone `ViT-B-16` (CLIP pre-trained, `PTH/ViT-B-16.pt`), stride `[16, 16]`
- IMFE: `PREFIX: True` (modality prefixes), `INVERSE: True` (InverseNet)
- CDA: `DA: True`, `DA_SHARE: True`
- Losses: softmax ID loss (weight 0.25, label smoothing on) + triplet (weight 1.0), BNNeck, `NECK_FEAT: before`
- Optimizer: Adam, `BASE_LR 0.001`, steps `(40, 70)`, `MAX_EPOCHS 200`, batch 32, seed 1111
- Test: `FEAT_NORM: yes`, `RE_RANKING: no`, `MISS: nothing`, distance = euclidean
- Checkpoint: `IDEAbest.pth` (all keys matched successfully on load)

### Per-dataset differences

| Setting | RGBNT201 | MSVR310 |
|---|---|---|
| Config | `configs/RGBNT201/IDEA.yml` | `configs/MSVR310/IDEA.yml` |
| Input size (H×W) | 256 × 128 | 128 × 256 |
| `TEXT_PROMPT` | 2 | 1 |
| `OFF_FAC` (offset magnitude) | 5.0 | 25.0 |
| Flip prob. / erase prob. | 0.3 / 0.5 | 0.5 / 0.5 |
| Warmup iters / eval period | 10 / 1 | 5 / 2 |
| Test batch size | 4 | 128 |
| Cameras | 4 | 8 |

### Dataset statistics

**RGBNT201**

| Subset | # IDs | # Images | # Cameras |
|---|---|---|---|
| train | 171 | 3951 | 4 |
| query | 30 | 836 | 2 |
| gallery | 30 | 836 | 2 |

**MSVR310** (`RGB_IR` loader)

| Subset | # IDs | # Images | # Cameras |
|---|---|---|---|
| train | 155 | 1032 | 8 |
| query | 52 | 591 | 8 |
| gallery | 155 | 1055 | 8 |

---

## 3. RGBNT201 — full search-pattern breakdown

### 3.1 Local feature testing

| # | Query ⇒ Gallery pattern | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|---|---|
| 1 | `T_RGB` | 27.0 | 23.2 | 35.4 | 45.0 |
| 2 | `T_NIR` | 19.7 | 15.8 | 28.7 | 35.9 |
| 3 | `T_TIR` | 25.1 | 23.4 | 36.2 | 45.8 |
| 4 | `T_RGB + T_NIR` | 44.5 | 42.7 | 56.9 | 64.4 |
| 5 | `T_RGB + T_TIR` | 50.9 | 49.5 | 62.2 | 69.7 |
| 6 | `T_NIR + T_TIR` | 37.2 | 35.8 | 51.4 | 60.8 |
| 7 | `T_RGB + T_NIR + T_TIR` | 60.0 | 61.4 | 68.2 | 76.1 |
| 8 | All V + all T + `LOCAL` | 60.0 | 61.8 | 68.5 | 76.3 |
| 9 | **`LOCAL`** | **60.1** | **61.7** | **69.1** | **76.6** |
| 10 | `LOCAL_v` (visual only) | 59.7 | 61.5 | 69.9 | 76.3 |
| 11 | `LOCAL_t` (text only) | 60.2 | 61.7 | 68.2 | 76.4 |

### 3.2 Combined feature testing

| Query ⇒ Gallery pattern | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|---|
| `V_RGB + V_NIR + V_TIR + T_RGB + T_NIR + T_TIR` | 59.8 | 61.5 | 68.1 | 75.8 |
| `V_RGB + V_NIR + V_TIR` | 59.4 | 61.0 | 69.3 | 76.0 |
| `T_RGB + T_NIR + T_TIR` | 60.0 | 61.4 | 68.2 | 76.1 |

### 3.3 Observations
- Single modality is weak (19.7–27.0 mAP); RGB and TIR are the strongest individually, NIR the weakest.
- Every pairwise combination beats its members; `RGB+TIR` (50.9) is the best pair, `NIR+TIR` (37.2) the worst — RGB carries most of the discriminative signal.
- Going from best pair to all three modalities adds **+9.1 mAP** (50.9 → 60.0), confirming the complementarity the fusion module is meant to exploit.
- The aggregated `LOCAL` feature (60.1) is the best single descriptor and matches the much larger concatenated pattern (60.0) — the CDA aggregation loses nothing while being far more compact.
- `LOCAL_v` (59.7) vs `LOCAL_t` (60.2): the text branch alone is marginally ahead of the visual branch, and combining them (`LOCAL`, 60.1) sits between them — the two branches are largely redundant at this checkpoint.

---

## 4. MSVR310 — full search-pattern breakdown

### 4.1 Local feature testing

| # | Query ⇒ Gallery pattern | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|---|---|
| 1 | `T_RGB` | 32.8 | 44.2 | 64.6 | 71.1 |
| 2 | `T_NIR` | 27.4 | 41.5 | 63.3 | 73.8 |
| 3 | `T_TIR` | 10.7 | 18.3 | 34.9 | 43.7 |
| 4 | `T_RGB + T_NIR` | 39.3 | 51.6 | 70.4 | 78.5 |
| 5 | `T_RGB + T_TIR` | 34.4 | 43.7 | 67.2 | 74.1 |
| 6 | `T_NIR + T_TIR` | 29.6 | 39.4 | 64.8 | 73.1 |
| 7 | `T_RGB + T_NIR + T_TIR` | 41.1 | 52.5 | 72.6 | 77.8 |
| 8 | All V + all T + `LOCAL` | 41.2 | 52.6 | 72.8 | 77.3 |
| 9 | **`LOCAL`** | **41.7** | **53.5** | 72.4 | 77.3 |
| 10 | `LOCAL_v` (visual only) | 40.8 | 52.1 | 71.1 | 77.3 |
| 11 | `LOCAL_t` (text only) | 41.9 | 53.3 | 72.6 | 77.8 |

### 4.2 Combined feature testing

| Query ⇒ Gallery pattern | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|---|
| `V_RGB + V_NIR + V_TIR + T_RGB + T_NIR + T_TIR` | 40.7 | 52.3 | 72.6 | 77.3 |
| `V_RGB + V_NIR + V_TIR` | 39.8 | 51.3 | 70.9 | 76.5 |
| `T_RGB + T_NIR + T_TIR` | 41.1 | 52.5 | 72.6 | 77.8 |

### 4.3 Observations
- TIR is dramatically weaker on vehicles (10.7 mAP) than on persons — thermal gives little identity signal for vehicles, and adding it to RGB (`RGB+TIR`, 34.4) barely improves over RGB alone (32.8).
- NIR is the useful complement here: `RGB+NIR` = 39.3 mAP, **+6.5** over RGB alone.
- Full three-modality fusion still gives the best result (41.1 → 41.7 with `LOCAL`), so even the weak TIR stream contributes once aggregated.
- The mAP/Rank-1 gap is much larger than on RGBNT201 (41.7 vs 53.5): the model finds a correct match early but ranks the remaining positives poorly, consistent with MSVR310's larger gallery (155 IDs) vs query (52 IDs) and its multi-view vehicle setting.
- As on RGBNT201, `LOCAL_t` ≈ `LOCAL` > `LOCAL_v` — the text-enhanced branch is at least as strong as pure visual.

---

## 5. Cross-dataset comparison (main `LOCAL` protocol)

| Metric | RGBNT201 | MSVR310 | Δ |
|---|---|---|---|
| mAP | 60.1 | 41.7 | −18.4 |
| Rank-1 | 61.7 | 53.5 | −8.2 |
| Rank-5 | 69.1 | 72.4 | +3.3 |
| Rank-10 | 76.6 | 77.3 | +0.7 |

MSVR310 is the harder benchmark by mAP but reaches comparable Rank-5/Rank-10 — the ranking quality degrades in the tail rather than at the top.

Modality contribution (mAP, three-modality fusion minus best single modality):
- RGBNT201: 60.0 − 27.0 = **+33.0**
- MSVR310: 41.1 − 32.8 = **+8.3**

Multi-modal fusion pays off far more on the person benchmark.

---

## 6. Retrieval video visualization

Both `test_log.txt` files record a visualization run using the `LOCAL` feature only:

| Dataset | Feature | mAP | Rank-1 | Output |
|---|---|---|---|---|
| RGBNT201 | `LOCAL` | 60.1 % | 61.7 % | `./IDEA_RGBNT201/rank_video_RGBNT201.mp4` |
| MSVR310 | `LOCAL` | 41.7 % | 53.5 % | `./IDEA_MSVR310/rank_video_MSVR310.mp4` |

These numbers match the `LOCAL` rows of the full evaluations above, confirming the visualization uses the same features and distance metric as the main test.

---

## 7. Notes and warnings from the logs

- **Checkpoint loading**: CLIP pre-training reported `missing_keys=['text_prompt', 'visual.new_positional_embedding']` — expected, these are IDEA-specific parameters initialized fresh. The final IDEA checkpoint loaded with `<All keys matched successfully>`.
- **Position embedding resize**: `[197, 768] → [129, 768]` (RGBNT201: 16×8 grid; MSVR310: 8×16 grid), matching the respective input sizes.
- **DataLoader warning** (both runs): `NUM_WORKERS: 14` exceeds the suggested maximum of 12 for this machine. Harmless for correctness; lower to ≤12 to avoid potential slowdowns.
- **Deprecated `addmm_` overload** (MSVR310 run, `utils/metrics.py:317`): `distmat.addmm_(1, -2, qf, gf.t())` uses the deprecated positional signature. Should be `distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)`. No effect on results, but will break on a future PyTorch release.
- **Config placeholder**: `PRETRAIN_PATH_T` is still `/path/to/your/vitb_16_224_21k.pth` in both configs; the actually loaded backbone is `PTH/ViT-B-16.pt` (CLIP), so this field is unused in this run.
- **Runtime**: feature extraction dominated the runs — ~21 s (RGBNT201) and ~22 s (MSVR310); all distance-matrix evaluations together took under 1 s / ~8 s respectively.
