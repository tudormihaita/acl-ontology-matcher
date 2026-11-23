# Final configuration (Phase 5)

Task: Bio-ML OMIM → ORDO equivalence, unsupervised (no reference mappings used for training).
Validation = Bio-ML `train.tsv` (1,116 refs, never trained on); test = Bio-ML `test.tsv` (2,605 refs)
with the official 100-negative candidate lists. **Every choice below was made on validation F1**
(ties: validation full-retrieval MRR); test numbers are reported only.

Config file: `configs/final.yaml`. All rows: `results/summary.csv`.

## Decision chain

| Step | Compared (validation F1) | Chosen | Note |
|---|---|---|---|
| E1 positives | `syn` 0.6310 · `syn_lex` 0.6308 | `syn` | tie; broken by val MRR (0.799 vs 0.794) |
| E2 hard negatives | none 0.6310 · sibling 0.6265 · **sibling_filtered 0.6326** · hierarchy 0.6219 | `sibling_filtered` | combined sibling+hierarchy run skipped: hierarchy did not beat "none" |
| E3a temperature | 0.02: 0.6248 · **0.05: 0.6326** · 0.1: 0.6232 | τ = 0.05 | τ = 0.02 had the best *test* F1 (0.685) but is not chosen: validation decides |
| E4 class repr. | **max 0.6329** · mean 0.4269 | `max` | averaging an abbreviation with a full name blurs both |
| E5 scoring × selection | cosine: argmax 0.633 · mutual 0.662 · stable 0.661; CSLS: argmax 0.650 · **mutual 0.671** · stable 0.669 | CSLS + mutual | largest gain after training itself |
| E5b neighbor λ | **0: 0.671** · 0.1: 0.665 · 0.2: 0.648 · 0.3: 0.639 | λ = 0 | parent similarity hurts |
| Pre-declared lexicon check | final 0.6709 · final + lex 0.6685 | `final` | lexicon neutral to slightly negative, in both comparisons |

**Final:** positives `syn`, one `sibling_filtered` hard negative per row, InfoNCE with τ = 0.05,
batch 128, 1 epoch, lr 2e-5, MiniLM-L6-v2; matching `max` over label pairs, CSLS (k = 10),
mutual nearest neighbors, no neighbor re-scoring, θ chosen on validation (≈ 0.31–0.33 in CSLS units).

## Main results (test)

Final: mean ± std over seeds 42, 43, 44. 95% CI: bootstrap (1,000 resamples) for seed 42.

| | Precision | Recall | F1 | Local MRR | Local Hits@1 | Full MRR | Full R@1 | Full R@10 |
|---|---|---|---|---|---|---|---|---|
| String match (B) | 0.891 | 0.415 | 0.567 | 0.816 | 0.770 | 0.756 | 0.716 | 0.830 |
| Pre-trained, cosine + argmax (E0) | 0.816 | 0.493 | 0.615 | 0.856 | 0.811 | 0.792 | 0.747 | 0.871 |
| Pre-trained, CSLS + mutual (E0) | 0.688 | 0.656 | 0.672 | 0.869 | 0.823 | 0.805 | 0.759 | 0.886 |
| **Final (3 seeds)** | **0.845 ± 0.009** | **0.636 ± 0.016** | **0.726 ± 0.007** | **0.870 ± 0.000** | **0.834 ± 0.000** | **0.811 ± 0.001** | **0.775 ± 0.001** | 0.867 ± 0.001 |

- Seed 42: F1 0.733, 95% CI 0.720–0.748; local MRR 95% CI in `summary.csv`.
- Embedding space (final, mean): alignment 0.457 (pre-trained 0.547; lower = gold pairs closer),
  uniformity −2.93 (pre-trained −3.24; slightly less uniform).
- Cost per run on a Mac mini M4 (MPS): training ≈ 160 s, encoding 37.7k labels ≈ 10 s, matching ≈ 4 s.

## Reading the results

1. **Contrastive training mainly improves precision at a threshold, not ranking.** With the same
   matching (CSLS + mutual), training lifts validation F1 from 0.561 to 0.669 and test F1 from
   0.672 to 0.726 (precision 0.688 → 0.845 at similar recall), but full-retrieval MRR only moves
   0.805 → 0.811. The pre-trained MiniLM already puts the gold target first for about 3 in 4
   sources; training makes true matches score clearly above non-matches, so one θ separates them.
2. **The matching step matters as much as the training.** CSLS + mutual adds about +0.04 F1 on top
   of the trained model by removing "hub" targets and one-sided matches. This is the
   DeepAlignment-style insight from RR1: distances alone are not alignments.
3. **What did not help:** lexicon positives (neutral), parent/child negatives and neighbor re-scoring
   (both hurt), mean class vectors (much worse). Sibling-type negatives give at most a small gain;
   differences of about 0.01 F1 are within the confidence intervals.
4. **Against string matching:** +0.16 F1 (0.567 → 0.726), from recall (0.415 → 0.636) at a moderate
   precision cost (0.891 → 0.845).

## Not run (optional "should"/"could" items)

E2b model-mined negatives, E3b batch size, E3c SupCon, E6 PubMedBERT. They can be added later
without changing the core table.
