# Relation ablation suite A–D — results

Branch `feat/reproduce-adamf-mat` (worktree `final-project-reproduce-adamf-mat`). All AdaMF runs: no MAT, L2-normalised modalities, VG ∩ COCO test (1,500 images). "rel" = top-1 accuracy among confusers on 1,408 relation queries (chance 0.214); "con" = same on the 734 queries with a different-relation confuser (chance 0.314); "→" = after masking the queried edge. Caption R@1 is CLIP rerank unless noted. Raw outputs in `runs/analysis/` and `runs/{kg,full}/analysis/` (local, `/runs/` ignored).

## Headline (5 seeds, D2)
| Config | Caption R@1 | rel → masked | con → masked |
|---|---|---|---|
| baseline `l2` | 0.519 | 0.277 → 0.277 | 0.336 → 0.337 |
| so0.3 (30 % structure-only rows) | 0.527 | 0.300 → 0.285 | 0.350 → 0.332 |
| sl0.1 (structure-only loss 0.1) | **0.531** | 0.300 → 0.285 | 0.349 → 0.331 |
| B5 so0.3 + sl0.1 | 0.529 | 0.300 → 0.284 | 0.349 → 0.330 |

D1 paired bootstrap vs baseline (pp, 95 % CI): rel +2.3 [1.2, 3.4] for all three; con +1.3–1.4 with CI crossing 0 (e.g. so0.3 +1.42 [−0.14, 3.08]). Masking drop on con: baseline −0.05 [−0.14, 0]; so0.3 +1.83 [1.28, 2.43]; sl0.1 +1.74; B5 +1.93.
→ Edge use is established; better *A-vs-B relation discrimination* is not (the 3-seed CI [0.3, 4.0] did not survive 5 seeds).

## B — training interventions (3 seeds)
| Item | Config | Caption R@1 | rel → masked | Note |
|---|---|---|---|---|
| B2 | so0.1 / 0.3 / 0.5 / 0.7 | .529 / .529 / .524 / .519 | .293 / .302 / .308 / .305, all drop ≥ .011 | 10 % is enough; ≥ 50 % trades caption for relation |
| B3 | sl0.1 / 0.3 / 1 | .532 / .529 / .526 | ≈ .297, drop ≈ .015 | best caption R@1 of the suite |
| B1 | selection on recall@1+relation | .516 (base), .524 (so0.3) | no change for base | detector works: baseline edge_sensitivity ≈ 0 every epoch |
| B4 | predicate swap 1 / 0.1 / so0.3+1 | .485 / .523 / .503 | no edge use | weight 1 destabilises (early stop at epoch 7–8); 0.1 inert |
| — | hn4 (same-pair batches) | .518 | no edge use | CLIP still separates the images |

## C — data and graph
| Item | Result |
|---|---|
| C1 train fraction 25/50/100 % | baseline never reads edges; so0.3 reads them at every size, rel .282 → .295 → .302; caption gain of so0.3 over baseline +0.7 / +1.2 / +0.8 |
| C2 full train set (45,817 images, same val/test, base vocabulary) | baseline caption .558 (+3.7 vs 7k) and now reads edges (rel .306 → .294, con .348 → .336); so0.3 .560, rel .315 → .295 — its caption edge over baseline vanishes |
| C3 WordNet IsA edges (+963 hypernyms, ~33 IsA triplets/image), 1,387 indirect queries | indirect R@1 (graph only): CLIP .038, base .053, KG .050; masking IsA drops KG to .041 (used) but no gain over base; caption −0.6 (base) / ±0 (so0.3) |
| C4 noise stand-in (no SGG on CPU) | mild detector-like noise: baseline .505 vs so0.3 .501 vs sl0.1 .503; heavy: .481 vs .455 vs .456 — edge-reading models are less robust |
| C5 caption ↔ triplet matching | train captions naming a triplet: exact 17.9 %, + WordNet synonyms 19.3 %, + hypernyms 22.0 % |

## Conclusions for the report
1. On 7k training images the baseline ignores edges (shortcut through the CLIP image vector). Structure-only dropout or a structure-only loss makes the model use them (significant masking drop) and adds ~1 pt caption R@1.
2. With ~6.5× more training data the baseline starts using edges on its own and caption R@1 rises 3.7 pt; the intervention's caption gain disappears. Earlier claim "more data will not remove the shortcut" is wrong.
3. Distinguishing relation A from B (con) is not significantly better than baseline for any intervention at 5 seeds.
4. Reading edges costs robustness when graphs are noisy (C4).
5. WordNet IsA edges add nothing over CLIP label text for hypernym queries (C3).

## Unresolved
- B4 at weights between 0.1 and 1, or applied to the structure-only vector, untested.
- C2 with so0.3/sl0.1 at 5 seeds and relation validation, and C2 + noise, untested.
- C4 uses simulated noise; real scene-graph generator not run (no GPU).
