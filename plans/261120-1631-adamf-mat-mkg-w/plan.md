# AdaMF-MAT on MKG-W

Status: proposed, not started · Branch: `feat/mkg-w-dataset` (no commits unless asked) · Report so far: `plans/reports/mkg-w-261120-1631-knowledge-graph-retrieval.html`

## Outcome
Run AdaMF-MAT (Zhang et al., COLING 2024) on MKG-W as the 2024+ method of level 2, and place it in the same table as CLIP, path score, GraphSAGE and the relation path scorer: indirect / direct R@1, R@10, MRR on the 9,301-entity gallery, per hop, 3 seeds.

## Why this differs from the VG ∩ COCO adaptation
On VG ∩ COCO the method had to be bent: one relation ("is described by"), GraphSAGE instead of per-entity structure embeddings, InfoNCE instead of RotatE, MAT on the head only. MKG-W is the paper's own benchmark, and our triples and id maps already come from the AdaMF-MAT repository (`configs/data/mkg_w.yaml`: 34,196 / 4,276 / 4,274 train / valid / test). So the method can run close to the paper:

| Part | Paper | Here |
|---|---|---|
| Entities | 15,000 MKG-W entities, all seen in training (transductive) | same |
| Structure modality | per-entity learned embedding | same |
| Image modality | pre-extracted visual feature per entity | mean CLIP ViT-B/32 vector of the entity's images (`images.npy`); 5,699 entities have no image |
| Text modality | pre-extracted description feature | CLIP text vector of the description (new embedding table); name vector when the description is missing (850 entities) |
| Fusion | AdaMF adaptive weights | same code idea as `adamf_mat.py` on `feat/reproduce-adamf-mat` (`modality_weights`, `fuse`) |
| Decoder | RotatE + self-adversarial negative sampling | same |
| MAT | synthetic head, tail, or both | same (possible here because the tail is an entity) |

## Two evaluations of one trained model
1. **Link prediction (reproduction check).** Filtered MRR, Hits@1/3/10 on the official test triples over all 15,000 entities, head and tail prediction. Compared with the paper's MKG-W row. This says whether our implementation is AdaMF-MAT or something weaker. Features differ (CLIP instead of the paper's extractors), so a small gap is expected; it is reported, not hidden.
2. **Image retrieval (our benchmark).** An indirect query carries `anchor_id`, `relation_id`, `direction`. Graph score of candidate c = RotatE score of (anchor, relation, c) or (c, relation, anchor), over the gallery, min-max normalised per query. Then the usual `α·CLIP + (1−α)·graph`, α and top-k chosen on validation, `segment_metrics` unchanged. Direct queries have no anchor: graph score 0, CLIP ranks them (same rule as the relation path scorer).

Test queries are built from test triples and the model trains on train triples only, so the asked fact is never in the training graph. Validation triples are used for early stopping only.

## What must be stated next to the numbers
- AdaMF-MAT reads `anchor_id` **and** `relation_id` from the query record. The relation path scorer reads only `anchor_id` and infers the relation from the query text. GraphSAGE reads neither. Three different input levels; the table gets a column for it.
- The paper's numbers are over 15,000 entities with link-prediction metrics; ours over the 9,301 entities with images. Not comparable to each other; evaluation 1 is the bridge.

## Runs
All CPU, seeds 13 / 37 / 97, same early-stopping rule (validation indirect R@1 for retrieval reporting; validation MRR for the link-prediction table).

| Id | Config | Purpose |
|---|---|---|
| R0 | RotatE, structure only | baseline KGC, no modalities |
| R1 | AdaMF-MAT (full) | the method |
| R2 | AdaMF w/o MAT | paper ablation |
| R3 | mean fusion + MAT (w/o AdaMF) | paper ablation |
| R4 | w/o image modality | does the image help the graph model |
| R5 | w/o text modality | same for the description |
| R6 | R1 on shuffled train triples | control, same shuffle seed as the other MKG-W controls |

R4–R6 only after R0–R3 are in. No sweep of dim / margin / negatives beyond the paper's MKG-W settings unless evaluation 1 is far off.

## Decision rules (fixed before running)
- Reproduction accepted when R1 test MRR is within 3 points (absolute, on the 0–100 scale) of the paper's MKG-W MRR and R1 ≥ R0. Otherwise debug before any retrieval claim.
- "AdaMF-MAT beats the relation path scorer" when indirect R@1 (reranked, test) exceeds 0.274 by more than the seed spread of both.
- "Modalities help" when R1 exceeds R0 on indirect R@1 by more than the seed spread; R4 / R5 say which one.

## Phases
1. **Scout the official code** (`github.com/zjukg/AdaMF-MAT`): MKG-W hyper-parameters (dim, margin, negatives, learning rates, epochs, μ), how missing modalities are filled, exact form of the three MAT replacements, the paper's MKG-W table. Record them in this plan. No code yet.
2. **Data**: embed descriptions for all 15,000 entities (extend `scripts/embed_mkg_w.py`, new table `descriptions`); per-entity image vector = mean of its image rows; presence masks for image and text.
3. **Model** `src/models/kg_adamf_mat.py`: entity encoder (three modalities → `dim`, adaptive or mean fusion, presence mask), RotatE scorer, two modality generators, discriminator / generator losses for head, tail and both. Config dataclass with the ablation switches; settings in `configs/retrieval/mkg_w.yaml` under `adamf_mat`.
4. **Training** `src/models/kg_adamf_train.py`: negative sampling with self-adversarial weights, alternating encoder / generator steps, early stopping, history per epoch. Tests: score shape, fusion weights sum to 1 and respect the mask, MAT off equals plain training, filtered ranks on a toy graph.
5. **Evaluation**: `src/evaluation/link_prediction.py` (filtered MRR / Hits@K) and `scripts/train_mkg_w_adamf_mat.py` writing `runs/mkg_w/adamf_mat/<run_name>/{seed-<n>/, benchmark.json}` in the existing report shape plus a `link_prediction` block.
6. **Smoke run** R0 and R1, seed 13: time per epoch, evaluation 1 against the decision rule. Stop here and report if the rule fails.
7. **Full runs** R0–R3, then R4–R6.
8. **Write-up**: add an AdaMF-MAT section to the report page and the journal; update `task.md` line for the 2024+ method.

## Phase 1 findings (official code and paper, read 2026-11-20)
- Released MKG-W settings (`scripts/run_mkgw.sh`, `args.py`, `run_adamf_mat.py`): dim 200 (entity 400), batch 1024, margin 12, epsilon 2, 128 negatives, adversarial temperature 2, lr 2e-5, generator lr 1e-4, 1000 epochs, noise 64, generator hidden 512. The script passes `-mu=0`; the paper says μ is searched but the grid is not readable in the HTML version.
- Model `AdvMixRotatE`: joint = softmax(a·tanh(e_m)) weighted sum over structure / image / text; image and text go through one Linear each to the entity size; no handling of missing modalities (the feature file has a row for every entity).
- MAT (`AdvMixTrainer`): generators take the structure embedding + noise and output entity-size embeddings; encoder loss adds μ × loss(true, synthetic) for synthetic head, tail and both; generator loss is the mirror, stepped with its own Adam.
- Paper, MKG-W test (MRR / Hits@1 / Hits@3 / Hits@10): AdaMF-MAT 35.85 / 29.04 / 39.01 / 48.42; AdaMF w/o MAT 34.27 / 27.21 / 37.86 / 47.21. Features there: BEiT (image), sentence-BERT (text).
- Our deviations: frozen CLIP ViT-B/32 features; zero image row for entities without images; uniform negatives alternating head / tail per batch, no Bernoulli trick, no filtering of accidental true triples; μ = 1 for the full model; learning rate and epoch budget chosen in the smoke run so that a CPU run converges (measured: 0.25 s per step, 34 steps per epoch, about 77 s per full validation, so the paper's 1000 epochs would take over 3 hours per run).
- One checkpoint per seed, picked on validation link-prediction MRR, is used for both evaluations (simpler than the two selection rules first planned).

## Cost
No agents needed; one session. Machine time is unknown until phase 6: RotatE on 34k triples is small, but full-entity filtered evaluation each epoch and the generator step add cost. Phase 6 measures seconds per epoch on CPU; if 7 configs × 3 seeds exceeds about 3 hours of wall-clock with two queues, cut R4–R6 to one seed and say so.

## Risks
- **Sparse graph** (median 3 edges per entity): RotatE-family models on MKG-W reach modest MRR even in the paper; retrieval gain over the path scorer is not guaranteed.
- **CLIP features instead of the paper's**: may move evaluation 1 either way.
- **Relation oracle**: with `relation_id` given, 1-hop-like cases become pure link prediction; the comparison with the path scorer is about input level as much as about the model.
- **Missing images** for 38 % of entities: the presence mask decides how AdaMF weights behave there; follow the official code.

## Not in scope
Text-conditioned variant (fused entity vector projected to CLIP space, InfoNCE against the query, as on VG ∩ COCO), explanation paths and fidelity for AdaMF-MAT, automatic entity linking, merging `feat/reproduce-adamf-mat` into this branch.

## Open questions
1. Use `relation_id` from the query record (paper-faithful, recommended), or predict the relation from the query text first so the input level matches the relation path scorer?
2. Text modality from descriptions (as in the paper, recommended) or names only? Descriptions of val/test entities are also direct queries, which is harmless here because direct queries get no graph score.
3. Start implementation on `feat/mkg-w-dataset` without committing, as so far?
