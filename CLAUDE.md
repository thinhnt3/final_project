# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A course project (brief: `references/Project03_Multimodal_XAI (2).pdf`) for text-to-image retrieval on the Visual Genome ∩ COCO subset: 10,000 images, and COCO captions as queries, each with exactly one relevant image. Frozen CLIP ViT-B/32 retrieves the top-k candidates. A GraphSAGE encoder turns each image's scene graph into a vector in CLIP space. The candidates are then reranked by `final = α·cos_CLIP + (1−α)·cos(query, graph_vector)`. Each result is explained by removing one relation at a time and measuring how much the graph score drops.

Docs, plans and journals are written in Vietnamese. Code, identifiers, log messages and comments are in English. `docs/README.md` is the walkthrough for human readers, covering the project and its current results. It links `docs/01`–`05`, and two appendices: `phu-luc-a-*` explains the background with formulas and worked numbers, and `phu-luc-b-*` is a hand calculation through the whole pipeline. When results change, update the numbers in `docs/04-ket-qua.md` and regenerate the figures with `python -m scripts.make_doc_figures`.

## Where the project state lives

- `plans/261117-0037-multimodal-graph-retrieval-roadmap/` is the live roadmap. `plan.md` lists the phases. Each `phase-*.md` has numbered steps, "Xong khi" (done-when) criteria and a "Tiến độ" (progress) table with evidence. Update that table when you finish a step.
- `docs/graphsage-design-decisions.md` records model decisions G1–G10 and the reasons for them. Its header status line was written before the code. Data and graph decisions D1–D7 are in `phase-02-*.md` and `plans/journals/`.
- `agents/AGENTS.md` and `plan-task/` come from an earlier 40-image pilot whose code was deleted. Their file paths and statuses are stale (for example, `src/retrieval/pipeline.py` and `configs/project_scope.yaml` no longer exist). Their guardrails still apply; see Invariants below.
- `runs.zip` at the root archives older experiments (`pilot_v2`, `coco2017_level2`, `vietnam_mmkg`, ...). The current scripts did not produce them. Root-level `*.zip` result archives are gitignored because they are too large to commit.

## Commands

Requires Python 3.12+ (the code uses PEP 695 generics). Run everything from the repo root, and run scripts as modules: config paths are relative to the root, and `src` and `scripts` are imported as packages.

```bash
pip install -r requirements.txt
pytest                                   # full suite: offline, no model download, no real data
pytest tests/retrieval/test_rerank.py::test_alpha_one_keeps_the_clip_ranking_for_any_k   # one test
pytest tests/models -k resume            # by keyword
ruff check .                             # lint; settings in pyproject.toml (line length 120)
mypy                                     # type-checks src/ and scripts/ (disallow_untyped_defs)
```

The experiment pipeline runs in this order. Each step reads the previous step's output from disk:

```bash
python -m scripts.embed_vg_coco                      # frozen CLIP -> data/processed/vg_coco/embeddings/<model>/
python -m scripts.run_benchmark --method clip        # also graph_lexical, graph_lexical_objects
python -m scripts.train_graphsage                    # every seed in the config; --seed N, --resume | --overwrite, --image-node clip_image
python -m scripts.run_benchmark --method graphsage   # --run <run-name> scores a run other than the config's architecture
python -m scripts.explain_retrieval --sample 5       # or --query-id coco-<id>
python -m scripts.evaluate_fidelity --output <new.json>   # --run, --seed, --split val|test, --sample 10, --budget 3
python -m scripts.make_doc_figures                   # redraw docs/images/*.svg from runs/ and data/processed/
```

`evaluate_fidelity` needs a finished `train_graphsage` run and `run_benchmark --method graphsage` for that run. It reuses the checkpoint and the α and k selected on validation, verifies the input hashes, and never overwrites `--output`.

- These are gitignored and not in the repo: the 10k images (1.4 GB), the CLIP embeddings, the checkpoints (`*.pt`), `graph_vectors.*`, and the data-build scripts `scripts/build_vg_coco.py` and `scripts/review_vg_coco.py`. The processed JSON is committed: `data/processed/vg_coco/*.json`, `graph/*.json` and `data/splits/*`. `run_benchmark --method graphsage` and `explain_retrieval` read `graph_vectors` and `best.pt`, so they need a local `train_graphsage` run first.
- `train_graphsage` reports to Weights & Biases, because `configs/retrieval/vg_coco.yaml` sets `wandb.mode: online`. Set it to `offline` or `disabled` to run without a W&B login. `WandbTracker` deliberately turns off W&B's host, git, hardware and console metadata, so leave those settings as they are.

## Architecture

Every stage writes files that the next stage reads, so each stage can be rerun on its own.

1. **`src/data/`** (together with the gitignored build script) builds the dataset:
   - It downloads the raw dumps and records their sha256 in `data/raw/MANIFEST.json`.
   - It joins VG with the Karpathy COCO captions.
   - It normalizes objects to WordNet synsets. Predicates are only lowercased.
   - It makes a stratified sklearn split. Test images come only from Karpathy test.
   - `graph_build.py` builds per-image graphs over a vocabulary taken from train images only, written to `data/processed/vg_coco/graph/{concepts,predicates,image_graphs}.json`.
   - The parameters go to `data/splits/vg_coco_graph.yaml`. That file is generated, so don't edit it by hand.
2. **`src/embeddings/`** holds `ClipEncoder` (frozen) and `EmbeddingTable`. A table is float32 unit rows plus ids in the same order, saved as `<name>.npy` and `<name>.ids.json`. `meta.json` records the sha256 of each input, and `verify_inputs_unchanged` makes every later script refuse stale embeddings.
3. **`src/models/`**:
   - `graph_data.py` builds one PyG `Data` per image:
     - Node 0 is the image, then one node per concept, then one *relation node* per triplet that carries the predicate. The relation node exists because SAGEConv cannot read edge labels.
     - Edges run image→object and subject→relation→object.
     - `feature_row` is each node's row in the frozen table of its kind. It is deliberately not named `*index*`: PyG batching shifts every attribute with "index" in its name.
   - `graphsage.py` holds `GraphImageEncoder`:
     - A directed layer is one SAGEConv along the edges plus one against them, summed. `directed: false` uses a single SAGEConv on symmetrized edges, for ablation.
     - ReLU sits between layers but not after the last one, because CLIP vectors have negative components.
     - The image node's state goes through a linear projection and L2 normalization.
     - `image_node: learned` is the main model: the graph vector comes from the graph alone. `clip_image` is the variant: its graph score already contains CLIP image information.
     - The frozen lookup tables are non-persistent buffers, so checkpoints hold only the trained weights.
   - `train.py` and `checkpoint.py` handle training:
     - The loss is symmetric in-batch InfoNCE against frozen CLIP caption vectors. The best epoch is picked by graph-only R@1 on validation.
     - Batches depend only on (seed, epoch), so `--resume` reproduces an uninterrupted run.
     - `last.pt` is written every epoch and `best.pt` on each improvement, both atomically through a temp file and rename.
     - A non-finite loss or gradient raises `TrainingError` before the weights are updated.
     - Resume is refused if any setting changed, except `max_epochs`, `patience` and `seeds`.
   - `explain.py` removes each triplet (keeping its objects), re-encodes the graph and ranks the relations by how much the score drops. `score_graph_variants` scores several variants of one image graph against one fixed query; both explanation and fidelity use it.
   - `ablations.py` holds pure graph perturbations on the same schema:
     - `remove_relations` removes triplets by position.
     - `objects_only` is ablation A2.
     - `rewire_relations` is ablation A4. It swaps relation targets while preserving each object's directed degree and the predicate counts. It rejects swaps that would create self-loops or duplicates, and reports how many swaps succeeded.
     - No script trains or scores the ablated graphs yet.
4. **`src/evaluation/`**:
   - `build_split_data` builds each split's gallery from that split's images only.
   - `ranks_of_relevant` breaks ties against the relevant image.
   - `aggregate_runs` gives the mean and the sample standard deviation.
   - `fidelity.py` measures relation fidelity with a fixed checkpoint:
     - It removes up to `budget` (capped at half the relations) of the relations whose leave-one-out drop is positive, all together.
     - It compares that against an equally sized, disjoint random control.
     - Drops are scaled by (1 − α), and set to 0 if the image is not a top-k candidate.
     - It also reports sparsity, plus a status: `controlled`, `insufficient_relations` or `no_supporting_relations`.
5. **`src/retrieval/`**:
   - `rerank.py` does the fusion and the α/k sweep. `clip_candidate_mask` is shared with the fidelity script, so both use the same top-k candidate set.
   - `lexical.py` is the B4 baseline: IDF-weighted word overlap, with IDF from train graphs only.
   - `config.py` holds `RetrievalConfig`.

Results are written here:

- `runs/benchmark/<method>.json` holds validation and test metrics, the selected α and k, and the input sha256.
- `runs/graphsage/<run>/seed<n>/` holds a run. `config.yaml`, `history.json` and `result.json` are committed.
- `runs/explanations/` holds the explanation output.
- `evaluate_fidelity` writes to the path given by `--output`. No fidelity or ablation results are committed yet.

Run names come from `GraphSageConfig.run_name()`: `<directed|undirected>-<learned|clip_image>-l<layers>-h<hidden>`.

## Invariants (enforced by the code and tests; keep them)

- Nothing is ever selected on test. α, k, the best epoch, hyperparameters, the vocabulary, IDF and thresholds all come from train or validation. `sweep_alpha_and_k` raises unless it is given the validation split. Test is scored once, with the setting selected on validation.
- Reranking rules:
  - Both score matrices must lie in [-1, 1].
  - Fusion runs in float64, because float32 minus the non-candidate offset creates ties.
  - α = 1 must reproduce the CLIP ranks exactly.
  - On a tie, the setting closest to plain CLIP wins: higher α first, then smaller k.
- Every method uses the same split, gallery and candidate pool. Trained methods report the mean ± sample std over seeds 13, 37 and 97. Deterministic methods report a single run; don't invent extra seeds for them.
- Pure functions in `src/` do no file I/O. I/O happens in the load/save helpers and in `scripts/`.
- Don't hard-code paths or thresholds. Settings live in `configs/data/vg_coco.yaml` and `configs/retrieval/vg_coco.yaml` and are parsed into frozen dataclasses that validate on load (`DataConfig.from_yaml`, `RetrievalConfig.from_yaml`).
- Raise a subclass from `src.core.exceptions` (`ConfigError`, `DataError`, `TrainingError`, ...) where the problem is detected. Scripts catch `AppError` once in `main()` and exit with code 1.
- Tests use small hand-built graphs and fake encoders. Earlier reviews tested the suite by planting bugs in the code, so make sure a new test can actually fail.
