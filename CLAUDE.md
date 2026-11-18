# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A course project (brief: `references/Project03_Multimodal_XAI (2).pdf`) for text-to-image retrieval on the Visual Genome ∩ COCO subset: 10,000 images, and COCO captions as queries, each with exactly one relevant image. Frozen CLIP ViT-B/32 retrieves the top-k candidates. A GraphSAGE encoder turns each image's scene graph into a vector in CLIP space. The candidates are then reranked by `final = α·cos_CLIP + (1−α)·cos(query, graph_vector)`.

- **Explanations** come from removing one relation at a time and measuring how much the graph score drops.
- **Ablations** (objects only, zeroed predicates, random endpoints, undirected), **joint-deletion fidelity**, relation sparsity / keep-only, and validity audits test whether the gain and the explanations really come from relations.

Docs, plans and journals are written in Vietnamese. Code, identifiers, log messages and comments are in English.

`docs/README.md` is the walkthrough for human readers. It links `docs/01`–`05` and two appendices:

- `phu-luc-a-*` explains the background, with formulas and worked numbers.
- `phu-luc-b-*` is a hand calculation through the whole pipeline.

When results change, update the numbers in `docs/04-ket-qua.md` and regenerate the figures with `python -m scripts.make_doc_figures`.

## Where the project state lives

- **Roadmap:** `plans/261117-0037-multimodal-graph-retrieval-roadmap/`. `plan.md` lists the phases. Each `phase-*.md` has numbered steps, "Xong khi" (done-when) criteria and a "Tiến độ" (progress) table with evidence. Update that table when you finish a step.
- **Assignments and acceptance status:** `task.md` says who owns what. `plan-task/requirements-checklist.md` tracks acceptance against the brief. The rest of `plan-task/` comes from the 40-image pilot and is stale.
- **Decision records:** `docs/graphsage-design-decisions.md` holds model decisions G1–G10, plus the ablation and fidelity protocol and the observed evidence. Data and graph decisions D1–D7 are in `phase-02-*.md` and `plans/journals/`. `agents/AGENTS.md` keeps the agent rules and a dated run log.
- **Other:** `docs/build_MMKG.md` is the guide for building the Vietnamese MMKG (Level 3). `runs.zip` at the root archives old pilot experiments.

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

# Ablations: one config per variant under configs/retrieval/ablations/ (full, objects_only, drop_predicates, random_edges, undirected)
python -m scripts.train_graphsage --config configs/retrieval/ablations/objects_only.yaml --resume
python -m scripts.run_benchmark --method graphsage --config configs/retrieval/ablations/objects_only.yaml
python -m scripts.analyze_ablation                   # paired, image-clustered bootstrap -> runs/analysis/ablation.json
python -m scripts.evaluate_fidelity --config configs/evaluation/fidelity.yaml   # -> runs/fidelity/full-joint-test.json
python -m scripts.evaluate_explanation_quality       # sparsity / keep-only; --labels review-v1.json summarizes validity labels
python -m scripts.prepare_validity_review --all      # blinded package for tools/validity-review/index.html
python -m scripts.make_doc_figures                   # redraw docs/images/*.svg from runs/ and data/processed/
```

- **Not in the repo** (gitignored): the 10k images (1.4 GB), the CLIP embeddings, the checkpoints (`*.pt`), `graph_vectors.*`, and the data-build scripts `scripts/build_vg_coco.py` and `scripts/review_vg_coco.py`. Every script that consumes a training run therefore needs a local `train_graphsage` run first.
- **Committed:** the processed JSON (`data/processed/vg_coco/*.json`, `graph/*.json`, `data/splits/*`) and the benchmark and run summaries already tracked under `runs/`.
- **What `.gitignore` covers:**
  - `/runs/` and `/private/`, so new experiment outputs, local reports and review packages stay local. Files already tracked in `runs/` stay tracked; add a new result file only on purpose (`git add -f`).
  - Root-level `*.zip` archives.
- **Weights & Biases:**
  - `configs/retrieval/vg_coco.yaml` sets `wandb.mode: online`, so `train_graphsage` reports to W&B. Set it to `offline` or `disabled` to run without a login. The ablation configs already use `disabled`.
  - `WandbTracker` deliberately turns off W&B's host, git, hardware and console metadata. Leave those settings as they are.

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
   - `graphsage.py` holds `GraphImageEncoder` and `GraphSageConfig`:
     - A directed layer is one SAGEConv along the edges plus one against them, summed.
     - ReLU sits between layers but not after the last one, because CLIP vectors have negative components.
     - The image node's state goes through a linear projection and L2 normalization.
     - `image_node: learned` is the main model: the graph vector comes from the graph alone. `clip_image` is the variant: its graph score already contains CLIP image information.
     - The config fields `ablation` and `ablation_seed` select an intervention. `drop_predicates` zeroes the relation-node features, and `undirected` uses the single-SAGEConv layer, which also has fewer parameters.
     - `run_name()` appends `-<ablation>-a<seed>` for every variant except `full`.
     - The frozen lookup tables are non-persistent buffers, so checkpoints hold only the trained weights.
   - `ablations.py`: `ablate_graph` copies a raw graph and applies the graph-level interventions.
     - `objects_only` deletes every triplet.
     - `random_edges` independently shuffles subjects and objects with a per-image RNG. This keeps each concept's in- and out-degree and the predicate counts, but allows self-loops and duplicate triplets.
     - The same intervention applies to training, validation, vector export and explanation.
   - `train.py` and `checkpoint.py` handle training:
     - The loss is symmetric in-batch InfoNCE against frozen CLIP caption vectors. The best epoch is picked by graph-only R@1 on validation.
     - Batches depend only on (seed, epoch), so `--resume` reproduces an uninterrupted run.
     - `last.pt` is written every epoch and `best.pt` on each improvement, both atomically through a temp file and rename.
     - A non-finite loss or gradient raises `TrainingError` before the weights are updated.
     - Resume is refused if any setting changed (except `max_epochs`, `patience` and `seeds`), or if a run marked finished is missing files.
   - `run_validation.py`: `validate_run` and `validate_artifact_inputs` reject checkpoints, graph vectors or benchmarks whose architecture, seed, CLIP model or input hashes do not match. Every evaluation script calls them before using an artifact.
   - `explain.py` removes each triplet (keeping its objects), re-encodes the graph and ranks the relations by signed score drop.
4. **`src/evaluation/`**:
   - `build_split_data` builds each split's gallery from that split's images only.
   - `ranks_of_relevant` breaks ties against the relevant image.
   - `aggregate_runs` gives the mean and the sample standard deviation.
   - `ablation_analysis.py` runs a paired bootstrap clustered by the true image (2,000 resamples). Its 95% intervals are conditional on the trained seeds.
   - `fidelity.py`: `joint_relation_fidelity` works on the top-1 predicted image with everything else frozen (CLIP scores, α/k, the candidate pool, rival scores).
     - It removes the top-b relations by single-deletion drop all together, where b = min(n−1, max(2, ⌈n·fraction⌉)).
     - It compares against 20 equal-size random groups drawn from all relations.
     - It reports the graph-score drop, the rank change and the loss of top-1.
     - Graphs with fewer than 3 relations are skipped. Negative drops are kept.
   - `explanation_quality.py` measures relation sparsity (1 − |E|/n) and keep-only retention, and summarizes validity labels. `validity_audit.py` packages blinded cases for the offline review form. Labels record whether a reviewer was an AI or a human; never relabel AI labels as human.
5. **`src/retrieval/`**:
   - `rerank.py` does the fusion and the α/k sweep.
   - `lexical.py` is the B4 baseline: IDF-weighted word overlap, with IDF from train graphs only.
   - `config.py` holds `RetrievalConfig`.

Results are written here:

- `runs/benchmark/<method>.json` holds validation and test metrics, the selected α and k, and the input sha256. GraphSAGE runs are named `graphsage-<run>.json`.
- `runs/graphsage/<run>/seed<n>/` holds a run (`config.yaml`, `history.json`, `result.json`).
- `runs/explanations/` holds the explanation output.

The analysis, fidelity and explanation-quality outputs under `runs/` stay local. Their headline numbers are recorded in `docs/graphsage-design-decisions.md`.

## Invariants (enforced by the code and tests; keep them)

- Nothing is ever selected on test. α, k, the best epoch, hyperparameters, the vocabulary, IDF and thresholds all come from train or validation. `sweep_alpha_and_k` raises unless it is given the validation split. Test is scored once, with the setting selected on validation. Each ablation variant selects its own α/k on validation.
- Reranking rules:
  - Both score matrices must lie in [-1, 1].
  - Fusion runs in float64, because float32 minus the non-candidate offset creates ties.
  - α = 1 must reproduce the CLIP ranks exactly.
  - On a tie, the setting closest to plain CLIP wins: higher α first, then smaller k.
- Every method uses the same split, gallery and candidate pool. Trained methods report the mean ± sample std over seeds 13, 37 and 97. Deterministic methods report a single run; don't invent extra seeds for them.
- Fidelity and quality diagnostics never retrain and never tune. They describe score sensitivity of the frozen model, not proof of semantic faithfulness. Keep that wording in reports.
- Pure functions in `src/` do no file I/O. I/O happens in the load/save helpers and in `scripts/`.
- Don't hard-code paths or thresholds. Settings live in YAML under `configs/` (data, retrieval, ablation and evaluation configs) and are parsed into frozen dataclasses that validate on load (`DataConfig.from_yaml`, `RetrievalConfig.from_yaml`).
- Raise a subclass from `src.core.exceptions` (`ConfigError`, `DataError`, `TrainingError`, ...) where the problem is detected. Scripts catch `AppError` once in `main()` and exit with code 1.
- Tests use small hand-built graphs and fake encoders. Earlier reviews tested the suite by planting bugs in the code, so make sure a new test can actually fail.
