# Folder plan

```text
.
├── agents/
│   └── AGENTS.md                 # luật nhận task và guardrail
├── references/
│   └── Project03_Multimodal_XAI (2).pdf
├── plan-task/
│   ├── README.md                 # scope, dependency, trạng thái
│   ├── 00-roadmap.md             # task detail và nghiệm thu
│   └── folder-plan.md            # quy ước thư mục
├── configs/                      # YAML cấu hình, không chứa secret
├── schemas/                      # schema JSON/CSV và version
├── data/
│   ├── raw/                      # download/manifest/checksum, không commit data lớn
│   ├── processed/                # normalized graph/caption/link
│   └── splits/                   # split cố định hoặc script tạo split
├── notebooks/                    # khám phá/demo, không làm source of truth
├── scripts/                      # download, preprocess, split, evaluate, reproduce
├── src/                          # chỉ code sau P0/P1 gate
│   ├── data/                     # loaders, validators
│   ├── models/                   # clip adapter, graph encoder
│   ├── retrieval/                # FAISS, rerank, scoring
│   ├── explain/                  # path/subgraph trace, perturbation
│   ├── evaluation/               # metrics, aggregation, ablation
│   └── cli/                      # entry points
├── tests/                        # unit, fixture, leakage, metric tests
├── artifacts/                    # embeddings/index/checkpoints, gitignored
├── runs/                         # logs/results per seed/config, gitignored
└── reports/                      # tables, plots, case studies, final assets
```

## Quy tắc thêm/xóa

- Mỗi artifact có `manifest.json` gồm source version, config hash, seed và timestamp.
- Xóa một implementation chỉ được xóa module/task branch; giữ schema, fixtures và kết quả đã dùng trong report.
- Không để notebook chứa logic duy nhất; logic phải nằm trong `src/` hoặc `scripts/` và được notebook gọi lại.
- `src/` hiện là vùng khóa. Mở khóa chỉ sau khi P0/P1 ghi `DONE` trong roadmap.
