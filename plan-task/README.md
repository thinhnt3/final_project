# Plan task - Text-to-image with graph reranking

## Phạm vi đề xuất

Mục tiêu an toàn để đạt Cấp độ 1 và tạo đường nâng lên Cấp độ 3:

- Nhánh: text-to-image.
- Dataset công khai ban đầu: Visual Genome vì có ảnh, caption/region và scene graph sẵn.
- Baseline: CLIP frozen + FAISS.
- Graph model: chọn **một** trong GCN/GAT/GraphSAGE sau benchmark smoke test; mặc định GraphSAGE vì dễ mở rộng và ít phụ thuộc hơn.
- Score: `final = alpha * clip_cosine + (1-alpha) * graph_score`; alpha chỉ chọn trên validation.
- Evaluation: Recall@1/5/10, MRR; tối thiểu 10 query cho Cấp độ 1, tối thiểu 20 query và direct/indirect split cho Cấp độ 2.
- Đích mở rộng: Cấp độ 3 với MMKG tiếng Việt nhỏ và fidelity/validity/sparsity trên tối thiểu 10 query.

Nếu muốn Cấp độ 2, thêm dataset thứ hai sau khi pipeline Visual Genome ổn định. Nếu muốn Cấp độ 4, chỉ chọn sau khi Cấp độ 3 có kết quả reproducible.

## Dependency graph

```text
P0 scope/data decision
        |
P1 contracts + split + leakage policy
   _____|____________________
  /      |        |          \
P2-data P2-clip P2-graph   P2-query
  \______|________|__________/
         P3 integration + smoke test
                    |
       P4 experiments (3 seeds)
          ____|___________
         /      |         \
   P4-ablation P4-xai  P4-custom-mmkg
         \______|________/
             P5 report/demo
```

## Có thể chạy song song

Sau P1: download/inspect dữ liệu, CLIP+FAISS baseline, graph schema, query-set design và environment lock có thể chạy song song nếu cùng tuân thủ contract.

Sau P3: metric implementation, ablation harness, explanation extraction và custom-MMKG collection có thể chạy song song; các nhánh chỉ dùng artifact versioned từ P3.

## Phải chạy đồng bộ

- P0 phải chốt dataset/model trước P2.
- P1 phải chốt split, ID mapping, graph schema trước mọi preprocessing/model.
- P3 phải tích hợp trên cùng candidate pool trước khi so sánh CLIP với graph.
- P4 experiment phải chạy sau smoke test và freeze config.
- P5 chỉ bắt đầu khi đủ logs 3 seed, ablation và query-level examples.

## Definition of done tổng

Có thể chạy lại từ một command/config; có split cố định; không leakage; có CLIP-only và CLIP+Graph; có ablation; có mean/std từ 3 seed; có bảng metric và 5 query thành công + 5 query thất bại; nếu nhận Cấp độ 3 thì có MMKG tiếng Việt, data card và fidelity/validity/sparsity.

## Trạng thái

| ID | Nhóm | Phụ thuộc | Trạng thái |
|---|---|---|---|
| P0 | Chốt phạm vi, dataset, graph model | - | DONE: Visual Genome, GraphSAGE pending |
| P1 | Contract, split, leakage, environment | P0 | DONE: fixed 28/10/10 split, no-overlap validation |
| P2-DATA | Download/normalize/index dữ liệu | P1 | DONE: official dumps + 40 JPG + graph subset |
| P2-CLIP | Frozen CLIP + FAISS baseline | P1 | DONE: real CLIP weights + FAISS run |
| P2-GRAPH | Graph schema + graph score prototype | P1 | DONE: scorer + trained GraphSAGE experiment; primary reranker still uses traceable scorer |
| P2-QUERY | Query benchmark trực tiếp/gián tiếp | P1 | DONE for C1: 40 graph-derived queries, 10 held-out test queries |
| P3 | Tích hợp reranking + explanation trace | P2-* | DONE: pipeline smoke |
| P4-EVAL | 3 seeds, metrics, alpha tuning | P3 | DONE on 10-query held-out test |
| P4-ABL | Edge ablation/random graph | P3 | DONE: drop-all and shuffled-relation runs |
| P4-XAI | Fidelity, validity, sparsity | P3 | OPTIONAL C3 |
| P4-MMKG | MMKG tiếng Việt 50-150 entities, 300-1000 triples, 200-500 images | P3 | OPTIONAL C3 |
| P4-EXT | Cấp độ 4, chọn một hướng | P4-EVAL + C3 nếu có | OPTIONAL |
| P5 | Report, notebook/demo, reproducibility | P4-* cần thiết | TODO |
