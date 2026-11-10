# Roadmap chi tiết

## P0 - Quyết định phạm vi (đồng bộ, blocker)

**Task:** Chốt text-to-image, Visual Genome, GraphSAGE mặc định; ghi rõ nếu nhóm chọn GCN/GAT hoặc dataset khác.

**Output:** `configs/project_scope.yaml`, biên bản quyết định, license/source notes.

**Nghiệm thu:** dataset có đủ image-text và graph links; có cách tạo positive/negative candidate; ước lượng chạy được trên máy nhóm; giảng viên/nhóm đồng ý level target.

## P1 - Contract và reproducibility (đồng bộ, trước mọi code model)

**Task:** Định nghĩa schema cho image/entity/relation/query, ID mapping, train/val/test split, seed protocol, config format, artifact naming và leakage checklist.

**Output:** `configs/`, `schemas/`, `data/splits/README.md`, `requirements.txt` hoặc lock file, checklist leakage.

**Nghiệm thu:** cùng một split được CLIP và graph dùng; test query không tham gia tuning; có command kiểm tra duplicate/image overlap và graph statistics.

## P2-DATA - Dữ liệu và preprocessing (song song sau P1)

**Task:** Download/verify dữ liệu; normalize annotation; tạo graph edge list và metadata; lưu checksum; tạo subset nhỏ để smoke test.

**Output:** `data/raw/` manifest, `data/processed/` manifest, stats report, loader contract.

**Nghiệm thu:** load được từ clean environment; không hard-code absolute path; có thống kê node/edge/image/caption và mapping coverage.

## P2-CLIP - Baseline truy vấn (song song sau P1)

**Task:** Trích frozen CLIP image/text embeddings; chuẩn hóa vector; FAISS top-k; CLIP-only ranking và cache.

**Output:** embedding manifest, index, top-k result file, baseline smoke metrics.

**Nghiệm thu:** deterministic với seed/config; candidate pool cố định cho graph rerank; Recall/MRR tính đúng trên toy fixture và một subset thật.

## P2-GRAPH - Graph encoder/score (song song sau P1, có thể bắt đầu bằng toy graph)

**Task:** Implement graph loader, một encoder đã chọn, image/query node representation hoặc path-based score; định nghĩa graph score trong [0,1] hoặc có normalization rõ ràng.

**Output:** model module, checkpoint/config contract, graph-score fixture và explanation trace fixture.

**Nghiệm thu:** score thay đổi khi thay đổi cạnh liên quan; trace trả về đúng node/relation/path dùng để tính score; không đọc test để build statistics.

## P2-QUERY - Bộ truy vấn đánh giá (song song sau P1)

**Task:** Tạo tối thiểu 10 query cho Cấp độ 1; mở rộng tối thiểu 20 query cho Cấp độ 2, chia direct/indirect; giữ test queries không dùng tuning.

**Output:** versioned query JSON/CSV, annotation rationale, query construction script.

**Nghiệm thu:** mỗi query có ground-truth image(s), loại direct/indirect, nguồn annotation và kiểm tra không trùng train/val.

## P3 - Tích hợp (đồng bộ)

**Task:** Kết nối CLIP top-k → graph score → alpha rerank → top-k + explanation. Chọn alpha trên validation; tạo một command end-to-end.

**Output:** inference API/CLI, result schema, smoke notebook, end-to-end log.

**Nghiệm thu:** chạy được trên toy và subset thật; CLIP-only và CLIP+Graph dùng cùng candidate pool; explanation lấy từ chính scoring path.

## P4-EVAL - Thực nghiệm chính (đồng bộ config, chạy song song theo seed được)

**Task:** Chạy CLIP-only, CLIP+Graph trên cùng split với tối thiểu 3 seed; aggregate mean/std cho Recall@1/5/10 và MRR; tách direct/indirect.

**Output:** `runs/`, metric JSON/CSV, bảng/plot, frozen experiment config.

**Nghiệm thu:** không có missing seed; có per-query result để audit; report nêu alpha và cách chọn trên validation.

## P4-ABL - Ablation (song song sau P3)

**Task:** Bỏ cạnh liên quan, shuffle/random graph hoặc edge-type ablation; giữ mọi thứ khác cố định.

**Output:** ablation matrix và per-query deltas.

**Nghiệm thu:** ablation chứng minh ảnh hưởng đến từ cấu trúc quan hệ, không chỉ từ candidate pool/model size.

## P4-XAI - XAI Cấp độ 3 (song song sau P3)

**Task:** Xuất path/subgraph explanation; perturb/remove explanation edges; đo fidelity, validity, sparsity tối thiểu 10 query.

**Output:** explanation JSON, perturbation logs, metric definitions và case studies.

**Nghiệm thu:** bỏ/nhiễu cạnh được giải thích làm score/rank thay đổi theo hướng hợp lý; không sinh explanation hậu kỳ không ảnh hưởng prediction.

## P4-MMKG - MMKG tiếng Việt Cấp độ 3 (song song sau P3, phụ thuộc data policy)

**Task:** Xây 50-150 entities, 300-1,000 triples, 200-500 images; data card, nguồn/licence, entity-image linking và split.

**Output:** versioned dataset, data card, build script, stats, transfer evaluation.

**Nghiệm thu:** tái tạo được dataset từ manifest/source; có kiểm tra chất lượng triple/link; chạy được phương pháp tốt nhất trên dữ liệu này.

## P4-EXT - Cấp độ 4 (sau khi core ổn định)

Chọn đúng một: thuật toán mới, text-to-video, user study đối chứng, công trình 2026, hoặc chuẩn bị submission. Phải có hypothesis, baseline và tiêu chí thành công trước khi implement.

## P5 - Đóng gói (đồng bộ cuối)

**Task:** Report 12-18 trang, slides, contribution table, notebook/demo, video dự phòng ≤5 phút, README reproduce và clean-run test.

**Nghiệm thu:** người mới clone/copy repo có thể chạy smoke test; demo trả top-k và explanation; báo cáo có 5 success + 5 failure, limitation, leakage và 3-seed mean/std.
