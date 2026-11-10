# Requirements checklist - Text-to-image Graph Retrieval

Quy ước:

- `[x] DONE`: đã có bằng chứng chạy/artefact trong repo.
- `[~] PARTIAL`: đã có prototype hoặc smoke test nhưng chưa đủ điều kiện nộp.
- `[ ] TODO`: chưa làm.

Reference: `references/Project03_Multimodal_XAI (2).pdf` và yêu cầu bài toán text-to-image.

## A. Phạm vi và dữ liệu

- [x] Chọn đúng một dataset cho Cấp độ 1: Visual Genome.
- [x] Tải scene graph chính thức.
- [x] Tải image metadata chính thức.
- [x] Tải ảnh JPG thật cho subset.
- [x] Kiểm tra 40/40 ảnh đọc được, định dạng JPEG, file size > 0.
- [x] Kiểm tra 40/40 image ID khớp graph và 40/40 query có ground-truth graph.
- [x] Có graph gồm image/object/relationship; subset có 1.322 relations.
- [~] Dùng toàn bộ dataset hoặc subset đủ đại diện: hiện mới pilot 40 ảnh.
- [ ] Ghi đầy đủ license, nguồn, checksum và data card.
- [ ] Đảm bảo dữ liệu cuối cùng phù hợp giới hạn nộp bài/cloud.

Evidence: `data/raw/visual_genome/`, `data/processed/visual_genome/`, `src/data/visual_genome.py`.

## B. Contract, split và chống leakage

- [x] Có ID mapping image/object/relation.
- [x] Có split cố định train/validation/test.
- [x] Có kiểm tra overlap giữa train/val/test.
- [~] Query test không được dùng để tune: script đã tách test, nhưng cần audit toàn bộ preprocessing.
- [ ] Chứng minh graph statistics/edge construction không đọc test.
- [ ] Có leakage checker tự động cho image/query/duplicate.
- [ ] Có environment lock đầy đủ, ví dụ `requirements.lock` hoặc environment file.

Evidence: `data/splits/visual_genome.json`, `src/data/splits.py`.

## C. Baseline CLIP frozen + FAISS

- [x] Dùng CLIP frozen thật: `openai/clip-vit-base-patch32`.
- [x] Không cập nhật gradient cho CLIP.
- [x] Trích image embedding.
- [x] Trích text/query embedding.
- [x] Dùng cosine/inner-product retrieval.
- [x] Có FAISS index.
- [x] Có NumPy fallback để smoke test khi không có FAISS.
- [ ] Cache embeddings thành artifact versioned để không phải encode lại.
- [ ] Baseline chạy trên dataset/subset cuối cùng được chốt để nộp.

Evidence: `src/models/embeddings.py`, `src/retrieval/index.py`, `runs/clip_visual_genome.json`.

## D. Graph method

- [x] Có graph loader và graph representation.
- [x] Có graph score có thể truy nguyên relation.
- [x] Có implementation GraphSAGE.
- [x] Có GraphSAGE training smoke bằng margin loss.
- [~] GraphSAGE đã chạy riêng nhưng chưa được nối hoàn toàn vào primary CLIP + graph reranker.
- [~] Primary reranker hiện dùng traceable lexical graph score, chưa phải learned GraphSAGE score.
- [ ] Chốt một graph scorer duy nhất dùng trong kết quả chính.
- [ ] Tune graph model/hyperparameter chỉ trên validation.
- [ ] Lưu checkpoint/config của graph model.

Evidence: `src/models/graphsage.py`, `src/retrieval/reranker.py`, `runs/graphsage_visual_genome.json`.

## E. Reranking

- [x] Có công thức kết hợp CLIP và graph score.
- [x] Có alpha.
- [x] Alpha được chọn trên validation trong experiment script.
- [x] CLIP-only và CLIP+Graph dùng cùng candidate pool.
- [x] Có top-k result và explanation relation.
- [~] Explanation đang gắn với lexical graph scorer; cần cập nhật nếu chuyển sang GraphSAGE scorer.
- [ ] Có một CLI/notebook end-to-end sạch cho người chấm chạy.

Evidence: `src/retrieval/pipeline.py`, `src/retrieval/reranker.py`.

## F. Query benchmark

- [x] Có tối thiểu 10 query cho pilot Cấp độ 1.
- [x] Có ground-truth image ID cho từng query.
- [~] Query hiện được sinh từ relation của scene graph, chưa có annotation thủ công độc lập.
- [ ] Phân loại direct/indirect có quy tắc và kiểm tra thủ công.
- [ ] Có bộ query final được đóng băng trước khi chạy test.
- [ ] Phân tích 5 query thành công và 5 query thất bại.

Evidence: `data/processed/visual_genome/queries.json`.

## G. Metrics và thực nghiệm chính

- [x] Recall@1.
- [x] Recall@5.
- [x] Recall@10.
- [x] MRR.
- [x] Có CLIP-only baseline.
- [x] Có CLIP+Graph result.
- [x] Đã chạy 3 seed theo protocol.
- [~] 3 seed cho kết quả giống nhau vì pipeline inference hiện deterministic, chưa có stochastic training trong primary reranker.
- [x] Có mean và standard deviation trong artifact.
- [ ] Chạy trên quy mô final, không chỉ 40 ảnh.
- [ ] Có bảng kết quả theo direct/indirect.

Evidence: `runs/clip_visual_genome.json`, `src/evaluation/metrics.py`.

## H. Ablation

- [x] Ablation bỏ relation.
- [x] Ablation shuffle predicate.
- [x] Có artifact metric cho ablation.
- [~] Kết quả hiện chưa ủng hộ graph scorer: bỏ relation tốt hơn trên pilot.
- [ ] Điều tra nguyên nhân và cải thiện graph score/query construction.
- [ ] Chạy ablation trên final dataset và 3 seed.

Evidence: `src/evaluation/ablations.py`, `runs/ablation_visual_genome.json`.

## I. XAI - chỉ bắt buộc nếu chọn Cấp độ 3

- [x] Có relation trace cho explanation cơ bản.
- [ ] Fidelity: perturb/remove explanation edge làm score/rank đổi hợp lý.
- [ ] Validity.
- [ ] Sparsity.
- [ ] Đánh giá tối thiểu 10 query theo ba metric.
- [ ] Phân tích 5 query thành công và 5 query thất bại có explanation.

## J. MMKG tiếng Việt - chỉ bắt buộc nếu chọn Cấp độ 3

- [ ] 50-150 entities.
- [ ] 300-1.000 triples.
- [ ] 200-500 ảnh.
- [ ] Entity-image linking.
- [ ] Data card, nguồn và license.
- [ ] Chạy phương pháp tốt nhất trên MMKG tự xây.
- [ ] Phân tích khả năng chuyển giao.

## K. Cấp độ 4 - phần mở rộng, chọn một

- [ ] Chọn một hướng: thuật toán mới / text-to-video / user study / công trình 2026 / dự kiến submission.
- [ ] Có hypothesis và tiêu chí thành công.
- [ ] Có baseline và ablation tương ứng.

## L. Sản phẩm nộp

- [~] Source code trong `src/` đã có loader, embedding, index, reranking, metrics, GraphSAGE.
- [ ] README chạy lại từ môi trường sạch.
- [ ] Notebook hoặc giao diện nhập query và trả top-k + explanation.
- [ ] Report kỹ thuật 12-18 trang.
- [ ] Slide trình bày.
- [ ] Bảng phân công đóng góp.
- [ ] Video demo dự phòng không quá 5 phút.
- [ ] Tập split hoặc script tạo split được nộp.
- [ ] Tài liệu phòng tránh leakage.
- [ ] Khai báo phần sử dụng mã nguồn mở/AI.
- [ ] Chuẩn bị demo chạy trực tiếp và câu hỏi vấn đáp.

## Kết luận trạng thái hiện tại

Pilot hiện đủ để chứng minh: dữ liệu thật tải được, CLIP thật chạy được, FAISS chạy được, graph reranking chạy được, metrics và 3-seed artifact tồn tại.

Pilot **chưa đủ để tuyên bố hoàn thành bài nộp Cấp độ 1** vì còn phải chốt graph scorer chính, chạy quy mô final, hoàn thiện query annotation, leakage audit, README/demo và phân tích kết quả. Cấp độ 3 và Cấp độ 4 chưa triển khai.
