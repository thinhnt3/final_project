# Requirements checklist — Text-to-image Graph Retrieval

Cập nhật theo code và bằng chứng chạy trên `main` + thay đổi local ngày 18/11/2026.
Checklist pilot 40 ảnh trước đây đã được thay bằng trạng thái VG ∩ COCO hiện tại.

- `[x]`: đã có code và bằng chứng kiểm tra/chạy tương ứng.
- `[~]`: đã làm một phần, còn giới hạn hoặc chưa đủ điều kiện nghiệm thu.
- `[ ]`: chưa triển khai hoặc chưa có bằng chứng.

Kết quả sinh trong `runs/` là artifact local; các snapshot cũ đã tracked vẫn tồn tại
trong Git. Report hướng dẫn tương tác nằm trong `private/`, không commit.

## A. Dữ liệu và phạm vi

- [x] Chốt text-to-image, CLIP frozen và GraphSAGE trên VG ∩ COCO.
- [x] Có annotation, graph JSON và caption COCO versioned theo image ID.
- [x] Chốt subset 10.000 ảnh: 7.000 train / 1.500 val / 1.500 test.
- [x] Khôi phục đủ 10.000 ảnh JPEG hợp lệ từ URL trong metadata; không có lượt tải thất bại cuối cùng.
- [x] Có 50.000 caption; mỗi caption trỏ đến một ảnh đúng.
- [x] Có graph card và báo cáo duyệt dữ liệu.
- [~] Có nguồn và checksum; cần hoàn thiện license/data card cho sản phẩm nộp.
- [ ] Chốt cách đóng gói dữ liệu/model theo giới hạn nộp bài hoặc cloud.

Evidence: `data/processed/vg_coco/`, `data/splits/vg_coco.json`,
`data/splits/vg_coco_graph.yaml`, `reports/data-review/`, `src/data/`.

## B. Split, provenance và chống leakage

- [x] Có mapping image/concept/predicate/query và split cố định.
- [x] Test chỉ lấy từ Karpathy test; có kiểm tra split và overlap trong data tests.
- [x] Từ vựng concept/predicate và statistics xây graph lấy từ train.
- [x] Chọn checkpoint trên val; α/k chỉ chọn trên val; test không dùng để tuning.
- [x] Snapshot lưu seed, architecture, input hashes; consumer kiểm tra cấu hình và hash.
- [x] Resume từ chối finished run thiếu artifact hoặc có settings khác.
- [~] Có kiểm tra ID/overlap/hash trong nhiều module; chưa có một CLI leakage audit tổng hợp.
- [ ] Có environment lock với phiên bản dependency đầy đủ.

Evidence: `src/data/graph_build.py`, `src/data/splits.py`,
`src/models/run_validation.py`, `src/models/checkpoint.py`, `tests/data/`.

## C. Baseline và embedding

- [x] Dùng CLIP ViT-B/32 thật, frozen; encode image, caption, concept và predicate.
- [x] Cache vector chuẩn hoá, ID và metadata/hash để không encode lại mỗi lượt.
- [x] Baseline cosine chạy trên gallery test 1.500 ảnh, 7.500 query.
- [x] Có lexical graph baseline để so với GraphSAGE.
- [~] Retrieval hiện dùng NumPy cosine; FAISS của pilot không còn là implementation hiện hành.
- [ ] Nếu đề yêu cầu FAISS bắt buộc: bổ sung index và kiểm tra tương đương với NumPy.
- [ ] Backbone OpenCLIP thứ hai.

Evidence: `src/embeddings/`, `src/evaluation/benchmark.py`,
`src/retrieval/lexical.py`, `scripts/run_benchmark.py`.

## D. GraphSAGE và training

- [x] Mỗi ảnh thành graph PyG gồm image/object/relation node.
- [x] Có GraphSAGE có hướng, 3 tầng, image token học được và projection sang CLIP space.
- [x] Có biến thể image node mang CLIP image vector; kết quả lịch sử còn trong repo.
- [x] Huấn luyện InfoNCE thật; CLIP và lookup embedding đóng băng.
- [x] Early stopping, checkpoint, resume, NaN checks và snapshot cấu hình.
- [x] Full và bốn ablation đã train/export đủ seed 13/37/97.
- [x] Kiểm tra 15 run: checkpoint có đủ, vector 10.000×512 hữu hạn và chuẩn hoá, input hashes nhất quán.
- [~] Đã có validation selection; chưa nghiên cứu rộng hyperparameter hoặc chứng minh kiến trúc tối ưu.

Evidence: `src/models/{graph_data,graphsage,train,checkpoint}.py`,
`scripts/train_graphsage.py`, output local `runs/graphsage/`.

## E. Reranking và explanation

- [x] Kết hợp raw cosine CLIP và graph theo α; cùng nguồn candidate CLIP cho phép so sánh tương ứng.
- [x] α/k chọn riêng cho từng seed/variant trên validation.
- [x] α=1 giữ thứ hạng CLIP; có conservative tie handling.
- [x] Explanation của GraphSAGE dùng đúng checkpoint và graph intervention tương ứng.
- [x] Xếp relation theo signed score drop khi xoá từng relation.
- [x] Có CLI embed/train/benchmark/explanation và hướng dẫn chạy.
- [~] Chưa có một giao diện demo hoặc một lệnh duy nhất gói trọn toàn bộ pipeline.

Evidence: `src/retrieval/rerank.py`, `src/models/explain.py`,
`scripts/explain_retrieval.py`, `docs/graphsage-design-decisions.md`.

## F. Query và phân tích trường hợp

- [x] Query lấy từ caption COCO, không sinh trực tiếp từ chữ trong graph.
- [x] Query/split versioned, ground truth image ID rõ ràng.
- [x] Phân tích paired wins/losses trên cùng query; bootstrap theo cụm ảnh.
- [~] Có 5 ca full thắng và 5 ca objects-only thắng, kèm caption/graph/hạng; còn thiếu bình luận thủ công và ảnh minh hoạ hoàn chỉnh.
- [ ] Định nghĩa và gán nhãn direct/indirect có kiểm tra thủ công.
- [ ] Đánh giá đại diện để kiểm chứng giả thuyết caption thiên về đối tượng hoặc annotation nhiễu.

Evidence: `scripts/analyze_ablation.py`, `src/evaluation/ablation_analysis.py`,
output local `runs/analysis/ablation.json`.

## G. Metrics và thực nghiệm

- [x] Recall@1/5/10, MRR, per-seed và mean/sample std.
- [x] Ba seed là ba lượt training thật, không nhân bản inference deterministic.
- [x] CLIP, lexical và GraphSAGE đã có benchmark trên subset final.
- [x] Phân biệt official benchmark với kiểm tra k=500 sau thực nghiệm; không dùng kiểm tra đó chọn lại model.
- [ ] Bảng theo direct/indirect và thí nghiệm trên dataset công khai thứ hai.

Evidence: `src/evaluation/{metrics,benchmark,ablation_analysis}.py`,
`scripts/run_benchmark.py`, output local `runs/benchmark/`.

## H. Ablation

- [x] Full, objects-only, zero predicate features, random endpoints và undirected.
- [x] Random endpoints giữ directed degree và predicate histogram; cho phép cạnh tự nối/trùng theo định nghĩa control.
- [x] Mỗi biến thể dùng cùng cache/split/training protocol; đủ ba seed.
- [x] Có paired analysis và khoảng bất định conditional trên ba checkpoint đã train.
- [~] Đã quét α/k để chọn cấu hình; chưa có biểu đồ sensitivity đầy đủ.
- [ ] Ablation xử lý hub.
- [ ] Direction ablation có số tham số khớp với full.
- [ ] Lặp lại ablation cần thiết trên dataset thứ hai.

Full có R@1 fusion cao nhất ở lượt chạy hiện tại, nhưng không thắng mọi metric.
Chưa có bằng chứng mạnh về lợi ích riêng của nhãn vị từ/cạnh thật qua các training seed mới.

Evidence: `src/models/ablations.py`, `configs/retrieval/ablations/`,
`docs/graphsage-design-decisions.md`.

## I. XAI/fidelity

- [x] Joint deletion trên full checkpoint cố định, so với random groups cùng số relation.
- [x] 200 query được lấy mẫu cố định, đủ ba seed; 198/195/196 query hợp lệ sau small-graph skips.
- [x] Đo signed graph-score drop, rank increase và mất top-1, lưu budget thực tế và skip reasons.
- [x] Giữ nguyên CLIP, α/k, candidate pool và điểm đối thủ khi perturb ảnh dự đoán gốc.
- [x] Ghi rõ selection dependence: đây là perturbation diagnostic, không phải chứng minh semantic faithfulness độc lập.
- [ ] Validity về ngữ nghĩa hoặc đánh giá bằng con người.
- [~] Có removed fraction trong trace; chưa có đánh giá sparsity đầy đủ hoặc lời giải thích tối thiểu.
- [ ] Hoàn tất đánh giá cả fidelity/validity/sparsity theo yêu cầu nộp.

Evidence: `src/evaluation/fidelity.py`, `scripts/evaluate_fidelity.py`,
`configs/evaluation/fidelity.yaml`, output local `runs/fidelity/`.

## J. MMKG và mở rộng

- [ ] MMKG tiếng Việt với entity/triple/image theo yêu cầu, linking, license và data card.
- [ ] Transfer evaluation trên MMKG tự xây.
- [ ] Chọn và định nghĩa giả thuyết/baseline cho phần mở rộng Cấp độ 4.

## K. Sản phẩm nộp

- [x] Source, config, tests và tài liệu định nghĩa ablation/fidelity.
- [x] Split cuối được versioned; README có lệnh chạy lại.
- [~] Clean-environment reproduction còn phụ thuộc ảnh/cache local và script khôi phục dữ liệu chưa được đóng gói đầy đủ.
- [ ] Demo/giao diện nhập query và xem top-k + explanation.
- [ ] Report kỹ thuật 12–18 trang, slides và bảng đóng góp.
- [ ] Video dự phòng ≤5 phút; chuẩn bị demo/vấn đáp.
- [ ] Khai báo mã nguồn mở/AI và hướng dẫn đóng gói artifact cuối cùng.

## Kết luận hiện tại

Pipeline cốt lõi, ablation trên VG ∩ COCO và fidelity dạng perturbation đã chạy.
Phase 4 còn OpenCLIP và phân tích ca hoàn chỉnh. Phase 5 còn dataset thứ hai,
hub ablation và các giới hạn kiểm chứng đã nêu. Chưa coi toàn bộ bài nộp hoặc
đánh giá XAI Cấp độ 3 là hoàn tất.
