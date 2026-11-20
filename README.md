# Truy hồi ảnh bằng văn bản: CLIP + xếp hạng lại bằng đồ thị

Cho một câu mô tả, hệ thống tìm ảnh khớp nhất. CLIP chọn ra các ứng viên, sau đó đồ thị cảnh của từng ảnh (vật thể và quan hệ giữa chúng) được dùng để xếp hạng lại:

```
điểm = α · cosine_CLIP + (1 − α) · điểm_đồ_thị
```

Điểm đồ thị lấy từ GraphSAGE: mạng này mã hoá đồ thị của mỗi ảnh thành một vector trong không gian CLIP.

Tài liệu giải thích đầy đủ, có hình minh hoạ và phụ lục công thức kèm ví dụ: [`docs/README.md`](docs/README.md).

## Kết quả

Bảng dưới là lượt chạy ban đầu ngày 17/11. Lượt khôi phục local ngày 18/11
đã chạy lại full và ablation; full R@1 mới là 48,12 ± 0,23%. Xem
[ablation và fidelity](docs/graphsage-design-decisions.md#ablation-và-fidelity) để đọc giao thức và kết quả mới.

Tập test: 1.500 ảnh, 7.500 câu truy vấn (caption COCO), mỗi câu có đúng một ảnh đúng. α và số ứng viên k được chọn trên tập validation; tập test chỉ được chấm sau khi đã chốt tham số.

| Phương pháp | R@1 | R@5 | R@10 | MRR | α | k |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | 43,89 | 72,19 | 82,64 | 56,76 | 1 | không dùng |
| **CLIP + GraphSAGE** (mô hình chính) | **48,20 ± 0,05** | 75,87 ± 0,18 | 86,39 ± 0,14 | 60,75 ± 0,10 | 0,81 / 0,83 / 0,78 | 1.500 / 1.500 / 500 |
| CLIP + GraphSAGE, đỉnh ảnh mang vector CLIP (lúc khởi tạo) | 49,43 ± 0,46 | 77,90 ± 0,47 | 87,52 ± 0,69 | 62,22 ± 0,37 | 0,66 / 0,63 / 0,68 | 500 / 500 / 20 |

Cột α và k ghi giá trị được chọn trên validation cho từng seed, theo thứ tự 13 / 37 / 97. α là trọng số của điểm CLIP trong điểm cuối, nên α càng thấp thì điểm đồ thị càng được tin: mô hình chính dành khoảng 19% cho điểm đồ thị, biến thể đỉnh ảnh mang CLIP khoảng 34%. Ở lượt khôi phục ngày 18/11, mô hình chính chọn α = 0,78 / 0,84 / 0,78 và k = 500 / 10 / 10.

Các dòng GraphSAGE là trung bình ± độ lệch chuẩn qua 3 seed (13, 37, 97). Đơn vị: %.

Khi chỉ dùng điểm đồ thị, không có CLIP:

| Phương pháp | R@1 | R@10 |
|---|---|---|
| So khớp từ với đồ thị | 12,63 | 39,93 |
| GraphSAGE (mô hình chính) | 27,78 ± 0,63 | 72,20 ± 0,36 |
| Mức ngẫu nhiên | 0,07 | 0,67 |

Cách đọc bảng:

- GraphSAGE nâng R@1 của CLIP thêm 4,3 điểm, và hơn cách so khớp từ 1,8 điểm.
- Ở mô hình chính, đỉnh ảnh không mang thông tin CLIP của ảnh, nên vector đồ thị chỉ đến từ cấu trúc đồ thị. Phần cải thiện vì thế quy được cho đồ thị.
- Biến thể "đỉnh ảnh mang vector CLIP" cho số cao hơn nhưng dao động giữa các seed nhiều hơn, và điểm đồ thị của nó đã lẫn thông tin CLIP.
- Các tham số huấn luyện là giá trị khởi đầu, chưa tinh chỉnh.

File kết quả: `runs/benchmark/*.json`. Lịch sử huấn luyện từng seed: `runs/graphsage/<tên lượt chạy>/seed<n>/history.json`.

### Chưa có

- Dataset thứ hai và ablation xử lý hub; ablation cốt lõi và fidelity perturbation đã có (xem tài liệu bên dưới).
- Backbone thứ hai (OpenCLIP) và tập dữ liệu công khai thứ hai.
- Phân tích 5 truy vấn thành công và 5 truy vấn thất bại.

## Dữ liệu

Phần giao của Visual Genome và COCO: ảnh và đồ thị cảnh lấy từ Visual Genome, câu truy vấn là caption COCO của cùng ảnh.

| | train | validation | test |
|---|---|---|---|
| Ảnh | 7.000 | 1.500 | 1.500 |
| Câu truy vấn | 35.000 | 7.500 | 7.500 |

- Ảnh test chỉ lấy từ tập test của cách chia Karpathy.
- Nhãn vật thể được gộp theo synset WordNet. Từ vựng của đồ thị (1.778 khái niệm, 493 vị từ) chỉ tính trên ảnh train.
- Thông số đồ thị: `data/splits/vg_coco_graph.yaml`. Báo cáo duyệt dữ liệu: `reports/data-review/index.html`.

## Cách hoạt động

1. Mỗi ảnh là một đồ thị con: một đỉnh ảnh, các đỉnh đối tượng, và các relation giữa đối tượng. Nhãn của relation (`riding`, `on`) được lưu trên một đỉnh phụ để GraphSAGE đọc được.
2. Đỉnh đối tượng và đỉnh phụ khởi đầu bằng vector CLIP text của chữ trên đỉnh.
3. Ba tầng GraphSAGE có hướng trộn thông tin giữa các đỉnh. Vector của đỉnh ảnh sau tầng cuối là vector đồ thị của ảnh.
4. Huấn luyện bằng InfoNCE: vector đồ thị của ảnh phải gần vector CLIP của caption mô tả ảnh đó. CLIP đóng băng.
5. Khi truy vấn: CLIP lấy top-k ảnh, điểm đồ thị là cosine giữa vector truy vấn và vector đồ thị, rồi xếp hạng lại theo công thức trên.

Chi tiết và lý do của từng quyết định: `docs/graphsage-design-decisions.md`. Minh hoạ từng bước: `docs/diagrams/graphsage-how-it-works.html`.

## Chạy lại

Cần Python 3.12. Chạy mọi lệnh từ thư mục gốc của repo.

```bash
pip install -r requirements.txt

python -m scripts.embed_vg_coco                          # vector CLIP của ảnh, caption, chữ trên đồ thị
python -m scripts.run_benchmark --method clip            # baseline CLIP
python -m scripts.run_benchmark --method graph_lexical   # CLIP + so khớp từ
python -m scripts.train_graphsage                        # huấn luyện 3 seed, khoảng 2 phút mỗi seed trên CPU
python -m scripts.run_benchmark --method graphsage       # CLIP + GraphSAGE
python -m scripts.explain_retrieval --sample 5           # kết quả kèm relation giải thích

pytest                                                   # 252 test, không cần mạng
```

Lưu ý:

- `embed_vg_coco` cần 10.000 file ảnh trong `data/processed/vg_coco/images/`. Ảnh (1,4 GB) không nằm trong repo. Script tải ảnh và xây dữ liệu (`scripts/build_vg_coco.py`) hiện cũng không nằm trong repo; các file JSON nó sinh ra (ảnh, truy vấn, đồ thị, split) thì có.
- `train_graphsage` báo cáo lên Weights & Biases. Đổi `wandb.mode` trong `configs/retrieval/vg_coco.yaml` thành `offline` hoặc `disabled` nếu không dùng.
- Biến thể đỉnh ảnh mang CLIP: `python -m scripts.train_graphsage --image-node clip_image`, rồi `python -m scripts.run_benchmark --method graphsage --run directed-clip_image-l3-h512`.

## Cấu trúc repo

```
configs/        tham số dữ liệu và thí nghiệm (YAML)
src/data/       xây dữ liệu: ghép VG với COCO, chuẩn hoá synset, chia split, xây đồ thị
src/embeddings/ CLIP đóng băng và kho lưu vector
src/models/     đồ thị con, GraphSAGE, huấn luyện, giải thích
src/retrieval/  xếp hạng lại, so khớp từ, đọc config
src/evaluation/ Recall@K, MRR, bộ chạy benchmark
scripts/        các lệnh chạy ở trên
runs/           kết quả benchmark và lượt huấn luyện
docs/           quyết định thiết kế và sơ đồ
plans/          kế hoạch theo phase và nhật ký công việc
tests/          test tự động
```

## Ablation và fidelity

Hướng dẫn chạy, định nghĩa phép can thiệp và giới hạn kết luận:
[docs/graphsage-design-decisions.md](docs/graphsage-design-decisions.md#ablation-và-fidelity).

```bash
python -m scripts.train_graphsage --config configs/retrieval/ablations/objects_only.yaml --resume
python -m scripts.run_benchmark --method graphsage --config configs/retrieval/ablations/objects_only.yaml
python -m scripts.analyze_ablation
python -m scripts.evaluate_fidelity --config configs/evaluation/fidelity.yaml
```

Report giải thích tương tác là tài liệu local trong `private/`, không thuộc repository.

Các output mới trong `runs/` và report local trong `private/` được Git-ignore.
Những file lịch sử đã tracked trong `runs/` vẫn được Git theo dõi.
Tiến độ nghiệm thu hiện tại: [requirements checklist](plan-task/requirements-checklist.md).
