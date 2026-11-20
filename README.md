# Truy hồi ảnh bằng văn bản: CLIP + xếp hạng lại bằng đồ thị

Cho một câu mô tả, hệ thống tìm ảnh khớp nhất. CLIP chọn ra các ứng viên, sau đó đồ thị cảnh của từng ảnh (vật thể và quan hệ giữa chúng) được dùng để xếp hạng lại:

```
điểm = α · cosine_CLIP + (1 − α) · điểm_đồ_thị
```

Điểm đồ thị lấy từ GraphSAGE: mạng này mã hoá đồ thị của mỗi ảnh thành một vector trong không gian CLIP.

## Kết quả cấp độ 1: GraphSAGE

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

## Cấp độ 2: AdaMF-MAT

Tái hiện AdaMF-MAT (trộn đa modality thích nghi + huấn luyện đối kháng) trên cùng bài toán và cùng cách đánh giá ở trên. Mỗi ảnh có ba modality, mỗi cái qua một lớp Linear về 256 chiều:

- **structure:** GraphSAGE 3 tầng có hướng chạy trên đồ thị cảnh; đỉnh ảnh khởi tạo bằng vector học được, không mang CLIP ảnh.
- **image:** vector CLIP ViT-B/32 của ảnh (đóng băng).
- **text:** trung bình vector CLIP text của nhãn đối tượng trong ảnh, không có vị từ.

Trọng số từng modality tính riêng cho từng ảnh: `w = softmax(aᵀ·tanh(e_m))`, vector trộn `e = Σ w_m·e_m`, rồi chiếu về không gian CLIP và huấn luyện bằng InfoNCE như GraphSAGE. MAT thêm một generator sinh vector image và text giả từ vector structure; encoder phải chấm ảnh thật cao hơn ảnh giả. Code: `src/models/adamf_mat.py`.

### So với cấp độ 1

Truy hồi theo caption, 1.500 ảnh test, 7.500 caption. Trung bình ± độ lệch chuẩn qua 3 seed (13, 37, 97), riêng các dòng ghi "5 seed" thêm seed 211 và 307. α và k chọn trên validation như cấp độ 1. Đơn vị %.

| Phương pháp | R@1 | R@5 | R@10 | MRR | Chỉ điểm mô hình, R@1 |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 43,89 | 72,19 | 82,64 | 56,76 | |
| CLIP + GraphSAGE (đỉnh ảnh học được) | 48,12 ± 0,23 | 75,33 ± 0,65 | 83,87 ± 2,12 | 60,42 ± 0,44 | 27,54 |
| CLIP + GraphSAGE (đỉnh ảnh mang CLIP) | 49,43 ± 0,46 | 77,90 ± 0,47 | 87,52 ± 0,69 | 62,22 ± 0,37 | 42,25 |
| CLIP + linear adapter (chỉ modality image, không đồ thị) | 49,76 ± 0,08 | 77,86 ± 0,06 | 87,33 ± 0,42 | 62,44 ± 0,13 | 46,25 |
| CLIP + AdaMF-MAT (bản đầy đủ của bài báo) | 51,42 ± 0,32 | 79,36 ± 0,37 | 88,91 ± 0,19 | 63,98 ± 0,31 | 48,48 |
| CLIP + AdaMF, không MAT, chuẩn hoá L2 từng modality (5 seed) | 51,93 ± 0,32 | 79,81 ± 0,17 | 89,23 ± 0,27 | 64,47 ± 0,24 | 48,62 |
| + structure-only dropout 30% (`so0.3`, 5 seed) | 52,74 ± 0,26 | 80,64 ± 0,12 | 89,68 ± 0,26 | 65,19 ± 0,20 | 49,77 |
| + structure-only loss, trọng số 0,1 (`sl0.1`, 5 seed) | **53,1** | | | | |

AdaMF chọn α khoảng 0,5–0,6 (GraphSAGE khoảng 0,8), tức điểm mô hình được tin gần ngang CLIP.

"Linear adapter" là AdaMF bỏ cả structure lẫn text: một lớp Linear trên vector CLIP ảnh, không dùng đồ thị. Nó là mốc để tách phần đóng góp của đồ thị: GraphSAGE đỉnh ảnh mang CLIP chỉ ngang adapter (49,43 so với 49,76), nghĩa là phép gộp hàng xóm không thêm gì; phần lớn mức tăng so với CLIP đến từ việc học lại vector ảnh.

### Ablation của AdaMF-MAT

R@1 sau khi xếp hạng lại, 3 seed:

| Cấu hình | R@1 | Cấu hình | R@1 |
|---|---|---|---|
| AdaMF-MAT đầy đủ | 51,42 | AdaMF không MAT | 51,63 |
| trộn bằng trung bình + MAT | 52,16 | trộn bằng trung bình, không MAT | 51,69 |
| bỏ structure | 51,92 | bỏ text | 50,54 |
| bỏ image | 47,75 | chỉ image (linear adapter) | 49,76 |

- Trọng số thích nghi và MAT đều không giúp: bật hay tắt, R@1 nằm trong 51,4–52,2.
- Image là modality quyết định: bỏ đi thì tụt 3,9 điểm so với AdaMF không MAT. Thêm text vào image được thêm 2,2 điểm (49,76 → 51,92). Bỏ structure thì không đổi.
- Bản gốc không chuẩn hoá: trọng số structure chỉ khoảng 0,0002 nhưng vector dài 84–2.011 lần, nên phần đóng góp thật khoảng 7%. Chuẩn hoá L2 từng modality trước khi trộn đưa phần đóng góp về đúng trọng số (`scripts/adamf_weight_stats.py`).

### Mô hình có đọc cạnh không

Caption COCO hiếm khi nhắc đúng quan hệ (chỉ 17,9% caption train chứa một triplet của ảnh), nên R@1 theo caption không cho biết mô hình có dùng cạnh hay không. Vì vậy dựng thêm bộ truy vấn quan hệ (`src/evaluation/relation_queries.py`):

- Mỗi truy vấn là câu sinh từ một triplet chỉ có ở đúng một ảnh test ("a man riding a horse").
- Ảnh gây nhiễu: các ảnh có cả hai vật thể nhưng không có triplet đó.
- Đo tỉ lệ ảnh đúng đứng đầu trong nhóm {ảnh đúng + ảnh gây nhiễu}.
- Bộ 1.408 truy vấn (mức ngẫu nhiên 21,4%); bộ con 734 truy vấn mà ảnh gây nhiễu có quan hệ khác giữa đúng hai vật thể đó (mức ngẫu nhiên 31,4%).
- "Che cạnh": xoá cạnh được hỏi khỏi đồ thị ảnh đúng rồi chấm lại. Độ đúng tụt nghĩa là mô hình có đọc cạnh.

Điểm chỉ dùng mô hình, không cộng CLIP, 5 seed:

| Cấu hình | 1.408 truy vấn | → che cạnh | 734 truy vấn | → che cạnh |
|---|---|---|---|---|
| CLIP (tham chiếu) | 28,8 | | 32,6 | |
| AdaMF không MAT, L2 (mốc) | 27,7 | 27,7 | 33,6 | 33,7 |
| so0.3 | 30,0 | 28,5 | 35,0 | 33,2 |
| sl0.1 | 30,0 | 28,5 | 34,9 | 33,1 |
| so0.3 + sl0.1 | 30,0 | 28,4 | 34,9 | 33,0 |

Bootstrap ghép cặp theo cụm ảnh, so với mốc, CI 95%:

- 1.408 truy vấn: +2,3 điểm [1,2; 3,4] cho cả ba cấu hình.
- 734 truy vấn: +1,3 đến +1,4 điểm, CI chứa 0 (ví dụ so0.3 +1,42 [−0,14; 3,08]).
- Mức tụt khi che cạnh trên 734 truy vấn: mốc −0,05 [−0,14; 0]; so0.3 +1,83 [1,28; 2,43]; sl0.1 +1,74; so0.3 + sl0.1 +1,93.

Cách đọc:

- AdaMF mốc **không đọc cạnh**: che cạnh không đổi gì, điểm còn thấp hơn CLIP. Mô hình đi đường tắt qua vector CLIP ảnh ở modality image.
- Ép modality structure tự gánh việc (so0.3 hoặc sl0.1) làm mô hình **bắt đầu đọc cạnh** (mức tụt có ý nghĩa thống kê), đồng thời R@1 caption tăng khoảng 1 điểm.
- Việc **phân biệt quan hệ A với quan hệ B** giữa cùng hai vật thể chưa tốt hơn mốc một cách có ý nghĩa thống kê, và cũng chỉ ngang CLIP. Kết quả 3 seed (CI [0,3; 4,0]) không còn đứng vững ở 5 seed.

### Ablation khác (3 seed nếu không ghi khác)

| Thí nghiệm | Kết quả |
|---|---|
| Tỉ lệ so 10/30/50/70% | 10% đã đủ để mô hình đọc cạnh; từ 50% trở lên, R@1 caption giảm để đổi lấy độ đúng quan hệ |
| Trọng số sl 0,1 / 0,3 / 1 | 0,1 tốt nhất (R@1 caption 53,2) |
| Chọn checkpoint theo R@1 + độ đúng quan hệ | Không cứu được mốc; bộ đo độ nhạy cạnh trên validation báo đúng là mốc ≈ 0 ở mọi epoch |
| Âm khó đổi vị từ (giống MAT) | Trọng số 1 làm huấn luyện mất ổn định (R@1 48,5); trọng số 0,1 không có tác dụng |
| Batch gom ảnh cùng cặp vật thể | Không đọc cạnh; CLIP ảnh vẫn đủ để tách các ảnh |
| Dữ liệu train 25 / 50 / 100% (7k ảnh) | Mốc không đọc cạnh ở mọi mức; so0.3 đọc cạnh ở mọi mức, độ đúng 28,2 → 29,5 → 30,2 |
| Toàn bộ VG ∩ COCO (45.817 ảnh train, giữ nguyên val/test) | Mốc đạt R@1 55,8 (+3,7) và **tự đọc cạnh** (30,6 → 29,4 khi che); so0.3 đạt 56,0, mất lợi thế so với mốc |
| Thêm cạnh IsA WordNet (2 tầng) + 1.387 truy vấn gọi tên khái niệm cha | Mô hình có dùng cạnh IsA (che thì tụt 5,0 → 4,1) nhưng không hơn bản không có IsA (5,3); CLIP 3,8 |
| Nhiễu đồ thị mô phỏng (bỏ đỉnh, đổi nhãn, bỏ cạnh) | Nhiễu nặng: mốc 48,1, so0.3 45,5, sl0.1 45,6. Mô hình đọc cạnh chịu nhiễu kém hơn |
| Caption nhắc tới triplet của ảnh | Khớp chính xác 17,9%; thêm từ đồng nghĩa WordNet 19,3%; thêm khái niệm cha 22,0% |

### Kết luận

1. AdaMF-MAT vượt GraphSAGE 3,3 điểm R@1 (51,4 so với 48,1), nhưng gần như toàn bộ đến từ modality image và nhãn đối tượng, không đến từ quan hệ.
2. Với 7k ảnh train, AdaMF bỏ qua cạnh vì modality image (CLIP) đã đủ để khớp caption.
3. Ép modality structure học riêng (so hoặc sl) làm mô hình dùng cạnh và tăng R@1 caption khoảng 1 điểm, cao nhất 53,1.
4. Khi có nhiều dữ liệu hơn (khoảng 6,5 lần), mô hình mốc tự dùng cạnh và R@1 tăng 3,7 điểm; lợi ích của can thiệp biến mất.
5. Chưa cấu hình nào phân biệt quan hệ A với B tốt hơn mốc hoặc CLIP một cách có ý nghĩa thống kê.
6. Dùng cạnh phải trả giá bằng độ bền khi đồ thị nhiễu. Cạnh IsA không thêm được gì so với chữ CLIP của nhãn.

### Chưa làm

- Âm khó đổi vị từ với trọng số giữa 0,1 và 1, hoặc áp riêng lên vector modality structure.
- Bộ dữ liệu đầy đủ với 5 seed, kèm so/sl và nhiễu.
- Nhiễu từ bộ sinh scene graph thật (cần GPU), thay cho nhiễu mô phỏng.

Số liệu R@K, MRR, α, k: `runs/benchmark/adamf_mat-*.json`. Các bảng truy vấn quan hệ, ablation khác và bootstrap lấy từ báo cáo [plans/reports/ablation-261120-0650-relation-suite.md](plans/reports/ablation-261120-0650-relation-suite.md); file kết quả thô của chúng không được lưu trong repo.

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

pytest                                                   # 334 test, không cần mạng
```

Lưu ý:

- `embed_vg_coco` cần 10.000 file ảnh trong `data/processed/vg_coco/images/`. Ảnh (1,4 GB) không nằm trong repo. Script tải ảnh và xây dữ liệu (`scripts/build_vg_coco.py`) hiện cũng không nằm trong repo; các file JSON nó sinh ra (ảnh, truy vấn, đồ thị, split) thì có.
- `train_graphsage` báo cáo lên Weights & Biases. Đổi `wandb.mode` trong `configs/retrieval/vg_coco.yaml` thành `offline` hoặc `disabled` nếu không dùng.
- Biến thể đỉnh ảnh mang CLIP: `python -m scripts.train_graphsage --image-node clip_image`, rồi `python -m scripts.run_benchmark --method graphsage --run directed-clip_image-l3-h512`.

Cấp độ 2 (AdaMF), mỗi seed khoảng 5 phút trên CPU:

```bash
python -m scripts.train_graphsage --model adamf_mat                                   # AdaMF-MAT đầy đủ
python -m scripts.run_benchmark --method adamf_mat --run adamf-mat-d256-mu1
python -m scripts.train_graphsage --model adamf_mat --no-mat --normalize-modalities \
    --structure-only-rate 0.3 --config configs/retrieval/vg_coco_5seeds.yaml          # so0.3, 5 seed
python -m scripts.run_benchmark --method adamf_mat --run adamf-d256-l2-so0.3 --config configs/retrieval/vg_coco_5seeds.yaml
python -m scripts.build_relation_queries                                              # bộ truy vấn quan hệ
python -m scripts.eval_relation_queries                                               # độ đúng trong nhóm gây nhiễu + che cạnh
python -m scripts.adamf_weight_stats --run adamf-mat-d256-mu1                         # trọng số và phần đóng góp từng modality
```

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
