# Truy hồi ảnh bằng văn bản: CLIP + xếp hạng lại bằng đồ thị

Cho một câu mô tả, hệ thống tìm ảnh khớp nhất. CLIP chọn ra các ứng viên, sau đó đồ thị cảnh của từng ảnh (vật thể và quan hệ giữa chúng) được dùng để xếp hạng lại:

```
điểm = α · cosine_CLIP + (1 − α) · điểm_đồ_thị
```

Điểm đồ thị lấy từ GraphSAGE: mạng này mã hoá đồ thị của mỗi ảnh thành một vector trong không gian CLIP.

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

- Dataset công khai thứ hai và ablation xử lý hub (bộ tự xây: xem [cấp độ 3](#cấp-độ-3-mmkg-tự-xây-vn-heritage-bản-lõi)); ablation cốt lõi và fidelity perturbation đã có (xem tài liệu bên dưới).
- Backbone thứ hai (OpenCLIP) và tập dữ liệu công khai thứ hai.
- Phân tích 5 truy vấn thành công và 5 truy vấn thất bại.

## Cấp độ 3: MMKG tự xây (VN Heritage, bản lõi)

Bộ dữ liệu tự xây về địa điểm và di tích Việt Nam: mỗi địa điểm có ảnh từ Wikimedia Commons và tri thức từ Wikidata và Wikipedia tiếng Việt. Bài toán vẫn là truy hồi ảnh bằng văn bản với cùng CLIP ViT-B/32 đóng băng và cùng công thức `α · cosine_CLIP + (1 − α) · điểm_đồ_thị`. Khác với Visual Genome ∩ COCO, ở đây **không có đồ thị cảnh**: đồ thị là một KG dùng chung, và ảnh nối vào KG qua cạnh "ảnh – mô tả – địa điểm".

### Dữ liệu

| | Số lượng |
|---|---|
| Địa điểm có ảnh (gallery) | 160 (train 97 · validation 22 · test 41) |
| Ảnh | 772, mỗi địa điểm 3–5 ảnh |
| Thực thể trong đồ thị | 849 (160 địa điểm + 689 đỉnh tri thức) |
| Claim Wikidata | 981, 18 quan hệ (gồm 92 cạnh phân cấp hành chính) |
| Cạnh rút từ bài Wikipedia | 902, đặt tên theo bốn sơ đồ (`generic`, `kind`, `section`, `cue`) |

- **Truy vấn trực tiếp:** tên địa điểm, chia theo split địa điểm (97 · 22 · 41).
- **Truy vấn gián tiếp:** giấu 40% claim Wikidata (15% validation, 25% test); mỗi claim bị giấu thành một câu, ví dụ `a heritage site located in Đà Lạt`. Claim train là cạnh của đồ thị, claim validation và test không bao giờ nằm trong đồ thị. Mỗi cách chia có khoảng 145 truy vấn gián tiếp trên test.
- **Chia theo cặp đỉnh:** mọi claim nối cùng hai đỉnh vào cùng một tập, nên sự kiện bị giấu không đọc lại được từ quan hệ nghịch.
- **Cạnh rút từ bài** không bao giờ bị hỏi, chỉ làm bằng chứng trong đồ thị.
- Truy vấn viết bằng tiếng Anh vì CLIP dùng chung chỉ học tiếng Anh; tên riêng giữ tiếng Việt.
- Data card: `data/processed/vn_heritage_core/retrieval/card.json` (và `split-<n>/card.json` cho từng cách chia claim). Thống kê bản lõi: `data/processed/vn_heritage_core/stats.json`.

### Cách hoạt động

1. Mỗi thực thể khởi đầu bằng vector CLIP text của tên. GraphSAGE 3 tầng có vector quan hệ (`src/models/kg_graphsage.py`) trộn thông tin dọc theo cạnh và ngược cạnh bằng hai bộ trọng số riêng.
2. Huấn luyện bằng InfoNCE: vector của địa điểm phải gần vector CLIP của câu truy vấn. Khi huấn luyện một truy vấn train, cạnh nói đúng sự kiện của nó bị bỏ khỏi đồ thị, vì lúc test sự kiện được hỏi không bao giờ có trong đồ thị.
3. Khi truy vấn: gallery là 772 ảnh. Một ảnh nhận điểm đồ thị của địa điểm mà nó mô tả, rồi xếp hạng lại theo công thức trên. α và k chọn trên validation.
4. Mỗi cặp (truy vấn, địa điểm trả lời) có một hạng: hạng của ảnh tốt nhất của địa điểm đó giữa ảnh của các địa điểm sai (`src/evaluation/image_gallery.py`).

### Kết quả

Tập test, gallery 772 ảnh. Trung bình ± độ lệch chuẩn qua 10 cách chia claim (seed 1–10); mỗi cách chia là trung bình của 5 seed huấn luyện (13, 37, 97, 211, 503). Đơn vị %.

| Phương pháp | Nhóm truy vấn | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | trực tiếp | 4,9 | 24,4 | 31,7 | 14,2 |
| CLIP ViT-B/32 | gián tiếp | 6,7 ± 1,7 | 17,8 ± 2,7 | 25,1 ± 3,2 | 13,3 ± 1,8 |
| **CLIP + GraphSAGE**, đồ thị Wikidata | trực tiếp | 5,9 ± 1,8 | 16,4 ± 6,7 | 24,3 ± 8,7 | 12,6 ± 3,1 |
| **CLIP + GraphSAGE**, đồ thị Wikidata | gián tiếp | 8,6 ± 1,0 | 17,3 ± 2,8 | 24,4 ± 3,9 | 14,1 ± 1,5 |

Cùng các vector thực thể, chấm trên gallery 160 địa điểm (điểm của địa điểm là điểm ảnh tốt nhất của nó):

| Phương pháp | R@1 gián tiếp | R@10 gián tiếp | R@10 trực tiếp |
|---|---|---|---|
| CLIP ViT-B/32 | 6,7 ± 1,7 | 30,5 ± 3,8 | 34,2 |
| CLIP + GraphSAGE, đồ thị Wikidata | 8,6 ± 0,9 | 35,8 ± 2,8 | 41,1 ± 3,7 |

Cách đọc bảng:

- **Chấm theo ảnh, đồ thị không vượt CLIP ở R@10** trên truy vấn gián tiếp: −0,7 ± 1,4 (4/6) điểm (trong ngoặc là số cách chia thắng/thua). Chấm theo địa điểm thì hơn CLIP +5,3 ± 1,1 (10/0) điểm.
- R@1 gián tiếp tăng +1,9 ± 0,5 (8/2) điểm, nhưng bản bỏ hết cạnh cũng tăng ngang vậy (bảng ablation), nên phần này không đến từ cạnh.
- CLIP gần như không biết các địa điểm này: gõ đúng tên chỉ đưa một ảnh đúng lên hạng 1 ở 4,9% truy vấn.
- α chọn trên validation có trung vị 0,70; k là cả gallery ở 41/50 lần chạy; 4/50 lần chạy chọn α = 1, tức bỏ hẳn điểm đồ thị.
- Các tham số huấn luyện lấy nguyên của MKG-W, chưa tinh chỉnh. Huấn luyện không tái lập từng bit: chạy lại có thể lệch dưới 1 điểm.

### Vì sao theo ảnh kém hơn theo địa điểm

Mọi ảnh của một địa điểm nhận cùng điểm đồ thị, nên một địa điểm sai được đồ thị đẩy lên sẽ kéo theo tất cả ảnh của nó. Đếm trên mọi cặp (truy vấn gián tiếp, địa điểm trả lời) của test, gộp 10 cách chia × 5 seed (`scripts/analyze_vn_heritage_image_crowding.py`):

| Điểm xếp hạng | Địa điểm sai đứng trên đáp án | Ảnh sai đứng trên đáp án | Ảnh mỗi địa điểm sai | Đáp án trong 10 ảnh đầu | Đáp án trong 10 địa điểm đầu |
|---|---|---|---|---|---|
| CLIP | 42,2 | 96,6 | 2,29 | 25,0 | 30,4 |
| CLIP + GraphSAGE | 37,8 | 122,0 | 3,23 | 24,4 | 35,6 |
| CLIP + GraphSAGE, điểm đồ thị chỉ cho ảnh tốt nhất của mỗi địa điểm | 37,8 | 52,3 | 1,38 | 35,2 | 35,6 |

- Đồ thị làm giảm số địa điểm sai đứng trên đáp án, nhưng mỗi địa điểm sai giờ mang theo nhiều ảnh hơn, nên số ảnh sai đứng trên lại tăng.
- Dòng cuối là phép thử chẩn đoán, không phải phương pháp được báo cáo: α và k của nó lấy lại từ dòng thứ hai chứ không chọn riêng trên validation. Khi mỗi địa điểm chỉ có một ảnh được cộng điểm đồ thị, kết quả theo ảnh trùng với kết quả theo địa điểm.
- Hệ quả: gallery địa điểm tương đương với việc trả về mỗi địa điểm một ảnh (ảnh khớp truy vấn nhất). Hai cách chấm chỉ khác nhau ở chỗ các ảnh trùng địa điểm có được tính là kết quả riêng hay không.

### Ablation: bỏ cạnh, cạnh ngẫu nhiên, bỏ nhãn quan hệ, không hướng

Cùng bộ phép can thiệp với cấp độ 1. Mỗi biến thể được huấn luyện lại trên 10 cách chia × 5 seed, α và k chọn riêng trên validation. Truy vấn gián tiếp, tập test. "So với bản đầy đủ" là chênh lệch ghép cặp bản đầy đủ − biến thể ± sai số chuẩn qua 10 cách chia, trong ngoặc là số cách chia thắng/thua.

| Biến thể | R@1 theo ảnh | R@10 theo ảnh | R@10 so với CLIP | Đầy đủ − biến thể, R@10 theo ảnh | R@10 theo địa điểm | Đầy đủ − biến thể, R@10 theo địa điểm | R@10 theo địa điểm, chỉ dùng điểm đồ thị |
|---|---|---|---|---|---|---|---|
| Đầy đủ (đồ thị Wikidata) | 8,6 ± 1,0 | 24,4 ± 3,9 | −0,7 |  | 35,8 ± 2,8 |  | 32,4 |
| Bỏ hết cạnh (`no_edges`) | 8,8 ± 0,8 | 23,3 ± 5,1 | −1,8 | +1,2 ± 1,8 (6/4) | 34,7 ± 2,4 | +1,1 ± 1,1 (7/3) | 31,7 |
| Cạnh ngẫu nhiên (`claims_shuffled`) | 9,0 ± 1,0 | 21,7 ± 4,3 | −3,4 | +2,8 ± 1,5 (8/2) | 31,4 ± 2,5 | +4,4 ± 1,0 (9/1) | 27,5 |
| Bỏ nhãn quan hệ (`claims_untyped`) | 8,4 ± 1,5 | 24,6 ± 3,1 | −0,5 | −0,2 ± 1,0 (5/4) | 34,8 ± 2,1 | +1,0 ± 0,4 (8/2) | 29,6 |
| Không hướng (`claims_undirected`) | 8,7 ± 1,0 | 24,9 ± 4,0 | −0,2 | −0,5 ± 1,2 (4/6) | 36,8 ± 2,1 | −1,0 ± 0,8 (4/6) | 31,5 |

- **Chỉ phép cạnh ngẫu nhiên tách được khỏi bản đầy đủ:** +2,8 ± 1,5 (8/2) điểm R@10 theo ảnh, +4,4 ± 1,0 (9/1) theo địa điểm. Việc "ai nối với ai" mang tín hiệu.
- **Bỏ hết cạnh gần ngang bản đầy đủ:** +1,2 ± 1,8 (6/4) theo ảnh, +1,1 ± 1,1 (7/3) theo địa điểm, đều trong khoảng hai sai số chuẩn. Phần lớn kết quả đến từ vector tên của thực thể và việc huấn luyện trên truy vấn. Cấp độ 1 cũng cho thấy điều tương tự: danh sách vật thể gánh phần lớn, cạnh góp ít.
- **Nhãn quan hệ và hướng cạnh gần như không được dùng:** theo ảnh, bỏ nhãn quan hệ −0,2 ± 1,0 (5/4), bỏ hướng −0,5 ± 1,2 (4/6). Theo địa điểm, bỏ nhãn quan hệ làm giảm nhẹ nhưng đều (+1,0 ± 0,4, 8/2); bỏ hướng thì không (−1,0 ± 0,8, 4/6).
- **Cạnh sai gây hại:** cạnh ngẫu nhiên kém cả bản bỏ hết cạnh (21,7% so với 23,3% R@10 theo ảnh).
- Sai số chuẩn lạc quan hơn thực tế vì 10 cách chia dùng chung 160 địa điểm và claim chồng lấn. Cột cuối chấm theo địa điểm vì trên gallery ảnh mọi ảnh của một địa điểm có cùng điểm đồ thị, nên chỉ dùng điểm đồ thị thì các ảnh hoà nhau.

### Ablation: thêm cạnh rút từ bài Wikipedia

Đồ thị Wikidata cộng 902 cạnh rút từ liên kết trong bài, dưới bốn cách đặt tên quan hệ. Gallery ảnh, tập test; cột thứ tư là chênh lệch ghép cặp so với đồ thị Wikidata.

| Đồ thị | R@1 gián tiếp | R@10 gián tiếp | So với đồ thị Wikidata, R@10 | R@10 trực tiếp |
|---|---|---|---|---|
| Đồ thị Wikidata, không có cạnh từ bài | 8,6 ± 1,0 | 24,4 ± 3,9 | | 24,3 ± 8,7 |
| + cạnh từ bài, `generic` | 8,4 ± 0,8 | 23,9 ± 4,9 | −0,5 ± 1,0 (3/7) | 25,7 ± 9,1 |
| + cạnh từ bài, `cue` | 8,6 ± 1,0 | 24,5 ± 5,2 | +0,0 ± 1,3 (5/5) | 25,3 ± 9,1 |
| + cạnh từ bài, `kind` | 8,6 ± 1,1 | 23,9 ± 4,3 | | 25,2 ± 7,2 |
| + cạnh từ bài, `section` | 8,5 ± 0,9 | 23,5 ± 3,2 | | 24,2 ± 7,4 |
| + cạnh từ bài, `cue`, cạnh ngẫu nhiên | 9,0 ± 1,2 | 21,9 ± 4,0 | | 24,9 ± 7,9 |

- Thêm cạnh từ bài không đổi gì đo được trên truy vấn gián tiếp.
- Bốn sơ đồ đặt tên không tách được nhau: chênh lệch nằm trong sai số chuẩn.
- Mẫu 100 cạnh do mô hình ngôn ngữ chấm (chưa có người kiểm): đích và loại đích đúng 98/100, nhưng quan hệ đoán bằng từ khoá chỉ đúng 11/21.
- Giới hạn: cạnh từ bài trùng cặp với một claim đã bị bỏ khi build, nên chưa thử vai trò "bài nhắc lại đúng sự kiện bị hỏi".

### Mốc phụ: AdaMF-MAT

Mô hình dự đoán liên kết của cấp độ 2, huấn luyện trên các triple của đồ thị. Chỉ chạy 3 cách chia claim (1, 2, 7) × 5 seed, chấm theo địa điểm, truy vấn gián tiếp trên test.

| Phương pháp | R@1 | R@10 | MRR dự đoán liên kết |
|---|---|---|---|
| CLIP | 7,1 | 30,8 | |
| CLIP + GraphSAGE, đồ thị Wikidata | 8,5 | 37,2 |  |
| CLIP + AdaMF-MAT, đồ thị Wikidata | 11,0 | 40,1 | 0,292 |
| CLIP + AdaMF-MAT, cạnh ngẫu nhiên | 5,5 | 29,0 | 0,080 |
| CLIP + AdaMF-MAT, + cạnh từ bài `cue` | 11,4 | 48,9 | 0,315 |
| CLIP + RotatE (bỏ ảnh và văn bản) | 6,5 | 26,6 | 0,084 |

Không so sánh ngang được với GraphSAGE: AdaMF-MAT được cho sẵn thực thể neo và quan hệ của truy vấn lấy từ metadata, còn GraphSAGE chỉ nhìn vector văn bản của câu truy vấn. Lượt chạy dừng ở 3 cách chia vì hướng này lệch khỏi bài toán truy hồi ảnh.

### Chưa có

- Fidelity, sparsity và validity của giải thích trên bộ này (cấp độ 1 đã có trên Visual Genome ∩ COCO).
- Người kiểm tra tay ảnh có đúng địa điểm không, và kiểm lại mẫu 100 cạnh rút từ bài.
- Tinh chỉnh siêu tham số; backbone đa ngữ để truy vấn bằng tiếng Việt.
- Huấn luyện bộ mã hoá trực tiếp trên ảnh. Hiện bộ mã hoá học xếp hạng địa điểm, chỉ bước chấm điểm là theo ảnh.

### Chạy lại

Cần thư mục ảnh của bản lõi (772 file, không nằm trong repo); đặt đường dẫn ở `configs/data/vn_heritage_queries.yaml` hoặc truyền `--image-dir`.

```bash
python -m scripts.build_vn_heritage_queries --split-seed 1                      # truy vấn, triple và tầng mật độ của một cách chia claim
python -m scripts.run_vn_heritage_retrieval --split-seed 1 --stage embed        # vector CLIP của ảnh, truy vấn, tên thực thể
python -m scripts.run_vn_heritage_retrieval --split-seed 1 --stage clip         # baseline CLIP, cả hai gallery
python -m scripts.run_vn_heritage_retrieval --split-seed 1 --stage graphsage \
    --graphs claims no_edges claims_shuffled claims_untyped claims_undirected   # mô hình chính và ablation
python -m scripts.run_vn_heritage_retrieval --split-seed 1 --stage graphsage \
    --graphs generic kind section cue cue_shuffled                              # ablation cạnh rút từ bài
python -m scripts.summarize_vn_heritage_splits --split-seeds 1 2 3 4 5 6 7 8 9 10 --gallery image
python -m scripts.analyze_vn_heritage_image_crowding --split-seeds 1 2 3 4 5 6 7 8 9 10
```

Mỗi cách chia mất khoảng 10–15 phút trên CPU. Kết quả ghi vào `runs/vn_heritage_core/split-<n>/` (Git-ignore): `benchmark_images.json` cho gallery ảnh, `benchmark.json` cho gallery địa điểm; bảng tổng hợp ở `runs/vn_heritage_core/splits_summary_<gallery>_test.json`.

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
