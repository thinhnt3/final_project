# Kết quả trên VN Heritage (MMKG tự xây, bản lõi)

Bộ dữ liệu tự xây về địa điểm và di tích Việt Nam: mỗi địa điểm có ảnh từ Wikimedia Commons và tri thức từ Wikidata và Wikipedia tiếng Việt. Bài toán vẫn là truy hồi ảnh bằng văn bản với cùng CLIP ViT-B/32 đóng băng và cùng công thức `α · cosine_CLIP + (1 − α) · điểm_đồ_thị`. Khác với Visual Genome ∩ COCO, ở đây **không có đồ thị cảnh**: đồ thị là một KG dùng chung, và ảnh nối vào KG qua cạnh "ảnh – mô tả – địa điểm".

Báo cáo đủ bảng về cách xây bộ dữ liệu (bản đầy đủ và bản lõi), cấu trúc đồ thị, rò rỉ và giấy phép: [plans/reports/vn-heritage-261120-0033-mmkg-report.html](../plans/reports/vn-heritage-261120-0033-mmkg-report.html) (mở bằng trình duyệt).

## Dữ liệu

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

## Cách hoạt động

1. Mỗi thực thể khởi đầu bằng vector CLIP text của tên. GraphSAGE 3 tầng có vector quan hệ (`src/models/kg_graphsage.py`) trộn thông tin dọc theo cạnh và ngược cạnh bằng hai bộ trọng số riêng.
2. Huấn luyện bằng InfoNCE: vector của địa điểm phải gần vector CLIP của câu truy vấn. Khi huấn luyện một truy vấn train, cạnh nói đúng sự kiện của nó bị bỏ khỏi đồ thị, vì lúc test sự kiện được hỏi không bao giờ có trong đồ thị.
3. Khi truy vấn: gallery là 772 ảnh. Một ảnh nhận điểm đồ thị của địa điểm mà nó mô tả, rồi xếp hạng lại theo công thức trên. α và k chọn trên validation.
4. Mỗi cặp (truy vấn, địa điểm trả lời) có một hạng: hạng của ảnh tốt nhất của địa điểm đó giữa ảnh của các địa điểm sai (`src/evaluation/image_gallery.py`).

## Kết quả

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

## Vì sao theo ảnh kém hơn theo địa điểm

Mọi ảnh của một địa điểm nhận cùng điểm đồ thị, nên một địa điểm sai được đồ thị đẩy lên sẽ kéo theo tất cả ảnh của nó. Đếm trên mọi cặp (truy vấn gián tiếp, địa điểm trả lời) của test, gộp 10 cách chia × 5 seed (`scripts/analyze_vn_heritage_image_crowding.py`):

| Điểm xếp hạng | Địa điểm sai đứng trên đáp án | Ảnh sai đứng trên đáp án | Ảnh mỗi địa điểm sai | Đáp án trong 10 ảnh đầu | Đáp án trong 10 địa điểm đầu |
|---|---|---|---|---|---|
| CLIP | 42,2 | 96,6 | 2,29 | 25,0 | 30,4 |
| CLIP + GraphSAGE | 37,8 | 122,0 | 3,23 | 24,4 | 35,6 |
| CLIP + GraphSAGE, điểm đồ thị chỉ cho ảnh tốt nhất của mỗi địa điểm | 37,8 | 52,3 | 1,38 | 35,2 | 35,6 |

- Đồ thị làm giảm số địa điểm sai đứng trên đáp án, nhưng mỗi địa điểm sai giờ mang theo nhiều ảnh hơn, nên số ảnh sai đứng trên lại tăng.
- Dòng cuối là phép thử chẩn đoán, không phải phương pháp được báo cáo: α và k của nó lấy lại từ dòng thứ hai chứ không chọn riêng trên validation. Khi mỗi địa điểm chỉ có một ảnh được cộng điểm đồ thị, kết quả theo ảnh trùng với kết quả theo địa điểm.
- Hệ quả: gallery địa điểm tương đương với việc trả về mỗi địa điểm một ảnh (ảnh khớp truy vấn nhất). Hai cách chấm chỉ khác nhau ở chỗ các ảnh trùng địa điểm có được tính là kết quả riêng hay không.

## Ablation: bỏ cạnh, cạnh ngẫu nhiên, bỏ nhãn quan hệ, không hướng

Cùng bộ phép can thiệp với hai dataset công khai (bảng ablation trong [README](../README.md#ablation-bỏ-cạnh-và-cạnh-ngẫu-nhiên)). Mỗi biến thể được huấn luyện lại trên 10 cách chia × 5 seed, α và k chọn riêng trên validation. Truy vấn gián tiếp, tập test. "So với bản đầy đủ" là chênh lệch ghép cặp bản đầy đủ − biến thể ± sai số chuẩn qua 10 cách chia, trong ngoặc là số cách chia thắng/thua.

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

## Ablation: thêm cạnh rút từ bài Wikipedia

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

## Mốc phụ: AdaMF-MAT

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

## Chưa có

- Fidelity, sparsity và validity của giải thích trên bộ này (cấp độ 1 đã có trên Visual Genome ∩ COCO).
- Người kiểm tra tay ảnh có đúng địa điểm không, và kiểm lại mẫu 100 cạnh rút từ bài.
- Tinh chỉnh siêu tham số; backbone đa ngữ để truy vấn bằng tiếng Việt.
- Huấn luyện bộ mã hoá trực tiếp trên ảnh. Hiện bộ mã hoá học xếp hạng địa điểm, chỉ bước chấm điểm là theo ảnh.

## Chạy lại

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
