# MKG-W caption subset: MKG-W theo khuôn VG ∩ COCO

Một tập con của MKG-W được dựng lại để có đúng ba thành phần như VG ∩ COCO: mỗi mục có một ảnh, caption do người viết và một đồ thị của riêng nó. Mục đích là có một dataset thứ hai đúng nghĩa text-to-image, chạy trên nguyên pipeline của dataset thứ nhất:

```
điểm = α · cosine_CLIP + (1 − α) · điểm_đồ_thị
```

Bản MKG-W đầy đủ, với câu hỏi gián tiếp và anchor, ở [mkg-w-results.md](mkg-w-results.md). Ở bản đó ảnh gần như không đóng góp vào câu hỏi gián tiếp (α = 0,15–0,20); tập con này được dựng để bù chỗ đó.

## Kết quả

Cùng bộ phương pháp, cùng lệnh và cùng tham số với bảng "So với cấp độ 1" trong [README](../README.md), chạy trên tập con này. Cột VG ∩ COCO chép từ README để đặt cạnh nhau.

Tập test của tập con: 441 mục, 882 câu truy vấn, mỗi câu có đúng một mục đúng (VG ∩ COCO: 1.500 ảnh, 7.500 caption). α và k chọn trên validation. Trung bình ± độ lệch chuẩn qua 3 seed (13, 37, 97); các dòng ghi "5 seed" thêm seed 211 và 307. Đơn vị %. "Chỉ mô hình" là R@1 khi chỉ dùng điểm của mô hình, không cộng CLIP.

| Phương pháp | VG ∩ COCO R@1 | R@10 | MRR | Chỉ mô hình | Tập con R@1 | R@10 | MRR | Chỉ mô hình |
|---|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | 43,89 | 82,64 | 56,76 | | 35,60 | 65,76 | 45,73 | |
| CLIP + so khớp từ với đồ thị | 46,41 | 85,41 | 59,29 | 12,63 | **45,01** | 75,40 | 55,61 | 15,19 |
| CLIP + GraphSAGE (đỉnh ảnh học được) | 48,12 ± 0,23 | 83,87 ± 2,12 | 60,42 ± 0,44 | 27,54 | 43,35 ± 0,54 | 80,76 ± 1,71 | **56,10 ± 0,52** | 28,19 |
| CLIP + GraphSAGE (đỉnh ảnh mang CLIP) | 49,43 ± 0,46 | 87,52 ± 0,69 | 62,22 ± 0,37 | 42,25 | 39,98 ± 0,40 | 77,63 ± 0,58 | 52,35 ± 0,29 | 28,95 |
| CLIP + linear adapter (chỉ modality image, không đồ thị) | 49,76 ± 0,08 | 87,33 ± 0,42 | 62,44 ± 0,13 | 46,25 | 35,75 ± 0,43 | 67,01 ± 2,16 | 46,46 ± 0,39 | 23,39 |
| CLIP + AdaMF-MAT (bản đầy đủ của bài báo) | 51,42 ± 0,32 | 88,91 ± 0,19 | 63,98 ± 0,31 | 48,48 | 41,16 ± 0,59 | 77,89 ± 0,20 | 53,04 ± 0,30 | 31,52 |
| CLIP + AdaMF, không MAT, chuẩn hoá L2 từng modality (5 seed) | 51,93 ± 0,32 | 89,23 ± 0,27 | 64,47 ± 0,24 | 48,62 | 41,70 ± 0,92 | 79,73 ± 0,22 | 54,13 ± 0,58 | 34,29 |
| + structure-only dropout 30% (`so0.3`, 5 seed) | 52,74 ± 0,26 | 89,68 ± 0,26 | 65,19 ± 0,20 | 49,77 | 43,29 ± 0,65 | **81,22 ± 0,65** | 55,87 ± 0,26 | **37,07** |
| + structure-only loss, trọng số 0,1 (`sl0.1`, 5 seed) | **53,1** | | | | 43,08 ± 0,59 | 81,16 ± 0,66 | 55,51 ± 0,28 | 36,42 |

Dòng so khớp từ của VG ∩ COCO không có trong bảng của README; số lấy từ `runs/benchmark/graph_lexical.json`.

R@5 trên tập con, theo thứ tự các dòng: 56,58 · 67,91 · 70,82 ± 1,03 · 66,89 ± 0,41 · 58,16 ± 0,34 · 66,97 ± 1,09 · 68,71 ± 0,53 · 70,61 ± 0,47 · 70,48 ± 0,34.

α và k được chọn trên tập con:

| Phương pháp | α theo seed | k |
|---|---|---|
| So khớp từ | 0,75 | 200 |
| GraphSAGE (đỉnh ảnh học được) | 0,51 / 0,59 / 0,72 | 440 |
| GraphSAGE (đỉnh ảnh mang CLIP) | 0,61 / 0,54 / 0,48 | 440 |
| Linear adapter | 0,73 / 0,69 / 0,81 | 10 / 10 / 440 |
| AdaMF-MAT đầy đủ | 0,66 / 0,59 / 0,63 | 440 |
| AdaMF không MAT, L2 | 0,51 / 0,55 / 0,54 / 0,55 / 0,50 | 440 |
| so0.3 | 0,48 / 0,46 / 0,41 / 0,42 / 0,58 | 440 |
| sl0.1 | 0,48 / 0,61 / 0,53 / 0,49 / 0,46 | 440 |

Khi chỉ dùng điểm đồ thị, không có CLIP (tập con):

| Phương pháp | R@1 | R@10 |
|---|---|---|
| So khớp từ với đồ thị | 15,19 | 43,20 |
| GraphSAGE (đỉnh ảnh học được) | 28,19 ± 1,08 | 73,05 ± 0,99 |
| Mức ngẫu nhiên | 0,23 | 2,27 |

Cách đọc bảng, đặt cạnh VG ∩ COCO:

- **Modality image gần như không giúp ở tập con.** Linear adapter chỉ hơn CLIP 0,15 điểm (35,75 so với 35,60); ở VG ∩ COCO nó hơn 5,9 điểm. Học lại vector ảnh không thay được thông tin mà ảnh không chứa.
- **Phần tăng ở tập con đến từ đồ thị.** GraphSAGE đỉnh ảnh học được, vốn không nhìn ảnh, hơn CLIP 7,8 điểm. Ở VG ∩ COCO, README kết luận ngược lại: gần như toàn bộ mức tăng của AdaMF đến từ modality image.
- **Đưa CLIP của ảnh vào đỉnh ảnh làm kém đi** (39,98 so với 43,35), trong khi ở VG ∩ COCO nó giúp (49,43 so với 48,12).
- **AdaMF-MAT không vượt GraphSAGE ở tập con** (41,16 so với 43,35). Ở VG ∩ COCO nó hơn 3,3 điểm.
- **MAT không giúp, giống VG ∩ COCO:** bỏ MAT và chuẩn hoá L2 được 41,70, trong khoảng dao động của bản đầy đủ.
- **Ép modality structure học riêng giúp ở cả hai dataset:** so0.3 thêm 1,6 điểm ở tập con (41,70 → 43,29), 0,8 điểm ở VG ∩ COCO. Điểm chỉ-mô-hình cao nhất cũng thuộc so0.3 (37,07).
- **So khớp từ có R@1 cao nhất ở tập con** (45,01). Caption hay nhắc nguyên văn tên hàng xóm và đồ thị chứa đúng tên đó. Các mô hình học hơn nó ở R@10 và MRR.
- **Độ lệch giữa các seed lớn hơn VG ∩ COCO** (0,4–0,9 so với 0,1–0,5 ở R@1) vì tập nhỏ. Các dòng từ 41 đến 43,4 cách nhau ít hơn hai lần độ lệch, nên thứ tự giữa GraphSAGE, so0.3 và sl0.1 chưa chắc.

### Vì sao AdaMF-MAT không vượt GraphSAGE ở tập con

AdaMF-MAT không vượt GraphSAGE ở tập con (41,16 so với 43,35), trái với VG ∩ COCO. Ba phép đo dưới đây chỉ ra nguyên nhân: modality image chiếm chỗ trong vector trộn mà không mang thêm thông tin.

| Cấu hình trên tập con (3 seed; dòng đầu 5 seed) | R@1 sau khi xếp hạng lại | Chỉ điểm mô hình, R@1 |
|---|---|---|
| AdaMF, không MAT, L2 | 41,70 ± 0,92 | 34,29 ± 1,34 |
| bỏ image | **44,37 ± 0,68** | 31,25 ± 0,26 |
| bỏ structure | 41,12 ± 0,80 | 32,62 ± 2,11 |
| bỏ text | 40,44 ± 1,62 | 31,97 ± 1,71 |
| GraphSAGE (đỉnh ảnh học được), để so | 43,35 ± 0,54 | 28,19 ± 1,08 |

- **Ảnh ở tập con không chứa điều caption hỏi.** Linear adapter, tức chỉ học lại vector CLIP của ảnh, đạt 35,75, ngang CLIP 35,60. Ở VG ∩ COCO nó đạt 49,76, hơn CLIP 5,9 điểm, vì caption COCO tả đúng thứ có trong ảnh.
- **AdaMF vẫn dành khoảng 40% vector trộn cho image.** Trọng số trung bình của ba modality trong bản L2 là structure 0,21, image 0,40, text 0,39 (`scripts/adamf_weight_stats.py`). Phần 40% đó lặp lại điều CLIP đã biết.
- **Điểm trùng với CLIP thì cộng vào CLIP không thêm gì.** Xét riêng điểm mô hình, AdaMF cao hơn GraphSAGE (34,29 so với 28,19). Nhưng điểm cuối là α·CLIP + (1 − α)·điểm mô hình, và điểm của AdaMF đã lẫn CLIP ảnh nên bổ sung cho CLIP ít hơn. GraphSAGE không nhìn ảnh, điểm của nó độc lập với CLIP nên cộng vào được nhiều hơn.
- **Bỏ image là đủ để AdaMF vượt GraphSAGE.** Điểm mô hình giảm (34,29 xuống 31,25) nhưng R@1 sau khi xếp hạng lại tăng 2,7 điểm, lên 44,37, hơn GraphSAGE 1,0 điểm. Ở VG ∩ COCO, bỏ image làm tụt 3,9 điểm.
- **Bản đầy đủ của bài báo còn thêm một vấn đề đã thấy ở VG ∩ COCO:** không chuẩn hoá thì trọng số structure chỉ khoảng 0,004 trong khi vector của nó dài gấp khoảng 40 lần hai modality kia, phần đóng góp thật khoảng 15%.

Giới hạn của kết luận: các dòng ablation chạy 3 seed trên 882 câu truy vấn, và khoảng cách 1,0 điểm giữa "bỏ image" và GraphSAGE nằm trong tầm hai lần độ lệch giữa các seed. Điều chắc hơn là chiều thay đổi: bỏ image làm tăng 2,7 điểm ở tập con và làm giảm 3,9 điểm ở VG ∩ COCO.

File kết quả: `runs/mkg_w_captions/benchmark/*.json`. Lịch sử huấn luyện từng seed: `runs/mkg_w_captions/<graphsage hoặc adamf_mat>/<tên lượt chạy>/seed<n>/history.json`.

### Chưa có

- Ablation bỏ cạnh và cạnh ngẫu nhiên trên tập con này.
- Ablation trộn bằng trung bình của AdaMF-MAT và bộ relation queries.
- Lát cắt chỉ gồm phim (khoảng 1.300 mục) để báo cáo riêng.
- Giải thích bằng quan hệ và fidelity.

## Dữ liệu

| | train | validation | test |
|---|---|---|---|
| Mục | 2.055 | 440 | 441 |
| Câu truy vấn | 4.110 | 880 | 882 |

Đối chiếu với VG ∩ COCO:

| | VG ∩ COCO | MKG-W caption subset |
|---|---|---|
| Mục trong gallery | 1 ảnh | 1 thực thể, đại diện bằng ảnh đầu tiên của nó |
| Caption | 5 caption COCO, người viết khi nhìn ảnh | 2 câu đầu của mô tả Wikipedia, người viết về thực thể |
| Đồ thị của mục | scene graph của ảnh, người gán nhãn | các triple Wikidata chạm vào thực thể |
| Đỉnh | vật thể trong ảnh | thực thể hàng xóm, và một đỉnh "this entity" cho chính mục đó |
| Quy mô | 10.000 mục, 50.000 caption | 2.936 mục, 5.872 caption |

### Cách thiết lập

Dựng từ bản MKG-W đã xử lý (`data/processed/mkg_w`), không tải thêm gì.

1. **Chọn mục.** Một thực thể thành một mục khi có ảnh, có mô tả, có ít nhất 5 triple trong đồ thị tri thức, và mô tả cho đủ 2 caption. Còn 2.936 thực thể.
2. **Ảnh.** Lấy ảnh đầu tiên của thực thể, lưu dạng JPEG với cạnh dài nhất 512 pixel.
3. **Caption.** Hai câu đầu của mô tả. Tên thực thể, cả dạng đầy đủ lẫn dạng bỏ phần trong ngoặc ("Signs (film)" và "Signs"), được thay bằng "this entity", để caption không khớp được với mục chỉ nhờ tiêu đề. Câu ngắn hơn 40 ký tự bị bỏ. Mọi mục có đúng 2 caption vì bước huấn luyện đòi số caption bằng nhau.
4. **Đồ thị của mục.** Mọi triple chạm vào thực thể, giữ nguyên chiều. Chính thực thể được thay bằng một đỉnh chung mang chữ "this entity"; hàng xóm mang tên đầy đủ; nhãn cạnh là tên quan hệ. Dùng triple của cả ba phần train / validation / test của đồ thị tri thức: đồ thị ở đây mô tả mục, như scene graph mô tả ảnh, không có sự kiện nào bị giấu để dự đoán.
5. **Chia.** Ngẫu nhiên theo mục, 70 / 15 / 15, seed 13.

Từ vựng đồ thị có 11.153 khái niệm (9.404 xuất hiện ở mục train) và 144 quan hệ. Khái niệm chỉ có ở validation hoặc test vẫn được giữ: đặc trưng của đỉnh là vector CLIP text đóng băng, không có tham số học riêng cho từng khái niệm. Trung vị 7 triple mỗi mục.

Thông số: `configs/data/mkg_w_captions.yaml`. Thống kê sau khi dựng: `data/splits/mkg_w_captions_card.yaml`.

### Giới hạn của cách thiết lập

- Che tên chưa kín. Câu thứ hai hay dùng họ ("Carey is the first artist…") hoặc tên khác của thực thể.
- Câu thứ hai thường kém vai trò caption ("Initially, no sequel was planned…") và không nói thực thể là gì.
- Ảnh không chứa đồ thị. Áp phích phim không cho thấy ai sản xuất phim; đồ thị bổ sung thông tin cho CLIP chứ không mô tả lại ảnh như scene graph.
- Một số quan hệ không có nhãn chữ trong dữ liệu và hiện mã Wikidata (ví dụ "P172"), vô nghĩa với CLIP text.
- Vài mục là đỉnh bậc rất cao; đồ thị lớn nhất có 1.319 triple.
- Ảnh thu thập tự động theo tên nên có ảnh sai thực thể.
- Tập test nhỏ: 441 mục, mỗi câu truy vấn đổi 0,11 điểm R@1.

## Chạy lại

Cần bản MKG-W đã dựng (`python -m scripts.build_mkg_w --stage all`). Các lệnh sau `build_mkg_w_captions` là script của VG ∩ COCO, chạy nguyên với config của tập con.

```bash
C=configs/retrieval/mkg_w_captions.yaml
C5=configs/retrieval/mkg_w_captions_5seeds.yaml

python -m scripts.build_mkg_w_captions                                   # dựng tập con
python -m scripts.embed_vg_coco --config $C                              # vector CLIP
python -m scripts.run_benchmark --method clip --config $C
python -m scripts.run_benchmark --method graph_lexical --config $C
python -m scripts.train_graphsage --config $C                            # GraphSAGE, 3 seed
python -m scripts.run_benchmark --method graphsage --config $C
python -m scripts.train_graphsage --config $C --image-node clip_image    # đỉnh ảnh mang CLIP
python -m scripts.train_graphsage --config $C --model adamf_mat --no-mat --drop-modality structure --drop-modality text   # linear adapter
python -m scripts.run_benchmark --method adamf_mat --run adamf-d256-no-structure-no-text --config $C
python -m scripts.train_graphsage --config $C --model adamf_mat          # AdaMF-MAT đầy đủ
python -m scripts.run_benchmark --method adamf_mat --run adamf-mat-d256-mu1 --config $C
python -m scripts.train_graphsage --config $C5 --model adamf_mat --no-mat --normalize-modalities
python -m scripts.run_benchmark --method adamf_mat --run adamf-d256-l2 --config $C5
python -m scripts.train_graphsage --config $C5 --model adamf_mat --no-mat --normalize-modalities --structure-only-rate 0.3
python -m scripts.run_benchmark --method adamf_mat --run adamf-d256-l2-so0.3 --config $C5
python -m scripts.train_graphsage --config $C5 --model adamf_mat --no-mat --normalize-modalities --structure-loss-weight 0.1
python -m scripts.run_benchmark --method adamf_mat --run adamf-d256-l2-sl0.1 --config $C5
for D in image structure text; do                                        # ablation bỏ từng modality
  python -m scripts.train_graphsage --config $C --model adamf_mat --no-mat --normalize-modalities --drop-modality $D
  python -m scripts.run_benchmark --method adamf_mat --run adamf-d256-l2-no-$D --config $C
done
python -m scripts.adamf_weight_stats --run adamf-d256-l2 --config $C     # trọng số từng modality
```

Chấm biến thể đỉnh ảnh mang CLIP cần một bản config có `graphsage.image_node: clip_image`; với config mặc định, `run_benchmark --run directed-clip_image-l3-h512` dừng với lỗi "architecture/ablation does not match the requested config".

File config dữ liệu mang vài mục của khuôn VG ∩ COCO (`sources`, `join`, `normalize`, `download`, `graph`) mà dataset này không dùng; chúng có mặt vì các script trên đọc file theo khuôn đó.
