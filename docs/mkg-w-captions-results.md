# MKG-W caption subset: MKG-W theo khuôn VG ∩ COCO

Một tập con của MKG-W được dựng lại để có đúng ba thành phần như VG ∩ COCO: mỗi mục có một ảnh, caption do người viết và một đồ thị của riêng nó. Mục đích là có một dataset thứ hai đúng nghĩa text-to-image, chạy trên nguyên pipeline của dataset thứ nhất:

```
điểm = α · cosine_CLIP + (1 − α) · điểm_đồ_thị
```

Bản MKG-W đầy đủ, với câu hỏi gián tiếp và anchor, ở [mkg-w-results.md](mkg-w-results.md). Ở bản đó ảnh gần như không đóng góp vào câu hỏi gián tiếp (α = 0,15–0,20); tập con này được dựng để bù chỗ đó.

## Kết quả

Tập test: 441 mục, 882 câu truy vấn, mỗi câu có đúng một mục đúng. α và số ứng viên k được chọn trên tập validation; tập test chỉ được chấm sau khi đã chốt tham số. Mô hình và tham số huấn luyện giữ nguyên như VG ∩ COCO, chưa tinh chỉnh.

| Phương pháp | R@1 | R@5 | R@10 | MRR | α | k |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | 35,60 | 56,58 | 65,76 | 45,73 | 1 | không dùng |
| CLIP + so khớp từ với đồ thị | **45,01** | 67,91 | 75,40 | 55,61 | 0,75 | 200 |
| CLIP + GraphSAGE | 43,35 ± 0,54 | **70,82 ± 1,03** | **80,76 ± 1,71** | **56,10 ± 0,52** | 0,51 / 0,59 / 0,72 | 440 / 440 / 440 |

Cột α và k ghi giá trị được chọn trên validation cho từng seed, theo thứ tự 13 / 37 / 97. Dòng GraphSAGE là trung bình ± độ lệch chuẩn qua 3 seed. Đơn vị: %.

Khi chỉ dùng điểm đồ thị, không có CLIP:

| Phương pháp | R@1 | R@10 |
|---|---|---|
| So khớp từ với đồ thị | 15,19 | 43,20 |
| GraphSAGE | 28,19 ± 1,08 | 73,05 ± 0,99 |
| Mức ngẫu nhiên | 0,23 | 2,27 |

Cách đọc bảng:

- CLIP một mình đạt 35,6 và α nằm trong khoảng 0,51–0,72: ảnh và đồ thị cùng đóng góp vào điểm cuối.
- GraphSAGE nâng R@1 của CLIP thêm 7,8 điểm. Ở VG ∩ COCO mức tăng là 4,3 điểm.
- So khớp từ hơn GraphSAGE 1,7 điểm ở R@1, ngược với VG ∩ COCO. Caption thường nhắc nguyên văn tên hàng xóm ("directed by Rudolph Maté") và đồ thị chứa đúng tên đó, nên khớp chữ đã lấy được phần lớn lợi ích. GraphSAGE hơn ở R@5, R@10 và MRR.
- Độ lệch giữa các seed ở R@1 là 0,54, lớn hơn nhiều so với 0,05 ở VG ∩ COCO. Với 882 câu truy vấn, khoảng cách 1,7 điểm giữa hai phương pháp đồ thị chưa đủ chắc để xếp hạng chúng.

File kết quả: `runs/mkg_w_captions/benchmark/*.json`. Lịch sử huấn luyện từng seed: `runs/mkg_w_captions/graphsage/directed-learned-l3-h512/seed<n>/history.json`.

### Chưa có

- Ablation bỏ cạnh và cạnh ngẫu nhiên trên tập con này.
- Biến thể đỉnh trung tâm mang vector CLIP của ảnh.
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
python -m scripts.build_mkg_w_captions                                                              # dựng tập con
python -m scripts.embed_vg_coco --config configs/retrieval/mkg_w_captions.yaml                      # vector CLIP
python -m scripts.run_benchmark --method clip --config configs/retrieval/mkg_w_captions.yaml        # baseline CLIP
python -m scripts.run_benchmark --method graph_lexical --config configs/retrieval/mkg_w_captions.yaml
python -m scripts.train_graphsage --config configs/retrieval/mkg_w_captions.yaml                    # 3 seed
python -m scripts.run_benchmark --method graphsage --config configs/retrieval/mkg_w_captions.yaml
```

File config dữ liệu mang vài mục của khuôn VG ∩ COCO (`sources`, `join`, `normalize`, `download`, `graph`) mà dataset này không dùng; chúng có mặt vì các script trên đọc file theo khuôn đó.
