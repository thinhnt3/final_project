# MKG-W: truy hồi ảnh của thực thể bằng CLIP + đồ thị tri thức

Dataset thứ hai của đề tài. Cho một câu hỏi, hệ thống tìm thực thể có ảnh khớp nhất. CLIP chấm mọi thực thể theo ảnh của nó, sau đó đồ thị tri thức (thực thể và quan hệ giữa chúng) được dùng để xếp hạng lại:

```
điểm = α · cosine_CLIP + (1 − α) · điểm_đồ_thị
```

Khác với VG ∩ COCO, ở đây có nhóm câu hỏi gián tiếp: câu hỏi không nêu tên đáp án mà nêu một thực thể khác và một quan hệ ("the performer of Doggfather"). Thực thể được nêu tên gọi là **anchor**. Điểm đồ thị tốt nhất lấy từ mô hình lan điểm dọc theo cạnh, xuất phát từ anchor.

Bản trình bày có hình: [plans/reports/mkg-w-261120-1631-knowledge-graph-retrieval.html](../plans/reports/mkg-w-261120-1631-knowledge-graph-retrieval.html).

## Kết quả

Tập test: 9.301 thực thể có ảnh làm gallery, 5.154 câu hỏi (2.710 trực tiếp, 2.444 gián tiếp). Một câu gián tiếp có thể có nhiều đáp án đúng, nên nhóm này được chấm trên 2.626 cặp (câu hỏi, đáp án). α và số ứng viên k được chọn trên tập validation; tập test chỉ được chấm sau khi đã chốt tham số.

Câu hỏi gián tiếp (2.626 cặp):

| Phương pháp | Đọc gì từ câu hỏi | R@1 | R@10 | MRR | α | k |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | câu chữ | 2,70 | 9,63 | 5,52 | 1 | không dùng |
| CLIP + GraphSAGE (GNN theo yêu cầu của đề) | câu chữ | 5,81 ± 0,16 | 24,42 ± 0,36 | 12,30 ± 0,13 | 0,70 / 0,70 / 0,70 | 9.301 |
| CLIP + path score (không huấn luyện) | anchor | 15,61 | 35,15 | 22,58 | 0,65 | 9.301 |
| CLIP + RotatE (chỉ cấu trúc, không ảnh và văn bản) | anchor + quan hệ | 26,18 ± 0,48 | 43,82 ± 0,63 | 32,22 ± 0,39 | 0,45 / 0,55 / 0,45 | 9.301 |
| CLIP + AdaMF-MAT (phương pháp 2024) | anchor + quan hệ | 24,77 ± 0,42 | **52,46 ± 0,45** | **34,77 ± 0,10** | 0,20 / 0,10 / 0,25 | 9.301 |
| **CLIP + relation path scorer** (R@1 cao nhất) | câu chữ + anchor | **27,44 ± 0,12** | 46,04 ± 0,07 | 33,89 ± 0,04 | 0,15 / 0,20 / 0,15 | 9.301 |

Cột "Đọc gì từ câu hỏi" ghi mức thông tin mỗi phương pháp nhận. Anchor và quan hệ được đọc từ nhãn của câu hỏi, không phải tự tìm trong câu chữ; xem mục [Kiểm tra lại](#kiểm-tra-lại-con-số-2744).

Câu hỏi trực tiếp (2.710 cặp) và toàn bộ test (5.336 cặp), R@1:

| Phương pháp | Trực tiếp | Toàn bộ |
|---|---|---|
| CLIP ViT-B/32 | 52,51 | 28,00 |
| CLIP + GraphSAGE | 60,96 ± 0,23 | 33,82 ± 0,17 |
| CLIP + GraphSAGE, không cạnh | **87,27 ± 0,30** | **46,63 ± 0,18** |
| CLIP + path score | 52,51 | 34,35 |
| CLIP + AdaMF-MAT | 52,51 | 38,86 ± 0,20 |
| CLIP + relation path scorer | 52,51 | 40,17 ± 0,06 |

Cột α và k ghi giá trị được chọn trên validation cho từng seed, theo thứ tự 13 / 37 / 97. k = 9.301 là cả gallery: validation luôn chọn mức này vì top-100 của CLIP chỉ chứa 28% đáp án của câu hỏi gián tiếp. Bước "CLIP lọc ứng viên" vì vậy không lọc gì trên dataset này.

Các dòng có ± là trung bình ± độ lệch chuẩn qua 3 seed (13, 37, 97). Path score không có tham số học nên chỉ có một giá trị. Đơn vị: %.

Khi chỉ dùng điểm đồ thị, không có CLIP (câu hỏi gián tiếp):

| Phương pháp | R@1 | R@10 |
|---|---|---|
| GraphSAGE | 6,13 ± 0,38 | 23,29 ± 0,09 |
| Path score | 14,13 | 32,64 |
| AdaMF-MAT | 24,20 ± 0,22 | 51,80 ± 0,30 |
| Relation path scorer | 26,45 ± 0,12 | 44,71 ± 0,16 |
| Mức ngẫu nhiên | 0,01 | 0,11 |

Cách đọc bảng:

- Relation path scorer nâng R@1 của CLIP trên câu hỏi gián tiếp từ 2,7 lên 27,4. Gần như toàn bộ phần tăng đến từ đồ thị: chỉ riêng điểm đồ thị đã đạt 26,5.
- GraphSAGE mã hoá mỗi thực thể thành một vector cố định và không biết câu hỏi nhắc tới thực thể nào. Trên câu hỏi gián tiếp nó chỉ hơn CLIP 3,1 điểm.
- AdaMF-MAT được cho biết cả quan hệ của câu hỏi nhưng R@1 vẫn thấp hơn relation path scorer 2,7 điểm. Ở R@10 nó hơn 6,4 điểm: nó đưa đáp án vào nhóm đầu thường xuyên hơn nhưng ít khi đặt lên vị trí đầu.
- Ba mô hình dùng anchor không chạm vào câu hỏi trực tiếp (không có anchor thì điểm đồ thị bằng 0), nên cột trực tiếp của chúng bằng CLIP.
- Trên câu hỏi trực tiếp, mô hình tốt nhất là GraphSAGE không cạnh: nó học khớp câu hỏi với tên thực thể, không dùng đồ thị.

File kết quả: `runs/mkg_w/**/benchmark.json`. Lịch sử huấn luyện từng seed: `runs/mkg_w/<mô hình>/<lượt chạy>/seed-<n>/history.json`.

### Ablation: bỏ cạnh và cạnh ngẫu nhiên

Câu hỏi gián tiếp, R@1 sau khi xếp hạng lại. "Xáo cạnh" giữ nguyên head và quan hệ của mỗi triple, hoán vị tail.

| Phương pháp | Đồ thị thật | Xáo cạnh | Không cạnh |
|---|---|---|---|
| GraphSAGE | 5,81 ± 0,16 | 5,03 ± 0,39 | 4,70 ± 0,21 |
| Path score | 15,61 | 2,44 | không áp dụng |
| Relation path scorer | 27,44 ± 0,12 | 2,75 ± 0,05 | không áp dụng |

- Khi xáo cạnh, hai mô hình dùng anchor rơi về mức CLIP (2,70). Phần tăng của chúng đến từ cạnh thật.
- GraphSAGE gần như không đổi khi xáo hay bỏ cạnh. Đã thử chọn epoch theo câu hỏi gián tiếp (5,14 ± 1,12) và ép học từ hàng xóm bằng structure-only dropout 30% (5,46 ± 0,81); cả hai không giúp.

### Theo khoảng cách từ anchor tới đáp án

R@1 sau khi xếp hạng lại, chia theo số cạnh ngắn nhất trong đồ thị train giữa anchor và đáp án:

| Số cạnh | Số cặp | CLIP | GraphSAGE | Path score | AdaMF-MAT | Relation path scorer |
|---|---|---|---|---|---|---|
| 1 | 597 | 3,52 | 9,88 | 59,13 | 65,33 | 83,14 |
| 2 | 311 | 5,47 | 9,11 | 10,61 | 34,08 | 31,83 |
| 3 | 1.128 | 1,77 | 3,81 | 1,24 | 11,49 | 10,40 |
| trên 3 hoặc không có đường | 590 | 2,20 | 3,79 | 1,69 | 4,18 | 1,36 |

Path score và relation path scorer chỉ đi tối đa 3 cạnh; 22% số cặp nằm ngoài tầm đó và ở mức CLIP. AdaMF-MAT không bị giới hạn này nên hơn ở nhóm xa, nhưng kém rõ ở nhóm 1 cạnh (65,3 so với 83,1).

### AdaMF-MAT: phương pháp từ 2024

AdaMF-MAT (Zhang et al., COLING 2024) là mô hình hoàn thiện đồ thị tri thức đa phương thức, và MKG-W là một trong các benchmark của chính bài báo. Trước khi dùng cho truy hồi, bản cài đặt được đối chiếu với bài báo trên bài toán gốc: link prediction trên triple test, xếp hạng trong cả 15.000 thực thể.

| Link prediction (test) | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|
| Bài báo | 35,85 | 29,04 | 39,01 | 48,42 |
| Bản cài đặt ở đây (3 seed) | 34,58 ± 0,07 | 25,78 ± 0,07 | 39,47 ± 0,26 | 49,18 ± 0,29 |
| RotatE chỉ cấu trúc, cùng cài đặt (3 seed) | 29,96 ± 0,14 | 24,54 ± 0,12 | 32,17 ± 0,26 | 40,35 ± 0,21 |

Luật đặt trước khi chạy: chấp nhận bản tái lập khi MRR cách bài báo không quá 3 điểm. Kết quả cách 1,3 điểm.

Khác bài báo:

- Đặc trưng ảnh và văn bản là CLIP ViT-B/32 đóng băng (bài báo: BEiT và sentence-BERT). Thực thể không có ảnh nhận vector 0.
- Learning rate 1e-3, tối đa 200 epoch, dừng sớm theo MRR validation; chạy trên CPU (bài báo: 2e-5, 1.000 epoch, GPU).
- Negative lấy đều, luân phiên thay head và tail theo batch.

Dùng cho truy hồi: câu hỏi gián tiếp cho anchor, quan hệ và chiều hỏi; mỗi thực thể trong gallery được chấm bằng điểm RotatE của triple (anchor, quan hệ, thực thể), chuẩn hoá về [0, 1] theo từng câu, rồi xếp hạng lại cùng CLIP. Mỗi seed dùng một checkpoint, chọn theo MRR link prediction trên validation.

Ảnh và văn bản có giúp không: trên link prediction thì có, MRR tăng 4,6 điểm so với RotatE chỉ cấu trúc. Trên truy hồi thì tuỳ thước đo. R@10 gián tiếp tăng từ 43,8 lên 52,5, nhưng R@1 giảm từ 26,2 xuống 24,8, lớn hơn độ lệch giữa các seed. Phần giảm nằm ở nhóm 1 cạnh (80,9 xuống 65,3); ở nhóm 2 cạnh và nhóm ngoài tầm thì AdaMF-MAT hơn (34,1 so với 24,8; 4,2 so với 0,1). Luật đặt trước là "phương thức giúp khi R@1 gián tiếp của AdaMF-MAT vượt RotatE quá độ lệch seed"; luật này không đạt.

Các ablation của bài báo (bỏ MAT, bỏ adaptive fusion, bỏ từng phương thức) và đối chứng xáo cạnh đang chạy, chưa có đủ 3 seed.

### Kiểm tra lại con số 27,44

Con số của relation path scorer có ba trợ giúp: anchor được đọc từ nhãn của câu hỏi, hệ thống biết trước câu nào là gián tiếp, và mỗi câu hỏi là đúng một triple viết theo mẫu. Bảng dưới bỏ từng trợ giúp (relation path scorer, R@1 sau khi xếp hạng lại).

| Bộ câu hỏi | Số cặp | CLIP | Anchor từ nhãn | Anchor tự tìm | Tự tìm, anchor cũng được điểm |
|---|---|---|---|---|---|
| Gián tiếp | 2.626 | 2,70 | 27,44 | 27,44 | 10,7 |
| Trực tiếp | 2.710 | 52,51 | 52,51 | 11,2 | 88,0 |
| Toàn bộ test | 5.336 | 28,00 | 40,17 | 19,2 | 49,9 |
| Chain: hai triple nối nhau | 3.943 | 0,96 | 3,9 | 3,9 | 1,6 |
| Viết tay, câu tự nhiên | 30 | 0,00 | 20,0 | 6,7 | 0,0 |

- **Anchor tự tìm** so khớp tên thực thể trong câu. Với câu theo mẫu nó tìm đúng 100%, nên cột gián tiếp không đổi. Nhưng câu chỉ nêu tên đáp án cũng bị coi là có anchor, và điểm bị đẩy sang hàng xóm của chính đáp án: trực tiếp tụt từ 52,5 xuống 11,2.
- **Anchor cũng được điểm** coi thực thể được nêu tên là một ứng viên; mức điểm, α và k chọn lại trên validation. Trực tiếp lên 88,0 và toàn bộ test lên 49,9, nhưng phần tăng đến từ khớp tên (câu hỏi là tên thực thể đạt 99,9), còn gián tiếp tụt xuống 10,7. Chưa có cấu hình nào giữ được cả hai nhóm.
- **Chain** ("a film written by the director of Scream 3"): triple đầu có trong đồ thị train, triple sau thì không, thực thể ở giữa không được nêu tên. R@1 3,9; R@10 33,9.
- **Viết tay**: 30 câu do người viết (14 câu một triple, 9 chain, 7 giao hai điều kiện), đáp án được kiểm tra bằng triple. Biết anchor thì đạt 20,0 (R@10 53,3). Tự tìm anchor chỉ đúng 60% số câu vì người viết "Signs" còn dataset ghi "Signs (film)". Bộ này nhỏ, mỗi câu đổi 3,3 điểm.

Bốn cột cuối của hai dòng chain và viết tay, và hai cột cuối của ba dòng đầu, lấy từ `runs/mkg_w/extra_queries/benchmark.json`.

### Giải thích bằng subgraph (relation path scorer)

Điểm của relation path scorer cho một thực thể bằng đúng tổng khối lượng các đường đi (tối đa 3 cạnh) từ anchor tới thực thể đó, nên liệt kê các đường đi là phân rã chính xác của điểm. Subgraph giải thích của một cặp (câu hỏi, thực thể) là các triple nằm trên 3 đường đi nặng nhất. Ví dụ, câu "the performer of Watch That Man" trả về Clint Black với đường đi `Watch That Man —[record label]→ RCA Records ←[record label]— Something That We Do —[performer]→ Clint Black`.

Đo trên 200 câu hỏi test có anchor (hạt giống lấy mẫu 2026), thực thể được giải thích là thực thể đứng đầu sau khi trộn CLIP với điểm đường đi, dù đúng hay sai. 194 câu được chấm; 6 câu bị bỏ qua vì thực thể đứng đầu không có đường đi nào (CLIP tự đưa nó lên). Số dưới đây là trung bình ba seed.

| Tiêu chí | Subgraph giải thích | Ngẫu nhiên trong các triple trên đường đi | Ngẫu nhiên trong các triple chạm anchor hoặc đáp án |
|---|---|---|---|
| Fidelity: mất vị trí đầu sau khi xoá | 97,8% | 31,5% | 37,4% |
| Fidelity: điểm đường đi giảm (thang 0–1) | 0,917 | 0,244 | 0,301 |

- **Fidelity**: xoá các triple của subgraph cho riêng câu hỏi đó rồi chấm lại mọi thực thể (đối thủ cũng đổi điểm). Hai nhóm đối chứng xoá cùng số triple, 20 lần mỗi câu. Đối chứng thứ nhất chỉ tính được khi số triple trên mọi đường đi lớn hơn subgraph: khoảng 165 trên 194 câu.
- **Sparsity**: subgraph có trung bình 3,3 triple trên 53,8 triple nằm trên mọi đường đi tới thực thể, tức sparsity 64,6%, và giữ 84,7% khối lượng điểm.
- **Validity (loại quan hệ)**: 71,3% subgraph có đi qua đúng quan hệ mà câu hỏi nêu. Tách theo kết quả của thực thể đứng đầu: 100% khi nó là đáp án đã có trong đồ thị train (86 câu), 61,4% khi nó là đáp án cần tìm (khoảng 34 câu), 42,6% khi nó sai (khoảng 74 câu).

Giới hạn: đường đi được chọn theo chính khối lượng mà mô hình gán, nên fidelity cao một phần là do cách dựng; đối chứng ngẫu nhiên cho biết phần vượt lên. Validity ở đây chỉ so loại quan hệ, chưa có người chấm xem đường đi có phải lập luận hợp lý không. Đường đi được phép quay lại thực thể cũ (như mô hình), nên có đường dạng A → B ← C → B. Bậc của đỉnh không được tính lại sau khi xoá cạnh, giống lúc huấn luyện.

Chạy lại: `python -m scripts.evaluate_mkg_w_path_explanations` (giao thức ở `configs/evaluation/path_explanation.yaml`); kết quả và subgraph từng câu ở `runs/mkg_w/path_explanation/test.json`.

### Chưa có

- Ablation của AdaMF-MAT (bỏ MAT, mean fusion, bỏ ảnh, bỏ text) và đối chứng xáo cạnh: đang chạy. AdaMF-MAT cũng chưa được chấm trên câu chain và câu viết tay, vì nó cần đúng một quan hệ cho mỗi câu hỏi.
- GraphSAGE có đánh dấu anchor theo từng câu hỏi (để mô hình tốt nhất là một GNN).
- Người chấm validity cho subgraph giải thích (hiện chỉ kiểm tra loại quan hệ bằng máy).
- Cách quyết định theo từng câu hỏi rằng thực thể được nêu tên là đáp án hay là anchor.
- Backbone thứ hai (OpenCLIP).

## Dữ liệu

MKG-W là đồ thị tri thức đa phương thức lấy từ Wikidata: 15.000 thực thể, 169 quan hệ, mỗi thực thể có tên, phần lớn có mô tả (14.150) và ảnh (9.301 thực thể, 27.884 ảnh).

| | train | validation | test |
|---|---|---|---|
| Triple | 34.196 | 4.276 | 4.274 |
| Câu hỏi trực tiếp | 12.672 | 2.711 | 2.710 |
| Câu hỏi gián tiếp | 14.441 | 2.462 | 2.444 |

Thông số dữ liệu: `configs/data/mkg_w.yaml`. Thống kê sau khi dựng: `data/splits/mkg_w_card.yaml`.

### Cách thiết lập

MKG-W gốc là dataset cho bài toán hoàn thiện đồ thị (link prediction), không có câu hỏi và không có nhãn truy hồi ảnh. Phần câu hỏi do nhóm dựng thêm, theo năm bước. Mọi ngưỡng đều tính trên triple train.

**1. Nguồn.** Triple và bảng id (entity2id, relation2id, train / valid / test) lấy từ repository của AdaMF-MAT, để cách chia trùng với cách chia mà phương pháp 2024 dùng. Tên, mô tả và ảnh của thực thể lấy từ bản phát hành MMRNS của nhóm tác giả dataset. File tải về được ghi mã băm vào `data/raw/MANIFEST.json`.

**2. Gắn ảnh với thực thể.** Mỗi thư mục trong file nén ảnh ứng với một thực thể. Thư mục được khớp với thực thể theo tên; nơi file nén đổi dấu câu trong tên thì khớp lần hai theo tên đã bỏ dấu câu, và chỉ nhận khi đúng một thực thể ứng với đúng một thư mục. Ảnh không giải mã được bị loại. Kết quả: 9.301 thực thể có ít nhất một ảnh. Đây là gallery; 5.699 thực thể còn lại vẫn nằm trong đồ thị nhưng không thể là đáp án.

**3. Câu hỏi trực tiếp.** Mỗi thực thể có ảnh sinh một câu hỏi là tên của nó, và thêm một câu là câu đầu của mô tả nếu có. Vì các câu này không gắn với triple nào, thực thể được chia ngẫu nhiên 70 / 15 / 15 (seed 13).

**4. Câu hỏi gián tiếp.** Mỗi triple có thể hỏi theo hai chiều: nêu head hỏi tail, hoặc nêu tail hỏi head. Một cặp (quan hệ, chiều) chỉ được dùng khi qua hai ngưỡng:

- có ít nhất 200 triple train (276 cặp bị loại vì quá ít);
- đáp án không phải là lớp: nếu luôn đoán 10 đáp án phổ biến nhất mà đã trúng hơn 30% triple train thì cặp đó bị loại (15 cặp, ví dụ "instance of" trúng 98,8%, "country of origin" 99,3%). Loại câu hỏi này đoán được mà không cần đọc anchor, và đáp án kiểu "Film" hay "United States" không phải ảnh có nghĩa để truy hồi.

Còn lại 45 cặp thuộc 30 quan hệ. Mỗi cặp có một mẫu câu viết tay trong config, ví dụ `the performer of {anchor}`. Sau đó:

- các triple cùng (anchor, quan hệ, chiều) gộp thành một câu hỏi có nhiều đáp án;
- câu hỏi có hơn 10 đáp án đúng bị bỏ, vì anchor khi đó là một lớp;
- câu hỏi đi theo cách chia của triple sinh ra nó. Đáp án đúng của một câu test là các đáp án đến từ triple test và có ảnh; các đáp án đúng khác (ở train, validation, hoặc không có ảnh) được bỏ qua khi tính thứ hạng, như trong filtered link prediction.

Vì câu test sinh từ triple test và đồ thị chấm điểm chỉ gồm triple train, sự kiện được hỏi không bao giờ có sẵn trong đồ thị.

**5. Nhãn khoảng cách.** Mỗi câu hỏi gián tiếp được ghi số cạnh ngắn nhất trong đồ thị train từ anchor tới một đáp án, tối đa 3. Nhãn này chỉ dùng để chia nhóm khi báo cáo, không lọc câu hỏi nào.

| Số cạnh từ anchor tới đáp án | train | validation | test |
|---|---|---|---|
| 1 | 3.636 | 568 | 567 |
| 2 | 1.991 | 294 | 276 |
| 3 | 5.332 | 1.031 | 1.034 |
| trên 3 hoặc không có đường | 3.482 | 569 | 567 |

### Hai bộ kiểm tra thêm

Chỉ có phần test, không dùng để huấn luyện hay chọn tham số.

- **Chain (3.516 câu):** ghép hai triple nối nhau bằng cách lồng hai mẫu câu, ví dụ "a film written by the director of Scream 3". Triple đầu (anchor → thực thể ở giữa) phải là triple train và phải có đúng một kết quả, để cụm bên trong chỉ một thực thể. Triple sau là triple test. Thực thể ở giữa không xuất hiện trong câu. Dùng cùng ngưỡng quan hệ và giới hạn 10 đáp án như trên.
- **Viết tay (30 câu):** câu tự nhiên do người viết, nằm ở `data/splits/mkg_w_handwritten_queries.yaml`. Mỗi câu khai báo đáp án và mẫu sự kiện định nghĩa đáp án (một triple, hai triple nối nhau, hoặc hai triple cùng chỉ về một đáp án). Script dựng kiểm tra rằng đáp án thoả mẫu, có ảnh, và không suy ra được chỉ từ đồ thị train; câu nào vi phạm thì dừng lại báo lỗi. Hai câu nháp đã bị loại ở bước này.

### Giới hạn của cách thiết lập

- Câu hỏi gián tiếp là câu theo mẫu, không phải câu người dùng thật. Tên quan hệ gần như nguyên văn trong câu và tên anchor trùng khít tên trong dataset.
- Một số tên thực thể mang phần chú thích của DBpedia ("Signs (film)") hoặc là tên lỗi ("Accumulation: None"); chúng đi thẳng vào câu hỏi.
- Đáp án là thực thể, không phải một ảnh cụ thể. Một thực thể được chấm bằng ảnh khớp nhất của nó.
- 38% thực thể không có ảnh nên không thể là đáp án, dù vẫn là đỉnh trong đồ thị.

## Cách hoạt động

1. Mỗi thực thể là một đỉnh, mỗi triple train là một cạnh có nhãn quan hệ.
2. CLIP chấm một thực thể bằng ảnh khớp câu hỏi nhất trong các ảnh của nó.
3. **Path score**: đặt 1 điểm lên anchor rồi lan theo cạnh tối đa 3 bước. Mỗi cạnh chia cho căn bậc của hai đầu, nên đường đi qua đỉnh bậc cao gần như không có trọng lượng. Không có tham số học.
4. **Relation path scorer**: như path score, nhưng mỗi bước nhân với một trọng số phụ thuộc vào quan hệ, hướng đi, số thứ tự bước và câu hỏi. Một lớp tuyến tính đọc vector CLIP của câu hỏi để sinh các trọng số này. Khi chưa huấn luyện, mọi trọng số bằng 1 và mô hình trùng với path score.
5. Huấn luyện trên câu hỏi gián tiếp của train: tăng phần điểm rơi vào đáp án. Triple đang được hỏi bị ẩn khỏi đồ thị cho riêng câu hỏi đó. CLIP đóng băng.
6. Khi truy vấn: điểm đồ thị của mỗi câu được chia cho giá trị lớn nhất, rồi xếp hạng lại theo công thức trên. Câu hỏi không có anchor nhận điểm đồ thị bằng 0.

GraphSAGE dùng lại kiến trúc của VG ∩ COCO: mỗi đỉnh khởi đầu bằng vector CLIP text của tên thực thể, ba tầng trộn thông tin từ hàng xóm, mỗi thực thể ra một vector, điểm đồ thị là cosine với vector câu hỏi. Vector này cố định cho mọi câu hỏi.

AdaMF-MAT học cho mỗi thực thể ba embedding (cấu trúc, ảnh, văn bản), trộn bằng trọng số thích nghi theo từng thực thể, chấm triple bằng RotatE, và dùng thêm embedding ảnh/văn bản tổng hợp làm mẫu đối kháng khi huấn luyện.

## Chạy lại

Cần Python 3.12. Chạy mọi lệnh từ thư mục gốc của repo.

```bash
python -m scripts.build_mkg_w --stage all                      # tải nguồn, giải nén ảnh, sinh câu hỏi
python -m scripts.embed_mkg_w                                  # vector CLIP của ảnh, câu hỏi, tên và mô tả
python -m scripts.benchmark_mkg_w_clip                         # baseline CLIP
python -m scripts.benchmark_mkg_w_paths                        # CLIP + path score, đồ thị thật và xáo cạnh
python -m scripts.train_mkg_w_graphsage --variant real         # real | shuffled_edges | no_edges
python -m scripts.train_mkg_w_relation_paths --variant real    # real | shuffled_edges, 3 seed
python -m scripts.build_mkg_w_extra_queries                    # câu chain và câu viết tay
python -m scripts.benchmark_mkg_w_extra_queries                # bảng "Kiểm tra lại"
python -m scripts.train_mkg_w_adamf_mat                        # AdaMF-MAT, khoảng 40-60 phút mỗi seed trên CPU

pytest                                                         # 393 test, không cần mạng
```

Tham số thí nghiệm: `configs/retrieval/mkg_w.yaml`.
