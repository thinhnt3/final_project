# Phụ lục B. Tính tay một ví dụ từ đầu đến cuối

Phụ lục này đi qua **mọi bước** của hệ thống trên một ví dụ đủ nhỏ để tính bằng tay:

1. dựng đồ thị con;
2. chạy hai tầng GraphSAGE có hướng;
3. tính vector đồ thị và điểm đồ thị;
4. trộn với điểm CLIP;
5. xếp hạng và đo;
6. giải thích bằng cách bỏ relation;
7. tính loss InfoNCE khi huấn luyện.

Công thức và khái niệm của từng bước được giải thích ở [Phụ lục A](phu-luc-a-kien-thuc-nen.md).

> **Số trong phụ lục này là số giả định**, chọn nhỏ và tròn để tính được tay. Mô hình thật khác ở ba điểm:
>
> - vector có 512 số thay vì 3;
> - có 3 tầng thay vì 2;
> - các ma trận trọng số được học chứ không phải ma trận đơn vị.
>
> Cách tính ở từng bước thì giống hệt code. Mọi kết quả dưới đây đều đã được tính lại bằng máy. Nếu tính tay với số làm tròn 4 chữ số, chữ số cuối có thể lệch 1 đơn vị.

## B0. Đề bài

Truy vấn: **"a man riding a horse"**. Có 3 ảnh trong gallery:

| Ảnh | Nội dung | Đồ thị cảnh |
|---|---|---|
| A | người đứng cạnh ngựa | man → next to → horse |
| B | người cưỡi ngựa (**ảnh đúng**) | man → riding → horse |
| C | ngựa trên đồng cỏ | horse → on → grass |

Ta dùng vector 3 chiều với ba trục được đặt tên cho dễ hình dung: (người; ngựa; cưỡi). Vector CLIP văn bản của các chữ trên đồ thị (giả định):

```
man     = (1;   0;  0)
horse   = (0;   1;  0)
riding  = (0;   0;  1)
next to = (0,6; 0; −0,8)      ← trục "cưỡi" âm: nghĩa ngược với cưỡi
```

Vector của câu truy vấn là tổng ba chữ "man", "horse", "riding" rồi chuẩn hoá về độ dài 1:

```
q = (1; 1; 1) / √3 = (0,57735; 0,57735; 0,57735)
```

## B1. CLIP chọn ứng viên (k = 2)

Giả định CLIP chấm ba ảnh như sau:

| Ảnh | Điểm CLIP $s^C$ | Hạng theo CLIP |
|---|---|---|
| A | 0,31 | 1 |
| B (đúng) | 0,30 | 2 |
| C | 0,28 | 3 |

CLIP thấy người và ngựa ở cả A lẫn B, nên điểm của hai ảnh gần nhau và ảnh sai A lại đứng đầu. Với k = 2, chỉ A và B được xếp lại; C nằm ngoài top-k.

## B2. Dựng đồ thị con

Mỗi ảnh có 4 đỉnh: ảnh, man, horse và một đỉnh phụ mang vị từ. Đỉnh ảnh bắt đầu bằng vector học được, dùng chung; trong ví dụ này lấy bằng (0; 0; 0).

```
Ảnh A:  ảnh → man,  ảnh → horse,  man → (next to) → horse
Ảnh B:  ảnh → man,  ảnh → horse,  man → (riding)  → horse
```

## B3. GraphSAGE, tầng 1

Mỗi tầng có hai `SAGEConv`: một lớp gom từ hàng xóm vào, một lớp gom từ hàng xóm ra (xem [A8](phu-luc-a-kien-thuc-nen.md#a8-mạng-nơ-ron-đồ-thị-gcn-gat-graphsage)). Để tính tay, cho mọi ma trận là ma trận đơn vị và mọi bias bằng 0. Mỗi `SAGEConv` khi đó trả về "vector của chính đỉnh + trung bình hàng xóm", nên tổng của hai lớp là:

```
h' = 2·h + trung bình(hàng xóm vào) + trung bình(hàng xóm ra)        (tập rỗng → vector 0)
```

Sau tầng 1 có ReLU: số âm thành 0.

**Ảnh B (riding):**

| Đỉnh | h ban đầu | Hàng xóm vào → trung bình | Hàng xóm ra → trung bình | h' = 2h + vào + ra | Sau ReLU |
|---|---|---|---|---|---|
| ảnh | (0; 0; 0) | — → (0; 0; 0) | man, horse → (0,5; 0,5; 0) | (0,5; 0,5; 0) | (0,5; 0,5; 0) |
| man | (1; 0; 0) | ảnh → (0; 0; 0) | riding → (0; 0; 1) | (2; 0; 1) | (2; 0; 1) |
| horse | (0; 1; 0) | ảnh, riding → (0; 0; 0,5) | — → (0; 0; 0) | (0; 2; 0,5) | (0; 2; 0,5) |
| riding | (0; 0; 1) | man → (1; 0; 0) | horse → (0; 1; 0) | (1; 1; 2) | (1; 1; 2) |

**Ảnh A (next to):**

| Đỉnh | h ban đầu | Hàng xóm vào → trung bình | Hàng xóm ra → trung bình | h' = 2h + vào + ra | Sau ReLU |
|---|---|---|---|---|---|
| ảnh | (0; 0; 0) | — → (0; 0; 0) | man, horse → (0,5; 0,5; 0) | (0,5; 0,5; 0) | (0,5; 0,5; 0) |
| man | (1; 0; 0) | ảnh → (0; 0; 0) | next to → (0,6; 0; −0,8) | (2,6; 0; −0,8) | (2,6; 0; 0) |
| horse | (0; 1; 0) | ảnh, next to → (0,3; 0; −0,4) | — → (0; 0; 0) | (0,3; 2; −0,4) | (0,3; 2; 0) |
| next to | (0,6; 0; −0,8) | man → (1; 0; 0) | horse → (0; 1; 0) | (2,2; 1; −1,6) | (2,2; 1; 0) |

Nhận xét:

- **Sau tầng 1, đỉnh ảnh của A và B giống hệt nhau**: cả hai đều là (0,5; 0,5; 0). Đỉnh ảnh mới chỉ biết "có man và horse"; thông tin về vị từ chưa tới được nó.
- Vị từ đã vào được man và horse. Ở B, man và horse có thành phần "cưỡi" dương (1 và 0,5). Ở A, thành phần này âm, rồi bị ReLU đưa về 0.

## B4. GraphSAGE, tầng 2 (tầng cuối, không có ReLU)

Chỉ cần tính đỉnh ảnh. Đỉnh ảnh không có hàng xóm vào; hàng xóm ra của nó là man và horse, với vector **sau tầng 1**.

```
Ảnh B:  2 × (0,5; 0,5; 0) + 0 + trung bình[(2; 0; 1), (0; 2; 0,5)]
      = (1; 1; 0) + (1; 1; 0,75)
      = (2; 2; 0,75)

Ảnh A:  2 × (0,5; 0,5; 0) + 0 + trung bình[(2,6; 0; 0), (0,3; 2; 0)]
      = (1; 1; 0) + (1,45; 1; 0)
      = (2,45; 2; 0)
```

Đến đây vị từ mới ảnh hưởng tới đỉnh ảnh. Ảnh B có thành phần "cưỡi" bằng 0,75, còn ảnh A bằng 0.

## B5. Tầng chiếu và chuẩn hoá: vector đồ thị

Tầng chiếu cũng là ma trận đơn vị, nên chỉ còn bước chuẩn hoá về độ dài 1.

```
Ảnh B:  ‖(2; 2; 0,75)‖ = √(4 + 4 + 0,5625) = √8,5625  = 2,92617
        g_B = (2; 2; 0,75) / 2,92617 = (0,68349; 0,68349; 0,25631)

Ảnh A:  ‖(2,45; 2; 0)‖ = √(6,0025 + 4)   = √10,0025 = 3,16267
        g_A = (2,45; 2; 0) / 3,16267 = (0,77466; 0,63238; 0)
```

## B6. Điểm đồ thị

Điểm đồ thị là cosine giữa q và g, tức tích vô hướng vì cả hai đều dài 1. Vì q = (1; 1; 1)/√3, cosine bằng tổng các thành phần của h chia cho √3 × ‖h‖:

```
s^G(B) = (2 + 2 + 0,75)    / (1,73205 × 2,92617) = 4,75 / 5,06828 = 0,93720
s^G(A) = (2,45 + 2 + 0)    / (1,73205 × 3,16267) = 4,45 / 5,47791 = 0,81235
```

Đồ thị của B ("riding") khớp truy vấn hơn hẳn đồ thị của A ("next to"): 0,9372 so với 0,8124.

## B7. Trộn điểm và xếp hạng lại (α = 0,8)

```
điểm cuối = α · s^C + (1 − α) · s^G           (chỉ cho ứng viên trong top-k)

A: 0,8 × 0,31 + 0,2 × 0,81235 = 0,24800 + 0,16247 = 0,41047
B: 0,8 × 0,30 + 0,2 × 0,93720 = 0,24000 + 0,18744 = 0,42744
C: ngoài top-2 → 0,28 − 3 = −2,72            (xuống dưới mọi ứng viên, giữ thứ tự CLIP giữa các ảnh ngoài top-k)
```

| Hạng | Trước (theo CLIP) | Sau khi xếp lại |
|---|---|---|
| 1 | A (0,31) | **B (0,4274)** ✓ |
| 2 | B (0,30) | A (0,4105) |
| 3 | C (0,28) | C (−2,72) |

Ảnh đúng B tăng từ hạng 2 lên hạng 1.

Nếu chọn α = 1, điểm cuối của A và B chỉ còn là điểm CLIP, nên thứ tự quay về đúng như CLIP. Đó là lý do phép kiểm "α = 1 phải cho đúng thứ hạng của CLIP" luôn phải đạt.

## B8. Đo kết quả cho truy vấn này

| | Hạng của ảnh đúng | R@1 | R@5 | MRR (1/hạng) |
|---|---|---|---|---|
| Chỉ CLIP | 2 | 0 | 1 | 0,5 |
| CLIP + đồ thị | 1 | 1 | 1 | 1 |

Trên tập test thật, các con số này được tính trung bình trên 7.500 truy vấn (xem [A3](phu-luc-a-kien-thuc-nen.md#a3-truy-hồi-và-cách-đo-hạng-rk-mrr)).

## B9. Giải thích: bỏ relation rồi tính lại

**Bỏ "riding" khỏi đồ thị B.** Đồ thị chỉ còn ảnh → man và ảnh → horse.

```
Tầng 1:  ảnh  = (0,5; 0,5; 0)
         man  = 2 × (1; 0; 0) + 0 + 0 = (2; 0; 0)
         horse = 2 × (0; 1; 0) + 0 + 0 = (0; 2; 0)
Tầng 2:  ảnh  = (1; 1; 0) + trung bình[(2; 0; 0), (0; 2; 0)] = (1; 1; 0) + (1; 1; 0) = (2; 2; 0)
s^G      = (2 + 2 + 0) / (1,73205 × 2,82843) = 4 / 4,89898 = 0,81650

Mức giảm Δ(riding) = 0,93720 − 0,81650 = +0,12070
```

**Bỏ "next to" khỏi đồ thị A.** Ta được đúng đồ thị rút gọn ở trên, nên s^G = 0,81650.

```
Mức giảm Δ(next to) = 0,81235 − 0,81650 = −0,00415
```

Cách đọc:

- **"riding" đóng góp dương** (+0,1207). Lời giải thích cho kết quả hạng 1 sẽ ghi `man → riding → horse`.
- **"next to" đóng góp âm** (−0,0041): bỏ nó đi, điểm đồ thị của A còn tăng nhẹ. Relation này đang kéo A ra xa truy vấn, đúng với trực giác.

Ở ví dụ thật (ảnh 2350042), mức giảm nhỏ hơn nhiều, chỉ khoảng 0,005–0,007. Một lý do có thể là đồ thị thật có tới 15 đối tượng và 6 relation, nên khi lấy trung bình, mỗi relation chỉ chiếm một phần nhỏ. Ablation ở tài liệu 4 cho kết quả cùng chiều: bỏ hết relation chỉ làm R@1 giảm khoảng 0,56 điểm, vì phần lớn mức tăng đến từ các đối tượng.

## B10. Huấn luyện: loss InfoNCE cho batch gồm A và B

Giả sử batch có 2 ảnh, A và B, mỗi ảnh đi kèm caption của chính nó:

```
caption của A: "a man next to a horse" → t_A = (man + horse + next to) chuẩn hoá
             = (1,6; 1; −0,8) / √4,2 = (1,6; 1; −0,8) / 2,04939 = (0,78072; 0,48795; −0,39036)
caption của B: "a man riding a horse"  → t_B = q = (0,57735; 0,57735; 0,57735)
```

**Bảng điểm S** (hàng là vector đồ thị, cột là caption):

```
g_A·t_A = 0,77466×0,78072 + 0,63238×0,48795 + 0×(−0,39036)         = 0,9134
g_A·t_B = s^G(A)                                                    = 0,8124
g_B·t_A = 0,68349×0,78072 + 0,68349×0,48795 + 0,25631×(−0,39036)    = 0,7671
g_B·t_B = s^G(B)                                                    = 0,9372

S = | 0,9134  0,8124 |      chia cho τ = 0,07  →  | 13,048  11,605 |
    | 0,7671  0,9372 |                            | 10,958  13,389 |
```

**Cross-entropy theo hàng** (ảnh tìm caption của mình). Với 2 lựa chọn, softmax của ô đúng là 1 / (1 + e^(logit sai − logit đúng)):

```
Hàng A: p = 1 / (1 + e^(11,605 − 13,048)) = 1 / (1 + e^(−1,443)) = 1 / 1,2362 = 0,8089    CE = −ln 0,8089 = 0,2121
Hàng B: p = 1 / (1 + e^(10,958 − 13,389)) = 1 / (1 + e^(−2,431)) = 1 / 1,0880 = 0,9191    CE = −ln 0,9191 = 0,0843
```

**Cross-entropy theo cột** (caption tìm ảnh của mình):

```
Cột A: p = 1 / (1 + e^(10,958 − 13,048)) = 1 / (1 + e^(−2,090)) = 0,8899    CE = 0,1166
Cột B: p = 1 / (1 + e^(11,605 − 13,389)) = 1 / (1 + e^(−1,784)) = 0,8561    CE = 0,1553
```

**Loss của batch:**

```
L = ½ × [ (0,2121 + 0,0843)/2  +  (0,1166 + 0,1553)/2 ]
  = ½ × [ 0,1482 + 0,1360 ]
  = 0,1421
```

Cách đọc:

- **Loss đã thấp.** Ở ví dụ này, mỗi vector đồ thị vốn đã gần caption của chính nó hơn caption còn lại: 0,9134 > 0,8124 và 0,9372 > 0,7671.
- **Huấn luyện làm gì.** Huấn luyện sẽ chỉnh các ma trận W, tức những thứ ta đã giả định là ma trận đơn vị, để nới rộng khoảng cách giữa đường chéo và phần còn lại, từ đó giảm loss.
- **Vai trò của nhiệt độ.** Cũng bảng S này, nếu τ = 1 thì loss là 0,6277. Với τ = 0,07, các khác biệt nhỏ về cosine được phóng to khoảng 14 lần (1 / 0,07 ≈ 14,3), nên mô hình bị phạt rõ ràng ngay cả khi cặp sai chỉ kém cặp đúng một chút.

## B11. Đối chiếu với hệ thống thật

| Bước | Ví dụ này | Hệ thống thật |
|---|---|---|
| Vector | 3 chiều, chọn tay | 512 chiều, do CLIP ViT-B/32 tạo |
| Đồ thị | 3 đỉnh + 1 đỉnh phụ | Trung vị 17–18 đối tượng và 12 relation mỗi ảnh |
| GraphSAGE | 2 tầng, ma trận đơn vị | 3 tầng, ma trận 512 × 512 được học (3,41 triệu tham số) |
| Đỉnh ảnh | (0; 0; 0) | Vector học được, dùng chung cho mọi ảnh |
| α, k | 0,8 và 2, chọn tay | Chọn trên validation, ví dụ α = 0,78 và k = 500 (lượt 18/11, seed 13) |
| Batch | 2 cặp | 256 cặp |
| Giải thích | Δ(riding) = +0,12 | Δ khoảng 0,005–0,045 |

Quay lại: [Phụ lục A](phu-luc-a-kien-thuc-nen.md) · [Mục lục tài liệu](README.md)
