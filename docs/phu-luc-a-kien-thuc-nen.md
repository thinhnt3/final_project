# Phụ lục A. Kiến thức nền, công thức và ví dụ

Phụ lục này giải thích mọi mảng kiến thức mà dự án dùng tới. Mỗi mục gồm bốn phần: **ý tưởng**, **công thức**, **ví dụ bằng số** và vị trí của nó **trong dự án**.

Số trong ví dụ có hai loại, và loại nào cũng được ghi rõ:

- **số thật:** lấy từ file trong `runs/` hoặc `data/`;
- **số giả định:** chọn nhỏ và tròn để tính tay được.

Mọi con số đều đã được tính lại bằng máy, không tính nhẩm. Phần tính trọn một vòng từ đầu đến cuối nằm ở [Phụ lục B](phu-luc-b-tinh-tay-tu-dau-den-cuoi.md).

Quy ước viết số: dấu phẩy ngăn phần thập phân (0,5), dấu chấm ngăn hàng nghìn (7.000). Các thành phần của vector được ngăn bằng dấu chấm phẩy, ví dụ (0,6; 0,8).

## Mục lục

| Mục | Chủ đề |
|---|---|
| [A1](#a1-vector-độ-dài-chuẩn-hoá-và-cosine) | Vector, độ dài, chuẩn hoá và cosine |
| [A2](#a2-clip-đặt-ảnh-và-câu-vào-cùng-một-không-gian) | CLIP |
| [A3](#a3-truy-hồi-và-cách-đo-hạng-rk-mrr) | Truy hồi và cách đo: hạng, R@K, MRR |
| [A4](#a4-đồ-thị-đỉnh-cạnh-có-hướng-edge_index) | Đồ thị: đỉnh, cạnh có hướng, `edge_index` |
| [A5](#a5-đồ-thị-cảnh-visual-genome-và-wordnet) | Đồ thị cảnh, Visual Genome và WordNet |
| [A6](#a6-idf-và-điểm-so-khớp-từ-b4) | IDF và điểm so khớp từ (B4) |
| [A7](#a7-tầng-tuyến-tính-relu-và-chuẩn-hoá) | Tầng tuyến tính, ReLU và chuẩn hoá |
| [A8](#a8-mạng-nơ-ron-đồ-thị-gcn-gat-graphsage) | Mạng nơ-ron đồ thị: GCN, GAT, GraphSAGE |
| [A9](#a9-ghép-nhiều-đồ-thị-thành-một-batch-pytorch-geometric) | Ghép nhiều đồ thị thành một batch (PyTorch Geometric) |
| [A10](#a10-softmax-cross-entropy-nhiệt-độ-và-infonce) | Softmax, cross-entropy, nhiệt độ và InfoNCE |
| [A11](#a11-tối-ưu-adamw-cắt-gradient-batch-dừng-sớm) | Tối ưu: AdamW, cắt gradient, batch, dừng sớm |
| [A12](#a12-train-validation-test-quá-khớp-rò-rỉ-và-chọn-tham-số) | Train/validation/test, quá khớp, rò rỉ, chọn tham số |
| [A13](#a13-seed-trung-bình-và-độ-lệch-chuẩn-mẫu) | Seed, trung bình và độ lệch chuẩn mẫu |
| [A14](#a14-số-thực-float32-và-float64) | Số thực float32 và float64 |
| [A15](#a15-sha256-và-tính-tái-lập) | sha256 và tính tái lập |
| [A16](#a16-giải-thích-mô-hình-bỏ-từng-relation-fidelity-sparsity-validity) | Giải thích mô hình: bỏ từng relation, fidelity, sparsity, validity |
| [A17](#a17-ablation-và-đảo-cạnh-giữ-bậc) | Ablation và đảo cạnh giữ bậc |
| [A18](#a18-hướng-mở-rộng-đồ-thị-tri-thức-r-gcn-hgt) | Hướng mở rộng: đồ thị tri thức, R-GCN, HGT |

## Bảng ký hiệu

| Ký hiệu | Nghĩa | Trong dự án |
|---|---|---|
| $d$ | Số chiều của vector | 512 |
| $\mathbf{q}$ | Vector CLIP của câu truy vấn | `queries.npy` |
| $\mathbf{v}_i$ | Vector CLIP của ảnh $i$ | `images.npy` |
| $\mathbf{g}_i$ | Vector đồ thị của ảnh $i$, do GraphSAGE tạo | `graph_vectors.npy` |
| $s^{C}_i$ | Điểm CLIP của ảnh $i$ = cos($\mathbf{q}$, $\mathbf{v}_i$) | |
| $s^{G}_i$ | Điểm đồ thị của ảnh $i$ = cos($\mathbf{q}$, $\mathbf{g}_i$) | |
| $\alpha$ | Trọng số của điểm CLIP trong điểm cuối | chọn trên validation |
| $k$ | Số ứng viên được xếp lại | chọn trên validation |
| $\tau$ | Nhiệt độ của InfoNCE | 0,07 |
| $B$ | Cỡ batch | 256 |
| $N$ | Số ảnh train | 7.000 |

---

## A1. Vector, độ dài, chuẩn hoá và cosine

**Ý tưởng.** Máy biểu diễn một ảnh hay một câu bằng một dãy số gọi là *vector* (hoặc *embedding*). Hai thứ có nghĩa giống nhau thì vector của chúng chỉ về gần cùng một hướng. Cosine đo mức "cùng hướng" đó.

**Công thức.**

$$
\mathbf{a}\cdot\mathbf{b}=\sum_{i=1}^{d} a_i b_i,\qquad
\lVert\mathbf{a}\rVert=\sqrt{\mathbf{a}\cdot\mathbf{a}},\qquad
\hat{\mathbf{a}}=\frac{\mathbf{a}}{\lVert\mathbf{a}\rVert},\qquad
\cos(\mathbf{a},\mathbf{b})=\frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}=\hat{\mathbf{a}}\cdot\hat{\mathbf{b}}
$$

- Cosine nằm trong khoảng từ −1 (ngược hướng) đến 1 (cùng hướng); bằng 0 khi hai vector vuông góc.
- Nếu hai vector đã được chuẩn hoá về độ dài 1 thì cosine chính là tích vô hướng.

**Ví dụ (số giả định).**

```
a = (1; 2; 2)        b = (2; 1; 2)
a·b = 1×2 + 2×1 + 2×2 = 8
‖a‖ = √(1 + 4 + 4) = 3          ‖b‖ = 3
cos(a, b) = 8 / (3 × 3) = 0,8889

â = a / 3 = (0,3333; 0,6667; 0,6667)
b̂ = b / 3 = (0,6667; 0,3333; 0,6667)
â·b̂ = 0,8889                      ← đúng bằng cos(a, b)
```

**Trong dự án.**

- Mọi vector được lưu ở độ dài 1. Hàm `l2_normalize` chuẩn hoá, còn `EmbeddingTable` từ chối bảng nào có vector lệch độ dài 1 quá 10⁻⁴.
- Nhờ vậy, toàn bộ điểm CLIP của một tập chỉ là một phép nhân ma trận, `query_vectors @ image_vectors.T` (hàm `clip_scores`), cho ra 7.500 × 1.500 điểm.
- Số thật: cosine trung bình giữa một caption và đúng ảnh của nó là 0,304, với một ảnh ngẫu nhiên là 0,152.

## A2. CLIP: đặt ảnh và câu vào cùng một không gian

**Ý tưởng.** CLIP (OpenAI, 2021) gồm hai bộ mã hoá:

- **bộ mã hoá ảnh** ViT-B/32;
- **bộ mã hoá văn bản**, một Transformer.

Mỗi bộ mã hoá kết thúc bằng một tầng chiếu ra vector 512 số. CLIP được huấn luyện trên khoảng 400 triệu cặp (ảnh, chú thích) lấy từ Internet. Mục tiêu huấn luyện là kéo vector của một ảnh lại gần vector chú thích của chính nó và đẩy xa chú thích của ảnh khác. Hàm mất mát giống hệt InfoNCE ở mục [A10](#a10-softmax-cross-entropy-nhiệt-độ-và-infonce). Nhờ cách học này, dùng CLIP để tìm ảnh không cần huấn luyện thêm; đó là cách dùng *zero-shot*.

**Chi tiết của ViT-B/32.**

- Ảnh được đưa về 224 × 224 điểm ảnh rồi cắt thành các ô 32 × 32, tức (224/32)² = 7 × 7 = 49 ô. Mỗi ô là một "từ" của Transformer.
- Câu được tách thành tối đa 77 token; phần dài hơn bị cắt bỏ (`truncation=True` trong `ClipEncoder.encode_texts`).

**Ví dụ (số thật, truy vấn `coco-659101`).**

| Ảnh | Điểm CLIP $s^C$ | Thứ hạng theo CLIP |
|---|---|---|
| 2350042 (ảnh đúng) | 0,3242 | 1 |
| 2363149 | 0,3154 | 2 |
| 2354907 | 0,3074 | 3 |

Điểm CLIP tuyệt đối thường chỉ quanh 0,3. Điều quan trọng là thứ tự giữa các ảnh, không phải độ lớn tuyệt đối của điểm.

**Trong dự án.** CLIP **đóng băng** (`requires_grad_(False)`). `scripts/embed_vg_coco.py` tính vector một lần cho 10.000 ảnh, 50.000 caption, 1.778 chữ khái niệm và 493 chữ vị từ. Đỉnh của đồ thị lấy vector ban đầu từ chính các chữ này.

## A3. Truy hồi và cách đo: hạng, R@K, MRR

**Ý tưởng.** Với mỗi truy vấn, hệ thống chấm điểm mọi ảnh trong gallery rồi sắp xếp giảm dần. *Hạng* là vị trí của ảnh đúng trong danh sách đó.

**Công thức.** Gọi $s_+$ là điểm của ảnh đúng và $Q$ là tập truy vấn.

$$
\operatorname{rank}=\bigl|\{\,j : s_j \ge s_+\,\}\bigr|,\qquad
\mathrm{R@}K=\frac{1}{|Q|}\sum_{q\in Q}\mathbf{1}\left[\operatorname{rank}(q)\le K\right],\qquad
\mathrm{MRR}=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{\operatorname{rank}(q)}
$$

Công thức hạng đếm mọi ảnh có điểm **lớn hơn hoặc bằng** ảnh đúng, kể cả chính ảnh đúng. Vì vậy khi có điểm hoà, ảnh đúng bị xếp dưới các ảnh hoà với nó. Đây là cách xử lý hoà bất lợi nhất cho phương pháp đang được đo.

**Ví dụ 1: điểm hoà (số giả định).** Gallery có 4 ảnh với điểm (0,30; 0,25; 0,30; 0,10), và ảnh đúng là ảnh đầu tiên. Có 2 ảnh có điểm ≥ 0,30, nên hạng = 2, dù trong mảng ảnh đúng đứng trước.

**Ví dụ 2: các chỉ số (số giả định).** Năm truy vấn có hạng lần lượt là 1, 3, 1, 12, 2.

```
R@1  = (số truy vấn có hạng ≤ 1)  / 5 = 2/5 = 40%
R@5  = (số truy vấn có hạng ≤ 5)  / 5 = 4/5 = 80%
R@10 = (số truy vấn có hạng ≤ 10) / 5 = 4/5 = 80%
MRR  = (1/1 + 1/3 + 1/1 + 1/12 + 1/2) / 5 = 2,9167 / 5 = 0,5833
```

**Mức ngẫu nhiên** với gallery 1.500 ảnh:

- R@K = K / 1.500, nên R@1 = 0,07%, R@5 = 0,33%, R@10 = 0,67%;
- MRR = (1/1.500) × (1/1 + 1/2 + … + 1/1.500) ≈ 0,00526, tức khoảng 0,53%.

**Trong dự án.** Code nằm ở `src/evaluation/metrics.py`. Số thật: CLIP đạt R@1 = 43,89% trên test, nghĩa là 3.292 trong 7.500 truy vấn có ảnh đúng đứng đầu.

## A4. Đồ thị: đỉnh, cạnh có hướng, `edge_index`

**Ý tưởng.** Đồ thị G = (V, E) gồm tập đỉnh V và tập cạnh E. Cạnh có hướng u → v đi từ *nguồn* u tới *đích* v.

- *Hàng xóm vào* của v: $\mathcal{N}_{in}(v)$ = mọi u có cạnh u → v.
- *Hàng xóm ra* của v: $\mathcal{N}_{out}(v)$ = mọi w có cạnh v → w.

PyTorch Geometric lưu các cạnh trong một ma trận 2 × |E| tên là `edge_index`: hàng trên là các nguồn, hàng dưới là các đích. Lệnh `edge_index.flip(0)` đổi chỗ hai hàng, tức đảo chiều mọi cạnh.

**Ví dụ (một phần đồ thị thật của ảnh 2350042).** Đánh số đỉnh như sau: 0 = ảnh, 1 = woman, 2 = cake, 3 = candle, 4 = (cutting), 5 = (on).

```
Cạnh:  0→1  0→2  0→3   (ảnh chứa đối tượng)
       1→4  4→2        (woman → cutting → cake)
       3→5  5→2        (candle → on → cake)

edge_index = [[0, 0, 0, 1, 4, 3, 5],    ← nguồn
              [1, 2, 3, 4, 2, 5, 2]]    ← đích

N_in(cake = 2)  = {0, 4, 5}   (ảnh, cutting, on)     N_out(cake)  = ∅
N_in(woman = 1) = {0}                                N_out(woman) = {4}
N_in(ảnh = 0)   = ∅                                  N_out(ảnh)   = {1, 2, 3}
```

Sau `flip(0)`, mọi cạnh đổi chiều: N_in(ảnh) trở thành {1, 2, 3}. Lớp `SAGEConv` thứ hai trong mỗi tầng dùng đúng phép đảo này (xem mục [A8](#a8-mạng-nơ-ron-đồ-thị-gcn-gat-graphsage)).

**Trong dự án.** Hàm `image_graph_to_data` trong `src/models/graph_data.py` dựng đồ thị này. Số thật trên 10.000 ảnh: 174.743 cạnh "chứa" cộng 2 × 127.812 cạnh của đỉnh phụ, ra đúng 430.367 cạnh.

## A5. Đồ thị cảnh, Visual Genome và WordNet

**Đồ thị cảnh.** Một đồ thị cảnh mô tả một bức ảnh bằng các đối tượng và các bộ ba *(chủ thể, vị từ, đối tượng)*, ví dụ (woman, cutting, cake). Visual Genome (Krishna và cộng sự, 2017) có 108.077 ảnh với đồ thị cảnh do người gán nhãn.

**WordNet synset.** WordNet là một từ điển nối các từ cùng nghĩa thành *synset*. Tên synset có dạng `lemma.loại_từ.số_thứ_tự_nghĩa`.

| Tên thô trong VG (số lần, đếm trên 10.000 ảnh) | Synset | Chữ hiển thị trong dự án |
|---|---|---|
| bike (652), bicycle (237), bikes (51), bicycles (26) | `bicycle.n.01` (danh từ, nghĩa thứ nhất) | "bike" |
| pants (1.477), pant (48), trouser (22), trousers (16) | `trouser.n.01` | "pants" |
| he | `helium.n.01` (gán sai) | bị bỏ vì nằm trong danh sách tên rác |
| it | `information_technology.n.01` (gán sai) | bị bỏ vì nằm trong danh sách tên rác |

Chữ hiển thị là tên thô phổ biến nhất của khái niệm, đếm trên ảnh train.

**Vị từ không dùng synset.** VG gán `ON` vào synset `along.r.01`, làm mất nghĩa. Vì vậy vị từ chỉ được chuẩn hoá bằng cách viết thường, gộp khoảng trắng và bỏ dấu câu ở hai đầu. Ví dụ: `" ON "` → `"on"`.

**Trong dự án.** Bước dựng dữ liệu gộp đối tượng theo synset; mã nằm ở `src/data/synsets.py` và `src/data/scene_graphs.py`. Sau đó `src/data/graph_build.py` lập từ vựng chỉ từ ảnh train: còn 1.778 khái niệm và 493 vị từ.

## A6. IDF và điểm so khớp từ (B4)

**Ý tưởng.** Phương pháp B4 chấm điểm một ảnh bằng tỉ lệ các từ trong câu truy vấn xuất hiện trong chữ của đồ thị ảnh. Không phải từ nào cũng quan trọng như nhau: từ hiếm (như `birthday`) cho nhiều thông tin hơn từ có ở khắp nơi (như `man`). IDF (*inverse document frequency*) gán trọng số theo độ hiếm đó.

**Công thức** (giống hệt `src/retrieval/lexical.py`). Gọi:

- $N$ = 7.000 là số ảnh train;
- df($w$) là số ảnh train có từ $w$ trong đồ thị;
- $W(q)$ là tập gốc từ của các từ nội dung trong truy vấn;
- $W(G)$ là tập gốc từ của chữ trên đồ thị ảnh (chữ khái niệm và chữ vị từ).

$$
\operatorname{idf}(w)=\ln\Bigl(1+\frac{N}{1+\operatorname{df}(w)}\Bigr),\qquad
\operatorname{idf}_{\text{unseen}}=\ln(1+N),\qquad
\operatorname{score}(q,G)=\frac{\sum_{w\in W(q)\cap W(G)}\operatorname{idf}(w)}{\sum_{w\in W(q)}\operatorname{idf}(w)}
$$

- *Từ nội dung* là các từ còn lại sau khi bỏ từ chức năng (the, is, to, as…).
- *Gốc từ*: chỉ bỏ một chữ `s` số nhiều. Ví dụ `friends` → `friend`, còn `glass` giữ nguyên.

**Ví dụ (số thật).** Truy vấn: *"Woman is just about to cut birthday cake as friends are around."*

| Gốc từ | df (số ảnh train có từ này) | idf |
|---|---|---|
| woman | 1.111 | 1,9872 |
| around | 344 | 3,0582 |
| cake | 107 | 4,1868 |
| cut | 11 | 6,3705 |
| about | 7 | 6,7754 |
| birthday | 6 | 6,9088 |
| friend | 5 | 7,0628 |
| just | 0 (chưa từng gặp) | 8,8538 = ln(7.001) |
| **Tổng** | | **45,2034** |

Tính thử một dòng: idf(cake) = ln(1 + 7.000 / 108) = ln(65,8148) = 4,1868.

| Ảnh | Gốc từ khớp | Điểm so khớp từ |
|---|---|---|
| 2350042 (ảnh đúng) | cake, woman | (4,1868 + 1,9872) / 45,2034 = **0,1366** |
| 2363149 | birthday, cake, woman | (6,9088 + 4,1868 + 1,9872) / 45,2034 = **0,2894** |
| 2354907 | cake | 4,1868 / 45,2034 = **0,0926** |

**Bài học từ ví dụ này.**

- **So khớp từ bỏ lỡ hành động chính.** Truy vấn có "cut", còn đồ thị ảnh đúng có "cutting". Hàm gốc từ không bỏ đuôi `-ing`, nên hai chữ không khớp.
- **Một từ hiếm ở ảnh khác lấn át ảnh đúng.** Ảnh 2363149 có chữ "birthday" (idf 6,91), nên điểm so khớp từ của nó gấp đôi ảnh đúng.
- **GraphSAGE so vector chứ không so chữ.** Vector CLIP của "cutting" và vector của một câu chứa "cut" có thể gần nhau dù chữ khác nhau. Đó là lý do để kỳ vọng GraphSAGE làm tốt hơn B4. Kết quả cho thấy đúng như vậy: riêng điểm đồ thị đạt R@1 27,78% so với 12,63%.

## A7. Tầng tuyến tính, ReLU và chuẩn hoá

**Công thức.**

$$
\mathbf{y}=W\mathbf{x}+\mathbf{b},\qquad
\operatorname{ReLU}(\mathbf{x})_i=\max(0,\,x_i)
$$

**Ví dụ (số giả định).** Với W = [[1, 2], [0, −1]], b = (0,5; 0) và x = (1; 3):

```
Wx         = (1×1 + 2×3;  0×1 + (−1)×3) = (7; −3)
Wx + b     = (7,5; −3)
ReLU(...)  = (7,5; 0)                 ← số âm bị thay bằng 0
chuẩn hoá  = (7,5; 0) / 7,5 = (1; 0)

Không ReLU: (7,5; −3) / √(56,25 + 9) = (7,5; −3) / 8,0777 = (0,9285; −0,3714)
```

**Vì sao sau tầng GraphSAGE cuối không có ReLU.** Vector CLIP có nhiều thành phần âm. Giả sử vector truy vấn là q = (0,6; −0,8). Một vector u không có thành phần âm (u₁, u₂ ≥ 0, dài 1) cho cos(q, u) = 0,6·u₁ − 0,8·u₂ ≤ 0,6. Như vậy dù học tốt đến đâu, vector đồ thị cũng không thể khớp truy vấn này hơn 0,6, trong khi chính vector (0,6; −0,8) cho cosine bằng 1. Code cũ của bản thử 40 ảnh mắc đúng lỗi này.

**Trong dự án.** Chiều ẩn là 512. Tầng chiếu cuối là `nn.Linear(512, 512)`, sau đó chuẩn hoá bằng `nn.functional.normalize`.

## A8. Mạng nơ-ron đồ thị: GCN, GAT, GraphSAGE

**Ý tưởng: truyền tin.** Ở mỗi tầng, mỗi đỉnh gom vector của các hàng xóm, trộn với vector của chính nó, rồi cho qua một tầng tuyến tính. Sau L tầng, một đỉnh "biết" về mọi đỉnh cách nó tối đa L bước.

**Ba họ phổ biến.**

GCN (Kipf và Welling, 2017) lấy trung bình có trọng số theo bậc của đỉnh:

$$
\mathbf{h}_i'=\sigma\Bigl(\sum_{j\in\mathcal{N}(i)\cup\{i\}}\frac{1}{\sqrt{\hat d_i\,\hat d_j}}\,W\mathbf{h}_j\Bigr)
$$

GAT (Veličković và cộng sự, 2018) học một trọng số chú ý cho từng hàng xóm:

$$
\alpha_{ij}=\operatorname{softmax}_j\Bigl(\operatorname{LeakyReLU}\bigl(\mathbf{a}^{\top}[W\mathbf{h}_i\,\Vert\,W\mathbf{h}_j]\bigr)\Bigr),\qquad
\mathbf{h}_i'=\sigma\Bigl(\sum_{j\in\mathcal{N}(i)}\alpha_{ij}W\mathbf{h}_j\Bigr)
$$

GraphSAGE (Hamilton và cộng sự, 2017), biến thể trung bình, giữ một ma trận riêng cho chính đỉnh. Đây đúng là công thức mà `SAGEConv` của PyTorch Geometric cài đặt:

$$
\mathbf{h}_i'=W_1\mathbf{h}_i+W_2\cdot\frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)}\mathbf{h}_j+\mathbf{b}
$$

Đỉnh không có hàng xóm nào thì phần trung bình bằng vector 0.

| | GCN | GAT | GraphSAGE (trung bình) |
|---|---|---|---|
| Cách gom hàng xóm | Trung bình có trọng số theo bậc | Trọng số chú ý được học | Trung bình đều |
| Tách riêng vector của chính đỉnh | Không (gộp chung) | Không | Có ($W_1$ riêng) |
| Ưu điểm với dự án | Đơn giản | Chọn được hàng xóm quan trọng | Đơn giản, chạy được trên đồ thị mới chưa gặp, có sẵn trong PyTorch Geometric |

Dự án chọn GraphSAGE (quyết định G5). GCN và GAT là các lựa chọn mà đề bài cho phép.

**Phiên bản có hướng của dự án.** Mỗi tầng gồm hai `SAGEConv` cộng lại:

- một lớp gom từ hàng xóm vào, tức theo chiều cạnh, dùng `edge_index`;
- một lớp gom từ hàng xóm ra, tức ngược chiều cạnh, dùng `edge_index.flip(0)`.

$$
\mathbf{h}_i'=W_1^{\rightarrow}\mathbf{h}_i+W_2^{\rightarrow}\,\overline{\mathbf{h}}_{\mathcal{N}_{in}(i)}+\mathbf{b}^{\rightarrow}
\;+\;W_1^{\leftarrow}\mathbf{h}_i+W_2^{\leftarrow}\,\overline{\mathbf{h}}_{\mathcal{N}_{out}(i)}+\mathbf{b}^{\leftarrow}
$$

Ở đây $\overline{\mathbf{h}}_{S}$ là trung bình vector của các đỉnh trong tập S, và bằng 0 nếu S rỗng.

**Ví dụ một tầng (số giả định, 2 chiều).** Để dễ tính, cho mọi ma trận W là ma trận đơn vị và mọi bias bằng 0. Khi đó công thức rút gọn thành h' = 2h + trung bình(vào) + trung bình(ra).

Đồ thị là ảnh → woman, ảnh → cake, woman → (cutting) → cake, với h(woman) = (1; 0), h(cake) = (0; 1), h(cutting) = (0,6; 0,8) và h(ảnh) = (0; 0).

```
woman': 2×(1; 0) + trung bình{ảnh} + trung bình{cutting}       = (2; 0) + (0; 0) + (0,6; 0,8) = (2,6; 0,8)
cake' : 2×(0; 1) + trung bình{ảnh, cutting} + trung bình{}       = (0; 2) + (0,3; 0,4) + (0; 0) = (0,3; 2,4)
ảnh'  : 2×(0; 0) + trung bình{} + trung bình{woman, cake}        = (0; 0) + (0; 0) + (0,5; 0,5) = (0,5; 0,5)
```

Sau tầng 1, đỉnh ảnh mới biết "có woman và cake". Vị từ `cutting` mới chỉ vào được woman và cake. Phải sang tầng 2, nó mới tới được đỉnh ảnh. [Phụ lục B](phu-luc-b-tinh-tay-tu-dau-den-cuoi.md) tính tiếp tầng 2.

**Vì sao cần 3 tầng.** Từ đỉnh ảnh tới đối tượng ở đầu kia của một relation có 3 bước: ảnh → woman → (cutting) → cake.

- Hai tầng đủ để ảnh biết "woman là chủ thể của một cutting".
- Cần tầng thứ ba để ảnh biết "chính cutting đó nối tới cake". Điều này quan trọng khi một ảnh có hai relation cùng vị từ, như "man riding bicycle, woman riding horse".

Test `test_two_layers_know_each_objects_role_and_three_layers_know_who_goes_with_whom` kiểm đúng điều này.

**Vì sao cần có hướng.** Xét hai đồ thị "man → riding → horse" và "horse → riding → man". Nếu bỏ hướng, man có cùng tập hàng xóm {ảnh, riding} trong cả hai đồ thị, horse cũng vậy, nên hai đồ thị cho cùng một vector. Khi có hướng thì khác:

- Ở đồ thị thứ nhất, man nhận "riding" qua lớp ngược chiều (vì man là chủ thể).
- Ở đồ thị thứ hai, man nhận "riding" qua lớp theo chiều (vì man là đối tượng).

Hai lớp có ma trận khác nhau nên hai vector khác nhau. Có hai test kiểm điều này: `test_directed_encoder_tells_which_object_is_the_subject` và `test_undirected_encoder_ignores_which_object_is_the_subject`.

## A9. Ghép nhiều đồ thị thành một batch (PyTorch Geometric)

**Ý tưởng.** Để huấn luyện nhanh, nhiều đồ thị con được ghép thành một đồ thị lớn gồm các mảnh rời nhau. Đỉnh của đồ thị sau được đánh số tiếp theo đỉnh của đồ thị trước. `batch.ptr` ghi vị trí bắt đầu của từng đồ thị.

**Ví dụ (số giả định).** Đồ thị A có 4 đỉnh, đồ thị B có 2 đỉnh (một ảnh với một đối tượng).

```
Đỉnh của A: 0, 1, 2, 3          Đỉnh của B: 4, 5   (đánh số lại, cộng thêm 4)
batch.ptr = [0, 4, 6]
Đỉnh ảnh của từng đồ thị = ptr[:-1] = [0, 4]      ← GraphImageEncoder lấy vector ở đúng hai vị trí này
Cạnh 0→1 của B trở thành 4→5 trong batch
```

**Cái bẫy về tên thuộc tính.** Khi ghép batch, PyTorch Geometric tự cộng thêm số đỉnh đã có vào **mọi thuộc tính có chữ "index" trong tên**, vì nó coi đó là chỉ số đỉnh.

- Ví dụ: B có số dòng tra bảng vector là [7, 3] (dòng 7 trong bảng ảnh, dòng 3 trong bảng khái niệm).
- Nếu thuộc tính tên là `feature_index`, trong batch nó sẽ thành [11, 7], và mô hình đọc nhầm vector.
- Vì vậy thuộc tính được đặt tên `feature_row`. Lỗi này được test bắt được ở bước 1.2 của Phase 1.

## A10. Softmax, cross-entropy, nhiệt độ và InfoNCE

**Softmax và cross-entropy.** Softmax biến một dãy điểm (*logit*) thành xác suất có tổng bằng 1. Cross-entropy đo mức "ngạc nhiên" khi đáp án đúng có xác suất thấp.

$$
\operatorname{softmax}(\mathbf{z})_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad
\operatorname{CE}(\mathbf{z},y)=-\ln\operatorname{softmax}(\mathbf{z})_y
$$

**Ví dụ (số giả định).** Logit (2; 1; 0), đáp án là vị trí 0.

```
e² = 7,389    e¹ = 2,718    e⁰ = 1      tổng = 11,107
softmax = (0,6652; 0,2447; 0,0900)
CE = −ln(0,6652) = 0,4076
```

**InfoNCE đối xứng** (van den Oord và cộng sự, 2018; CLIP dùng dạng này). Một batch có B cặp (ảnh, caption) đúng. $\mathbf{g}_i$ là vector đồ thị của ảnh i, $\mathbf{t}_j$ là vector CLIP của caption j.

$$
S_{ij}=\frac{\mathbf{g}_i\cdot\mathbf{t}_j}{\tau},\qquad
\mathcal{L}=\frac{1}{2}\Bigl[\frac{1}{B}\sum_{i=1}^{B}\operatorname{CE}(S_{i,:},\,i)+\frac{1}{B}\sum_{j=1}^{B}\operatorname{CE}(S_{:,j},\,j)\Bigr]
$$

- **Hàng i:** ảnh i phải nhận ra caption của mình trong B caption. Đây là chiều từ ảnh tìm caption.
- **Cột j:** caption j phải nhận ra ảnh của mình trong B ảnh. Đây là chiều từ caption tìm ảnh, cũng là chiều của bài toán truy hồi.
- **Mức đoán bừa:** loss = ln B. Với B = 256, đó là ln 256 = 5,545.

Số thật (seed 13): loss train ở vòng 1 là 5,06, chỉ thấp hơn mức đoán bừa một chút; đến vòng 27 còn 1,53.

**Nhiệt độ τ.** Chia cho τ nhỏ phóng đại khác biệt giữa các điểm, nên softmax "nhọn" hơn và mô hình bị phạt nặng hơn khi cặp đúng không vượt hẳn các cặp sai. Ví dụ batch 2 cặp ở Phụ lục B, cùng một bảng điểm:

| τ | Logit của hàng 1 | Loss |
|---|---|---|
| 0,07 (dự án dùng) | (13,048; 11,605) | 0,1421 |
| 1 | (0,913; 0,812) | 0,6277 |

**Trong dự án.** Hàm `info_nce` nằm trong `src/models/train.py`, với nhiệt độ lấy từ `training.temperature: 0.07`.

## A11. Tối ưu: AdamW, cắt gradient, batch, dừng sớm

**Gradient.** Gradient của loss theo một tham số cho biết loss tăng nhanh thế nào khi tham số tăng. Hạ gradient (*gradient descent*) đi ngược hướng đó: θ ← θ − η · gradient, với η là tốc độ học.

**AdamW** (Loshchilov và Hutter, 2019) tự điều chỉnh bước đi cho từng tham số theo trung bình động của gradient ($\mathbf{m}$) và của bình phương gradient ($\mathbf{v}$). Thành phần *weight decay* kéo tham số nhẹ về 0, và được tách riêng khỏi gradient (chữ "W" trong tên).

$$
\mathbf{m}_t=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\mathbf{g}_t,\qquad
\mathbf{v}_t=\beta_2\mathbf{v}_{t-1}+(1-\beta_2)\mathbf{g}_t^2
$$

$$
\hat{\mathbf{m}}_t=\frac{\mathbf{m}_t}{1-\beta_1^{\,t}},\qquad
\hat{\mathbf{v}}_t=\frac{\mathbf{v}_t}{1-\beta_2^{\,t}},\qquad
\theta_t=\theta_{t-1}(1-\eta\lambda)-\eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}
$$

Dự án dùng η = 0,001 và λ = 0,0001. Các giá trị còn lại là mặc định của PyTorch: β₁ = 0,9, β₂ = 0,999, ε = 10⁻⁸.

**Ví dụ một bước (số giả định).** θ = 0,5, gradient g = 0,2, bước thứ nhất (t = 1):

```
weight decay:  0,5 × (1 − 0,001 × 0,0001) = 0,49999995
m = 0,1 × 0,2 = 0,02             v = 0,001 × 0,04 = 0,00004
m̂ = 0,02 / 0,1 = 0,2             v̂ = 0,00004 / 0,001 = 0,04      √v̂ = 0,2
bước = 0,001 × 0,2 / 0,2 = 0,001
θ mới = 0,49999995 − 0,001 = 0,49899995
```

Điều đáng chú ý: ở bước đầu, độ lớn của bước đi xấp xỉ η, bất kể gradient lớn hay nhỏ. Adam tự chuẩn hoá độ lớn của gradient.

**Cắt gradient.** Nếu độ dài của toàn bộ gradient vượt ngưỡng c, gradient được thu nhỏ về đúng độ dài c. Dự án dùng c = 1,0.

```
g = (3; 4)   ‖g‖ = 5 > 1   →   g' = (3; 4) × 1/5 = (0,6; 0,8)
```

**Batch và vòng (epoch).** Một vòng đi qua cả 7.000 ảnh train, chia thành các batch 256 ảnh: 27 batch đủ 256 ảnh và 1 batch 88 ảnh (27 × 256 = 6.912; 7.000 − 6.912 = 88), tổng 28 batch. Thứ tự ảnh và caption được chọn ở mỗi vòng chỉ phụ thuộc vào (seed, số vòng). Nhờ vậy, chạy tiếp bằng `--resume` cho đúng kết quả như khi chạy liền mạch.

**Dừng sớm (số thật, seed 13, patience 5).**

| Vòng | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 |
|---|---|---|---|---|---|---|---|---|---|
| R@1 val (%) | 27,48 ★ | 27,37 | 27,13 | **27,76 ★** | 27,51 | 27,23 | 26,76 | 27,13 | 27,55 |

★ đánh dấu vòng lập kỷ lục mới. Trước khi bắt đầu vòng 28, script kiểm tra 28 − 1 − 22 = 5 ≥ patience = 5, nên dừng. Mô hình dùng về sau là `best.pt` của vòng 22, không phải vòng 27. Đó là lý do bảng huấn luyện ghi "22 / 27".

## A12. Train, validation, test: quá khớp, rò rỉ và chọn tham số

**Vai trò của ba tập.**

- **Train** dùng để học.
- **Validation** dùng để chọn: vòng tốt nhất, α, k.
- **Test** chỉ dùng để chấm điểm, một lần, sau khi mọi thứ đã chốt.

Nếu chọn bất cứ thứ gì dựa trên test, con số trên test không còn đo được khả năng của hệ thống trên dữ liệu mới.

**Quá khớp (số thật, seed 13).** Từ vòng 11 đến vòng 27, loss train vẫn giảm từ 1,98 xuống 1,53, nhưng loss validation đi ngang quanh 1,98–2,08. Mô hình bắt đầu "học thuộc" train. Dừng sớm cắt đúng lúc đó.

**Rò rỉ dữ liệu: các ví dụ dự án đã chặn.**

| Nếu làm thế này | Thì rò rỉ vì | Cách dự án chặn |
|---|---|---|
| Tính từ vựng, IDF hoặc ngưỡng lọc trên cả 10.000 ảnh | Từ hiếm chỉ có ở test sẽ ảnh hưởng tới cách chấm điểm | Chỉ tính trên 7.000 ảnh train; có test kiểm rằng từ vựng không đổi khi sửa ảnh val/test |
| Chọn α hoặc k trên test | Tham số được chỉnh cho chính tập dùng để chấm | `sweep_alpha_and_k` báo lỗi nếu không nhận tập validation |
| Trộn caption vào đồ thị ảnh | Ảnh test mang sẵn đáp án | Quyết định D7 |
| Một tấm ảnh nằm ở hai tập | Mô hình đã thấy ảnh test lúc học | Bỏ 580 ảnh VG dùng chung mã COCO (290 mã, mỗi mã 2 ảnh) |

**Chọn tham số có luật hoà.** Lưới α có 101 giá trị (0; 0,01; …; 1), lưới k có 7 giá trị, tổng cộng 707 cấu hình được chấm trên validation. Cấu hình tốt nhất được chọn như sau:

1. R@1 cao nhất;
2. nếu hoà, MRR cao nhất;
3. nếu vẫn hoà, α lớn nhất (gần CLIP hơn);
4. nếu vẫn hoà, k nhỏ nhất.

Ví dụ (số giả định):

| α | R@1 | MRR | |
|---|---|---|---|
| 0,7 | 40% | 0,55 | |
| 0,8 | 42% | 0,57 | hoà với 0,9 |
| 0,9 | 42% | 0,57 | **được chọn**, vì α lớn hơn nên gần CLIP thuần hơn |
| 1,0 | 38% | 0,52 | |

**Chia tập phân tầng (số thật).** Các ảnh được chia thành 4 tầng theo số relation, với mốc là các tứ phân vị (9; 14; 20):

- tầng 0: ≤ 9 relation;
- tầng 1: 10–14;
- tầng 2: 15–20;
- tầng 3: ≥ 21.

Ví dụ, một ảnh có 12 relation thuộc tầng 1. Mỗi tập được rút sao cho tỉ lệ các tầng giống kho mà nó được lấy ra; độ lệch tối đa đo được là 1,64 điểm phần trăm. Nhờ vậy val và test không vô tình toàn ảnh đơn giản hoặc toàn ảnh phức tạp.

## A13. Seed, trung bình và độ lệch chuẩn mẫu

**Ý tưởng.** Huấn luyện có yếu tố ngẫu nhiên: trọng số khởi tạo, thứ tự batch, caption được chọn. Chạy với nhiều seed rồi báo cáo trung bình ± độ lệch chuẩn cho biết kết quả có ổn định không.

**Công thức** (độ lệch chuẩn *mẫu*, chia cho n − 1, như `np.std(ddof=1)` trong `aggregate_runs`):

$$
\bar x=\frac{1}{n}\sum_{i=1}^{n}x_i,\qquad
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar x)^2}
$$

**Ví dụ (số thật): R@1 trên test của CLIP + GraphSAGE.**

```
seed 13: 48,24     seed 37: 48,20     seed 97: 48,15
trung bình = (48,24 + 48,20 + 48,15) / 3 = 48,1967
độ lệch    = +0,0433; +0,0033; −0,0467
bình phương: 0,00188 + 0,00001 + 0,00218 = 0,00407
s = √(0,00407 / 2) = √0,00203 = 0,045        → báo cáo 48,20 ± 0,05
```

**Cách đọc.**

- Mức tăng so với CLIP (4,31 điểm) gần gấp 100 lần độ lệch giữa các seed, nên không phải do may rủi lúc khởi tạo.
- Lưu ý: seed chỉ thay đổi phần ngẫu nhiên của huấn luyện; tập test thì giữ nguyên. Độ lệch chuẩn này không đo được việc kết quả sẽ dao động bao nhiêu nếu chọn một tập test khác.

## A14. Số thực float32 và float64

**Ý tưởng.** float32 giữ được khoảng 7 chữ số có nghĩa, float64 khoảng 16 chữ số. Quanh giá trị 0,3, hai số float32 liền nhau cách nhau khoảng 3 × 10⁻⁸. Quanh 2,7 thì khoảng cách đó là 2,4 × 10⁻⁷.

**Ví dụ (số thật, tính bằng máy).** Hai điểm CLIP float32 liền nhau:

```
s1 = 0,30000001192092896      s2 = 0,30000004172325134      (s1 ≠ s2)
Trừ 3 bằng float32:  s1 − 3 = −2,700000047683716     s2 − 3 = −2,700000047683716   ← BẰNG NHAU
Trừ 3 bằng float64:  s1 − 3 = −2,699999988079071     s2 − 3 = −2,6999999582767487  ← vẫn khác nhau
```

**Trong dự án.** Ảnh ngoài top-k bị đẩy xuống bằng cách trừ 3 vào điểm. Nếu làm phép trừ bằng float32, hai ảnh vốn khác điểm có thể thành hoà điểm, và luật hoà bất lợi làm thứ hạng thay đổi. Khi kiểm, một truy vấn validation đã tụt từ hạng 38 xuống 39. Vì vậy `rerank.py` đổi điểm sang float64 trước khi tính.

## A15. sha256 và tính tái lập

**Ý tưởng.** Hàm băm sha256 biến một file thành một chuỗi 64 ký tự. Chỉ cần đổi một byte của file là chuỗi băm thay đổi hoàn toàn. So chuỗi băm là cách nhanh để biết một file có bị thay đổi hay không.

**Ví dụ (số thật).**

```
sha256("abc") = ba7816bf8f01cfea…
sha256("abd") = a52d159f262b2c6d…      ← chỉ khác một chữ, chuỗi băm khác hẳn
```

**Trong dự án.**

| File | Ghi sha256 của | Dùng để |
|---|---|---|
| `data/raw/MANIFEST.json` | Các file thô đã tải | Báo lỗi nếu tải lại mà file khác đi |
| `embeddings/<model>/meta.json` | JSON đầu vào và các bảng vector | `verify_inputs_unchanged` từ chối vector cũ hơn dữ liệu |
| `runs/graphsage/<run>/seed<n>/config.yaml` | Dữ liệu dùng khi huấn luyện | `--resume` từ chối nếu dữ liệu đã đổi |
| `runs/benchmark/*.json` | Split và embedding | Biết con số được tính trên dữ liệu nào |

## A16. Giải thích mô hình: bỏ từng relation, fidelity, sparsity, validity

**Bỏ từng relation (leave-one-out, một dạng giải thích bằng nhiễu).** Đóng góp của relation r là mức giảm điểm đồ thị khi bỏ r khỏi đồ thị (giữ nguyên hai đối tượng của nó):

$$
\Delta_r=s(G)-s(G\setminus r)
$$

- Δ lớn và dương: điểm dựa nhiều vào relation đó.
- Δ âm: relation đó đang kéo điểm xuống.

**Ví dụ (số thật).** Với ảnh 2350042, s(G) = 0,4142:

- Δ(woman → cutting → cake) = 0,0073;
- Δ(woman → wearing → shirt) = 0,0063;
- Δ(candle → on → cake) = 0,0052.

**Ba tiêu chí mà đề bài nhắc tới (Cấp độ 3).**

| Tiêu chí | Hỏi điều gì | Trạng thái trong repo |
|---|---|---|
| Fidelity (độ trung thực) | Bỏ các relation được dùng để giải thích có làm điểm giảm nhiều hơn bỏ ngẫu nhiên cùng số relation không? | Có code (`src/evaluation/fidelity.py`, `scripts/evaluate_fidelity.py`), chưa có kết quả |
| Sparsity (độ gọn) | Lời giải thích dùng ít hay nhiều relation? | Có trong code fidelity |
| Validity (độ hợp lý) | Con người có thấy lời giải thích hợp lý không? | Chưa có. Code ghi rõ "no semantic validity claim" |

**Fidelity được tính thế nào** (theo `evaluate_fidelity`). Gọi R là số relation của ảnh và b là ngân sách (mặc định 3).

1. m = min(b, ⌊R/2⌋). Giới hạn này bảo đảm luôn còn đủ relation khác để làm nhóm đối chứng.
2. Tính Δ cho từng relation. Lấy tối đa m relation có Δ > 0, ưu tiên Δ lớn nhất: đây là tập giải thích S.
3. Chọn ngẫu nhiên |S| relation trong số còn lại làm nhóm đối chứng C. Ngẫu nhiên này cố định theo seed và mã ảnh.
4. Bỏ **cùng lúc** cả nhóm S và tính lại điểm, rồi làm tương tự với nhóm C. Kết quả được tính lại chứ không cộng các Δ, vì hiệu ứng chung của nhiều relation không bằng tổng hiệu ứng riêng.

$$
\operatorname{drop}(S)=(1-\alpha)\bigl[s(G)-s(G\setminus S)\bigr],\qquad
\operatorname{drop}(C)=(1-\alpha)\bigl[s(G)-s(G\setminus C)\bigr],\qquad
\text{sparsity}=1-\frac{|S|}{R}
$$

Hệ số (1 − α) quy mức giảm của điểm đồ thị ra mức giảm của **điểm cuối**. Nếu ảnh không nằm trong top-k thì điểm cuối của nó không dùng điểm đồ thị, nên cả hai mức giảm bằng 0. Mỗi kết quả có một trạng thái:

- `controlled`: đã đo đủ cả hai nhóm;
- `insufficient_relations`: ảnh có ít hơn 2 relation;
- `no_supporting_relations`: không relation nào có Δ > 0.

**Ví dụ (số giả định; dùng Δ thật của 3 relation đầu, 3 relation còn lại là giả định).** R = 6, b = 3, nên m = min(3, 3) = 3.

```
Δ = (0,0073; 0,0063; 0,0052; 0,0010; −0,0004; 0)
S = {r1, r2, r3}   (3 Δ dương lớn nhất)        C = 3 relation còn lại = {r4, r5, r6}
Giả định: s(G \ S) = 0,3950    s(G \ C) = 0,4130    s(G) = 0,4142    α = 0,81
drop(S) = 0,19 × (0,4142 − 0,3950) = 0,19 × 0,0192 = 0,003648
drop(C) = 0,19 × (0,4142 − 0,4130) = 0,19 × 0,0012 = 0,000228
sparsity = 1 − 3/6 = 0,5
```

Ở ví dụ này, bỏ tập giải thích làm điểm cuối giảm gấp khoảng 16 lần so với bỏ ngẫu nhiên, nên lời giải thích có fidelity tốt. Một kết luận thật cần trung bình trên nhiều truy vấn: mặc định là 10 truy vấn, đúng mức tối thiểu đề bài yêu cầu.

## A17. Ablation và đảo cạnh giữ bậc

**Ý tưởng.** Ablation bỏ hoặc phá một thành phần rồi đo lại. Nếu kết quả giảm, thành phần đó có đóng góp. Kế hoạch Phase 5 có các biến thể sau:

| Mã | Biến thể | Trả lời câu hỏi | Code trong repo |
|---|---|---|---|
| A1 | Đầy đủ | Mốc so sánh | Mô hình hiện tại |
| A2 | Chỉ đối tượng (bỏ mọi relation) | Relation có giúp không? | `objects_only` trong `src/models/ablations.py` |
| A3 | Bỏ vị từ, giữ cạnh | Nhãn vị từ có giúp không? | Chưa có |
| A4 | Cạnh ngẫu nhiên, giữ bậc | Cấu trúc thật hay chỉ là thêm đặc trưng? | `rewire_relations` trong `src/models/ablations.py` |
| A5 | Không GNN (so khớp từ) | GNN có giá trị không? | Đã có kết quả (B4) |
| A6 | Quét α và k | Kết quả nhạy với tham số đến đâu? | Đường quét của B4 đã lưu |
| A7 | Có hoặc không xử lý hub | Quyết định giữ hub có đúng không? | Chưa có |
| — | Tắt hướng | Hướng của relation có giúp không? | Cờ `graphsage.directed: false` |

Hiện chưa có script nào chạy các biến thể A2 và A4 rồi ghi kết quả. Mới có các hàm biến đổi đồ thị và test cho chúng.

**Đảo cạnh giữ bậc (A4).** Lấy hai bộ ba (s, p, o) và (u, q, v), rồi đổi đối tượng của chúng cho nhau, thành (s, p, v) và (u, q, o). Phép đảo bị từ chối nếu:

- tạo ra vòng tự thân, tức s trùng v hoặc u trùng o;
- s trùng u, hoặc o trùng v;
- tạo ra một bộ ba đã có sẵn.

**Ví dụ (số thật, hai bộ ba của ảnh 2350042).**

```
Trước:  (woman, wearing, shirt)    (girl, on, chair)
Sau:    (woman, wearing, chair)    (girl, on, shirt)
```

Kiểm điều kiện: woman không trùng girl hay chair; shirt khác chair; girl khác shirt; hai bộ ba mới chưa có trong ảnh. Vậy phép đảo hợp lệ. Sau khi đảo:

- woman vẫn là chủ thể của đúng một relation, và girl cũng vậy;
- shirt và chair vẫn là đối tượng của đúng một relation;
- vị từ `wearing` và `on` vẫn xuất hiện mỗi loại một lần.

Như vậy chỉ có *ai nối với ai* bị xáo trộn. Nếu mô hình vẫn đạt điểm như cũ trên đồ thị đã đảo, nghĩa là nó không thật sự dùng cấu trúc quan hệ. Hàm này ghi lại số lần đảo được yêu cầu, số lần thành công và số relation bị đổi, vì đồ thị nhỏ có thể không đảo được lần nào.

## A18. Hướng mở rộng: đồ thị tri thức, R-GCN, HGT

**Đồ thị tri thức (KG) và MMKG.** Đồ thị tri thức nối các *khái niệm* bằng các quan hệ chung, ví dụ (minh hoạ) `woman —IsA→ person` hay `slice —PartOf→ cake`. ConceptNet và Wikidata là hai nguồn KG công khai. MMKG (đồ thị tri thức đa phương thức) gắn thêm ảnh và văn bản vào các khái niệm. Phase 7 dự định thêm các cạnh IsA, RelatedTo, PartOf vào đồ thị hiện có.

**R-GCN** (Schlichtkrull và cộng sự, 2018) dùng một ma trận riêng cho mỗi loại quan hệ r:

$$
\mathbf{h}_i'=\sigma\Bigl(W_0\mathbf{h}_i+\sum_{r\in\mathcal{R}}\sum_{j\in\mathcal{N}_i^{r}}\frac{1}{|\mathcal{N}_i^{r}|}W_r\mathbf{h}_j\Bigr)
$$

Với dự án, R-GCN là một cách khác để mô hình hiểu vị từ: thay vì đặt vị từ thành đỉnh phụ, mỗi vị từ có ma trận riêng. Cái giá là 493 vị từ thì cần 493 ma trận.

**HGT** (Hu và cộng sự, 2020) là một Transformer trên đồ thị dị thể: trọng số chú ý phụ thuộc vào loại đỉnh nguồn, loại cạnh và loại đỉnh đích.

Cả R-GCN và HGT đều thuộc Phase 6 và chưa làm.

---

Xem tiếp: [Phụ lục B. Tính tay một ví dụ từ đầu đến cuối](phu-luc-b-tinh-tay-tu-dau-den-cuoi.md).
