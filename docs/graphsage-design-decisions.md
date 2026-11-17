# Quyết định kỹ thuật: GraphSAGE trong pipeline truy hồi

Cập nhật: 18/11/2026. Trạng thái: pipeline và ablation đã triển khai; fidelity perturbation đã chạy. Tài liệu này là nơi ghi quyết định hiện hành cho Giai đoạn 1; plan chi tiết nằm ở `plans/261117-0037-multimodal-graph-retrieval-roadmap/`.

Minh hoạ:

- `docs/diagrams/graphsage-how-it-works.html`: đi từng bước trên đồ thị của một ảnh thật (3 tầng, huấn luyện, chấm điểm).
- `docs/diagrams/graphsage-pipeline.html`: vị trí của GraphSAGE trong cả pipeline.

## 1. Đề bài yêu cầu gì

| Yêu cầu | Cách đáp ứng |
|---|---|
| Đỉnh: ảnh, đối tượng, khái niệm | Đỉnh ảnh và đỉnh đối tượng có từ Giai đoạn 1. Đỉnh khái niệm dùng chung thêm ở Giai đoạn 7 (xem G3) |
| Cạnh: ảnh – chứa – đối tượng; quan hệ giữa các đối tượng; quan hệ giữa khái niệm từ KG ngoài | Hai loại đầu có từ Giai đoạn 1. Cạnh ConceptNet thuộc Giai đoạn 7 |
| Cấp độ 1: một đồ thị tĩnh từ dữ liệu có sẵn | Đồ thị xây từ nhãn Visual Genome ở bước 2.7, không đổi khi chạy |
| Truy vấn → CLIP → top-k cosine → điểm đồ thị → xếp hạng lại → giải thích → fidelity | Giữ nguyên thứ tự này |
| Điểm = α·cosine + (1−α)·điểm đồ thị; điểm đồ thị từ nhúng GCN/GAT/GraphSAGE | Dùng GraphSAGE |
| α và siêu tham số chọn trên validation | Không dùng tập test để chọn bất cứ thứ gì |
| Được điều chỉnh cấu trúc nếu giải thích được cơ chế | Ba điều chỉnh: đỉnh phụ mang nhãn relation (G2), truy vấn không qua đồ thị (G1), hai lớp gộp theo hai chiều cạnh (G9) |

## 2. GraphSAGE làm gì

GraphSAGE ở đây là bộ mã hoá **đồ thị thành vector**. Nó không sinh đồ thị cảnh từ ảnh (SGG); dự án không có bước SGG. Đồ thị cảnh lấy từ nhãn do người gán của Visual Genome.

- **Chuẩn bị trước:** đồ thị của ảnh → GraphSAGE → một vector cho mỗi ảnh ("vector đồ thị"), cùng không gian với CLIP. Tính một lần cho cả 10.000 ảnh rồi lưu.
- **Lúc chạy:** truy vấn → CLIP text → top-k ảnh theo cosine với vector CLIP ảnh → điểm đồ thị của từng ứng viên = cosine(vector truy vấn, vector đồ thị) → xếp hạng lại.

## 3. Các quyết định

| # | Quyết định | Trạng thái |
|---|---|---|
| G1 | Truy vấn mã hoá bằng **CLIP text nguyên câu**, không phân tích thành đồ thị | Chốt |
| G2 | Đối tượng là đỉnh, relation là cạnh; nhãn relation lưu trên một **đỉnh phụ** giữa cạnh | Chốt |
| G3 | Cài theo kiểu **ghép batch đồ thị con của từng ảnh**; đỉnh khái niệm dùng chung để Giai đoạn 7 | Chốt |
| G4 | 3 tầng GraphSAGE, gộp bằng trung bình | Chốt (hệ quả của G2) |
| G5 | Dùng **`SAGEConv` của PyTorch Geometric**, không tự code lớp | Chốt |
| G6 | Vector ban đầu: CLIP text cho đối tượng và đỉnh phụ; đỉnh ảnh làm hai biến thể | Chốt |
| G7 | Huấn luyện bằng **InfoNCE trong batch**; CLIP đóng băng | Chốt |
| G8 | Giải thích bằng cách bỏ từng relation rồi đo điểm đồ thị giảm bao nhiêu | Chốt |
| G9 | **Mô hình hoá hướng của relation ngay ở Giai đoạn 1:** mỗi tầng có hai `SAGEConv`, một cho chiều xuôi và một cho chiều ngược | Chốt |
| G10 | Xoá toàn bộ code thuật toán cũ, viết mới | Chốt |

### G1. Truy vấn dùng CLIP text

Điểm đồ thị so vector CLIP text của truy vấn với vector đồ thị của ảnh. GraphSAGE được huấn luyện để vector đồ thị rơi vào không gian của CLIP text.

- **Lý do:** không cần bộ phân tích câu, không có lỗi phân tích, truy vấn giữ nguyên thông tin về hành động và màu sắc.
- **Phương án chưa làm:** truy vấn → SceneGraphParser → đồ thị → GraphSAGE, rồi so đồ thị với đồ thị. Phương án này cho phần giải thích trực quan hơn (bộ ba của truy vấn khớp bộ ba của ảnh). Quay lại nếu còn thời gian, dưới dạng một biến thể so sánh ở Giai đoạn 4.

### G2. Cấu trúc đồ thị của một ảnh

- Một **đỉnh ảnh**, nối với mọi đối tượng của ảnh (cạnh "chứa").
- **Đỉnh đối tượng:** mỗi khái niệm có trong ảnh một đỉnh (hai người trong một ảnh gộp thành một đỉnh `man`, theo kết quả bước 2.7).
- **Relation** `man —riding→ bicycle` là cạnh giữa hai đối tượng. Lớp `SAGEConv` không đọc được nhãn trên cạnh, nên cạnh được lưu thành `man — (riding) — bicycle`, với đỉnh phụ `(riding)` mang vector của chữ "riding".

Đỉnh phụ là cách lưu, không đổi ý nghĩa: relation vẫn là quan hệ giữa hai đối tượng. Nhờ nó, "man riding bicycle" và "man next to bicycle" cho vector khác nhau, và ablation "bỏ vị từ" ở Giai đoạn 5 đo được thật.

### G3. Ghép batch đồ thị con của từng ảnh

Mỗi ảnh là một đồ thị con. Khi huấn luyện, nhiều đồ thị con được ghép thành một batch và chạy qua **cùng một bộ trọng số**.

- **Vì sao không dựng một đồ thị chung ngay:** đối tượng là đỉnh riêng của từng ảnh, chỉ đỉnh khái niệm là dùng chung. Thông tin từ ảnh khác phải đi qua đỉnh khái niệm dùng chung, mà đỉnh đó chỉ có ích khi có cạnh ConceptNet giữa các khái niệm (Giai đoạn 7). Trước lúc đó, đồ thị chung không thêm thông tin gì cho vector của ảnh, trong khi phải lấy mẫu hàng xóm và phải có quy tắc chống rò rỉ giữa ảnh train và ảnh test.
- **Hệ quả tốt:** ảnh test không thể ảnh hưởng tới ảnh train, vì không có cạnh nào nối hai đồ thị con.

### G4. Ba tầng

Mỗi tầng đưa thông tin đi thêm một bước trên đồ thị:

- **Tầng 1:** ảnh biết có những đối tượng nào.
- **Tầng 2:** ảnh biết vai của từng đối tượng trong relation: `man` là chủ thể của `riding`, `bicycle` là đối tượng của `riding`. Với ảnh chỉ có một relation `riding`, từng đó đã đủ để nhận ra "man riding bicycle".
- **Tầng 3:** ảnh biết hai đầu của cùng một relation đi với nhau. Tầng này cần khi ảnh có nhiều relation cùng vị từ: "man riding bicycle, woman riding horse" khác "man riding horse, woman riding bicycle". Với 2 tầng, hai ảnh đó cho cùng vector (có test).

Công thức mỗi tầng là của paper GraphSAGE (biến thể trung bình):

```
h = W·x + W'·trung_bình(hàng xóm)
```

Sau tầng cuối không dùng ReLU: vector CLIP có cả thành phần âm, nên đầu ra của GraphSAGE cũng phải có. Code cũ mắc lỗi này.

### G5. Thư viện

Dùng `SAGEConv` và tiện ích ghép batch của PyTorch Geometric. Lý do: lớp đã được kiểm thử rộng rãi, báo cáo chỉ cần trích thư viện, và Giai đoạn 6 (R-GCN/HGT) cũng có sẵn trong đó. `.venv` hiện có torch 2.15, chưa cài PyG; cần thêm vào `requirements.txt`.

### G6. Vector ban đầu của các đỉnh

- **Đối tượng, đỉnh phụ:** vector CLIP text của chữ hiển thị ("man", "riding"). Tính một lần, lưu sẵn.
- **Đỉnh ảnh, mô hình chính:** một vector học được, dùng chung cho mọi ảnh. Vector đồ thị vì thế chỉ đến từ cấu trúc, và bảng "CLIP so với CLIP + Graph" cho biết phần cải thiện đến từ đồ thị.
- **Đỉnh ảnh, biến thể so sánh:** vector CLIP của chính ảnh đó. Đây là cách dùng GraphSAGE thông thường và có thể cho số cao hơn, nhưng điểm đồ thị khi đó đã lẫn thông tin CLIP ảnh.

Cả hai biến thể được huấn luyện và đưa vào bảng kết quả.

### G7. Huấn luyện

- Một batch gồm B cặp (ảnh, caption) từ tập train; mỗi ảnh xuất hiện một lần trong batch.
- Tính bảng điểm B × B giữa vector đồ thị của ảnh và vector CLIP text của caption. InfoNCE đẩy đường chéo lên, kéo phần còn lại xuống, theo cả hai chiều (ảnh → caption và caption → ảnh).
- Chỉ GraphSAGE và tầng chiếu được học. CLIP đóng băng; vector CLIP đọc từ bộ lưu sẵn.
- Dừng huấn luyện theo kết quả trên tập validation.

### G8. Giải thích kết quả

Với mỗi ứng viên được trả về: lần lượt bỏ từng relation khỏi đồ thị của ảnh, tính lại điểm đồ thị, và báo các relation làm điểm giảm nhiều nhất. Cách này cho đúng thứ đề bài gọi là "quan hệ đóng góp vào điểm đồ thị", và chính phép đo "bỏ đi thì điểm giảm bao nhiêu" là cơ sở để đánh giá fidelity ở Giai đoạn 5.

### G9. Hướng của relation

Nếu thông tin truyền hai chiều qua cạnh bằng cùng một bộ trọng số thì "man riding bicycle" và "bicycle riding man" cho cùng vector. Để phân biệt, mỗi tầng dùng **hai `SAGEConv`**:

- một lớp gộp hàng xóm theo chiều xuôi của cạnh (chủ thể → đỉnh phụ → đối tượng; ảnh → đối tượng);
- một lớp gộp theo chiều ngược;
- đầu ra của tầng là tổng của hai lớp.

```
h = SAGE_xuôi(x, cạnh) + SAGE_ngược(x, cạnh đảo chiều)
```

Ví dụ đã chạy thử (PyTorch Geometric 2.8.1):

```python
import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class DirectedSAGELayer(nn.Module):
    """One GraphSAGE layer that keeps edge direction: one SAGEConv per direction, summed."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.forward_conv = SAGEConv(in_dim, out_dim, aggr="mean")   # messages along the edge
        self.backward_conv = SAGEConv(in_dim, out_dim, aggr="mean")  # messages against the edge

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.forward_conv(x, edge_index) + self.backward_conv(x, edge_index.flip(0))


# man(0) -> (riding)(1) -> bicycle(2)
x = torch.randn(3, 8)                               # one row per node
man_rides_bike = torch.tensor([[0, 1], [1, 2]])     # row 0: source nodes, row 1: target nodes
bike_rides_man = torch.tensor([[2, 1], [1, 0]])

layer = DirectedSAGELayer(8, 4)
layer(x, man_rides_bike)   # shape (3, 4)
layer(x, bike_rides_man)   # a different result: direction is visible to the model
```

`edge_index` là ma trận 2 × số cạnh: hàng trên là đỉnh nguồn, hàng dưới là đỉnh đích. `edge_index.flip(0)` đổi chỗ hai hàng, tức đảo chiều mọi cạnh. Với một `SAGEConv` duy nhất chạy trên cạnh hai chiều, hai đồ thị trên cho cùng kết quả; với lớp có hướng thì khác nhau (đã kiểm bằng chạy thử).

Mỗi lớp vẫn là `SAGEConv` nguyên bản; chỉ có cách ghép hai lớp là thêm vào. Đỉnh vẫn nhận thông tin từ cả hai phía, nhưng biết thông tin đến từ phía nào. Config có cờ tắt hướng (một lớp, cạnh hai chiều) để đo đóng góp của hướng bằng ablation.

### G10. Code cũ

Xoá toàn bộ và viết mới: `src/models/`, `src/retrieval/`, `src/evaluation/`, các script `run_graphsage.py`, `run_clip_retrieval.py`, `run_ablation.py`, `smoke_retrieval.py`, hai script chuẩn bị tập 40 ảnh, và ba file trong `runs/`. Code cũ vẫn còn trong lịch sử git. Phần so khớp từ vựng sẽ được viết lại thành baseline B4 có tên rõ ràng, không còn đóng vai điểm đồ thị.

## 4. Vì sao code cũ phải viết lại

| Chỗ | Vấn đề |
|---|---|
| `src/retrieval/reranker.py`, `graph_score` | Điểm đồ thị chỉ đếm từ trùng giữa truy vấn và bộ ba. GraphSAGE chưa từng được nối vào xếp hạng lại |
| `src/models/graphsage.py`, tầng cuối | Có ReLU trước khi so cosine với vector CLIP |
| `src/models/graphsage.py`, `graph_inputs` | Bỏ hết nhãn relation |
| `scripts/run_graphsage.py`, dòng 36–38 | Margin loss với một ảnh sai cố định cho gần như mọi truy vấn; 8 vòng; không có validation |
| `scripts/run_graphsage.py`, phần đánh giá | Xếp hạng trên cả ảnh train |
| `scripts/run_clip_retrieval.py` | "3 seed" cho ba lần chạy giống hệt nhau, độ lệch chuẩn luôn bằng 0 |
| Tập 40 ảnh | Truy vấn sinh từ chính đồ thị, nên so khớp từ vựng được lợi sẵn |

Lớp `MeanGraphSAGELayer` cũ đúng công thức của paper; nó được thay bằng `SAGEConv` chỉ vì G5.

## 5. Siêu tham số chọn trên validation

Đã có cấu hình khởi đầu và lựa chọn α/k trên tập xác thực; chưa khảo sát rộng mọi siêu tham số. Các tham số có thể nghiên cứu tiếp: số chiều ẩn, cỡ batch, nhiệt độ của InfoNCE, tốc độ học, số vòng, k của top-k, α.

## 6. Giới hạn đã biết

- Ảnh nào cũng phải có sẵn đồ thị cảnh. Ảnh ngoài Visual Genome không chạy được nếu không có SGG.
- Đồ thị nắm "có vật gì và quan hệ gì", gần như không có màu sắc, kích thước, bối cảnh (thuộc tính VG chưa đưa vào).
- Từ gần nghĩa chưa gộp (`guy` và `man` là hai đỉnh khác nhau).

## 7. Còn mở

- Pipeline xây đồ thị đã có trên `main`; ghi chú nhánh chưa gộp trước đây không còn áp dụng.
- Định dạng `image_graphs.json` của bước 2.7 đã đủ để dựng đồ thị con theo G2 (khái niệm và bộ ba theo id); không cần chạy lại bước 2.7.


## Ablation và fidelity

Chạy từ thư mục gốc, với Python 3.12 và các thư viện trong `requirements.txt`.
Ảnh VG ∩ COCO, vector CLIP và điểm lưu mô hình là dữ liệu local, không nằm trong Git.

## Reproduce

```bash
python -m scripts.embed_vg_coco  # Chỉ cần khi thiếu bộ nhớ đệm vector
for variant in full objects_only drop_predicates random_edges undirected; do
  python -m scripts.train_graphsage --config "configs/retrieval/ablations/${variant}.yaml" --resume || break
  python -m scripts.run_benchmark --method graphsage --config "configs/retrieval/ablations/${variant}.yaml" || break
done
python -m scripts.analyze_ablation
python -m scripts.evaluate_fidelity --config configs/evaluation/fidelity.yaml
```

Cấu hình có các hạt giống huấn luyện 13, 37, 97; đã tắt W&B. Tên mặc định phân
biệt biến thể và hạt giống hoán vị đồ thị. Không dùng `--name` để ghi một biến thể
vào thư mục của bản đầy đủ. `--resume` từ chối lượt đã báo xong nhưng thiếu tệp
cần thiết hoặc có cấu hình khác. Các lệnh đánh giá/giải thích kiểm tra mã băm để
tránh dùng nhầm đầu vào hoặc kết quả cũ.

Bản ghi lượt đầy đủ ngày 18/11 ghi `private/ablation-execution/full.yaml`, đúng
đường dẫn đã dùng lúc chạy. `configs/retrieval/ablations/full.yaml` cung cấp cùng
thiết lập mô hình/huấn luyện để chạy lại mà không cần tệp riêng tư. Không sửa bản
ghi lịch sử để biến nó thành cấu hình đầu vào. Kết quả ngày 17/11 còn trong lịch
sử Git; kết quả đầy đủ hiện tại là lượt khôi phục ngày 18/11, không giống từng byte.

## Intervention contract

- `objects_only`: xoá bộ ba trước khi dựng đồ thị PyG; giữ đỉnh đối tượng và cạnh ảnh → đối tượng. Biến thể vẫn dùng mạng đồ thị, không phải bộ so khớp từ.
- `drop_predicates`: đặt đặc trưng ban đầu của đỉnh quan hệ bằng 0; giữ đỉnh, cạnh, hướng và truyền thông tin. Không loại bỏ toàn bộ thông tin quan hệ.
- `random_edges`: hoán vị chủ thể và tân ngữ độc lập bằng bộ sinh số riêng theo mã ảnh và `ablation_seed`; giữ bậc vào/ra, số quan hệ và phân bố vị từ. Cho phép bộ ba trùng và quan hệ tự nối.
- `undirected`: dùng lớp không hướng sẵn có. Số tham số ít hơn bản đầy đủ, nên chưa tách riêng hoàn toàn ảnh hưởng của hướng và dung lượng mô hình.

Tệp chú giải gốc không bị ghi đè. Cùng phép biến đổi áp dụng cho huấn luyện, tập
xác thực, xuất vector và giải thích. α/k chọn riêng trên tập xác thực; không dùng
tập kiểm thử để điều chỉnh tham số.

## Analysis and interpretation

`runs/analysis/ablation.json` lưu chỉ số từng hạt giống, các truy vấn thắng/thua
theo cặp, mười trường hợp minh hoạ và khoảng bất định khám phá 95%. Phép lấy mẫu
lại thực hiện 2.000 lần với hạt giống 310, gom các mô tả theo ảnh đúng để giữ
phụ thuộc giữa chúng. Ba mô hình đã huấn luyện được giữ cố định. Khoảng này
không mô tả mọi hạt giống huấn luyện có thể có và chưa hiệu chỉnh đa so sánh.

Kiểm tra bổ sung k=500 giữ α đã chọn. Đây là phân tích sau thực nghiệm, không
thay kết quả chính thức hoặc chọn một hệ thống mới. Hai lượt đầy đủ chọn k=10,
giải thích việc R@10 của chúng không đổi so với CLIP. Chỉ từ R@10 kết hợp không
thể kết luận bộ mã hoá đồ thị đầy đủ kém hơn.

## Fidelity protocol

`configs/evaluation/fidelity.yaml` cố định 200 truy vấn kiểm thử lấy mẫu bằng hạt
giống 2026, ba mô hình đã huấn luyện, mức yêu cầu xoá 0,10/0,25 và 20 nhóm ngẫu
nhiên đối chứng. Không huấn luyện lại hay tìm tham số. Mục tiêu là ảnh dự đoán
đứng đầu ban đầu, dù đúng hay sai. Giữ truy vấn, α/k, điểm CLIP, tập ứng viên và
điểm mọi đối thủ; chỉ thay đổi đồ thị của ảnh mục tiêu.

Xếp quan hệ theo mức giảm điểm khi xoá từng quan hệ, rồi xoá đồng thời nhóm được
chọn và so với các nhóm ngẫu nhiên cùng số lượng. Xoá ít nhất hai quan hệ và
chừa ít nhất một; vì thế tỷ lệ thực tế có thể khác mức yêu cầu ở đồ thị nhỏ.
Đồ thị ít hơn ba quan hệ được ghi nhận bỏ qua và nêu lý do. Giữ cả mức giảm âm:
xoá một quan hệ gây nhiễu có thể làm điểm tăng.

`runs/fidelity/full-joint-test.json` lưu vị trí quan hệ được xoá, nhóm ngẫu nhiên,
điểm đồ thị, thay đổi thứ hạng, mất vị trí đầu, mã băm và kết quả tổng hợp.

**Giới hạn:** việc chọn explanation vẫn dựa trên điểm của phép xoá. Xoá nhóm và
đo hạng bổ sung bằng chứng nhưng không độc lập với cách chọn này. Đây là kiểm
tra bằng can thiệp, không chứng minh lời giải thích đúng về ngữ nghĩa hay quan
hệ là nguyên nhân chính của mức tăng truy hồi. Xoá đỉnh quan hệ còn làm đổi cấu
trúc và cách tổng hợp thông tin lân cận.

## Observed evidence, 18/11/2026

Bản đầy đủ hơn chỉ đối tượng 0,56 điểm phần trăm R@1; khoảng 95% có điều kiện là
[0,05; 1,06]. So với bỏ vị từ, chênh lệch chưa làm tròn khoảng 0,053 điểm, khoảng
[−0,35; 0,45]. So với cạnh ngẫu nhiên, khoảng 0,329 điểm, khoảng [−0,09; 0,74].
Chưa xác lập lợi thế rõ của nhãn vị từ hoặc cấu trúc thật qua các lần huấn luyện mới.

Độ trung thực đánh giá được 198/195/196 trong 200 truy vấn theo hạt giống
13/37/97. Trung bình qua ba mô hình, xoá nhóm được chọn làm mất vị trí đầu
17,81% so với ngẫu nhiên 4,31% ở mức yêu cầu 10%; tương ứng 25,45% so với 6,06%
ở mức 25%. Điểm đồ thị giảm 0,02200 so với 0,00142 và 0,03040 so với 0,00216.
Tỷ lệ xoá thực tế và truy vấn bỏ qua được ghi lại. Không coi ba phép đo của cùng
một truy vấn là ba truy vấn độc lập.

Các trường hợp minh hoạ có chú giải khó diễn giải hoặc có thể không liên quan,
như `skateboard has arm`, `person with platform`. Mười ca được chọn chưa đủ chứng
minh mức phổ biến hay tác động tổng thể của nhiễu chú giải. Giả thuyết về loại
mô tả và khả năng CLIP phân biệt chủ thể/tân ngữ cần một phép đánh giá riêng.

## Validation and remaining scope

Chạy `pytest`, `ruff check src scripts tests` và `mypy src scripts`. Kiểm thử bao
phủ phép biến đổi, ghép lô, lan truyền ngược, điểm lưu/xuất vector, tiếp tục chạy,
mã băm đầu vào, số quan hệ xoá, mức giảm âm, xử lý hoà điểm và phân tích theo cặp.
Bộ mã cuối đã qua 263 kiểm thử. Hai bài thực hành lỗi cố ý nằm trong clone riêng
ở `private/bug-lab/`, không phải lỗi của mã chính hay sự cố thực nghiệm thật.

Các lệnh phân tích/độ trung thực đã chạy trên dữ liệu local thật; không chỉnh
tham số truy hồi sau khi thấy kết quả kiểm thử. Giai đoạn 5 còn tập dữ liệu thứ
hai và thí nghiệm xử lý đỉnh bậc cao. Đối chứng hướng có cùng số tham số và kiểm
tra lời giải thích độc lập là các phần mở rộng còn hữu ích.

## Source formatting and local artifacts

Tệp Python mới được định dạng toàn bộ; tệp cũ chỉ sửa vùng thay đổi. Các bước xử
lý được nhóm và có chú thích ngắn; danh sách nhiều tham số được tách dòng. Chú
thích trong mã vẫn giữ tiếng Anh theo quy ước dự án; tài liệu diễn giải dùng
tiếng Việt. Phần nhập mô-đun tuân theo Ruff.

`runs/` và `private/` bỏ qua tệp mới trong Git. Tệp lịch sử đã được theo dõi vẫn
được theo dõi: `.gitignore` không xoá hay giấu thay đổi của chúng. Chính sách này
không xoá hoặc ngừng theo dõi đầu ra cũ. Rà thay đổi kết quả riêng với thay đổi
mã; dùng các lệnh trên để tạo lại đầu ra.

Checklist hiện hành: `plan-task/requirements-checklist.md`; đã thay trạng thái
bản thử 40 ảnh và tách rõ bằng chứng can thiệp với tính đúng ngữ nghĩa cũng như
các sản phẩm nộp chưa hoàn tất.
