# Quyết định kỹ thuật: GraphSAGE trong pipeline truy hồi

Cập nhật: 18/11/2026. Trạng thái: pipeline và ablation đã triển khai; fidelity perturbation đã chạy. Tài liệu này là nơi ghi quyết định hiện hành cho Phase 1; plan chi tiết nằm ở `plans/261117-0037-multimodal-graph-retrieval-roadmap/`.

Minh hoạ:

- `docs/diagrams/graphsage-how-it-works.html`: đi từng bước trên đồ thị của một ảnh thật (3 tầng, huấn luyện, chấm điểm).
- `docs/diagrams/graphsage-pipeline.html`: vị trí của GraphSAGE trong cả pipeline.

## 1. Đề bài yêu cầu gì

| Yêu cầu | Cách đáp ứng |
|---|---|
| Đỉnh: ảnh, đối tượng, khái niệm | Đỉnh ảnh và đỉnh đối tượng có từ Phase 1. Đỉnh khái niệm dùng chung thêm ở Phase 7 (xem G3) |
| Cạnh: ảnh – chứa – đối tượng; quan hệ giữa các đối tượng; quan hệ giữa khái niệm từ KG ngoài | Hai loại đầu có từ Phase 1. Cạnh ConceptNet thuộc Phase 7 |
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
| G3 | Cài theo kiểu **ghép batch đồ thị con của từng ảnh**; đỉnh khái niệm dùng chung để Phase 7 | Chốt |
| G4 | 3 tầng GraphSAGE, gộp bằng trung bình | Chốt (hệ quả của G2) |
| G5 | Dùng **`SAGEConv` của PyTorch Geometric**, không tự code lớp | Chốt |
| G6 | Vector ban đầu: CLIP text cho đối tượng và đỉnh phụ; đỉnh ảnh làm hai biến thể | Chốt |
| G7 | Huấn luyện bằng **InfoNCE trong batch**; CLIP đóng băng | Chốt |
| G8 | Giải thích bằng cách bỏ từng relation rồi đo điểm đồ thị giảm bao nhiêu | Chốt |
| G9 | **Mô hình hoá hướng của relation ngay ở Phase 1:** mỗi tầng có hai `SAGEConv`, một cho chiều xuôi và một cho chiều ngược | Chốt |
| G10 | Xoá toàn bộ code thuật toán cũ, viết mới | Chốt |

### G1. Truy vấn dùng CLIP text

Điểm đồ thị so vector CLIP text của truy vấn với vector đồ thị của ảnh. GraphSAGE được huấn luyện để vector đồ thị rơi vào không gian của CLIP text.

- **Lý do:** không cần bộ phân tích câu, không có lỗi phân tích, truy vấn giữ nguyên thông tin về hành động và màu sắc.
- **Phương án chưa làm:** truy vấn → SceneGraphParser → đồ thị → GraphSAGE, rồi so đồ thị với đồ thị. Phương án này cho phần giải thích trực quan hơn (bộ ba của truy vấn khớp bộ ba của ảnh). Quay lại nếu còn thời gian, dưới dạng một biến thể so sánh ở Phase 4.

### G2. Cấu trúc đồ thị của một ảnh

- Một **đỉnh ảnh**, nối với mọi đối tượng của ảnh (cạnh "chứa").
- **Đỉnh đối tượng:** mỗi khái niệm có trong ảnh một đỉnh (hai người trong một ảnh gộp thành một đỉnh `man`, theo kết quả bước 2.7).
- **Relation** `man —riding→ bicycle` là cạnh giữa hai đối tượng. Lớp `SAGEConv` không đọc được nhãn trên cạnh, nên cạnh được lưu thành `man — (riding) — bicycle`, với đỉnh phụ `(riding)` mang vector của chữ "riding".

Đỉnh phụ là cách lưu, không đổi ý nghĩa: relation vẫn là quan hệ giữa hai đối tượng. Nhờ nó, "man riding bicycle" và "man next to bicycle" cho vector khác nhau, và ablation "bỏ vị từ" ở Phase 5 đo được thật.

### G3. Ghép batch đồ thị con của từng ảnh

Mỗi ảnh là một đồ thị con. Khi huấn luyện, nhiều đồ thị con được ghép thành một batch và chạy qua **cùng một bộ trọng số**.

- **Vì sao không dựng một đồ thị chung ngay:** đối tượng là đỉnh riêng của từng ảnh, chỉ đỉnh khái niệm là dùng chung. Thông tin từ ảnh khác phải đi qua đỉnh khái niệm dùng chung, mà đỉnh đó chỉ có ích khi có cạnh ConceptNet giữa các khái niệm (Phase 7). Trước lúc đó, đồ thị chung không thêm thông tin gì cho vector của ảnh, trong khi phải lấy mẫu hàng xóm và phải có quy tắc chống rò rỉ giữa ảnh train và ảnh test.
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

Dùng `SAGEConv` và tiện ích ghép batch của PyTorch Geometric. Lý do: lớp đã được kiểm thử rộng rãi, báo cáo chỉ cần trích thư viện, và Phase 6 (R-GCN/HGT) cũng có sẵn trong đó. `.venv` hiện có torch 2.15, chưa cài PyG; cần thêm vào `requirements.txt`.

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

Với mỗi ứng viên được trả về: lần lượt bỏ từng relation khỏi đồ thị của ảnh, tính lại điểm đồ thị, và báo các relation làm điểm giảm nhiều nhất. Cách này cho đúng thứ đề bài gọi là "quan hệ đóng góp vào điểm đồ thị", và chính phép đo "bỏ đi thì điểm giảm bao nhiêu" là cơ sở để đánh giá fidelity ở Phase 5.

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

Chưa có giá trị nào được chốt. Các tham số sẽ quét: số chiều ẩn, cỡ batch, nhiệt độ của InfoNCE, tốc độ học, số vòng, k của top-k, α.

## 6. Giới hạn đã biết

- Ảnh nào cũng phải có sẵn đồ thị cảnh. Ảnh ngoài Visual Genome không chạy được nếu không có SGG.
- Đồ thị nắm "có vật gì và quan hệ gì", gần như không có màu sắc, kích thước, bối cảnh (thuộc tính VG chưa đưa vào).
- Từ gần nghĩa chưa gộp (`guy` và `man` là hai đỉnh khác nhau).

## 7. Còn mở

- Nhánh `feat/vg-coco-graph-build` (code pipeline và dữ liệu đồ thị) chưa gộp vào `main`; Phase 1 cần nó.
- Định dạng `image_graphs.json` của bước 2.7 đã đủ để dựng đồ thị con theo G2 (khái niệm và bộ ba theo id); không cần chạy lại bước 2.7.


## Ablation và fidelity

Run from the repository root with Python 3.12 and the requirements installed.
CLIP embeddings and all VG-COCO images are local inputs, not Git artifacts.

## Reproduce

```bash
python -m scripts.embed_vg_coco  # only if the embedding cache is missing
for variant in full objects_only drop_predicates random_edges undirected; do
  python -m scripts.train_graphsage --config "configs/retrieval/ablations/${variant}.yaml" --resume || break
  python -m scripts.run_benchmark --method graphsage --config "configs/retrieval/ablations/${variant}.yaml" || break
done
python -m scripts.analyze_ablation
python -m scripts.evaluate_fidelity --config configs/evaluation/fidelity.yaml
```

Training configs contain seeds 13, 37 and 97. W&B is disabled. Default names
separate each variant and topology seed; do not override `--name` to reuse a
baseline directory for a different intervention. `--resume` rejects incomplete
finished artifacts or incompatible settings. An exported run is reusable only
with its original embedding cache, graph and split hashes. Benchmark/explanation
consumers reject mismatched or stale inputs.

The full training snapshots from the 18/11 restoration record
`private/ablation-execution/full.yaml`, the historical path actually used.
`configs/retrieval/ablations/full.yaml` provides the same model/training settings
publicly; replay does not require the private file. Snapshots are records, not
input configs to edit. The original 17/11 baseline remains in Git history;
current full run artifacts are the 18/11 restoration, not a byte-identical rerun.

## Intervention contract

- `objects_only`: remove triples before PyG construction; concept nodes and
  image→object edges remain. This is still a GNN, not the lexical baseline.
- `drop_predicates`: zero initial relation features; retain relation nodes,
  connectivity, direction and message passing. This does not remove all relation information.
- `random_edges`: independently permute subjects and objects using a private RNG
  keyed by image ID and ablation_seed. Preserve directed degrees, relation count
  and predicate histogram. Duplicate triples/self-relations are permitted.
- `undirected`: use the existing undirected layer; fewer parameters than full,
  so this is not a parameter-matched direction experiment.

Annotation files are never overwritten. Each intervention is used consistently
for training, validation, vector export and explanation. Alpha/k are selected
independently on validation; test is not used for tuning.

## Analysis and interpretation

`runs/analysis/ablation.json` includes official per-seed metrics, paired top-1
wins/losses, ten deterministic case studies, and an exploratory 95% image-cluster
bootstrap interval (2,000 draws, seed 310). Captions of the same true image remain
together; training checkpoints are fixed. Intervals are conditional on these
three models, not uncertainty over all possible training seeds, and are not
adjusted for multiple comparisons.

The fixed-k=500 diagnostic retains each original alpha; it is post-hoc and does
not replace the official benchmark or define a new tuned system. Two full seeds
selected k=10, which explains their unchanged R@10 versus CLIP. Do not infer that
full necessarily has a worse graph encoder from that fused R@10 alone.

## Fidelity protocol

`configs/evaluation/fidelity.yaml` fixes 200 test queries sampled with seed 2026,
all three trained seeds, removal fractions 0.10/0.25, and 20 random controls.
There is no retraining or parameter search. The target is each query's original
predicted top-1 image, whether correct or wrong. Query, alpha/k, CLIP scores,
candidate pool and all rival graph scores are frozen. Only the target graph changes.

Relations are ordered by single-deletion graph score drop, then the selected
group is deleted jointly and compared with equal-size random groups. At least
two relations are removed and one remains: requested budget fractions can
therefore differ from actual fractions for small graphs. Graphs with fewer than
three relations are explicitly skipped; the same sampled queries remain in the
output, including skip reasons. Negative score drops are retained.

`runs/fidelity/full-joint-test.json` stores per-query indices, random groups,
graph score changes, target rank changes, top-1 losses, hashes and summaries.
Joint deletion and downstream rank are additional diagnostics, but selection
still used deletion scores. This is selection-dependent perturbation evidence,
not independent proof of semantic faithfulness or causal superiority of relations.
Deleting relation nodes also changes topology and message normalization.

## Observed evidence, 18/11/2026

Full vs objects-only top-1 gain is 0.56 pp; exploratory conditional image-cluster
95% CI [0.05, 1.06] pp. Full vs masked predicates is about 0.053 pp, CI
[-0.35, 0.45]; full vs random edges 0.329 pp, CI [-0.09, 0.74]. These do not
establish a clear label/connectivity advantage across arbitrary retraining runs.

Fidelity evaluated 198/195/196 of the 200 sampled queries for seeds 13/37/97.
Across seed means, selected-group top-1 loss was 17.81% vs random 4.31% for the
10% requested budget, and 25.45% vs 6.06% for 25%. Actual integer budgets and
skipped queries are recorded. Graph score decreases were 0.02200 vs 0.00142 and
0.03040 vs 0.00216 respectively. Do not count the three measurements of one query
as three independent queries.

Examples expose potentially noisy or irrelevant graph annotations (e.g.
`skateboard has arm`, `person with platform`), but ten selected cases do not prove
the prevalence of annotation noise or that it caused aggregate performance.
Caption-type prevalence and CLIP's sensitivity to subject/object roles remain
hypotheses requiring a separately defined evaluation. Fidelity does not resolve them.

## Validation and remaining scope

Run `pytest`, `ruff check src scripts tests`, and `mypy src scripts` with the
requirements installed. Tests cover transformations, batching, backward,
checkpoint/export, resume integrity, stale provenance, joint perturbation budgets,
negative drops, conservative ranks and clustered paired analysis.

The analysis and fidelity commands were run on real local artifacts, without
changing retrieval hyperparameters after seeing test results. Phase 5 still
requires the second dataset and hub-policy ablation; an independent explanation
sanity check and parameter-matched direction control remain useful extensions.

## Source formatting and local artifacts

New ablation/analysis/fidelity files use grouped processing steps, short English
operation comments and multiline parameters for substantial calls. Existing
modules are formatted only in changed regions; unrelated legacy code keeps its
original formatting. Imports follow Ruff's import-order rules.

`/runs/` and `/private/` are ignored for newly generated artifacts. Historical
files already tracked under `runs/` remain tracked: `.gitignore` does not hide
or remove their modifications. No tracked result, checkpoint or local output is
deleted or untracked by this policy. Review tracked experiment changes separately
from source changes; regenerate local outputs using the commands above.

The current requirements checklist is `plan-task/requirements-checklist.md`.
It replaces the obsolete 40-image pilot status and explicitly separates measured
perturbation fidelity from unverified semantic validity and final deliverables.
