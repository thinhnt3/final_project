# 5. Chạy lại và bản đồ code

## Chuẩn bị

- Python 3.12 trở lên, vì code dùng cú pháp generic có từ Python 3.12.
- Cài thư viện bằng `pip install -r requirements.txt`. Danh sách gồm torch, torch_geometric, transformers, faiss-cpu, scikit-learn, wandb, pytest, ruff, mypy và một số thư viện khác.
- Chạy mọi lệnh **từ thư mục gốc của repo**, theo dạng `python -m scripts.<tên>`, vì đường dẫn trong config được tính từ thư mục gốc.
- `train_graphsage` gửi báo cáo lên Weights & Biases, vì `configs/retrieval/vg_coco.yaml` đặt `wandb.mode: online`. Nếu không dùng W&B, đổi giá trị này thành `offline` hoặc `disabled`.

## Lệnh và file sinh ra

![Lệnh chạy và file sinh ra](images/lenh-va-file.svg)

```bash
python -m scripts.embed_vg_coco                          # vector CLIP của ảnh, caption và chữ trên đồ thị (khoảng 3 phút trên MPS)
python -m scripts.run_benchmark --method clip            # baseline CLIP
python -m scripts.run_benchmark --method graph_lexical   # CLIP + so khớp từ (graph_lexical_objects: chỉ dùng khái niệm)
python -m scripts.train_graphsage                        # 3 seed; thêm --seed 13 để chỉ chạy một seed
python -m scripts.run_benchmark --method graphsage       # CLIP + GraphSAGE, đọc vector đồ thị của mọi seed
python -m scripts.explain_retrieval --sample 5           # 5 truy vấn test ngẫu nhiên, kèm relation giải thích
python -m scripts.evaluate_fidelity --output runs/fidelity/learned-seed13-test.json   # đo fidelity (tên file tuỳ chọn)
python -m scripts.make_doc_figures                       # vẽ lại các hình trong docs/images/ từ file kết quả
pytest                                                   # toàn bộ test, không cần mạng
```

Các tuỳ chọn của `evaluate_fidelity`:

| Tuỳ chọn | Mặc định | Tác dụng |
|---|---|---|
| `--output` | (bắt buộc) | File kết quả mới. Script từ chối nếu file đã tồn tại |
| `--run` | kiến trúc trong config | Lượt huấn luyện cần đo |
| `--seed` | seed đầu tiên trong config | Seed của checkpoint |
| `--split` | `test` | `val` hoặc `test` |
| `--sample` | 10 | Số truy vấn được lấy mẫu ngẫu nhiên (cố định theo seed) |
| `--budget` | 3 | Số relation tối đa trong tập giải thích |

Script cần đã chạy `train_graphsage` và `run_benchmark --method graphsage` cho lượt đó. Nó kiểm tra sha256 của dữ liệu, của checkpoint và của file benchmark, rồi mới đo.

Biến thể có đỉnh ảnh mang vector CLIP:

```bash
python -m scripts.train_graphsage --image-node clip_image
python -m scripts.run_benchmark --method graphsage --run directed-clip_image-l3-h512
```

Các tuỳ chọn hay dùng của `train_graphsage`:

| Tuỳ chọn | Tác dụng |
|---|---|
| `--resume` | Bỏ qua seed đã xong; seed bị ngắt thì chạy tiếp từ `last.pt` và cho kết quả trùng với lượt chạy liền mạch |
| `--overwrite` | Xoá lượt cũ của seed rồi chạy lại từ đầu |
| `--seed N` | Chỉ chạy seed N (có thể lặp lại để chạy nhiều seed) |
| `--image-node clip_image` | Chạy biến thể có đỉnh ảnh mang vector CLIP |

Nếu thư mục của lượt chạy đã tồn tại, script từ chối chạy, trừ khi có `--resume` hoặc `--overwrite`. Với `--resume`, script cũng từ chối nếu config đã thay đổi, trừ các trường `max_epochs`, `patience` và `seeds`.

## Cái gì có sẵn trong repo, cái gì phải tự tạo

| Có sẵn trong repo | Phải tự tạo |
|---|---|
| JSON dữ liệu đã xử lý: `data/processed/vg_coco/*.json`, `graph/*.json`, `data/splits/` | 10.000 file ảnh (1,4 GB) trong `data/processed/vg_coco/images/` |
| Kết quả benchmark: `runs/benchmark/*.json` | Vector CLIP, bằng `embed_vg_coco` |
| `config.yaml`, `history.json`, `result.json` của từng seed | Checkpoint `*.pt` và `graph_vectors.*`, bằng `train_graphsage` |
| Kết quả giải thích: `runs/explanations/*.json` | |

Hai script hiện không nằm trong repo: script tải ảnh và dựng dữ liệu (`scripts/build_vg_coco.py`) và script duyệt dữ liệu (`scripts/review_vg_coco.py`). Các module mà chúng dùng (`src/data/`) và test của các module đó thì có trong repo. Từ đó:

- Muốn chạy `embed_vg_coco` thì phải có sẵn 10.000 file ảnh.
- Muốn chạy `run_benchmark --method graphsage` và `explain_retrieval` thì phải chạy `train_graphsage` trên máy trước, vì checkpoint và vector đồ thị không được commit.
- Nếu gặp lỗi SSL khi tải ảnh bằng Python cài qua uv, đặt biến `SSL_CERT_FILE` trỏ tới file `certifi/cacert.pem`.

Các file nén kết quả đặt ở thư mục gốc, ví dụ `runs.zip` (bản lưu các thí nghiệm cũ `pilot_v2`, `coco2017_level2`, `vietnam_mmkg`…), được `.gitignore` bỏ qua. Chúng quá lớn để đưa vào git, nên hãy chia sẻ qua kênh khác.

## Bản đồ code

| Thư mục | Vai trò | File chính |
|---|---|---|
| `configs/` | Chứa mọi tham số dưới dạng YAML; trong code không có số nào viết cứng | `data/vg_coco.yaml` (dựng dữ liệu); `retrieval/vg_coco.yaml` (CLIP, benchmark, GraphSAGE, huấn luyện, W&B) |
| `src/core/` | Các ngoại lệ dùng chung và hàm đọc config | `exceptions.py`, `config_parsing.py` |
| `src/data/` | Dựng dữ liệu: tải, ghép, chuẩn hoá synset, chia tập, xây đồ thị | `graph_build.py`, `splits.py`, `synsets.py`, `vg_coco_join.py` |
| `src/embeddings/` | CLIP đóng băng; bảng vector lưu trên đĩa (gồm ma trận, danh sách id và sha256) | `clip_encoder.py`, `store.py` |
| `src/models/` | Đồ thị con, GraphSAGE, huấn luyện, checkpoint, giải thích, biến đổi đồ thị cho ablation | `graph_data.py`, `graphsage.py`, `train.py`, `checkpoint.py`, `explain.py`, `ablations.py` |
| `src/retrieval/` | Trộn điểm và quét α, k; so khớp từ; đọc config | `rerank.py`, `lexical.py`, `config.py` |
| `src/evaluation/` | Tính R@K, MRR; chuẩn bị dữ liệu và chấm điểm cho từng tập; đo fidelity | `metrics.py`, `benchmark.py`, `fidelity.py` |
| `src/connectors/` | Kết nối Weights & Biases | `wandb_tracker.py` |
| `scripts/` | Các lệnh chạy ở trên; `make_doc_figures.py` vẽ hình cho `docs/` | |
| `tests/` | Test, chia thư mục theo cấu trúc của `src/`; dùng đồ thị dựng tay và bộ mã hoá giả | |

Quy ước của code:

- Hàm thuần không đọc hay ghi file. Việc đọc/ghi nằm trong các hàm load/save và trong `scripts/`.
- Lỗi được báo bằng ngoại lệ riêng (`ConfigError`, `DataError`, `TrainingError`) ngay tại nơi phát hiện. Mỗi script bắt lỗi một lần và trả mã lỗi 1.
- Config được đọc thành dataclass bất biến và được kiểm tra hợp lệ ngay khi đọc.
- Tên biến, log và comment viết bằng tiếng Anh; tài liệu và kế hoạch viết bằng tiếng Việt.

## Kiểm tra chất lượng

```bash
pytest                              # toàn bộ test
pytest tests/retrieval              # test của một thư mục
pytest tests/retrieval/test_rerank.py::test_alpha_one_keeps_the_clip_ranking_for_any_k   # một test
ruff check .                        # lint
mypy                                # kiểm tra kiểu cho src/ và scripts/
```

Chất lượng của bộ test đã được kiểm bằng cách cố ý gài lỗi vào code. Ví dụ, test xếp hạng lại bắt được 12/12 lỗi gài, test huấn luyện bắt được 17/17.
