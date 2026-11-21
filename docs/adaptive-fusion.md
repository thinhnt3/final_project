# Kết hợp điểm thích nghi theo từng truy vấn

Đề xuất cải tiến của nhóm, thuộc hướng (a) của Cấp độ 4 trong đề: *cơ chế xếp hạng lại hoặc kết hợp điểm thích nghi*.

**Trạng thái:** code và test đã xong, **chưa có kết quả**. Chạy theo mục [Cách chạy](#cách-chạy), rồi điền số vào [bảng kết quả](#bảng-kết-quả-điền-sau-khi-chạy).

| Thành phần | File |
|---|---|
| Thuật toán | `src/retrieval/adaptive_fusion.py` |
| Script chạy | `scripts/evaluate_adaptive_fusion.py` |
| Config | `configs/evaluation/adaptive_fusion.yaml` |
| Test | `tests/retrieval/test_adaptive_fusion.py` |

## Ý tưởng

Hệ thống hiện dùng **một** α và một k cho mọi truy vấn:

```
điểm cuối = α · điểm CLIP + (1 − α) · điểm đồ thị        (trong k ứng viên đầu của CLIP)
```

Nhưng không phải truy vấn nào cũng cần đồ thị như nhau:

- **Khi CLIP đã chắc chắn**, tức ảnh hạng 1 bỏ xa ảnh hạng 2, điểm đồ thị dễ làm hỏng một kết quả vốn đã đúng.
- **Khi CLIP phân vân**, tức hai ảnh đầu sát điểm nhau, điểm đồ thị là thông tin bổ sung có ích nhất.

Đề xuất của nhóm là chia truy vấn thành vài nhóm theo độ tự tin của CLIP, và cho mỗi nhóm một α, k riêng.

## Giả thuyết

- **H1.** Cho mỗi nhóm truy vấn một α riêng, theo độ tự tin của CLIP, làm R@1 trên test cao hơn α cố định. Nhóm CLIP phân vân sẽ chọn α thấp hơn, tức dựa vào đồ thị nhiều hơn, so với nhóm CLIP chắc chắn.
- **Đối chứng.** Nếu chia nhóm **ngẫu nhiên** cũng tăng tương tự, thì mức tăng chỉ do có thêm tham số để khớp với validation, chứ không do độ tự tin của CLIP. Đối chứng ngẫu nhiên có cùng kích thước nhóm và cùng số tham số được chọn.

## Cách làm

1. **Đặc trưng.** Với mỗi truy vấn, tính *margin* = điểm CLIP của ảnh hạng 1 − điểm CLIP của ảnh hạng 2. Đặc trưng này không dùng nhãn, nên tính được cả lúc chạy thật.
2. **Chia nhóm trên validation.** Cắt validation thành B nhóm tại các phân vị của margin, với B ∈ {2, 3, 4}. Ví dụ B = 2 cắt tại trung vị. Mốc so sánh là α cố định, tức bộ xếp hạng chính thức; script luôn chạy mốc này.
3. **Chọn tham số cho từng nhóm.** Mỗi nhóm chọn (α, k) riêng bằng **đúng** hàm quét chính thức `sweep_alpha_and_k`: lưới α bước 0,01, lưới k {10, …, 1.500}. Luật hoà cũng giữ nguyên: R@1, rồi MRR, rồi α lớn hơn, rồi k nhỏ hơn.
4. **Chấm test.** Tính margin cho từng truy vấn test, xếp truy vấn vào nhóm theo **mốc chia của validation**, rồi dùng (α, k) của nhóm đó. Test chỉ được chấm một lần.

Với truy vấn q thuộc nhóm b(q):

$$
\text{điểm}_q=\alpha_{b(q)}\, s^{C}_q+\bigl(1-\alpha_{b(q)}\bigr)\, s^{G}_q\quad\text{trong }k_{b(q)}\text{ ứng viên đầu của CLIP}
$$

**Vì sao chọn riêng từng nhóm vẫn là tối ưu chung.** Điểm cuối của một truy vấn chỉ phụ thuộc vào hàng điểm của chính nó. Vì vậy R@1 trên cả validation là tổng có trọng số của R@1 từng nhóm, và chọn tốt nhất cho từng nhóm cũng là chọn tốt nhất cho toàn bộ.

## Vì sao không rò rỉ dữ liệu test

- **Mốc chia và (α, k) chỉ lấy từ validation.** Hàm `fit_adaptive_fusion` từ chối mọi tập khác validation.
- **Margin không dùng nhãn.** Nó chỉ dùng điểm CLIP của chính truy vấn, thông tin vốn có sẵn lúc chạy thật.
- **Mốc α cố định là bộ xếp hạng chính thức.** Script chọn (α, k) bằng đúng hàm quét chính thức trên toàn bộ validation, so với lựa chọn trong file benchmark của `run_benchmark` cho từng seed, và **dừng nếu khác**. Nhờ vậy mọi so sánh đều dùng đúng mốc chính thức.
- **Test chấm một lần,** sau khi đã chốt mọi thứ trên validation.

Các test tự động kiểm những điều trên:

- một nhóm cho ra đúng lựa chọn và đúng điểm của bộ xếp hạng chính thức;
- tập test dùng mốc chia của validation, không tự tính lại;
- việc chọn tham số bị từ chối trên tập khác validation;
- α = 1 ở mọi nhóm giữ nguyên thứ hạng CLIP;
- nhóm quá nhỏ bị từ chối;
- chạy cả script trên dữ liệu dựng tay: nếu (α, k) của mốc α cố định khác file benchmark thì script dừng với exit 1 và không ghi file kết quả; chênh lệch ghép cặp được tính đúng chiều (cách chia − α cố định).

Một test dựng sẵn ví dụ mà **không α cố định nào phục vụ được cả hai loại truy vấn**: loại "CLIP phân vân" cần α < 0,834, loại "CLIP chắc chắn" cần α > 0,952. Ở ví dụ này, α cố định chỉ đạt R@1 50%, còn chia nhóm đạt 100%.

## Cách chạy

Điều kiện: máy phải có vector CLIP (`data/processed/vg_coco/embeddings/`), và đã chạy `train_graphsage` lẫn `run_benchmark --method graphsage` cho config ghi ở `retrieval_config`. Mặc định là `configs/retrieval/ablations/full.yaml`, tức mô hình chính.

```bash
python -m scripts.evaluate_adaptive_fusion                 # dùng configs/evaluation/adaptive_fusion.yaml
pytest tests/retrieval/test_adaptive_fusion.py             # test, không cần dữ liệu thật
```

Script không huấn luyện lại và không cần GPU. Với config mặc định, mỗi seed quét lưới α, k trên toàn bộ validation 7 lần: 1 lần cho α cố định, cộng 2 đặc trưng × 3 số nhóm. Một lần quét trên 7.500 truy vấn validation mất khoảng 25–30 giây trên CPU (đo trên Apple M3), nên cả lượt chạy 3 seed mất khoảng 10–15 phút. Kết quả được ghi ra `runs/analysis/adaptive_fusion.json`; thư mục này bị `.gitignore` bỏ qua, nên muốn commit thì dùng `git add -f`.

## Đọc file kết quả

`results` có mục `fixed` cho α cố định, và một mục cho mỗi cách chia: `clip_margin-b2`, `clip_margin-b3`, …, `random-b2`, … Mỗi mục gồm:

| Trường | Nội dung |
|---|---|
| `seeds.<seed>.fit` | Với `fixed`: α, k chọn trên toàn bộ validation. Với cách chia: mốc chia (`edges`) và α, k của từng nhóm (`buckets`) |
| `seeds.<seed>.val` và `.test` | `metrics`: R@1/5/10, MRR. Với cách chia, thêm `buckets`, theo đúng thứ tự trong `fit.buckets`: số truy vấn, R@1 của cách chia này và của α cố định trên **cùng** các truy vấn đó |
| `summary` | Trung bình ± độ lệch chuẩn qua 3 seed |
| `paired_vs_fixed_test` | Chỉ có ở các cách chia. Chênh lệch R@1 so với α cố định trên test (`delta_r1_pp`, đơn vị điểm phần trăm) và khoảng 95% của bootstrap ghép cặp, gom theo ảnh đúng. Đây là cùng thủ tục mà `analyze_ablation` dùng. Trong `per_seed`, "full" là cách chia và "variant" là α cố định |

## Bảng kết quả (điền sau khi chạy)

| Cách chia | Số nhóm | R@1 test (± độ lệch chuẩn) | Chênh lệch với α cố định | Khoảng 95% |
|---|---|---|---|---|
| α cố định | 1 | | — | — |
| Theo margin của CLIP | 2 | | | |
| Theo margin của CLIP | 3 | | | |
| Theo margin của CLIP | 4 | | | |
| Ngẫu nhiên (đối chứng) | 2 | | | |
| Ngẫu nhiên (đối chứng) | 3 | | | |
| Ngẫu nhiên (đối chứng) | 4 | | | |

Kết quả **ủng hộ H1** khi cả ba điều sau cùng đúng:

1. Chia theo margin cho chênh lệch dương, với khoảng 95% không chứa 0.
2. Chia ngẫu nhiên không cho mức tăng tương tự.
3. Nhóm margin nhỏ chọn α thấp hơn nhóm margin lớn. Xem trong `fit.buckets`.

Nếu chỉ validation tăng còn test không tăng, cách chia đang khớp quá mức với validation. Nên báo cáo thẳng điều đó, kèm số nhóm mà hiện tượng bắt đầu xảy ra.

## Giới hạn và hướng mở rộng

- **Nhiều nhóm hơn thì nhiều tham số hơn**, nên dễ khớp quá mức với validation. Đối chứng ngẫu nhiên và kết quả trên test là hai cách kiểm điều này. Tham số `min_bucket_queries` (mặc định 500) chặn các nhóm quá nhỏ.
- **Margin chỉ là một đặc trưng.** Có thể thêm: độ lệch giữa thứ hạng CLIP và thứ hạng đồ thị; số từ chỉ quan hệ trong câu; hoặc học α(q) bằng một hàm logistic trên validation.
- **Cấp độ 4 hướng (a) yêu cầu thực nghiệm trên ít nhất 2 tập công khai.** Code không phụ thuộc tập dữ liệu, chỉ cần config retrieval của tập kia, nên chạy lại được ngay khi có tập thứ hai.
- **Hiện script chỉ đọc run GraphSAGE** ghi trong `retrieval_config`. Muốn áp dụng cho mô hình khác, ví dụ AdaMF-MAT, cần thêm phần đọc vector của mô hình đó; hàm `fit_adaptive_fusion` và `adaptive_scores` thì dùng lại được nguyên vẹn.
