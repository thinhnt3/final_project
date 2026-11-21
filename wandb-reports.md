# Report W&B

Cập nhật: 21/11/2026.

## Link

- VG ∩ COCO, cấp độ 1 đến 3: https://forge.coreweave.com/wandb/thinhnt3-vnggames/vg-coco-graph-retrieval/reports/x--VmlldzoxODM0MjcyMA
- Project chứa các run của report: https://forge.coreweave.com/wandb/thinhnt3-vnggames/vg-coco-graph-retrieval

## Nội dung report VG ∩ COCO

1. Cấp độ 1: CLIP + GraphSAGE
2. Ablation đồ thị của GraphSAGE và fidelity (lượt 18/11)
3. Cấp độ 2: AdaMF-MAT so với GraphSAGE
4. Ablation của AdaMF-MAT
5. Mô hình có đọc cạnh không: relation queries
6. Ablation khác, kết luận, chưa làm
7. Đánh giá giải thích: fidelity, sparsity, validity (lượt 21/11)

## Cập nhật report

Phần chữ nằm trong `docs/wandb-report/*.md`. Hình của mục 7 là file trong `reports/explanation-quality/figures/` trên nhánh `main`.

```bash
python -m scripts.upload_report_runs                       # run tổng hợp + đường học lên W&B, bỏ qua run đã có
python -m scripts.build_wandb_report --url <link report>   # viết lại report tại chỗ; bỏ --url để tạo report mới
```

Khi tạo report mới, thêm link vào mục "Link" ở trên.

## Chưa có report

- MKG-W (dataset thứ hai)
