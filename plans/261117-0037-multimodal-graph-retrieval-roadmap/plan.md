---
title: "Multimodal Graph Retrieval Roadmap"
description: "Truy hồi ảnh bằng CLIP + đồ thị: 8 nhiệm vụ từ pipeline cơ bản tới MMKG và demo"
status: in-progress
priority: P1
effort: ""
tags: [retrieval, clip, gnn, visual-genome]
created: 2026-11-17
---

# Multimodal Graph Retrieval Roadmap

## Overview

Tám nhiệm vụ theo thứ tự nhóm đã chốt. Nhiệm vụ 1–4 hoàn tất cấp độ 1 trên một tập công khai; Nhiệm vụ 5 thêm tập thứ hai, ablation và fidelity; Nhiệm vụ 6–8 là phần mở rộng.

## Quyết định dữ liệu

| Dùng cho | Tập dữ liệu | Quy mô | Truy vấn lấy từ |
|---|---|---|---|
| Nhiệm vụ 1–4, 6, 7 | Visual Genome ∩ COCO (51.498 ảnh có `coco_id`); test chỉ từ Karpathy test; nhãn gộp theo synset WordNet | 10.000 ảnh (7.000/1.500/1.500), 7.500 truy vấn test | Caption COCO (5 câu/ảnh) |
| Nhiệm vụ 5 | FB15k-237-IMG | tập con thực thể có ảnh | Cần chốt ở Giai đoạn 5 |
| Nhiệm vụ 8 (tuỳ chọn) | Một tập video nhỏ, ví dụ MSR-VTT | tập con | Caption của video |

Bắt buộc 2 tập công khai; tập video chỉ cần nếu làm phần text-to-video.

## Goals

| # | Mục tiêu | Ưu tiên |
|---|------|----------|
| 1 | Hoàn tất cấp độ 1: CLIP vs CLIP + Graph trên cùng cách chia | P1 |
| 2 | Chứng minh cải thiện đến từ cấu trúc quan hệ (ablation, fidelity) | P1 |
| 3 | Mở rộng: đồ thị dị thể, MMKG, video, demo | P2 |

## Phases

| # | Giai đoạn | Trạng thái |
|---|-------|--------|
| 1 | [Giai đoạn 1: Pipeline cơ bản: CLIP + GraphSAGE + Rerank](./phase-01-pipeline-clip-graphsage-rerank.md) | Đang thực hiện |
| 2 | [Giai đoạn 2: Dữ liệu VG ∩ COCO + pipeline xây đồ thị](./phase-02-xay-do-thi-tu-dong.md) | Đã triển khai và chạy dữ liệu |
| 3 | [Giai đoạn 3: Benchmark đánh giá, hoàn tất nhiệm vụ 1](./phase-03-benchmark-cap-do-1.md) | Đã triển khai và chạy đánh giá |
| 4 | [Giai đoạn 4: So sánh baseline vs CLIP + Graph](./phase-04-so-sanh-clip-vs-clip-graph.md) | Đang thực hiện |
| 5 | [Giai đoạn 5: Tập công khai thứ hai, ablation, fidelity](./phase-05-tap-thu-hai-ablation-fidelity.md) | Đang thực hiện |
| 6 | [Giai đoạn 6: Thuật toán cấp cao trên đồ thị dị thể](./phase-06-do-thi-di-the-thuat-toan-nang-cao.md) | Chưa làm |
| 7 | [Giai đoạn 7: Xây MMKG nhỏ + benchmark](./phase-07-mmkg-nho.md) | Chưa làm |
| 8 | [Giai đoạn 8: Cải tiến, text-to-video, demo](./phase-08-cai-tien-text-to-video-demo.md) | Chưa làm |

## Success Criteria

- [ ] Bảng CLIP vs CLIP + Graph với sai số qua nhiều seed, trên 2 tập công khai
- [~] Đã chạy thí nghiệm loại bỏ thành phần trên tập thứ nhất; chưa chứng minh ưu thế quan hệ trên cả hai tập. Không ép thứ tự kết quả theo kỳ vọng.
- [ ] 5 truy vấn thành công + 5 thất bại có đường đi giải thích
- [ ] Demo chạy được

<!-- slug: multimodal-graph-retrieval-roadmap -->
