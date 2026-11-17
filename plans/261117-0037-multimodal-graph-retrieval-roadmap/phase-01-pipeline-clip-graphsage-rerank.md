---
title: "Phase 1: Pipeline cơ bản: CLIP + GraphSAGE + Rerank"
status: in-progress
---

# Phase 1: Pipeline cơ bản: CLIP + GraphSAGE + Rerank

## Overview

Một pipeline chạy đầu-cuối trên 1 tập công khai (Visual Genome): CLIP đóng băng → top-k cosine → điểm đồ thị từ nhúng GraphSAGE → xếp hạng lại theo α.

## Requirements

- [x] Baseline CLIP đóng băng + tìm kiếm cosine (`scripts/run_clip_retrieval.py`)
- [x] Rerank `α·cosine + (1−α)·điểm đồ thị` (`src/retrieval/reranker.py`)
- [ ] Điểm đồ thị phải lấy từ nhúng GraphSAGE, không chỉ khớp từ khoá
- [ ] Một lệnh chạy đầu-cuối, cấu hình qua `configs/`

## Todo

- [x] Chạy thử trên 40 ảnh
- [ ] Kiểm tra `graph_score` đang dùng nhúng GNN hay khớp token
- [ ] Nối nhúng GraphSAGE vào điểm đồ thị
- [ ] Trả về quan hệ/đường đi đã dùng cho mỗi ứng viên

## Success Criteria

Một lệnh cho ra bảng xếp hạng CLIP và CLIP + Graph cùng lời giải thích, trên cùng tập ứng viên.
