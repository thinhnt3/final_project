---
title: "Phase 3: Benchmark đánh giá, hoàn tất nhiệm vụ 1"
status: todo
---

# Phase 3: Benchmark đánh giá, hoàn tất nhiệm vụ 1

## Overview

Bộ đo Recall@1/5/10 và MRR trên tập test cố định, nhiều seed, α chọn trên validation.

## Requirements

- [ ] Cùng tập truy vấn, cùng tập ảnh, cùng metric cho mọi phương pháp
- [ ] α và siêu tham số chỉ chọn trên validation
- [ ] Báo cáo mean ± std qua ít nhất 3 seed

## Todo

- [ ] Kiểm thử hàm metric trên dữ liệu đồ chơi
- [ ] Script quét α trên validation
- [ ] Xuất kết quả ra `runs/*.json` + bảng trong `reports/`

## Success Criteria

Chạy lại ra đúng cùng con số; kết quả test không bị dùng để chỉnh tham số.
