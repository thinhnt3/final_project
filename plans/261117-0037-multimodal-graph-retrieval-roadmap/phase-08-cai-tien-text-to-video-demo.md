---
title: "Phase 8: Cải tiến, text-to-video, demo"
status: todo
---

# Phase 8: Cải tiến, text-to-video, demo

## Overview

Cải tiến cuối, thử mở rộng sang truy hồi video, và một demo gõ truy vấn tự do có hiển thị đường đi giải thích.

## Requirements

- [ ] Demo: gõ truy vấn → ảnh top-k + quan hệ/đường đi
- [ ] Text-to-video: tập con nhỏ (ví dụ MSR-VTT), khung hình làm đỉnh ảnh

## Todo

- [ ] Chọn cải tiến dựa trên phân tích thất bại ở Phase 4
- [ ] Thử nghiệm text-to-video trên tập con
- [ ] Dựng demo
- [ ] Hoàn thiện báo cáo

## Nhánh song song: bộ dữ liệu di tích & điểm du lịch Việt Nam

Bộ ảnh–caption–KG thật, không sinh bằng AI, để tái dùng cho demo và cho MMKG miền du lịch.

- Nguồn: Wikidata (thực thể, quan hệ), Wikimedia Commons (ảnh, caption, license), Wikipedia vi/en (đoạn mở đầu bài viết).
- Code: `configs/data/vn_heritage.yaml`, `src/connectors/wikimedia.py`, `src/data/vn_heritage*.py`, `scripts/build_vn_heritage.py`, test ở `tests/data/test_vn_heritage.py`.
- Đầu ra: `data/processed/vn_heritage/` (places.jsonl, images.jsonl, kg_triples.tsv, kg_nodes.json, stats.json). Ảnh nằm ở `data/raw/vn_heritage/images/` (git-ignore).
- Datasheet: `docs/datasets/vn-heritage.md`.

- [ ] ≥ 1.000 địa điểm có ít nhất 1 ảnh kèm caption người viết và license
- [ ] Chia train/val/test theo địa điểm, ảnh của một địa điểm không rơi vào hai tập
- [ ] KG có cạnh loại, tỉnh, di sản, tôn giáo, kiến trúc
- [ ] Datasheet ghi nguồn, license, quy trình lọc và giới hạn

## Success Criteria

Demo chạy được trước lớp; có kết quả sơ bộ cho video.
