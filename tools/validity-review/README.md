# Form chấm validity cho người audit

Mở [index.html](index.html) bằng trình duyệt rồi chọn file bộ chấm JSON local.
Form không cần server, không gửi ảnh/nhãn lên mạng, không chứa dữ liệu hoặc
đáp án máy khi clone repo. Số query và relation lấy từ file nạp, **không cố định
20 query hoặc 42 relation**.

## Xuất bộ chấm

Từ thư mục gốc repo, sau khi có trace fidelity và ảnh/cache/checkpoint tương ứng:

```bash
# Mặc định lấy tất cả query hợp lệ của seed 13, budget 10%.
python -m scripts.prepare_validity_review --all

# Tuỳ chọn: pilot cân bằng 10 query mỗi nhóm đúng/khác ID đích.
python -m scripts.prepare_validity_review --per-stratum 10 \
  --output private/reviews/validity/pilot.json

# Có thể đổi seed, budget hoặc nguồn trace.
python -m scripts.prepare_validity_review --all --seed 37 --fraction 0.25

# Dùng lại manifest v2 cũ để giữ nguyên ID/hash của nháp đã có.
python -m scripts.prepare_validity_review --manifest existing-manifest.json \
  --output private/reviews/validity/legacy-package.json
```

Lệnh mặc định tạo `runs/explanation-quality/validity-audit.json` và manifest
cùng tên đuôi `.manifest.json`. Nạp file `validity-audit.json` vào form. Các
đường dẫn output trong ví dụ đều local, không cần đưa ảnh hoặc nhãn vào Git.

Exporter chỉ lấy nhóm relation đã chọn trong trace; không chạy model mới hoặc
tự chấm validity. Nó kiểm hash/graph, nhúng ảnh và lọc bỏ score, correctness,
nhãn AI. Graph không đủ điều kiện fidelity vẫn bị bỏ qua, nên “200 query lấy
mẫu” không có nghĩa luôn xuất đủ 200 query hợp lệ. Chạy `--help` để xem tuỳ chọn.

## Người chấm làm gì?

1. Nạp bộ JSON, nhập tên/mã reviewer, khai báo có từng xem nhãn AI từng item chưa.
2. Đọc rubric trong form. Với từng relation, xem ảnh và chấm **đúng về ảnh**:
   supported, refuted hoặc uncertain. Ghi căn cứ nhìn thấy hoặc điều chưa rõ.
3. Đọc query và chấm **query support**: direct support, context only, unrelated
   hoặc uncertain. Nếu direct support, copy cụm query nguyên văn và giải thích.
4. Chấm đủ số relation mà form hiển thị. Uncertain vẫn là nhãn hợp lệ nếu có
   lý do; không ép trường hợp ảnh mờ thành supported/refuted.
5. Tải nháp để sao lưu hoặc chuyển máy, dùng Mở lại nháp để tiếp tục. Khi đủ
   nhãn, xác nhận tự đối chiếu rồi xuất completed. File xuất có reviewer/thời
   gian trong tên; nên lưu vào `private/reviews/validity/submissions/`.

Form tự lưu theo manifest trong trình duyệt. Khi đổi trình duyệt/đường dẫn mở
form, nên tải nháp rồi import lại. Không cung cấp nhãn máy trong lượt chấm đầu.

## Rubric và ví dụ biên

Đúng về ảnh phải xét cả đối tượng, quan hệ và hướng. Chỉ thấy người và xe chưa
đủ xác nhận người đang cưỡi xe. Refuted cần bằng chứng đối lập rõ; không nhìn
thấy vì bị che là uncertain.

Relation đúng ảnh chưa chắc hỗ trợ query: `person wearing shoes` thường chỉ là
bối cảnh khi query hỏi người cưỡi ngựa; `person riding horse` hỗ trợ trực tiếp.
Một relation không cần chứng minh toàn query, nhưng chung một từ cũng chưa đủ:
phải nêu nó hỗ trợ nội dung/ràng buộc được hỏi ra sao.

## Giới hạn và bước sau audit

- Mẫu AI 20 query/42 relation được nhắc trong báo cáo là lượt thử nghiệm đã làm,
  không phải giới hạn công cụ. Exporter không tự mở rộng nhãn AI từ 20 sang 200.
- Máy/người/bản phân xử giữ riêng. Sau khi có nhãn người, mới xem bất đồng với
  máy và ghi lý do xử lý. Không tự cho máy thắng hoặc lấy đa số từ đúng hai phiếu.
- `completed` chỉ xác nhận file đủ trường, không chứng minh nhãn đúng tuyệt đối.
- Form dùng schema **validity-v2**. CLI `evaluate_explanation_quality --labels`
  hiện tổng hợp schema **v1**, chưa nhận file form v2. Chưa có runner VLM hoặc
  CLI đối chiếu v2; không đổi tên field để ép file vào v1.
- Rubric/prompt của judge cần pilot trên validation và pin model/revision trước
  khi chạy thật; không tune bằng kết quả test. Nếu chưa chạy judge/thu nhãn người
  thì không báo đã hoàn tất validity hoặc có human–machine agreement.
