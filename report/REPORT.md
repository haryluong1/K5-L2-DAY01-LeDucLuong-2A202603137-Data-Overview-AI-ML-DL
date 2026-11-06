# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**  2026-11-06

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** 

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?  
Record này nói về cả bức ảnh (Image Classification), không phải một vật thể riêng lẻ. Vì Image Classification phân loại nội dung tổng thể của ảnh thành một class duy nhất có điểm dự đoán cao nhất. Model không xác định vị trí cụ thể của từng vật thể trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?  
Danh sách tên lớp được định nghĩa sẵn trong taxonomy đi kèm với model/checkpoint, ở đây là ImageNet-1K. Model không tự nghĩ ra tên lớp mà chỉ dự đoán dựa trên các class đã được định nghĩa.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- class_id: Xác định lớp bằng mã số, thuận tiện cho máy xử lý.
- class_name: Giúp con người đọc và hiểu tên lớp.
- taxonomy_name: Xác định hệ thống phân loại đang được sử dụng, tránh nhầm lẫn khi các taxonomy có thể sử dụng ID hoặc tên lớp khác nhau.
Cần lưu cả ba thông tin để đảm bảo tính nhất quán và truy xuất chính xác nhãn.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?  
Guideline cần quy định rõ tiêu chí chọn lớp chính, chẳng hạn ưu tiên chủ thể nổi bật nhất, chiếm diện tích lớn nhất hoặc là nội dung chính của bức ảnh. Đồng thời cần có quy tắc xử lý khi ảnh chứa nhiều đối tượng hoặc không xác định được chủ thể chính để các annotator đưa ra kết quả nhất quán.

6. Vì sao không được lấy `score` của model làm ground truth?  
Vì score chỉ thể hiện mức độ tự tin của model đối với kết quả dự đoán, không đảm bảo đó là nhãn chính xác. Ground truth phải được con người gán dựa trên guideline và được kiểm tra chất lượng. Vì vậy, không thể sử dụng trực tiếp score của model làm ground truth.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?  
Ảnh có kích thước 640 × 427 pixel.
Box trên được hiểu như sau:
- Góc trên-trái: (385.33, 69.24).
- Góc dưới-phải: (498.92, 348.92).
- Chiều rộng: 113.58 pixel.
- Chiều cao: 279.68 pixel.
Nói đơn giản, model phát hiện một người và vẽ một hình chữ nhật bao quanh người đó.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?  
Ngưỡng càng thấp thì model giữ lại càng nhiều prediction có confidence thấp.
| Ngưỡng | Số prediction |
   | --- | --- |
   | 0.35 | 10 |
   | 0.60 | 6 |
- Ở ngưỡng `0.35`, có 10 prediction được giữ lại trong file JSON.
- Ở ngưỡng 0.60, có 6 prediction được giữ lại vì `score` của chúng lớn hơn hoặc bằng 0.60.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?  
Khi hạ ngưỡng, số lượng prediction tăng lên, bao gồm cả những box có confidence thấp.

Reviewer cần:
- Kiểm tra những prediction có score thấp.
- Phát hiện các box sai class hoặc box không thực sự chứa vật thể.
- Kiểm tra các vật thể bị bỏ sót hoặc bị phát hiện trùng lặp.
- Đối chiếu với ảnh gốc và guideline để xác định nhãn chính xác.
Mục đích là đảm bảo chất lượng Ground Truth, không phụ thuộc hoàn toàn vào prediction của model.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).  
Người gán nhãn cần vẽ Bounding Box sát với phạm vi vật thể nhìn thấy trong ảnh, đảm bảo bao phủ toàn bộ phần vật thể nhìn thấy và không bao gồm quá nhiều vùng nền xung quanh. Không tự suy đoán phần bị che khuất nếu guideline không quy định.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?  
Nguyên tắc xử lý:
- Nếu guideline đã quy định cách gán nhãn cho vật thể bị che khuất hoặc cắt mép thì annotator làm theo.
- Nếu chưa có quy định rõ, nên escalation (hỏi Lab Coach) thay vì tự quyết định.
Ví dụ: Một người bị chiếc bàn che mất phần thân dưới. Annotator không nên tự suy đoán vị trí phần cơ thể bị che để mở rộng Bounding Box nếu chưa có quy tắc.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[
  [446.0, 70.0],
  [445.0, 71.0],
  [444.0, 71.0],
  [443.0, 72.0],
  [442.0, 72.0]
]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?  
Bounding Box chỉ xác định phạm vi vật thể bằng một hình chữ nhật, gồm tọa độ góc trên-trái và góc dưới-phải.
Trong khi đó, Polygon gồm nhiều điểm tọa độ `[x, y] nối với nhau để tạo thành đường biên của vật thể.
Polygon giúp xác định hình dạng thực tế của vật thể chính xác hơn, loại bỏ phần nền không thuộc vật thể và thể hiện được những vật thể có hình dạng phức tạp.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?  
instance_id dùng để định danh riêng từng đối tượng được phát hiện trong một ảnh, giúp phân biệt các object với nhau ngay cả khi chúng thuộc cùng một class.
Ví dụ:
- kitchen-002: một object thuộc class bowl.
- kitchen-003: một object khác cũng thuộc class bowl.
Cả hai có cùng class_name nhưng có instance_id khác nhau và polygon riêng.
instance_id không phải class ID vì nó không đại diện cho loại đối tượng. Nó cũng không phải tracking ID vì không dùng để theo dõi cùng một đối tượng qua nhiều frame hoặc nhiều thời điểm khác nhau.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.  
Người gán nhãn cần vẽ polygon sát theo đường biên của vật thể nhìn thấy trong ảnh, bao phủ đầy đủ phần vật thể và hạn chế lấy thêm vùng nền bên ngoài. Không tự suy đoán hoặc vẽ thêm những phần bị che khuất nếu guideline chưa quy định.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?  
Trên ảnh kitchen, có một số chiếc bát (bowl) nằm gần nhau trên bàn, khiến đường biên giữa các vật thể có thể khó xác định chính xác.
- Nếu guideline đã quy định cách xử lý các vật thể chạm nhau hoặc bị che khuất thì annotator cần thực hiện theo quy tắc đó.
- Nếu guideline chưa có quy định rõ ràng, cần escalation (hỏi Lab Coach) để được hướng dẫn thay vì tự quyết định.
Mục đích là đảm bảo các annotator gán nhãn nhất quán và không tự đưa ra những quy tắc khác nhau.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | Một hoặc nhiều class label đại diện cho nội dung toàn bộ ảnh, tùy guideline. | Ảnh traffic có nhiều loại phương tiện. Model dự đoán `cab` đứng hạng 1 với score `0.510915`, nhưng vẫn có các class khác như `minibus` (`0.164284`) và `police_van` (`0.085848`). | Quan sát toàn bộ ảnh, chọn class phù hợp theo taxonomy và guideline. Không dựa hoàn toàn vào prediction của model. | Kiểm tra class có phù hợp với nội dung ảnh không, nhãn có chính xác và nhất quán với guideline không. |
| Phát hiện vật thể | Mỗi vật thể có một class label và một Bounding Box (`bbox_xyxy`) bao quanh vật thể. | Ảnh traffic có nhiều phương tiện nằm gần nhau, ở xa hoặc bị che khuất, dễ dẫn đến bỏ sót object hoặc vẽ box không chính xác. | Xác định từng vật thể, chọn class phù hợp và vẽ Bounding Box bao quanh phần vật thể nhìn thấy theo guideline. | Kiểm tra class, vị trí và kích thước Bounding Box; phát hiện vật thể bị bỏ sót, box sai hoặc trùng lặp. |
| Instance segmentation | Mỗi vật thể có một class label, một `instance_id` và một polygon/mask riêng. | Ảnh kitchen có nhiều chiếc bowl nằm gần nhau trên bàn, khiến đường biên giữa các vật thể khó xác định. | Vẽ polygon sát đường biên phần vật thể nhìn thấy, phân biệt từng instance và escalation nếu guideline chưa quy định cách xử lý. | Kiểm tra class, instance ID, độ chính xác của polygon và các vùng mask bị thiếu hoặc lấn sang vật thể khác. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của tôi cho bài lab này là: Chỉ sử dụng dữ liệu đúng phạm vi được cung cấp, không tự ý chia sẻ, sao chép hoặc công khai dữ liệu cá nhân và thông tin nhạy cảm; không đưa dữ liệu lên các nền tảng bên ngoài khi chưa được cho phép.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý, không tải xuống hoặc chia sẻ thêm, đồng thời báo cho Lab Coach hoặc người phụ trách để được hướng dẫn.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
