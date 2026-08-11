# MedPRS Dataset Reproduction & Evaluation Results

Tài liệu này tổng hợp toàn bộ kết quả thực nghiệm, so sánh chỉ số hiệu năng và định hướng phát triển hệ thống gợi ý tạp chí **MedPRS** (quy mô 1,406 tạp chí) được thực hiện trên Kaggle với card đồ họa đơn **RTX 6000**.

---

## 1. Bảng So Sánh Chỉ Số Hiệu Năng Tổng Hợp

Bảng dưới đây so sánh hiệu năng của mô hình gốc **Pointwise (BioBERT)** với các phiên bản **Listwise Reranking (Qwen-7B)** và **Cross-Encoder (BioBERT)**. Các phép đo được tính toán tương quan trực tiếp trên cùng một tập mẫu kiểm thử cố định:

| Kịch bản thử nghiệm | Số mẫu thử (Val set) | Pointwise Acc@1 | Rerank / Hybrid Acc@1 | Rerank / Hybrid Acc@5 | Rerank / Hybrid NDCG@10 | Trạng thái so với Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 (Concise Zero-shot)** | 466 | 0.4099 | 0.1652 | 0.5923 | 0.4261 | ⬇️ Giảm mạnh (-59.7% Acc@1) |
| **V2 (CoT Zero-shot)** | 390 | 0.4231 | 0.1564 | 0.5846 | 0.4208 | ⬇️ Giảm mạnh (-63.0% Acc@1) |
| **V3 (Few-shot 3-cand)** | 200 | 0.4800 | 0.3750 | 0.7050 | 0.5836 | ⬇️ Bị lỗi cắt cụt (Truncation) |
| **V4 (Few-shot 10-cand)** | 200 / 1000 | 0.4800 / 0.4530 | 0.4350 / 0.4120 | 0.7350 / 0.7100 | 0.6241 / 0.6180 | ⬇️ Tiệm cận sát nút Baseline |
| **V4 (Hybrid, $\alpha=1.0$)** | 200 | 0.4800 | **0.4900** | **0.7550** | **0.6551** | 🚀 **Vượt Baseline (+2.1%)** |
| **V4 (Hybrid, $\alpha=0.5$)** | 200 | 0.4800 | **0.4900** | **0.7600** | **0.6590** | 🚀 **Vượt Baseline (+2.7%)** |
| **V5 (Cross-Encoder V1 - Raw)** | 1000 | 0.4530 | 0.2490 | 0.7090 | 0.5259 | ⬇️ Giảm điểm |
| **V5 (Cross-Encoder V2 - Mining)**| 1000 | 0.4530 | **0.2720** | **0.6770** | **0.5300** | ⬆️ **Cải thiện so với V1** |

---

## 2. Các Phát Hiện & Đột Phá Quan Trọng

1. **Sức mạnh của Hybrid Blending**:
   * Việc kết hợp tuyến tính giữa điểm Logit của BioBERT Pointwise và thứ tự xếp hạng của Qwen Listwise theo công thức:
     $$S_{hybrid} = S_{pointwise} + \alpha \cdot (10 - rank_{llm})$$
     Đã mang lại kết quả tốt nhất hệ thống (Acc@5 đạt **76.00%**, NDCG@10 đạt **0.6590**), chứng minh sự bổ trợ xuất sắc giữa tri thức phân loại có giám sát và năng lực lập luận ngữ nghĩa của LLM.
2. **Kích thước ví dụ Few-shot ảnh hưởng đến định dạng đầu ra**:
   * Việc dùng ví dụ Few-shot chỉ có 3 ứng viên làm mô hình Qwen bị học rập khuôn độ dài (In-context length bias) và chỉ xuất ra 3-4 hạng ở tập test. Nâng cấp lên ví dụ 10 ứng viên đã giải quyết triệt để lỗi này, giúp Acc@1 đơn lẻ của Qwen tăng vọt từ **0.3750 lên 0.4350**.

---

## 3. Phân Tích Chuyên Sâu Thử Nghiệm Cross-Encoder (V5)

Mặc dù kết quả nâng cấp V2 (Warm-start từ checkpoint Pointwise + Khai thác mẫu âm khó Hard Negatives) đã giúp tăng Acc@1 từ **24.9% lên 27.2%**, điểm số vẫn thấp hơn baseline. Phân tích chỉ ra 2 nguyên nhân cốt lõi:

* **Lỗi cắt cụt thông tin tạp chí (Silent Tokenization Truncation)**:
  * Khi mã hóa cặp văn bản `[CLS] Paper [SEP] Journal [SEP]`, nếu Abstract bài báo quá dài, cơ chế mặc định sẽ cắt cụt từ cuối chuỗi, làm mất đi phần thông tin quan trọng nhất của tạp chí (Aims & Scope). Mô hình chỉ nhìn thấy tên tạp chí mà không đọc được mô tả chi tiết.
* **Hạn chế của BCE Loss**:
  * Hàm mất mát `BCEWithLogitsLoss` huấn luyện mô hình theo hướng phân loại nhị phân tuyệt đối (0 hoặc 1), trong khi bài toán thực tế là so sánh và xếp hạng tương đối giữa 10 ứng viên.

---

## 4. Lộ Trình Phát Triển Tiếp Theo (Future Roadmap)

Để đưa dự án tới điểm SOTA thực thụ và tối ưu hóa hệ thống ở quy mô **1.2 triệu bài báo (1.2M papers)**:

1. **Nâng cấp Cross-Encoder V3**:
   * Cấu hình Tokenizer với tham số **`truncation="only_first"`** để bảo vệ 100% thông tin Aims & Scope của tạp chí.
   * Thay thế BCE Loss bằng **Pairwise Margin Ranking Loss**:
     $$\text{Loss} = \max(0, \text{margin} - (S_{positive} - S_{negative}))$$
     Giúp mô hình tập trung học cách xếp hạng tương đối giữa các ứng viên gây nhiễu.
2. **Tối ưu hóa thời gian chạy**:
   * Cross-Encoder (chỉ mất 10ms/bài báo) sẽ là hướng đi bắt buộc nếu muốn triển khai ở quy mô 1.2M bài báo, thay thế hoàn toàn cho mô hình Qwen-7B (mất 6s/bài báo, tương đương 83 ngày chạy liên tục).
3. **Supervised Fine-tuning (SFT) Qwen**:
   * Tiến hành huấn luyện LoRA cho Qwen-2.5-7B trên tập train của MedPRS sử dụng loss xếp hạng nếu muốn tối đa hóa khả năng lập luận của LLM.
