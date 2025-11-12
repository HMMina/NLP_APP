# Báo Cáo Lab 06: Phân Loại Văn Bản với RNNs (lab6_rnns_text_classification.ipynb)

## 📋 Mục Lục
1. [Giải Thích Các Bước Triển Khai](#1-giải-thích-các-bước-triển-khai)
2. [Hướng Dẫn Thực Thi Mã](#2-hướng-dẫn-thực-thi-mã)
3. [Phân Tích Kết Quả](#3-phân-tích-kết-quả)
4. [Thách Thức và Giải Pháp](#4-thách-thức-và-giải-pháp)
5. [Hướng Phát Triển](#5-hướng-phát-triển)

---
## 1. Giải Thích Các Bước Triển Khai
### Task 0: Data Loading & Label Encoding
- **Mục đích**: Chuẩn bị dữ liệu cho quá trình huấn luyện.
- **Bước thực hiện**:
  - Đọc dữ liệu từ các file CSV (train.csv, val.csv, test.csv) trong folder hwu.
  - Sử dụng `LabelEncoder` để chuyển đổi nhãn "category" từ dạng text sang số.
  - Kiểm tra kích thước và cấu trúc dữ liệu.

### Task 1: TF-IDF + Logistic Regression
- **Mục đích**: Xây dựng baseline model sử dụng phương pháp truyền thống.
- **Bước thực hiện**:
  - Sử dụng `TfidfVectorizer` để chuyển văn bản thành vector.
  - Kết hợp với `LogisticRegression` trong một pipeline.
  - Huấn luyện trên tập train và đánh giá trên tập test.

### Task 2: Word2Vec (Average) + Dense Layer
- **Mục đích**: Sử dụng word embeddings nhưng chưa xử lý được tính tuần tự của văn bản.
- **Bước thực hiện**:
  - Huấn luyện mô hình Word2Vec trên dữ liệu training.
  - Chuyển đổi mỗi câu thành vector trung bình của các từ.
  - Xây dựng mạng neural với Dense layers để phân loại.

### Task 3: Pre-trained Embedding + LSTM
- **Mục đích**: Sử dụng LSTM với embedding đã được pre-train từ Word2Vec.
- **Bước thực hiện**:
  - Tokenize văn bản và padding sequences.
  - Tạo embedding matrix từ Word2Vec đã huấn luyện.
  - Xây dựng mô hình Sequential: Embedding (frozen) → LSTM → Dense.
  - Sử dụng EarlyStopping để tránh overfitting.

### Task 4: Scratch Embedding + LSTM
- **Mục đích**: So sánh hiệu quả khi embedding layer được học từ đầu.
- **Bước thực hiện**:
  - Sử dụng cùng architecture như Task 3 nhưng embedding layer trainable.
  - Huấn luyện end-to-end với EarlyStopping.

### Task 5: Evaluation & Analysis
- **Mục đích**: So sánh định lượng và định tính giữa các mô hình.
- **Bước thực hiện**:
  - Tính macro F1-score và test loss cho cả 4 mô hình.
  - Phân tích qualitative trên các câu khó có cấu trúc phủ định/phức tạp.
  - Tạo bảng so sánh và nhận xét.

## 2. Hướng Dẫn Thực Thi Mã
### 2.1 Yêu Cầu Hệ Thống
```bash
pip install pandas numpy scikit-learn gensim tensorflow jupyter
```

### 2.2 Cấu Trúc Thư Mục
```
Lab06/
├── lab6_rnns_text_classification.ipynb
├── hwu/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── rnns_text_classification.md
```

### 2.3 Cách Chạy
1. **Mở Jupyter Notebook**:
   ```bash
   jupyter notebook lab6_rnns_text_classification.ipynb
   ```

2. **Chạy từng cell theo thứ tự**:
   - Cell 1: Import libraries và setup random seed.
   - Cell 2: Load dữ liệu từ CSV files.
   - Cell 3: Label encoding.
   - Cell 4-5: Task 1 (TF-IDF + LR).
   - Cell 6-7: Task 2 (Word2Vec + Dense).
   - Cell 8-9: Task 3 (Pre-trained Embedding + LSTM).
   - Cell 10-11: Task 4 (Scratch Embedding + LSTM).
   - Cell 12-13: Task 5 (Evaluation & Analysis).

## 3. Phân Tích Kết Quả
### 3.1 Kết quả Task 1
```
                accuracy                           0.84      1076
               macro avg       0.85      0.83      0.84      1076
            weighted avg       0.84      0.84      0.84      1076
```
- Accuracy 0.84 : Đây là mức tốt cho baseline TF-IDF + LR, nhất là nếu dữ liệu có nhiều nhãn hoặc câu ngắn.
- Macro avg ~ Weighted avg : Cho thấy các lớp được cân bằng tốt, không có lớp nào bị bỏ quên.

### 3.2 Kết quả Task 2
```
                accuracy                           0.35      1076
               macro avg       0.34      0.33      0.30      1076
            weighted avg       0.35      0.35      0.32      1076
```
- Accuracy 0.35 : Thấp hơn nhiều so với TF-IDF + LR, cho thấy việc mất thông tin về thứ tự từ ảnh hưởng lớn.
- Tập train nhỏ có thể không đủ để học embeddings tốt.
- Vector trung bình làm mất ngữ cảnh.
- Mạng nông không đủ mạnh để bù đắp.

### 3.3 Kết quả Task 3
```
                accuracy                           0.40      1076
               macro avg       0.39      0.39      0.38      1076
            weighted avg       0.40      0.40      0.39      1076
```
- Accuracy 0.40 : Cải thiện so với Task 2, cho thấy LSTM giúp nắm bắt thông tin tuần tự.
- Pre-trained embeddings giúp mô hình học nhanh hơn và hiệu quả hơn.
- Tuy nhiên, vẫn chưa vượt trội so với TF-IDF + LR có thể do tập dữ liệu nhỏ và cấu trúc câu đơn giản.

### 3.4 Kết quả Task 4
```
                accuracy                           0.27      1076
               macro avg       0.16      0.25      0.18      1076
            weighted avg       0.17      0.27      0.20      1076
```
- Accuracy 0.27 : Thấp hơn so với Task 3, cho thấy việc học embedding từ đầu gặp khó khăn với tập dữ liệu nhỏ.
- Sử dụng early stopping giúp tránh overfitting nhưng mô hình vẫn chưa học được biểu diễn tốt.
- Cần nhiều dữ liệu hơn để embedding layer học hiệu quả.

### 3.5 Kết quả Task 5
- Bảng Tổng Hợp Kết Quả
```
| Pipeline | F1-score (Macro) | Test Loss | Nhận Xét |
|----------|------------------|-----------|----------|
| TF-IDF + Logistic Regression | 0.835298 | 1.050197 | Hiệu quả bất ngờ, đơn giản và nhanh. |
| Word2Vec (Avg) + Dense | 0.304154 | 2.452722 | Mất thông tin thứ tự, mạng nông. |
| Pre-trained Embedding + LSTM | 0.376478 | 2.108491 | Xử lý tuần tự ổn, tuy nhiên cần tuning thêm. |
| Scratch Embedding + LSTM | 0.178246 | 2.868297 | Flexible nhưng cần nhiều dữ liệu hơn. |
```

- Nhận Xét Chung:
  - TF-IDF + LR vẫn là lựa chọn tốt cho tập dữ liệu nhỏ và câu đơn giản.
  - Mô hình dựa trên embeddings và LSTM cần nhiều dữ liệu hơn để phát huy hiệu quả.
  - Pre-trained embeddings giúp cải thiện so với học từ đầu, nhưng vẫn chưa đủ để vượt qua phương pháp truyền thống.

- Phân Tích Định Tính Các Câu Khó
```
| Sentence | True Intent | TF-IDF + LR | W2V Avg + Dense | Pretrained LSTM | Scratch LSTM |
|-----------|--------------|--------------|------------------|------------------|---------------|
| can you remind me to not call my mom | reminder_create | calendar_set | general_quirky | takeaway_query | email_sendemail |
| is it going to be sunny or rainy tomorrow | weather_query | weather_query | qa_maths | qa_maths | takeaway_order |
| find a flight from new york to london but not ... | flight_search | general_negate | transport_query | email_sendemail | calendar_set |
```

- **Câu 1:** "can you remind me to not call my mom" với phủ định "not call"
  - TF-IDF không hiểu phủ định dễ nhầm “remind” với “calendar_set”
  - W2V Avg mất ngữ cảnh, chọn intent chung chung.
  - Pre-trained LSTM Nắm bắt phủ định tốt hơn, nhưng vẫn nhầm với “takeaway_query”.
  - Scratch LSTM Không học được biểu diễn ngữ nghĩa, chủ yếu dựa vào từ khóa.

- **Câu 2:** "is it going to be sunny or rainy tomorrow"
  - TF-IDF + LR đúng intent nhờ từ khóa “sunny”, “rainy”.
  - W2V Avg + Dense không hiểu ngữ cảnh, chọn intent không liên quan.
  - Pre-trained LSTM và Scratch LSTM chưa nắm bắt được ý định chính xác, dễ nhầm lẫn giữa các lớp hỏi đáp.

- **Câu 3:** "find a flight from new york to london but not ..."
  - TF-IDF + LR không hiểu ngữ cảnh “find a flight”, nhầm sang intent “general_negate”.
  - W2V Avg + Dense không nắm bắt được ý định chính xác.
  - Pre-trained LSTM và Scratch LSTM đều không hiểu phủ định “but not”, dẫn đến nhầm lẫn.

## 4. Thách Thức và Giải Pháp
### 4.1 Thách Thức
- Dữ liệu nhỏ hạn chế khả năng học biểu diễn tốt.
- Các câu phức tạp với phủ định, mệnh đề đa nghĩa.
- Overfitting do mô hình phức tạp với dữ liệu hạn chế.

### 4.2 Giải Pháp
- Sử dụng pre-trained embeddings để tận dụng kiến thức ngôn ngữ có sẵn.
- Fine tuning mô hình ngữ cảnh.
- Kết hợp nhiều đặc trưng (TF-IDF + embeddings) hoặc (TF-IDF + LSTM) để tận dụng ưu điểm của cả hai.
- Data augmentation để tăng kích thước tập huấn luyện.
- Regularization techniques như Dropout, Early Stopping để giảm overfitting.
- Hyperparameter tuning để tìm cấu hình tối ưu.


## 5 Hướng Phát Triển
- Thử nghiệm với Transformer models (BERT, DistilBERT).
- Ensemble methods kết hợp multiple approaches.
- Advanced preprocessing (spelling correction, normalization).
- Hyperparameter optimization với tools như Optuna.

---
---
