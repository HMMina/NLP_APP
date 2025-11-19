# Báo Cáo Lab 06: Gán Nhãn Từ Loại (POS Tagging) với RNN (lab6_rnn_for_pos_tagging.ipynb)

## 📋 Mục Lục
1. [Giải Thích Các Bước Triển Khai](#1-giải-thích-các-bước-triển-khai)
2. [Hướng Dẫn Thực Thi Mã](#2-hướng-dẫn-thực-thi-mã)
3. [Phân Tích Kết Quả](#3-phân-tích-kết-quả)
4. [Thách Thức và Giải Pháp](#4-thách-thức-và-giải-pháp)
5. [Hướng Phát Triển](#5-hướng-phát-triển)

---

## 1. Giải Thích Các Bước Triển Khai
### Task 1: Tải và Tiền xử lý Dữ liệu
- **Mục đích**: Đọc dữ liệu từ định dạng CoNLL-U và xây dựng từ điển.
- **Bước thực hiện**:
  - Đọc file `.conllu` (train/dev) để lấy danh sách các câu, mỗi câu gồm các cặp `(word, upos)`.
  - Xây dựng `word_to_ix` (ánh xạ từ -> index) và `tag_to_ix` (ánh xạ nhãn -> index).
  - Thêm các token đặc biệt: `<PAD>` (đệm), `<UNK>` (từ lạ).
  - **Kết quả**:
    - Train sentences: 12,544 câu.
    - Dev sentences: 2,001 câu.
    - Vocab size: 6,733 từ (min_freq=3).
    - Tag size: 18 nhãn (bao gồm `<PAD>`).

### Task 2: Tạo PyTorch Dataset và DataLoader
- **Mục đích**: Chuẩn bị dữ liệu dạng batch để huấn luyện mô hình.
- **Bước thực hiện**:
  - Tạo class `POSDataset` kế thừa từ `torch.utils.data.Dataset`.
  - Viết hàm `collate_fn` sử dụng `pad_sequence` để đệm các câu trong batch về cùng độ dài.
  - Tạo `DataLoader` cho tập train (shuffle=True) và dev (shuffle=False) với `batch_size=64`.

### Task 3: Xây dựng Mô hình RNN
- **Mục đích**: Xây dựng mô hình sequence labeling sử dụng RNN.
- **Kiến trúc mô hình**:
  1. **Embedding Layer**: Chuyển đổi index của từ thành vector (dim=100).
  2. **RNN Layer**: Bidirectional RNN (hidden_dim=128) để nắm bắt ngữ cảnh hai chiều.
  3. **Dropout**: Giảm overfitting (p=0.1).
  4. **Linear Layer**: Ánh xạ output của RNN sang không gian nhãn (output_dim=18).
- **Kỹ thuật đặc biệt**: Sử dụng `pack_padded_sequence` và `pad_packed_sequence` để RNN bỏ qua các token padding, tăng hiệu quả tính toán.

### Task 4: Huấn luyện Mô hình
- **Cấu hình**:
  - Optimizer: Adam (lr=1e-3).
  - Loss function: CrossEntropyLoss (ignore_index=PAD_TAG).
  - Epochs: 20.
- **Quy trình**:
  - Forward pass -> Tính loss -> Backward pass -> Update weights.
  - Theo dõi loss và accuracy trên tập dev sau mỗi epoch.
  - Lưu lại trạng thái mô hình tốt nhất (best dev accuracy).

### Task 5: Đánh giá Mô hình
- **Mục đích**: Kiểm tra hiệu suất mô hình trên tập dev và dự đoán câu mới.
- **Phương pháp**:
  - Tính accuracy trên các token thực (bỏ qua padding).
  - Viết hàm `predict_sentence` để gán nhãn cho câu nhập vào bất kỳ.

---

## 2. Hướng Dẫn Thực Thi Mã
### 2.1 Yêu Cầu Hệ Thống
```bash
pip install torch numpy
```

### 2.2 Cấu Trúc Thư Mục
```
Lab06/
├── lab6_rnn_for_pos_tagging.ipynb
├── rnn_for_pos_tagging.md
└── ../UD_English-EWT/
    ├── en_ewt-ud-train.conllu
    └── en_ewt-ud-dev.conllu
```

### 2.3 Cách Chạy
1. **Mở Jupyter Notebook**:
   ```bash
   jupyter notebook lab6_rnn_for_pos_tagging.ipynb
   ```
2. **Chạy tuần tự các cell**:
   - Cell 1-2: Import và setup.
   - Cell 3-5: Task 1 (Data Loading & Vocab).
   - Cell 6-8: Task 2 (Dataset & DataLoader).
   - Cell 9-10: Task 3 (Model Definition).
   - Cell 11-13: Task 4 (Training).
   - Cell 14-15: Task 5 (Evaluation & Demo).

---

## 3. Phân Tích Kết Quả

### 3.1 Quá Trình Huấn Luyện Chi Tiết
Dưới đây là kết quả chi tiết của quá trình huấn luyện qua 20 epochs:

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Nhận Xét |
|:-----:|:----------:|:--------:|:------------:|:---------|
| 1     | 67.9458    | 33.3044  | 0.7603       | Khởi đầu tốt, mô hình học nhanh các quy luật cơ bản. |
| 2     | 36.7162    | 25.0620  | 0.8155       | Loss giảm mạnh (~50%), accuracy tăng >5%. |
| 3     | 27.3685    | 20.9647  | 0.8466       | |
| 4     | 21.4835    | 18.4810  | 0.8632       | |
| 5     | 17.7275    | 16.6868  | 0.8767       | Kết thúc giai đoạn học nhanh. |
| 6     | 15.0122    | 15.5063  | 0.8876       | |
| 7     | 12.7403    | 14.9340  | 0.8937       | Dev loss bắt đầu ổn định quanh mức 14-15. |
| 8     | 11.3818    | 14.6995  | 0.8957       | |
| 9     | 10.1980    | 14.7442  | 0.8952       | |
| 10    | 9.1291     | 15.0488  | 0.8945       | |
| 11    | 8.2330     | 14.5638  | 0.8999       | Tiệm cận mức 90%. |
| 12    | 7.4461     | 14.9955  | 0.8995       | |
| 13    | 6.8169     | 14.9773  | 0.9035       | Đạt độ chính xác cao nhất (Best Model). |
| 14    | 6.2771     | 15.7560  | 0.8972       | Dev loss bắt đầu tăng -> Dấu hiệu Overfitting. |
| 15    | 5.5244     | 16.7355  | 0.8951       | |
| 16    | 5.0551     | 16.2119  | 0.9021       | |
| 17    | 4.5238     | 16.8760  | 0.9006       | |
| 18    | 3.9932     | 17.4442  | 0.9014       | |
| 19    | 3.6125     | 18.6773  | 0.8960       | |
| 20    | 3.2173     | 18.6980  | 0.9000       | Train loss rất thấp nhưng Dev loss cao nhất. |

### 3.2 Nhận Xét

#### 1. Hiệu Suất Tổng Thể
- **Đỉnh cao (Peak Performance)**: Mô hình đạt độ chính xác tốt nhất là **90.35%** tại Epoch 13. Đây là kết quả rất khả quan cho một mô hình RNN đơn giản (không dùng pre-trained embeddings hay kiến trúc phức tạp như Transformer).
- **Tốc độ hội tụ**: Mô hình hội tụ khá nhanh. Chỉ sau 5 epochs đầu tiên, độ chính xác đã đạt 87.67%. Các epochs sau đó chủ yếu tinh chỉnh các trường hợp khó (như từ đa nghĩa, từ hiếm).

#### 2. Phân Tích Overfitting
- **Giai đoạn 1 (Epoch 1-11)**: Cả Train Loss và Dev Loss đều giảm. Đây là giai đoạn "Learning" hiệu quả nhất.
- **Giai đoạn 2 (Epoch 12-13)**: Dev Loss đi ngang (khoảng 14.9), trong khi Train Loss tiếp tục giảm. Đây là điểm tối ưu ("Sweet Spot").
- **Giai đoạn 3 (Epoch 14-20)**:
    - **Train Loss** giảm sâu xuống 3.2 (mô hình học thuộc lòng dữ liệu huấn luyện).
    - **Dev Loss** tăng ngược lại lên 18.7 (mô hình mất khả năng tổng quát hóa).
    - **Kết luận**: Việc huấn luyện thêm sau epoch 13 không mang lại lợi ích về độ chính xác và làm giảm tính tổng quát của mô hình. Cơ chế **Early Stopping** nên được kích hoạt tại đây.

#### 3. Vai Trò Của Bidirectional RNN
- Việc accuracy đạt >90% chứng tỏ kiến trúc 2 chiều (Bidirectional) rất hiệu quả.
- Ví dụ: Trong câu "I **book** a flight" và "A good **book**", từ "book" có embedding giống nhau nhưng ngữ cảnh (context) khác nhau. RNN 2 chiều giúp mô hình nhìn thấy "a flight" (tương lai) để quyết định "book" là VERB, và nhìn thấy "A good" (quá khứ) để quyết định "book" là NOUN.

#### 4. So Sánh Với Baseline
- Nếu gán nhãn ngẫu nhiên hoặc gán nhãn phổ biến nhất (Most Frequent Tag), độ chính xác thường chỉ đạt khoảng 80-85%.
- Kết quả 90.35% cho thấy mô hình thực sự học được cấu trúc ngữ pháp chứ không chỉ nhớ vẹt.

### 3.3 Ví Dụ Dự Đoán
Câu: *"I love NLP ."*
- **Dự đoán**: `[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'PROPN'), ('.', 'PUNCT')]`
- **Phân tích**:
  - "I" -> PRON (Đại từ): Chính xác.
  - "love" -> VERB (Động từ): Chính xác.
  - "NLP" -> PROPN (Danh từ riêng): Chính xác.
  - "." -> PUNCT (Dấu câu): Chính xác.

---

## 4. Thách Thức và Giải Pháp
### 4.1 Thách Thức
- **Từ lạ (OOV - Out of Vocabulary)**: Các từ không xuất hiện trong tập train sẽ bị gán là `<UNK>`, làm giảm độ chính xác.
- **Từ đa nghĩa**: Một từ có thể có nhiều nhãn tùy ngữ cảnh (vd: "book" có thể là NOUN hoặc VERB).
- **Overfitting**: Sau khoảng 13 epochs, mô hình bắt đầu học thuộc lòng tập train (loss train giảm sâu nhưng loss dev tăng).

### 4.2 Giải Pháp
- **Early Stopping**: Dừng huấn luyện khi accuracy trên tập dev không cải thiện sau một số epoch nhất định (trong trường hợp này là sau epoch 13).
- **Xử lý OOV**: Sử dụng token `<UNK>` và thay thế các từ tần suất thấp bằng `<UNK>` khi training để mô hình học cách xử lý từ lạ.
- **Ngữ cảnh**: Sử dụng **Bidirectional RNN** để xem xét ngữ cảnh toàn cục.
- **Padding & Packing**: Sử dụng `pad_sequence` kết hợp `pack_padded_sequence` để xử lý batch hiệu quả mà không tính toán trên phần đệm.

---

## 5. Hướng Phát Triển
### 5.1 Cải Tiến Mô Hình
- **LSTM/GRU**: Thay thế RNN thường bằng LSTM hoặc GRU để xử lý phụ thuộc xa tốt hơn (tránh vanishing gradient).
- **CRF (Conditional Random Fields)**: Thêm lớp CRF lên trên RNN để mô hình hóa sự phụ thuộc giữa các nhãn liên tiếp (vd: ADJ thường đứng trước NOUN).
- **Pre-trained Embeddings**: Sử dụng GloVe hoặc Word2Vec thay vì học embedding từ đầu.

### 5.2 Tối Ưu Hóa
- **Hyperparameter Tuning**: Thử nghiệm với learning rate, hidden dim, số layers khác nhau.
- **Data Augmentation**: Tăng cường dữ liệu để mô hình tổng quát hóa tốt hơn.

---
**Kết luận**: Mô hình RNN đơn giản đã giải quyết tốt bài toán POS Tagging với độ chính xác ấn tượng (~90%). Việc áp dụng các kỹ thuật xử lý chuỗi chuẩn (padding, packing, masking) và Bidirectional RNN là chìa khóa cho hiệu suất này.
