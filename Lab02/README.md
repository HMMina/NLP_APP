# Báo Cáo Lab: Spark NLP Pipeline với TF-IDF
## Các Bước Triển Khai

### 1. Thiết Lập Môi Trường
- **Phiên bản Scala**: 2.12.18
- **Phiên bản Spark**: 3.5.1
- **Phiên bản Java**: 19.0.1
- **Phiên bản SBT**: 1.11.0

### 2. Tải Dữ Liệu
```scala
val df = spark.read.json("../../c4-train.00000-of-01024-30K.json")
  .limit(1000) // Giới hạn 1000 bản ghi để thực thi nhanh hơn
val textDF = df.select("text").na.drop()
```
- Tải thành công bộ dữ liệu C4 vào Spark DataFrame
- Áp dụng làm sạch dữ liệu bằng cách loại bỏ giá trị null
- Giới hạn kích thước dữ liệu cho mục đích demo

### 3. Triển Khai Pipeline
```scala
val pipeline = new Pipeline()
  .setStages(Array(tokenizer, remover, hashingTF, idf))
```

**Các Giai Đoạn Pipeline:**
1. **RegexTokenizer**: Phân tách văn bản sử dụng pattern "\\s+|[.,;!?()\"']"(ký tự không phải từ)
2. **StopWordsRemover**: Lọc bỏ các từ dừng tiếng Anh phổ biến
3. **HashingTF**: Chuyển đổi tokens thành vectors tần suất từ (20,000 features)
4. **IDF**: Áp dụng trọng số tần suất nghịch đảo văn bản

### 4. Chiến Lược Tokenization
- **Chính**: RegexTokenizer với pattern "\\s+|[.,;!?()\"']"
- **Thay thế**: Basic Tokenizer (dựa trên khoảng trắng) có sẵn dưới dạng comment
- RegexTokenizer xử lý tốt hơn dấu câu và ký tự đặc biệt

### 5. Quá Trình Vector Hóa
- **HashingTF**: 
  - Không gian đặc trưng: 20,000 chiều
  - Xử lý va chạm hash cho từ vựng lớn
  - Sử dụng bộ nhớ hiệu quả
- **IDF**: 
  - Giảm tầm quan trọng của các từ xuất hiện thường xuyên
  - Nhấn mạnh các từ độc đáo, mang tính thông tin

## Cách Chạy Code
### Các Bước Thực Thi
```bash
# Di chuyển đến thư mục dự án
cd C:\Users\ADMIN\.vscode\NLP_APP\Lab02\spark_labs

# Tạo các thư mục cần thiết
mkdir log, results

# Biên dịch dự án
sbt compile

# Chạy ứng dụng
sbt run
```

### Cấu Trúc Kết Quả Mong Đợi
```
log/
├── pipeline_YYYYMMDD_HHMMSS.log    # Log quá trình với timestamps
results/
├── lab17_pipeline_output.txt       # Kết quả chính thức theo yêu cầu
```

### Nội Dung Log File
```
Pipeline started at: 2025-10-01 19:55:14
SparkSession created successfully
Loading data from: ../../c4-train.00000-of-01024-30K.json
Loaded 1000 records
Tokenizer configured
StopWordsRemover configured
HashingTF configured with 20000 features
IDF configured
Pipeline configured
Starting pipeline fitting...
Pipeline fitting completed
Saving results to file...
Results saved successfully to results/lab17_pipeline_output.txt
Total documents processed: 1000
Vocabulary size (HashingTF): 20000
Pipeline completed successfully at: 2025-10-01 19:55:24
Spark session closed successfully

```

## Phân Tích Kết Quả

### Thống Kê Dữ Liệu
- **Tổng số văn bản được xử lý**: 1,000
- **Kích thước vector đặc trưng**: 20,000 chiều

### Định Dạng Kết Quả Mẫu
```
Document 1:
Text: Beginners BBQ Class Taking Place in Missoula!...
Features: (20000,[264,298,673,717,829,1271,1466,1499,1600...],[....])

Document 2:
Text: Text: Discussion in 'Mac OS X Lion (10.7)'....
Features: (20000,[406,643,651,1023,1349,1695,1845,1850,1994...],[....])

....
```

### Kiểm Tra Kết Quả
```bash
# Kiểm tra file output đã được tạo
ls results/lab17_pipeline_output.txt
```

### Phân Tích Vector Đặc Trưng
- Biểu diễn thưa thớt giảm thiểu footprint bộ nhớ
- Giá trị khác không cho thấy mức độ quan trọng của từ sau khi áp dụng TF-IDF
- Giá trị cao hơn đại diện cho các từ đặc trưng hơn của mỗi văn bản

## Khó Khăn Gặp Phải và Giải Pháp
### 1. Tương Thích Phiên Bản Java
**Vấn đề**: 
- Vấn đề tương thích Java 19 với Spark 3.5.x
- Hạn chế truy cập module trong các phiên bản Java mới hơn

**Giải pháp**: 
- Thêm JVM arguments trong `build.sbt`:
```scala
javaOptions ++= Seq(
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  // ... thêm các opens khác
)
```

### 2. Giải Quyết Đường Dẫn
**Vấn đề**: 
- Vấn đề đường dẫn tương đối khi chạy từ các thư mục khác nhau
- Nhầm lẫn vị trí file dữ liệu

**Giải pháp**: 
- Sử dụng đường dẫn tương đối `../../c4-train.00000-of-01024-30K.json` từ thư mục spark_labs
- Thêm logging phù hợp để theo dõi trạng thái tải file

### 3. Triển Khai Logging
**Vấn đề**: 
- Cần logging quá trình toàn diện như yêu cầu

**Giải pháp**: 
- Triển khai custom logging sử dụng Java PrintWriter
- Thêm timestamps và theo dõi lỗi
- Tạo các file log riêng biệt cho mỗi lần chạy

## Kiến Trúc Code
### Cấu Trúc Class
```scala
object TextPipelineExample {
  def main(args: Array[String]): Unit = {
    // 1. Khởi tạo và thiết lập logging
    // 2. Tạo Spark session
    // 3. Tải và tiền xử lý dữ liệu
    // 4. Cấu hình các stage của pipeline
    // 5. Thực thi pipeline
    // 6. Lưu kết quả và logging
    // 7. Dọn dẹp và xử lý lỗi
  }
}
```

### Chiến Lược Xử Lý Lỗi
- Khối try-catch để quản lý ngoại lệ
- Graceful shutdown của Spark session
- Logging lỗi toàn diện
- Dọn dẹp tài nguyên trong khối finally

## Mở Rộng và Bài Tập

Codebase hỗ trợ bốn bài tập như được chỉ định:

1. **Chuyển Đổi Tokenizer**: Comment/uncomment các triển khai tokenizer thay thế
2. **Điều Chỉnh Feature Vector**: Sửa đổi tham số `numFeatures` (20000 → 1000)
3. **Mở Rộng Mô Hình ML**: Framework sẵn sàng để thêm LogisticRegression
4. **Tích Hợp Word2Vec**: Phương pháp vectorization thay thế đã được bao gồm

## Kết Luận

Spark NLP pipeline đã triển khai thành công tất cả các thành phần yêu cầu:
- ✅ Nhập dữ liệu từ bộ dữ liệu C4
- ✅ Tiền xử lý văn bản với tokenization và loại bỏ từ dừng
- ✅ Vector hóa TF-IDF cho biểu diễn số
- ✅ Logging toàn diện và lưu trữ kết quả
- ✅ Xử lý lỗi và quản lý tài nguyên
