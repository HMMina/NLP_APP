# NLP_APP_Lab1
## Cấu trúc dự án
```
 Lab1/
    src/
       core/
          interfaces.py         # Định nghĩa các interface Tokenizer và Vectorizer
          dataset_loaders.py    # Loader cho dữ liệu UD_English-EWT
       preprocessing/
          simple_tokenizer.py   # Tokenizer đơn giản
          regex_tokenizer.py    # Tokenizer sử dụng regex
       representations/
           count_vectorizer.py   # Vector hóa văn bản
    test/
        lab1_test.py              # Test cho Lab 1
        lab2_test.py              # Test cho Lab 2
 UD_English-EWT/                   # Dữ liệu tiếng Anh
```

## Lab 1: Tokenization
Lab 1 tập trung vào việc xây dựng các tokenizer khác nhau để phân tách văn bản thành các token riêng lẻ:
1. Xây dựng các interface cần thiết trong `core/interfaces.py`:
   - Interface `Tokenizer` định nghĩa phương thức `tokenize` để phân tách văn bản

2. Cài đặt các tokenizer trong `preprocessing/` cụ thể:
   - `SimpleTokenizer`: Tokenizer đơn giản dựa trên việc sử dụng replace để thêm khoảng trắng vào trước và sau kí tự đặc biệt và sau đó phân tách văn bản dựa trên khoảng trắng.
   - `RegexTokenizer`: Tokenizer nâng cao sử dụng regular expressions "\w+|[^\w\s]" để lọc ra các token thỏa mãn điều kiện.

## Lab 2: Vector hóa văn bản (Count Vectorizer)
Lab 2 tập trung vào việc vector hóa văn bản sử dụng kỹ thuật Count Vectorizer:
1. Xây dựng interface `Vectorizer` trong `core/interfaces.py`
2. Cài đặt `CountVectorizer` trong `representations/` để khởi tạo từ điển từ tập hợp các token và chuyển đổi văn bản thành vector đặc trưng dựa trên tần suất xuất hiện của các token trong văn bản.

## Cách chạy code
```bash
pip install -r requirements.txt
```
Run Python File trực tiếp trong VSCode:
- Mở file `lab1_test.py` trong thư mục `test/`
- Mở file `lab2_test.py` trong thư mục `test/`

## Log kết quả
### Lab 1: Kết quả Tokenization
#### Testing Task 2:
Original: Hello, world! This is a test.
SimpleTokenizer: ["hello", ",", "world", "!", "this", "is", "a", "test", "."]
RegexTokenizer: ["hello", ",", "world", "!", "this", "is", "a", "test", "."]

Original: NLP is fascinating... isn't it?
SimpleTokenizer: ["nlp", "is", "fascinating", ".", ".", ".", "isn", "'", "t", "it", "?"]    
RegexTokenizer: ["nlp", "is", "fascinating", ".", ".", ".", "isn", "'", "t", "it", "?"]     

Original: Let's see how it handles 123 numbers and punctuation!
SimpleTokenizer: ["let", "'", "s", "see", "how", "it", "handles", "123", "numbers", "and", "punctuation", "!"]
RegexTokenizer: ["let", "'", "s", "see", "how", "it", "handles", "123", "numbers", "and", "punctuation", "!"]

#### Testing Task 3:
Testing Task 3:
--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...
SimpleTokenizer Output (first 20 tokens): ["al", "-", "zaman", ":", "american", "forces", "killed", "shaikh", "abdullah", "al", "-", "ani", ",", "the", "preacher", "at", "the", "mosque", "in", "the"]
RegexTokenizer Output (first 20 tokens): ["al", "-", "zaman", ":", "american", "forces", "killed", "shaikh", "abdullah", "al", "-", "ani", ",", "the", "preacher", "at", "the", "mosque", "in", "the"]

### Lab 2: Kết quả Count Vectorizer
#### Testing Corpus:
Count Vectorizer Test Corpus:
Vocabulary: {".", "a", "ai", "i", "is", "love", "nlp", "of", "programming", "subfield"}
Document-Term Matrix:
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]

## Giải thích kết quả thu được
### Lab 1: Tokenization
- **SimpleTokenizer và RegexTokenizer** đều hiện thực hóa việc phân tách văn bản thành các token riêng lẻ
- Kết quả cho thấy cả hai tokenizer đều phân tách chính xác văn bản dựa trên khoảng trắng và dấu câu
- SimpleTokenizer sử dụng phương pháp đơn giản bằng cách thêm khoảng trắng trước và sau các ký tự đặc biệt, sau đó tách chuỗi dựa trên khoảng trắng và lowercase
- RegexTokenizer sử dụng biểu thức chính quy để tìm kiếm các token dựa trên mẫu `\w+|[^\w\s]`, giúp phân tách được cả từ và ký tự đặc biệt

### Lab 2: Count Vectorizer
- **CountVectorizer** chuyển đổi văn bản thành vector đặc trưng dựa trên tần suất xuất hiện của các token
- Vocabulary được xây dựng từ tất cả các token duy nhất trong corpus
- Document-Term Matrix hiển thị tần suất xuất hiện của mỗi token trong từng văn bản
- Kết quả cho thấy văn bản đã được chuyển đổi thành dạng số học, có thể sử dụng cho các thuật toán học máy

## Khó khăn gặp phải và cách giải quyết
### Vấn đề import module
**Khó khăn:** 
- Khi chạy file test từ terminal, gặp lỗi `ModuleNotFoundError: No module named 'src'` hoặc `ModuleNotFoundError: No module named 'Lab1'`
- Các module không thể tìm thấy do đường dẫn import không phù hợp với cấu trúc thư mục

**Giải pháp:**
- Thêm đường dẫn gốc của workspace vào `sys.path` để Python có thể tìm thấy các module
- Sử dụng `import sys` và `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))` để thêm thư mục cha vào đường dẫn tìm kiếm module

