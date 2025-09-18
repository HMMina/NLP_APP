# NLP_APP

Dự án thực hành xử lý ngôn ngữ tự nhiên (NLP) bao gồm các lab về tokenization, vector hóa văn bản, và các kỹ thuật xử lý ngôn ngữ tự nhiên cơ bản.

## Các bước triển khai

### Lab 1: Tokenization

Lab 1 tập trung vào việc xây dựng các tokenizer khác nhau để phân tách văn bản thành các token riêng lẻ:

1. Xây dựng các interface cần thiết trong `core/interfaces.py`:
   - Interface `Tokenizer` định nghĩa phương thức `tokenize` để phân tách văn bản

2. Cài đặt các tokenizer cụ thể:
   - `SimpleTokenizer`: Tokenizer đơn giản dựa trên khoảng trắng và dấu câu
   - `RegexTokenizer`: Tokenizer nâng cao sử dụng regular expressions

### Lab 2: Vector hóa văn bản (Count Vectorizer)

Lab 2 tập trung vào việc vector hóa văn bản sử dụng kỹ thuật Count Vectorizer:

1. Xây dựng interface `Vectorizer` trong `core/interfaces.py`
2. Cài đặt `CountVectorizer` để chuyển đổi văn bản thành vector đặc trưng

## Cách chạy code

### Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### Lab 1: Chạy các tokenizer

```bash
# Di chuyển đến thư mục gốc dự án
cd NLP_APP

# Chạy test cho Lab 1
python Lab1/test/lab1_test.py
```

### Lab 2: Chạy Count Vectorizer

```bash
# Di chuyển đến thư mục gốc dự án
cd NLP_APP

# Chạy test cho Lab 2
python Lab1/test/lab2_test.py
```

## Log kết quả

### Lab 1: Kết quả Tokenization

Khi chạy các tokenizer, bạn sẽ thấy kết quả phân tách văn bản thành các token:

```
Input: "I love NLP."
SimpleTokenizer: ['I', 'love', 'NLP', '.']
RegexTokenizer: ['I', 'love', 'NLP', '.']

Input: "She's studying at university!"
SimpleTokenizer: ['She', "'s", 'studying', 'at', 'university', '!']
RegexTokenizer: ['She', "'s", 'studying', 'at', 'university', '!']
```

### Lab 2: Kết quả Count Vectorizer

Khi chạy Count Vectorizer, bạn sẽ nhận được Document-Term Matrix (DTM):

```
Corpus:
["I love NLP.", "I love programming.", "NLP is a subfield of AI."]

Vocabulary: {'I': 0, 'NLP': 1, 'a': 2, 'is': 3, 'love': 4, 'of': 5, 'programming': 6, 'subfield': 7, 'AI': 8, '.': 9}

Document-Term Matrix:
[1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
[1, 0, 0, 0, 1, 0, 1, 0, 0, 1]
[0, 1, 1, 1, 0, 1, 0, 1, 1, 1]
```

## Cấu trúc dự án

```
NLP_APP/
├── Lab1/
│   ├── src/
│   │   ├── core/
│   │   │   ├── interfaces.py         # Định nghĩa các interface
│   │   │   └── dataset_loaders.py    # Các loader cho dữ liệu
│   │   ├── preprocessing/
│   │   │   ├── simple_tokenizer.py   # Tokenizer đơn giản
│   │   │   └── regex_tokenizer.py    # Tokenizer sử dụng regex
│   │   └── representations/
│   │       └── count_vectorizer.py   # Vector hóa văn bản
│   └── test/
│       ├── lab1_test.py              # Test cho Lab 1
│       └── lab2_test.py              # Test cho Lab 2
└── UD_English-EWT/                   # Dữ liệu tiếng Anh
```

## Giải thích kết quả thu được

[Phần này sẽ được bổ sung sau khi có kết quả đầy đủ]

## Khó khăn gặp phải và cách giải quyết

[Phần này sẽ được bổ sung sau khi hoàn thành dự án]

## Nguồn tham khảo

[Phần này sẽ được bổ sung nếu có tham khảo từ các nguồn bên ngoài]

## Model sử dụng

[Phần này sẽ được bổ sung nếu sử dụng các model tạo sẵn]
