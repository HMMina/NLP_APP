# NLP_APP_Lab1

## Lab 1: Tokenization
Lab 1 t?p trung vào vi?c xây d?ng các tokenizer khác nhau d? phân tách van b?n thành các token riêng l?:
1. Xây d?ng các interface c?n thi?t trong `core/interfaces.py`:
   - Interface `Tokenizer` d?nh nghia phuong th?c `tokenize` d? phân tách van b?n

2. Cài d?t các tokenizer c? th?:
   - `SimpleTokenizer`: Tokenizer don gi?n d?a trên vi?c s? d?ng replace d? thêm kho?ng tr?ng vào tru?c và sau kí t? d?c bi?t và sau dó phân tách van b?n d?a trên kho?ng tr?ng.
   - `RegexTokenizer`: Tokenizer nâng cao s? d?ng regular expressions "\w+|[^\w\s]" d? l?c ra các token th?a mãn di?u ki?n.

## Lab 2: Vector hóa van b?n (Count Vectorizer)
Lab 2 t?p trung vào vi?c vector hóa van b?n s? d?ng k? thu?t Count Vectorizer:
1. Xây d?ng interface `Vectorizer` trong `core/interfaces.py`
2. Cài d?t `CountVectorizer` d? kh?i t?o t? di?n t? t?p h?p các token và chuy?n d?i van b?n thành vector d?c trung d?a trên t?n su?t xu?t hi?n c?a các token trong van b?n.

## Cách ch?y code
```bash
pip install -r requirements.txt
```
Run Python File tr?c ti?p trong VSCode:
- M? file `lab1_test.py` trong thu m?c `test/`
- M? file `lab2_test.py` trong thu m?c `test/`

## Log k?t qu?
### Lab 1: K?t qu? Tokenization
#### Testing Task 2:
Testing Task 2:
Original: Hello, world! This is a test.
SimpleTokenizer: ["hello", ",", "world", "!", "this", "is", "a", "test", "."]
RegexTokenizer: ["hello", ",", "world", "!", "this", "is", "a", "test", "."]

Original: NLP is fascinating... isn"t it?
SimpleTokenizer: ["nlp", "is", "fascinating", ".", ".", ".", "isn", """, "t", "it", "?"]    
RegexTokenizer: ["nlp", "is", "fascinating", ".", ".", ".", "isn", """, "t", "it", "?"]     

Original: Let"s see how it handles 123 numbers and punctuation!
SimpleTokenizer: ["let", """, "s", "see", "how", "it", "handles", "123", "numbers", "and", "punctuation", "!"]
RegexTokenizer: ["let", """, "s", "see", "how", "it", "handles", "123", "numbers", "and", "punctuation", "!"]

#### Testing Task 3:
Testing Task 3:
--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...
SimpleTokenizer Output (first 20 tokens): ["al", "-", "zaman", ":", "american", "forces", "killed", "shaikh", "abdullah", "al", "-", "ani", ",", "the", "preacher", "at", "the", "mosque", "in", "the"]
RegexTokenizer Output (first 20 tokens): ["al", "-", "zaman", ":", "american", "forces", "killed", "shaikh", "abdullah", "al", "-", "ani", ",", "the", "preacher", "at", "the", "mosque", "in", "the"]

### Lab 2: K?t qu? Count Vectorizer
#### Testing Corpus:
Count Vectorizer Test Corpus:
Vocabulary: {".": 0, "a": 1, "ai": 2, "i": 3, "is": 4, "love": 5, "nlp": 6, "of": 7, "programming": 8, "subfield": 9}
Document-Term Matrix:
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]

#### Testing UD_English-EWT Data (5 dòng d?u tiên):
Count Vectorizer New Corpus from UD_English-EWT:
Vocabulary: {"!": 0, ",": 1, "-": 2, ".": 3, "2": 4, "3": 5, ":": 6, "[": 7, "]": 8, "a": 9, "abdullah": 10, "al": 11, "american": 12, "ani": 13, "announced": 14, "at": 15, "authorities": 16, "baghdad": 17, "be": 18, "being": 19, "border": 20, "busted": 21, "by": 22, "causing": 23, "cells": 24, "cleric": 25, "come": 26, "dpa": 27, "for": 28, "forces": 29, "had": 30, "in": 31, "interior": 32, "iraqi": 33, "killed": 34, "killing": 35, "ministry": 36, "moi": 37, "mosque": 38, "near": 39, "of": 40, "officials": 41, "operating": 42, "preacher": 43, "qaim": 44, "respected": 45, "run": 46, "shaikh": 47, "syrian": 48, "terrorist": 49, "that": 50, "the": 51, "them": 52, "they": 53, "this": 54, "to": 55, "town": 56, "trouble": 57, "two": 58, "up": 59, "us": 60, "were": 61, "will": 62, "years": 63, "zaman": 64}
Document-Term Matrix:
[0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

## C?u trúc d? án

```
NLP_APP/
 Lab1/
    src/
       core/
          interfaces.py         # Ð?nh nghia các interface
          dataset_loaders.py    # Các loader cho d? li?u
       preprocessing/
          simple_tokenizer.py   # Tokenizer don gi?n
          regex_tokenizer.py    # Tokenizer s? d?ng regex
       representations/
           count_vectorizer.py   # Vector hóa van b?n
    test/
        lab1_test.py              # Test cho Lab 1
        lab2_test.py              # Test cho Lab 2
 UD_English-EWT/                   # D? li?u ti?ng Anh
```

## Gi?i thích k?t qu? thu du?c

### Lab 1: Tokenization

- **SimpleTokenizer và RegexTokenizer** d?u hi?n th?c hóa vi?c phân tách van b?n thành các token riêng l?
- K?t qu? cho th?y c? hai tokenizer d?u phân tách chính xác van b?n d?a trên kho?ng tr?ng và d?u câu
- SimpleTokenizer s? d?ng phuong pháp don gi?n b?ng cách thêm kho?ng tr?ng tru?c và sau các ký t? d?c bi?t, sau dó tách chu?i d?a trên kho?ng tr?ng và lowercase
- RegexTokenizer s? d?ng bi?u th?c chính quy d? tìm ki?m các token d?a trên m?u `\w+|[^\w\s]`, giúp phân tách du?c c? t? và ký t? d?c bi?t

### Lab 2: Count Vectorizer

- **CountVectorizer** chuy?n d?i van b?n thành vector d?c trung d?a trên t?n su?t xu?t hi?n c?a các token
- Vocabulary du?c xây d?ng t? t?t c? các token duy nh?t trong corpus
- Document-Term Matrix hi?n th? t?n su?t xu?t hi?n c?a m?i token trong t?ng van b?n
- K?t qu? cho th?y van b?n dã du?c chuy?n d?i thành d?ng s? h?c, có th? s? d?ng cho các thu?t toán h?c máy

## Khó khan g?p ph?i và cách gi?i quy?t

### V?n d? import module

**Khó khan:** 
- Khi ch?y file test t? terminal, g?p l?i `ModuleNotFoundError: No module named "src"` ho?c `ModuleNotFoundError: No module named "Lab1"`
- Các module không th? tìm th?y do du?ng d?n import không phù h?p v?i c?u trúc thu m?c

**Gi?i pháp:**
- Thêm du?ng d?n g?c c?a workspace vào `sys.path` d? Python có th? tìm th?y các module
- S? d?ng `import sys` và `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))`
- S?a l?i các import d? phù h?p v?i c?u trúc thu m?c

### Ð?nh nghia Interface

**Khó khan:**
- L?i import `from ABC import ABC, abstractmethod` thay vì `from abc import ABC, abstractmethod`

**Gi?i pháp:**
- S?a l?i dúng cú pháp import t? thu vi?n chu?n Python

## Tài li?u tham kh?o

- [Python Documentation on Regular Expressions](https://docs.python.org/3/library/re.html)
- [NLTK Documentation](https://www.nltk.org/)
- [Universal Dependencies - English Web Treebank](https://universaldependencies.org/)
