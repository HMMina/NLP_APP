# NLP_APP_Lab1

## Lab 1: Tokenization
Lab 1 t?p trung v�o vi?c x�y d?ng c�c tokenizer kh�c nhau d? ph�n t�ch van b?n th�nh c�c token ri�ng l?:
1. X�y d?ng c�c interface c?n thi?t trong `core/interfaces.py`:
   - Interface `Tokenizer` d?nh nghia phuong th?c `tokenize` d? ph�n t�ch van b?n

2. C�i d?t c�c tokenizer c? th?:
   - `SimpleTokenizer`: Tokenizer don gi?n d?a tr�n vi?c s? d?ng replace d? th�m kho?ng tr?ng v�o tru?c v� sau k� t? d?c bi?t v� sau d� ph�n t�ch van b?n d?a tr�n kho?ng tr?ng.
   - `RegexTokenizer`: Tokenizer n�ng cao s? d?ng regular expressions "\w+|[^\w\s]" d? l?c ra c�c token th?a m�n di?u ki?n.

## Lab 2: Vector h�a van b?n (Count Vectorizer)
Lab 2 t?p trung v�o vi?c vector h�a van b?n s? d?ng k? thu?t Count Vectorizer:
1. X�y d?ng interface `Vectorizer` trong `core/interfaces.py`
2. C�i d?t `CountVectorizer` d? kh?i t?o t? di?n t? t?p h?p c�c token v� chuy?n d?i van b?n th�nh vector d?c trung d?a tr�n t?n su?t xu?t hi?n c?a c�c token trong van b?n.

## C�ch ch?y code
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

#### Testing UD_English-EWT Data (5 d�ng d?u ti�n):
Count Vectorizer New Corpus from UD_English-EWT:
Vocabulary: {"!": 0, ",": 1, "-": 2, ".": 3, "2": 4, "3": 5, ":": 6, "[": 7, "]": 8, "a": 9, "abdullah": 10, "al": 11, "american": 12, "ani": 13, "announced": 14, "at": 15, "authorities": 16, "baghdad": 17, "be": 18, "being": 19, "border": 20, "busted": 21, "by": 22, "causing": 23, "cells": 24, "cleric": 25, "come": 26, "dpa": 27, "for": 28, "forces": 29, "had": 30, "in": 31, "interior": 32, "iraqi": 33, "killed": 34, "killing": 35, "ministry": 36, "moi": 37, "mosque": 38, "near": 39, "of": 40, "officials": 41, "operating": 42, "preacher": 43, "qaim": 44, "respected": 45, "run": 46, "shaikh": 47, "syrian": 48, "terrorist": 49, "that": 50, "the": 51, "them": 52, "they": 53, "this": 54, "to": 55, "town": 56, "trouble": 57, "two": 58, "up": 59, "us": 60, "were": 61, "will": 62, "years": 63, "zaman": 64}
Document-Term Matrix:
[0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

## C?u tr�c d? �n

```
NLP_APP/
 Lab1/
    src/
       core/
          interfaces.py         # �?nh nghia c�c interface
          dataset_loaders.py    # C�c loader cho d? li?u
       preprocessing/
          simple_tokenizer.py   # Tokenizer don gi?n
          regex_tokenizer.py    # Tokenizer s? d?ng regex
       representations/
           count_vectorizer.py   # Vector h�a van b?n
    test/
        lab1_test.py              # Test cho Lab 1
        lab2_test.py              # Test cho Lab 2
 UD_English-EWT/                   # D? li?u ti?ng Anh
```

## Gi?i th�ch k?t qu? thu du?c

### Lab 1: Tokenization

- **SimpleTokenizer v� RegexTokenizer** d?u hi?n th?c h�a vi?c ph�n t�ch van b?n th�nh c�c token ri�ng l?
- K?t qu? cho th?y c? hai tokenizer d?u ph�n t�ch ch�nh x�c van b?n d?a tr�n kho?ng tr?ng v� d?u c�u
- SimpleTokenizer s? d?ng phuong ph�p don gi?n b?ng c�ch th�m kho?ng tr?ng tru?c v� sau c�c k� t? d?c bi?t, sau d� t�ch chu?i d?a tr�n kho?ng tr?ng v� lowercase
- RegexTokenizer s? d?ng bi?u th?c ch�nh quy d? t�m ki?m c�c token d?a tr�n m?u `\w+|[^\w\s]`, gi�p ph�n t�ch du?c c? t? v� k� t? d?c bi?t

### Lab 2: Count Vectorizer

- **CountVectorizer** chuy?n d?i van b?n th�nh vector d?c trung d?a tr�n t?n su?t xu?t hi?n c?a c�c token
- Vocabulary du?c x�y d?ng t? t?t c? c�c token duy nh?t trong corpus
- Document-Term Matrix hi?n th? t?n su?t xu?t hi?n c?a m?i token trong t?ng van b?n
- K?t qu? cho th?y van b?n d� du?c chuy?n d?i th�nh d?ng s? h?c, c� th? s? d?ng cho c�c thu?t to�n h?c m�y

## Kh� khan g?p ph?i v� c�ch gi?i quy?t

### V?n d? import module

**Kh� khan:** 
- Khi ch?y file test t? terminal, g?p l?i `ModuleNotFoundError: No module named "src"` ho?c `ModuleNotFoundError: No module named "Lab1"`
- C�c module kh�ng th? t�m th?y do du?ng d?n import kh�ng ph� h?p v?i c?u tr�c thu m?c

**Gi?i ph�p:**
- Th�m du?ng d?n g?c c?a workspace v�o `sys.path` d? Python c� th? t�m th?y c�c module
- S? d?ng `import sys` v� `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))`
- S?a l?i c�c import d? ph� h?p v?i c?u tr�c thu m?c

### �?nh nghia Interface

**Kh� khan:**
- L?i import `from ABC import ABC, abstractmethod` thay v� `from abc import ABC, abstractmethod`

**Gi?i ph�p:**
- S?a l?i d�ng c� ph�p import t? thu vi?n chu?n Python

## T�i li?u tham kh?o

- [Python Documentation on Regular Expressions](https://docs.python.org/3/library/re.html)
- [NLTK Documentation](https://www.nltk.org/)
- [Universal Dependencies - English Web Treebank](https://universaldependencies.org/)
