# PyTorch Tensor

## 1. Tensor কী?

PyTorch-এ **Tensor** হলো এমন একটি data structure, যার মধ্যে numerical data রাখা যায় এবং যার ওপর বিভিন্ন mathematical operation করা যায়। Tensor-কে NumPy array-এর মতো ভাবা যায়, তবে PyTorch Tensor দিয়ে machine learning ও deep learning-এর কাজও করা যায়।

```python
import torch

x = torch.tensor([1, 2, 3, 4, 5])
print(x)
```

এখানে `torch.tensor()` একটি Python list-কে PyTorch Tensor-এ রূপান্তর করেছে।

## 2. Tensor তৈরি করা

`torch.tensor()` ব্যবহার করে সরাসরি Tensor তৈরি করা যায়।

```python
x = torch.tensor([1, 2, 3, 4, 5])
```

এখানে `x` একটি **1D Tensor**। এর মধ্যে পাঁচটি element আছে।

## 3. Dimension এবং Shape

Tensor-এর গঠন বোঝার জন্য `ndim`, `shape` এবং `size()` ব্যবহার করা হয়।

```python
print(x.ndim)
print(x.shape)
print(x.size())
print(x.dtype)
```

- `ndim` Tensor-এর dimension-এর সংখ্যা জানায়।
- `shape` প্রতিটি dimension-এ কতটি element আছে তা জানায়।
- `size()` shape-এর তথ্য দেয়।
- `dtype` Tensor-এর element-গুলোর data type জানায়।

`x`-এর ক্ষেত্রে `ndim` হবে `1` এবং `shape` হবে `torch.Size([5])`। অর্থাৎ এটি একটি dimension-এ পাঁচটি element রাখে।

## 4. 2D Tensor তৈরি করা

একটি Tensor-এর মধ্যে একাধিক row থাকলে সেটি 2D হতে পারে।

```python
y = torch.tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

print(y)
print(y.ndim)
print(y.shape)
print(y.size())
print(y.dtype)
```

এখানে দুটি row এবং প্রতিটি row-তে পাঁচটি element আছে। তাই `y.ndim` হলো `2` এবং `y.shape` হলো `torch.Size([2, 5])`। Shape-এর প্রথম সংখ্যা row-এর সংখ্যা এবং দ্বিতীয় সংখ্যা column-এর সংখ্যা বোঝায়।

## 5. Tensor-এর Data Type

Tensor-এর data type বোঝায় এর element-গুলো কী ধরনের value ধারণ করছে। যেমন integer বা floating-point number।

```python
x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([1.0, 2.0, 3.0, 4.0])

print(x.dtype)
print(y.dtype)
```

সাধারণত integer value দিয়ে তৈরি Tensor-এর type `torch.int64` এবং decimal value দিয়ে তৈরি Tensor-এর type `torch.float32` হতে পারে। নির্দিষ্ট data type প্রয়োজন হলে `dtype` argument দিয়ে তা উল্লেখ করা যায়।

## 6. Tensor Indexing

Indexing দিয়ে Tensor-এর নির্দিষ্ট element access করা যায়। Python-এর মতো PyTorch-এও index `0` থেকে শুরু হয়।

```python
x = torch.tensor([10, 20, 30, 40, 50])

print(x[0])
print(x[2])
print(x[-1])
```

এখানে `x[0]` প্রথম element `10`, `x[2]` তৃতীয় element `30`, আর `x[-1]` শেষ element `50` দেবে। Negative indexing দিয়ে শেষ দিক থেকে element access করা যায়।

## 7. Tensor Slicing

Slicing দিয়ে Tensor-এর একটি অংশ নেওয়া যায়।

```python
x = torch.tensor([10, 20, 30, 40, 50])

print(x[1:3])
```

Output হবে `tensor([20, 30])`। এখানে শুরু index `1` অন্তর্ভুক্ত, কিন্তু শেষ index `3` অন্তর্ভুক্ত নয়। তাই index `1` ও `2`-এর element পাওয়া যায়।

## 8. 2D Tensor Indexing

2D Tensor থেকে element নিতে row ও column index ব্যবহার করা যায়।

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])

print(x[0, 0])
print(x[1, 2])
print(x[0])
```

- `x[0, 0]` প্রথম row-এর প্রথম element, অর্থাৎ `10`।
- `x[1, 2]` দ্বিতীয় row-এর তৃতীয় element, অর্থাৎ `60`।
- `x[0]` প্রথম row-এর সব element দেয়।

Row ও column—দুটোর index-ই `0` থেকে শুরু হয়।

## 9. Tensor-এর Mathematical Operations

একই shape-এর Tensor-এর ওপর `+`, `-` এবং `*` ব্যবহার করে element-wise operation করা যায়।

```python
a = torch.tensor([1, 2])
b = torch.tensor([2, 4])

print(a + b)
print(a - b)
print(a * b)
```

এখানে প্রতিটি operation একই অবস্থানের element-গুলোর মধ্যে হয়।

- Addition: `[1, 2] + [2, 4]` → `[3, 6]`
- Subtraction: `[1, 2] - [2, 4]` → `[-1, -2]`
- Multiplication: `[1, 2] * [2, 4]` → `[2, 8]`

এখানে `*` element-wise multiplication করে; এটি matrix multiplication নয়।

## 10. NumPy Array থেকে Tensor

NumPy array-কে `torch.from_numpy()` দিয়ে PyTorch Tensor-এ রূপান্তর করা যায়।

```python
import numpy as np
import torch

array = np.array([1, 2, 3, 4, 5])
tensor = torch.from_numpy(array)

print(tensor)
```

এভাবে NumPy ও PyTorch-এর data নিয়ে একসঙ্গে কাজ করা যায়। মনে রাখতে হবে, `torch.from_numpy()` দিয়ে তৈরি Tensor এবং মূল NumPy array সাধারণত একই memory share করে। তাই একটির data পরিবর্তন করলে অন্যটিতেও পরিবর্তন দেখা যেতে পারে।

## 11. অন্য Tensor-এর মতো Tensor তৈরি করা

কোনো Tensor-এর shape অনুসরণ করে নতুন Tensor তৈরি করতে `torch.ones_like()` ও `torch.rand_like()` ব্যবহার করা যায়।

### `torch.ones_like()`

```python
x = torch.ones_like(tensor)
print(x)
```

এটি `tensor`-এর মতো একই shape-এর Tensor তৈরি করে, যার সব element `1`।

### `torch.rand_like()`

```python
x_rand = torch.rand_like(tensor, dtype=torch.float)
print(x_rand)
```

এটি একই shape-এর Tensor তৈরি করে, যার element-গুলো random floating-point value। Random value প্রতিবার আলাদা হতে পারে।

## 12. Random, Ones ও Zeros Tensor

নির্দিষ্ট shape দিয়ে random value, সব `1` অথবা সব `0`-এর Tensor তৈরি করা যায়।

```python
shape = (2, 3)

rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)
```

`shape = (2, 3)` মানে দুটি row ও তিনটি column।

- `torch.rand(shape)` সাধারণত `0` থেকে `1`-এর মধ্যে random floating-point value তৈরি করে।
- `torch.ones(shape)` সব element `1` করে।
- `torch.zeros(shape)` সব element `0` করে।

## 13. Tensor Join করা — `torch.cat()`

`torch.cat()` দিয়ে একই dimension-এ compatible Tensor-গুলোকে একসঙ্গে জোড়া লাগানো যায়।

```python
a = torch.cat([ones_tensor, ones_tensor], dim=1)
print(a)
```

এখানে `dim=1` মানে column dimension বরাবর Tensor-গুলো জোড়া লাগানো। দুটি `(2, 3)` Tensor এভাবে জোড়া লাগালে ফলাফলের shape হয় `(2, 6)`। `torch.cat()` ব্যবহার করার সময় যে dimension-এ join করা হচ্ছে, সেটি ছাড়া অন্য dimension-গুলোর size মিলতে হবে।

## 14. Tensor-এর প্রতিটি Element-এর সঙ্গে Number যোগ করা

একটি scalar number Tensor-এর প্রতিটি element-এর সঙ্গে যোগ করা যায়।

```python
x = torch.tensor([
    [1, 2],
    [3, 4]
])

print(x + 5)
```

এখানে `5` প্রতিটি element-এর সঙ্গে যোগ হয়, ফলে ফলাফল হবে:

```text
tensor([[6, 7],
        [8, 9]])
```

## 15. In-place Operation — `add_()`

PyTorch-এ underscore (`_`) দিয়ে শেষ হওয়া অনেক operation মূল Tensor-টিকেই পরিবর্তন করে। যেমন `add_()`।

```python
x = torch.tensor([
    [1, 2],
    [3, 4]
])

x.add_(5)
print(x)
```

`x.add_(5)` `x`-এর প্রতিটি element-এর সঙ্গে `5` যোগ করে এবং একই Tensor পরিবর্তন করে। সাধারণ `x + 5` expression-এর সঙ্গে এর পার্থক্য হলো, `add_()` মূল Tensor-এ পরিবর্তন আনে। Autograd বা gradient calculation-এর সময় in-place operation ব্যবহারে সতর্ক থাকা প্রয়োজন।

## 16. Tensor-এর সব Element-এর Sum

`sum()` দিয়ে Tensor-এর সব element যোগ করা যায়।

```python
x = torch.tensor([
    [6, 7],
    [8, 9]
])

total = x.sum()
print(total)
```

ফলাফল হবে `tensor(30)`, কারণ `6 + 7 + 8 + 9 = 30`। এখানে `sum()` একটি scalar Tensor ফেরত দেয়।
