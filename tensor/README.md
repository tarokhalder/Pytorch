# PyTorch Tensor

## PyTorch ও NumPy Import করা

PyTorch-এর কাজের জন্য `torch` এবং NumPy array নিয়ে কাজ করার জন্য `numpy` import করা হয়। PyTorch-এর কোন version ব্যবহার হচ্ছে, তা `torch.__version__` দিয়ে দেখা যায়।

```python
import torch
import numpy as np

print(torch.__version__)
```

## 1. Tensor কী এবং তৈরি করা

PyTorch-এর **Tensor** হলো সংখ্যাসহ বিভিন্ন data রাখার একটি structure। এটি NumPy array-এর মতো, তবে PyTorch Tensor machine learning ও deep learning-এর হিসাবেও ব্যবহার করা হয়।

```python
x = torch.tensor([1, 2, 3, 4, 5])
print(x)
```

`torch.tensor()` দিয়ে Python list থেকে Tensor তৈরি করা যায়। এখানে `x` একটি **1D Tensor**, কারণ এতে একটি dimension আছে।

## 2. Tensor-এর Dimension, Shape ও Data Type

Tensor-এর গঠন বুঝতে `ndim`, `shape`, `size()` এবং `dtype` কাজে লাগে।

```python
print(x.ndim)
print(x.shape)
print(x.size())
print(x.dtype)
```

- `ndim`: Tensor-এ কয়টি dimension আছে তা জানায়।
- `shape`: প্রতিটি dimension-এ কতটি element আছে তা জানায়।
- `size()`: Tensor-এর shape-এর তথ্য দেয়।
- `dtype`: element-গুলোর data type জানায়।

উপরের `x`-এর জন্য `ndim` হবে `1`, আর `shape` হবে `torch.Size([5])`। অর্থাৎ এক dimension-এ পাঁচটি element রয়েছে।

## 3. 2D Tensor তৈরি করা

2D Tensor-কে row ও column-সহ একটি টেবিলের মতো ভাবা যায়।

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

এখানে দুটি row এবং প্রতিটি row-তে পাঁচটি element আছে। তাই `y.ndim` হলো `2` এবং `y.shape` হলো `torch.Size([2, 5])`। Shape-এর প্রথম সংখ্যা row এবং দ্বিতীয় সংখ্যা column বোঝায়।

## 4. Tensor-এর Data Type

Tensor-এর `dtype` বলে দেয় data integer, floating-point number, নাকি অন্য কোনো type-এর।

```python
x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([1.0, 2.0, 3.0, 4.0])

print(x)
print(x.dtype)
print(y)
print(y.dtype)
```

সাধারণত integer value দিয়ে তৈরি Tensor-এর type `torch.int64` এবং decimal value দিয়ে তৈরি Tensor-এর type `torch.float32` হয়—যদি অন্য `dtype` নির্দিষ্ট না করা থাকে।

## 5. Tensor Indexing ও Slicing

**Indexing** দিয়ে নির্দিষ্ট element বের করা হয়। Python-এর মতো PyTorch-এও index `0` থেকে শুরু হয়; negative index দিয়ে শেষ দিক থেকেও element নেওয়া যায়।

```python
x = torch.tensor([10, 20, 30, 40, 50])

print(x[-5])
print(x[1:3])
```

`x[-5]` প্রথম element `10` দেয়। `x[1:3]` index `1` থেকে শুরু করে index `3`-এর আগ পর্যন্ত element দেয়, তাই ফলাফল হবে `tensor([20, 30])`। Slicing-এ শেষ index অন্তর্ভুক্ত হয় না।

## 6. 2D Tensor থেকে Element নেওয়া

2D Tensor-এ row ও column-এর index দিয়ে element access করা যায়।

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
- `x[0]` প্রথম row-এর সম্পূর্ণ data দেয়।

Row ও column—দুটির index-ই `0` থেকে শুরু হয়।

## 7. Tensor-এর Mathematical Operations

একই shape-এর Tensor-এর element-গুলোর ওপর `+`, `-` ও `*` দিয়ে element-wise operation করা যায়।

```python
a = torch.tensor([1, 2])
b = torch.tensor([2, 4])

print(a + b)
print(a - b)
print(a * b)
```

ফলাফল হবে যথাক্রমে `tensor([3, 6])`, `tensor([-1, -2])` এবং `tensor([2, 8])`। এখানে `*` element-wise multiplication করে; এটি matrix multiplication নয়।

## 8. NumPy Array থেকে Tensor

NumPy array-কে `torch.from_numpy()` দিয়ে PyTorch Tensor-এ রূপান্তর করা যায়।

```python
array = np.array([1, 2, 3, 4, 5])
tensor = torch.from_numpy(array)
print(tensor)
```

এভাবে NumPy ও PyTorch-এর data একসঙ্গে ব্যবহার করা যায়। `torch.from_numpy()` দিয়ে তৈরি Tensor এবং মূল NumPy array সাধারণত একই memory share করে; তাই একটির data বদলালে অন্যটিতেও পরিবর্তন দেখা যেতে পারে।

## 9. অন্য Tensor-এর মতো Tensor তৈরি করা

কোনো Tensor-এর shape অনুসরণ করে নতুন Tensor তৈরি করতে `torch.ones_like()` ও `torch.rand_like()` ব্যবহার করা যায়।

```python
x = torch.ones_like(tensor)
print(x)

x_rand = torch.rand_like(tensor, dtype=torch.float)
print(x_rand)
```

- `torch.ones_like(tensor)`: একই shape-এর Tensor তৈরি করে, যার সব element `1`।
- `torch.rand_like(tensor, dtype=torch.float)`: একই shape-এর Tensor তৈরি করে, যার element-গুলো `0` থেকে `1`-এর মধ্যে random floating-point value।

## 10. Random, Ones ও Zeros Tensor

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

- `torch.rand(shape)`: সাধারণত `0` থেকে `1`-এর মধ্যে random floating-point value তৈরি করে।
- `torch.ones(shape)`: সব element `1` করে।
- `torch.zeros(shape)`: সব element `0` করে।

## 11. Tensor জোড়া লাগানো — `torch.cat()`

`torch.cat()` দিয়ে একাধিক Tensor-কে নির্দিষ্ট dimension বরাবর জোড়া লাগানো যায়।

```python
a = torch.cat([ones_tensor, ones_tensor], dim=1)
print(a)
```

এখানে `dim=1` মানে column dimension বরাবর জোড়া লাগানো। দুটি `(2, 3)` Tensor জোড়া লাগালে ফলাফলের shape হবে `(2, 6)`। যে dimension-এ join করা হচ্ছে, সেটি ছাড়া অন্য dimension-গুলোর size একই হতে হবে।

## 12. Tensor-এর প্রতিটি Element-এর সঙ্গে Number যোগ করা

একটি scalar number Tensor-এর প্রতিটি element-এর সঙ্গে যোগ করা যায়।

```python
x = torch.tensor([
    [1, 2],
    [3, 4]
])

print(x + 5)
```

ফলাফল হবে:

```text
tensor([[6, 7],
        [8, 9]])
```

এখানে `5` প্রতিটি element-এর সঙ্গে যোগ হয়েছে।

## 13. In-place Operation — `add_()`

`add_()` Tensor-এর মূল data-তেই পরিবর্তন আনে। PyTorch-এ underscore (`_`) দিয়ে শেষ হওয়া অনেক operation in-place operation।

```python
x = torch.tensor([
    [1, 2],
    [3, 4]
])

x.add_(5)
print(x)
```

এখানে `x.add_(5)` মূল `x`-এর প্রতিটি element-এর সঙ্গে `5` যোগ করে এবং `x`-কেই পরিবর্তন করে। সাধারণ `x + 5` expression-এর তুলনায় এটাই মূল পার্থক্য। Gradient calculation-এর সময় in-place operation ব্যবহারে সতর্ক থাকা প্রয়োজন।

## 14. Tensor-এর সব Element-এর Sum

`sum()` দিয়ে Tensor-এর সব element যোগ করা যায়।

```python
sum = x.sum()
print(sum)
```

এই নোটবুকের আগের cell-এ `x.add_(5)` চালানোর পর `x` হবে `[[6, 7], [8, 9]]`; তাই sum হবে `tensor(30)`। সাধারণভাবে, `tensor.sum()` সব element যোগ করে একটি scalar Tensor ফেরত দেয়।
