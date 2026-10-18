# 🔵 Student Performance Prediction Using PyTorch

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Deep%20Learning-5B9BD5?style=for-the-badge" alt="Deep Learning" />
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab" />
</p>

---

## 📘 Overview

This project uses **PyTorch** to build a Feedforward Neural Network that predicts students' exam scores using numerical student performance factors.

The goal is to practice the complete deep learning workflow, including data preprocessing, feature scaling, neural network construction, training, loss visualization, and regression evaluation.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Neural network construction and training |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Scikit-learn | Data preprocessing and evaluation |
| Matplotlib | Training loss visualization |
| Google Colab | Development environment |

## 📂 Project Structure

```text
student-performance-prediction/
├── student_performance.ipynb
└── README.md
```

## 📊 Dataset

**Dataset:** Student Performance Factors

- **Target variable:** `Exam_Score`
- **Input features:** 6 numerical features
- **Train-test split:** 80% training, 20% testing
- **Feature scaling:** `StandardScaler`

Only numerical columns were selected for this version of the project.

## 🧠 Model Architecture

The neural network was implemented using `torch.nn.Module` and `nn.ModuleList`.

| Component | Configuration |
|---|---|
| Input features | 6 |
| Hidden layers | 2 |
| Neurons per hidden layer | 4, 4 |
| Activation function | ReLU |
| Output neurons | 1 |
| Loss function | `nn.MSELoss()` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training epochs | 10000 |

### Training Workflow

1. Forward propagation
2. Mean Squared Error calculation
3. Gradient reset using `optimizer.zero_grad()`
4. Backpropagation using `loss.backward()`
5. Parameter updates using `optimizer.step()`
6. Training loss visualization using Matplotlib

## 📈 Model Evaluation

The model was evaluated using the held-out test dataset.

| Evaluation Metric | Result |
|---|---:|
| Within-5-marks accuracy | 99.32% |
| Mean Absolute Error (MAE) | 1.28 |
| Root Mean Squared Error (RMSE) | 2.27 |
| R² Score | 0.6340 |

> **Note:** The 99.32% result means that 99.32% of test predictions were within ±5 marks of the actual exam scores. This is a tolerance-based metric, not standard regression accuracy.

## 📉 Training Loss Graph

The training loss was recorded across epochs and plotted with Matplotlib to observe how the model's error changed during training.

## 🎯 Key Learnings

- Preparing numerical data for deep learning
- Applying feature scaling correctly
- Building neural networks with PyTorch
- Using `nn.ModuleList` to manage hidden layers
- Understanding forward propagation and backpropagation
- Training models with the Adam optimizer
- Visualizing training loss
- Evaluating regression models with MAE, RMSE, and R²

## 👨‍💻 Author

**Tarak Chandra Halder**

GitHub: [@tarokhalder](https://github.com/tarokhalder)

---

<p align="center">
  <i>Part of my journey to learn PyTorch and build practical deep learning projects.</i>
</p>
