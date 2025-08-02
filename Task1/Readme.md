# Task 1: From Scratch Neural Network Regression with NumPy

This project implements a fully connected neural network for regression from scratch using only NumPy — **no external deep learning libraries** like TensorFlow or PyTorch are used. It covers forward propagation, backpropagation, and weight updates using stochastic gradient descent (SGD).

---

## 📌 Task Objective

To build and train a neural network (with 1 or more hidden layers) that learns to predict a **noisy cubic function**. All components (layers, activations, loss, optimizer) are implemented manually to build foundational understanding.

---

## 🧠 Architecture

- **Input:** 1 feature (X)
- **Hidden Layer 1:** 64 neurons, ReLU activation
- **Hidden Layer 2:** 64 neurons, ReLU activation
- **Output Layer:** 1 neuron (no activation)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Stochastic Gradient Descent (SGD)

---

## 📁 Files

| File | Description |
|------|-------------|
| `model.py` | Contains all classes: Dense Layer, ReLU, Sigmoid, Loss (MSE), Optimizer (SGD) |
| `train.ipynb` | Training loop, data generation, loss curve plotting, prediction visualization |
| `README.md` | This file |

---

## ⚙️ How to Run

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Task1

2. Install dependencies
Only numpy and matplotlib are required:
pip install numpy matplotlib
3. Run the notebook

bash
Copy
Edit
jupyter notebook train.ipynb
Make sure model.py is in the same directory

Run all cells in order

📈 Output
Loss Curve: Shows model convergence

Prediction Plot: Visual comparison between ground truth and model output

Example:



🔍 Notes
Mini-batch size: 32

Epochs: 500

Learning Rate: 0.01

No external ML libraries used

Gradient calculations implemented by hand

📚 References
https://numpy.org/doc/
[MIT OpenCourseWare: Computational Thinking](https://ocw.mit.edu/courses/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/)


---

### 📂 Optional: Add These Files (Images)

You can also add:
- `loss_curve.png` → Save the `plt.plot(losses)` output using:
  ```python
  plt.savefig("loss_curve.png")

prediction_plot.png → Save the final prediction comparison plot:

python
Copy
Edit
plt.savefig("prediction_plot.png")