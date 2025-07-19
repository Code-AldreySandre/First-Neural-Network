# Neural Network from Scratch – MNIST Classification with PyTorch

This is my first neural network project, built from scratch using **PyTorch** to classify handwritten digits from the **MNIST dataset**.

The project aims to provide a hands-on understanding of how a simple feedforward neural network works — from data preprocessing to model training and evaluation.

---

## 📚 Project Structure

- **Data loading:** Uses `torchvision.datasets.MNIST` to download and load training images.
- **Data transformation:** Images are transformed into tensors using `ToTensor()`.
- **Data visualization:** Displays sample images from the dataset with `matplotlib`.
- **Model definition:** A basic fully-connected neural network built with `torch.nn.Module`.
- **Training loop:** Custom training function using `SGD` optimizer with momentum.
- **Evaluation:** Measures accuracy of the model on training data (can be extended to include validation).

---

## 🔧 Technologies Used

- Python 3.x
- [PyTorch](https://pytorch.org/)
- torchvision
- matplotlib
- NumPy

---

## 🧠 Model Architecture

```text
Input layer:      784 neurons (28x28 pixels flattened)
Hidden layer:     128 neurons (Linear + ReLU)
Output layer:     10 neurons (Linear + LogSoftmax)
