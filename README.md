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

```

---

## 💻 Sample Code Snippets
### Loading and Visualizing the Data
```
from torchvision import datasets, transforms

transform = transforms.ToTensor()
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

Defining the Neural Network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.linear2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear1(x))
        x = F.log_softmax(self.linear2(x), dim=1)
        return x
```
---
## 📸 Output Example
```
dataiter = iter(trainloader)
images, labels = next(dataiter)
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
```
---
## 📷 Expected output:

![Sample Output](imgs/MnistExamples.png)
