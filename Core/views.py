from django.shortcuts import render
# Create your views here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import sys
import re

plt.style.use('ggplot')

def load_model(model_name):
    mnist = fetch_openml(model_name, as_frame=False, cache=False)
    return mnist

mnist = load_model("mnist_784")

# Preprocessing Data
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# to avoid big weights
X /= 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def visualize_data(X, y,n):
    """Plot the first 100 images in a 10x10 grid."""
    plt.figure(figsize=(15, 15))  # Set figure size to be larger (you can adjust as needed)
    n=int(n**(0.5))
    for i in range(n):  # For 10 rows
        for j in range(n):  # For 10 columns
            index = i * n + j
            plt.subplot(n, n, index + 1)  # 10 rows, 10 columns, current index
            plt.imshow(X[index].reshape(28, 28))  # Display the image
            plt.xticks([])  # Remove x-ticks
            plt.yticks([])  # Remove y-ticks
            plt.title(y[index], fontsize=40)  # Display the label as title with reduced font size

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing (you can modify as needed)
    plt.tight_layout()  # Adjust the spacing between plots for better visualization
    plt.savefig('./Core/static/images/output_plot.png')
    #plt.show()  # Display the entire grid


# Build Neural Network with PyTorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim/8)
output_dim = len(np.unique(mnist.target))
# A Neural network in PyTorch's framework.
class ClassifierModule(nn.Module):
    def __init__(
        self,
        input_dim=mnist_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.5,
        ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X


def About(request):
    return render(request,'about.html')

def Dataset(request):
    size = mnist.data.shape[0]
    val=16
    visualize_data(X, y, val)
    return render(request,'dataset.html',{'size':size})

def Model(request):
    return render(request,'model.html')

def Train(request):
    # skorch allows to use PyTorch's networks in the SciKit-Learn setting:
    print(1)
    from skorch import NeuralNetClassifier
    print(2)
    torch.manual_seed(0)
    print(3)
    net = NeuralNetClassifier(
        ClassifierModule,
        max_epochs=10,
        lr=0.1,
        device=device,
    )
    print(3)
    original_stdout = sys.stdout
    with open('filename.txt','w') as f:
        sys.stdout=f
        net.fit(X_train, y_train)
        sys.stdout=original_stdout
        print(4)
    filename = 'filename.txt'
    print(5)
    with open(filename, 'r') as file:
        lines = file.readlines()
    print(6)
    data = [re.split(r'\s+', re.sub(r'\x1b\[[0-9;]*m', '', line.strip())) for line in lines[2:]]
    print(data)
    print(7)
    columns = ['epoch', 'train_loss', 'valid_acc', 'valid_loss', 'dur']
    df = pd.DataFrame(data, columns=columns)
    print(8)
    plt.plot(df['epoch'], df['valid_acc'], marker='o')
    print(9)
    plt.title('Epoch vs. Valid Accuracy')
    print(10)
    plt.xlabel('Epoch')
    print(11)
    plt.ylabel('Valid Accuracy')
    print(12)

    # Use Django chat frame
    #plt.savefig('./Core/static/images/m1.png')

    print(14)
    return render(request,'train.html')
    
def Predict(request):
    test_image=Image.open('./Core/static/images/zero.png')
    test_image=test_image.resize((128,128))
    image_np=np.array(test_image)
    threshold = 200
    white_pixels = np.all(image_np >= threshold, axis=-1)
    white_pixel_count = np.sum(white_pixels)
    total_pixels = np.prod(image_np.shape[:2])
    white_percentage = (white_pixel_count / total_pixels) * 100
    if white_percentage > 60:
        invert_image = np.where(image_np >= 200, 30,170)
        test_image =Image.fromarray(invert_image.astype(np.uint8))
    test_image=test_image.convert("L")
    test_image=test_image.resize((28,28))
    test_image=np.array(test_image)
    test_image=test_image.flatten()
    test_image=test_image.astype('float32')      
    normalized_test_image = (test_image / 255.0)
    from skorch import NeuralNetClassifier
    torch.manual_seed(0)
    net = NeuralNetClassifier(
        ClassifierModule,
        max_epochs=10,
        lr=0.1,
        device=device,
    )
    net.fit(X_train, y_train)
    X_test[0]=normalized_test_image
    y_pred=net.predict(X_test)
    return render(request,'predict.html',{"pred":y_pred[0]})