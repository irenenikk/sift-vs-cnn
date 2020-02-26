import torch.optim as optim
import argparse
from models.baseline_cnn import BaselineCNN
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier

def save_model(model, path):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    full_model_path = os.path.join(curr_dir, path)
    torch.save(model.state_dict(), full_model_path)

def train_baseline_net(trainloader, epochs=100, learning_rate=0.001):
    # GPU Stuff
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baselineNet = BaselineCNN()
    baselineNet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(baselineNet.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = baselineNet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    save_model(baselineNet, 'saved_models/baseline_net_model')

def find_hyperparameters(training_images, training_labels):
    net = NeuralNetClassifier(
        BaselineCNN,
        max_epochs=10,
        lr=[0.1, 0.005],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    params = {
        'lr': [0.01, 0.02],
        'max_epochs': [10, 20],
        'module__maxpool_kernel_size': [2, 4, 8],
        'module__out1': [6, 12, 18],
        'module__kernel1': [2, 4, 8],
        'module__in2': [2, 6, 12],
        'module_out2': [8, 16, 24],
        'module__kernel2': [2, 4, 8]
    }
    gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')
    # TODO: X and y sizes are said not to match
    gs.fit(training_images, training_labels)
    print(gs.best_score_)
    return gs.best_params_
