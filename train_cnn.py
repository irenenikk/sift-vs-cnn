import torch.optim as optim
import argparse
from models.baseline_cnn import BaselineCNN
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='Obtain SIFT features for training set')
parser.add_argument("-root", "--image-root", type=str, default="data/images_small",
                    help="The path to the image data folder")
parser.add_argument("-train-idx", "--training_index-file", type=str, default="data/Butterfly200_train_release.txt",
                    help="The path to the file with training indices")
parser.add_argument("-s", "--species-file", type=str, default="data/species.txt",
                    help="The path to the file with mappings from index to species name")

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