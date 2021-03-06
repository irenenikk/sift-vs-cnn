import sys
sys.path.append('./')

import copy
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from torch.optim import lr_scheduler

from models.baseline_cnn import BaselineCNN
from data_pipeline.imagenet_pretrained import PretrainedImagenet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    full_model_path = os.path.join(curr_dir, path)
    torch.save(model, full_model_path)

def save_checkpoint(model, epoch, optimiser, model_filepath):
    print('Saving model to', model_filepath)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                }, model_filepath)

def train_neural_net(model, model_filepath, trainloader, evalloader, criterion, optimiser, scheduler, epochs=20, resume=True):
    # GPU Stuff
    epoch = 0
    print_interval = 50
    if resume:
        print('Loading checkpoint from', model_filepath)
        checkpoint = torch.load(model_filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        epoch = checkpoint['epoch']
    model.to(device)
    best_accuracy = 0
    best_model = copy.deepcopy(model.state_dict())
    print('Training a neural network with a trainset of size', len(trainloader.dataset))
    for _ in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            if (i+1) % print_interval == 0:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / print_interval))
                running_loss = 0.0
        scheduler.step()
        # evaluate after each epoch
        print('Evaluating model')
        model.eval()
        with torch.no_grad():
            loss, acc = evaluate_model_accuracy(model, evalloader, criterion)
        print('Evaluation loss', loss, 'evaluation accuracy', acc)
        if acc > best_accuracy:
            print('New best accuracy')
            best_model = copy.deepcopy(model.state_dict())
            best_accuracy = acc
            save_checkpoint(model, epoch, optimiser, model_filepath)
        epoch += 1

def evaluate_model_accuracy(model, evalloader, criterion):
    acc = 0
    loss = 0
    for x, y in evalloader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        preds = torch.argmax(outputs, axis=1)
        acc += (y == preds).sum().item()
        loss += criterion(outputs, y).item()
    acc /= len(evalloader.dataset)
    loss /=  len(evalloader)
    return loss, acc

def train_classifier(neural_net, params, model_path, trainloader, evalloader, resume, epochs):
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(params, lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1)
    train_neural_net(neural_net, model_path, trainloader, \
                        evalloader, criterion, optimiser, scheduler, epochs=epochs, resume=resume)

def run_transfer_learning(model_name, checkpoint_path, trainloader, evalloader, last_layer_size, resume, epochs):
    neural_net = PretrainedImagenet.get_resnet_feature_extractor_for_transfer(model_name, last_layer_size)
    neural_net.to(device)
    train_classifier(neural_net, list(neural_net.fc.parameters()) + list(neural_net.fc2.parameters()), checkpoint_path + '_' + model_name, trainloader, evalloader, resume, epochs)

def run_baseline_training(neural_net, checkpoint_path, trainloader, evalloader, resume, epochs):
    neural_net.to(device)
    train_classifier(neural_net, neural_net.parameters(), checkpoint_path, trainloader, evalloader, resume, epochs)

