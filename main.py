from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
import os
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
from model import ResNet
from utils import *
from tqdm import tqdm
import mlflow



def train(lr, weight_decay, epochs, batch_size, mlflow_experiment_name, mlflow_run_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    batch_size = batch_size
    data_root = "diffusion/pizza-not-pizza"

    dataset = DatasetGenerator(data_root)
    train_set_length = int(0.7 * len(dataset))
    test_set_length = len(dataset) - train_set_length
    #split the dataset 
    train_set, test_set = random_split(dataset, [train_set_length, test_set_length])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)
    class_num = 2
    ce_loss=nn.CrossEntropyLoss()

    #load model
    model = ResNet(class_num=class_num)
    model = model.to(device)

    #define optimizer
    opt_model = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    epoch = epochs
    model.train()
    torch.set_grad_enabled(True)
    
    mlflow.set_experiment(mlflow_experiment_name)
    with mlflow.start_run(run_name=mlflow_run_name):
        for epo in range(1,epoch+1):
            correct_model = 0
            print("Epoch {}/{} \n".format(epo, epoch))
            with tqdm(total=len(train_loader), desc="Train") as pb:

                for batch_num, (img, img_label) in enumerate(train_loader):
                    opt_model.zero_grad()
                    img = img.to(device) 
                    img_label = img_label.to(device)       
                    outputs = model(img)
                    correct_model += (torch.argmax(outputs, dim=1)==img_label).sum().item()
                    loss = ce_loss(outputs, img_label)
                    loss.backward()
                    #loss_model.append(loss)
                    opt_model.step()
                    pb.update(1)
        train_accuracy = correct_model/len(train_set)
        val_accuracy = evaluate(model, test_loader, device)
        mlflow.log_param("learning rate", lr)
        mlflow.log_param("weight decay", weight_decay)
        mlflow.log_metric("valdiation accuracy", val_accuracy)
        mlflow.log_metric("training accuracy", train_accuracy)
        #mlflow.log_dict(dict(model.state_dict()), "model_state_dict")
        mlflow.pytorch.log_model(model, "model")
        
def evaluate(model, test_loader, device):
    correct_model, num_predictions = 0, 0
    model.eval()
    for batch_num, (img, img_label) in enumerate(test_loader):
        img = img.to(device)
        img_label = img_label.to(device)
        predictions = model(img)
        correct_model += (torch.argmax(predictions, dim=1)==img_label).sum().item()
        num_predictions += predictions.shape[0]

    accuracy_model = correct_model / num_predictions
    return accuracy_model

def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--mlflow_experiment_name', type=str, default="Default", help='mlflow experiment name')
    parser.add_argument('--mlflow_run_name', type=str, default="mlflow run", help='mlflow run name')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    train(**vars(opt))