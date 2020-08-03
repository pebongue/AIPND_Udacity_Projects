# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: Placide E.
# DATE CREATED: 24 July 2020                                 
# REVISED DATE: 24 July 2020
# PURPOSE: Train a network on a set of flower images
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py data_dir <directory with images> --arch <model>
#             --save_dir <directory where to save the model>
#   Example call:
#    python train.py data_dir flowers/ --arch vgg19 --save_dir checkpoints/
##

# Imports python modules
from time import time, sleep

# Imports all used modules
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import numpy as np
import torch
import PIL
import json
import os

import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict

# Imports functions created for this program
from get_train_input_args import get_input_args


# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    
    # TODO 1: Train the network on default arguments
    in_arg = get_input_args()

    # Function that checks command line arguments for data_directory using in_arg  
    def print_model_default_folder():
        # Accesses values of Argument 1 by printing it
        print("If no data directory is provided, you can use the default folder: flowers")
    
    print_model_default_folder()
    
    # TODO 2: check data_directory
        
    def set_data_directory_arguments(dir_args):
        if not os.path.isdir(dir_args):
            print("Directory or folder path provided was not found")
            exit(1)
        
        data_dir = dir_args
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        return data_dir, train_dir, valid_dir, test_dir
    
    
    data_dir, train_dir, valid_dir, test_dir = set_data_directory_arguments(in_arg.data_dir)

    # Function that checks for saved directory
    def check_saved_directory(save_dir_args):
        if not os.path.isdir(save_dir_args):
            print(f'Directory {save_dir_args} was not found. Creating using the saved one.')
        
        return os.makedirs(save_dir_args)
    
    #saved_data_dir = check_saved_directory(in_arg.save_dir)


    # TODO 3: Prepare data loader for model
    normalised_means = [0.485, 0.456, 0.406]
    normalised_std = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(30),
                                      transforms.ToTensor(),
                                      transforms.Normalize(normalised_means, normalised_std)])
    
    data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(normalised_means, normalised_std)])
    
    img_trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
    img_validset = datasets.ImageFolder(valid_dir, transform=data_transforms)
    img_testset = datasets.ImageFolder(test_dir, transform=data_transforms)
    
    train_dataloader = torch.utils.data.DataLoader(img_trainset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(img_validset, batch_size=64)
    test_dataloader = torch.utils.data.DataLoader(img_testset, batch_size=64, shuffle=True)

    # Function that checks Results Dictionary using results
    input_size = 0
    
    def building_model(arch_args):
        if not (arch_args.startswith("vgg") or arch_args.startswith("densenet")):
            print("Only supporting VGG or DenseNet trained models.")
            exit(1)
        
        print(f"Using the following pre-trained {arch_args} network.")
        model = models.__dict__[arch_args](pretrained=True)
        model.name = arch_args
        
        if arch_args.startswith("vgg"):
            input_size = model.classifier[0].in_features
        
        if arch_args.startswith("densenet"):
            input_size = 1920
        
        return model, input_size
    
    
    model, input_size = building_model(in_arg.arch)    
    
    #Prevent back propagation
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    # TODO 4: build the classifier
    def build_classifier(hidden_units_args):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units_args)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear (hidden_units_args, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        
    build_classifier(in_arg.hidden_units)
    model.zero_grad()

    # Function that set the torch device to be used
    def set_torch_device(gpu_args):
        if gpu_args and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("The gpu is not available. Sytem will use the cpu")
            device = torch.device("cpu")
            
        return device
    
    device = set_torch_device(in_arg.gpu)
    model = model.to(device)

    # TODO 5: Define loss and optimizer
    #Loss function
    criterion = nn.NLLLoss()
    
    def loss_optimizer(lrn_rate_args):
        if lrn_rate_args > 0 : 
            optimizer = optim.Adam(model.classifier.parameters(), lr=lrn_rate_args)
        else:
            optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
         
        return optimizer
        
        
    optimizer = loss_optimizer(in_arg.learning_rate)

    # Function for the validation process
    def validation(a_model, a_dataloader, a_criterion):
        test_loss = 0
        accuracy = 0
        for images, labels in a_dataloader:

            #images.resize_(images.shape[0], 50176)
            images, labels = images.to(device), labels.to(device)

            output = a_model.forward(images)
            test_loss += a_criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return test_loss, accuracy
    

    # TODO 6: Define function to train the network and display loss, loss validation and accuracy
    def train_print_loss_accuracy():
        epochs = in_arg.epochs
        steps = 0
        running_loss = 0
        print_every = 40

        for e in range(epochs):
            running_loss = 0
            model.train()

            for ii, (inputs, labels) in enumerate(train_dataloader):
                steps += 1

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()

                    with torch.no_grad():
                        valid_loss, accuracy = validation(model, valid_dataloader, criterion)

                    print("Epoch: {}/{} | ".format(e+1, epochs),
                          "Training Loss: {:.4f} | ".format(running_loss/print_every),
                          "Validation Loss: {:.4f} | ".format(valid_loss/len(valid_dataloader)),
                          "Validation Accuracy: {:.4f}".format(accuracy/len(valid_dataloader)))

                    running_loss = 0
                    model.train()
        
    train_print_loss_accuracy()
    
    #Testing and displaying the accuracy of the model using the test_datasets
    def test_model():
        correct_result = 0
        total = 0

        model.eval()

        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                output = model(images)

            _,ps = torch.max(output.data, 1)
            total += labels.size(0)
            correct_result += (ps == labels).sum().item()

        # Test output
        print('The trained network achieved an accuracy level of : %d%%' % (100 * correct_result / total))
        
    #Display the accuracy of the tested model
    test_model()
    
    
    #Saving the model
    def saving_model(checkpoint_dir, u_epochs):
        model.to("cpu")
        model.class_to_idx = img_trainset.class_to_idx
        
        checkpoint = {'input_size': 25088,
                      'output_size': 102,
                      'model_name': model.name,
                      'epochs': u_epochs,
                      'state_dict': model.state_dict(),
                      'classifier': model.classifier,
                      'optimizer_state': optimizer.state_dict(),
                      'class_to_idx': model.class_to_idx,
                      'arch': in_arg.arch}
        
        if checkpoint_dir:
            torch.save(checkpoint, checkpoint_dir + '/checkpoint.pth')
        else:
            torch.save(checkpoint, 'checkpoint.pth')
    
    saving_model(in_arg.save_dir, in_arg.epochs)
    
    # TODO 0: Measure total program runtime by collecting end time
    end_time = time()
    
    # TODO 0: Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
