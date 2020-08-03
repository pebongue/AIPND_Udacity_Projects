# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: Placide E.
# DATE CREATED: 24 July 2020                                 
# REVISED DATE: 24 July 2020
# PURPOSE: Predict the trained model from a saved checkpoint by sending a single image
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --image_path <image to use for prediction> --checkpoint_path <where the trained model checkpoint is saved>
#             --gpu <whether to use gpu or not>
#   Example call:
#    python predict.py --image_path pet_images/my_image.jpg --gpu True --checkpoint_path checkpoint.pth
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

#import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

from torch import nn
from torch import optim
import torch.nn.functional as F

#from collections import OrderedDict

# Imports functions created for this program
from get_predict_input_args import get_input_args


# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    
    # TODO 1: Train the network on default arguments
    in_arg = get_input_args()
    
    
    # Function that set the torch device to be used
    def set_torch_device(gpu_args):
        if gpu_args and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("The gpu is not available. Sytem will use the cpu")
            device = torch.device("cpu")
                
        #model = model.to(device)
        return device
    
    device = set_torch_device(in_arg.gpu)
    
    #Load categories from the JSON file
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # TODO: Write a function that loads a checkpoint and rebuilds the model
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)

        model = models.__dict__[checkpoint['model_name']](pretrained=True)
        model = model.to(device)
        
        model.name = checkpoint['model_name']
        model.epochs = checkpoint['epochs']
        model.optimizer = checkpoint['optimizer_state']
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

        return model
    
    model = load_checkpoint(in_arg.checkpoint_path)
    
    # Function that help process an image before prediction
    #Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    def process_image(image_path):
        # TODO: Process a PIL image for use in a PyTorch model
        normalised_mean = [0.485, 0.456, 0.406]
        normalised_std = [0.229, 0.224, 0.225]

        p_image = PIL.Image.open(image_path).convert("RGB")

        # Process image the same as trained images
        img_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(normalised_mean, normalised_std)])
        p_image = img_transforms(p_image)

        #convert the image to numpy to keep color channel as float
        np_image = np.array(p_image)

        return np_image
    
    # Function to predict an image
    def predict(image_path, a_model, topk):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

        a_model.to(device)
        a_model.eval()

        tensor_image = torch.from_numpy(process_image(image_path))

        inputs = tensor_image

        if in_arg.gpu and torch.cuda.is_available():
            inputs = tensor_image.float().cuda()         

        inputs = inputs.unsqueeze(dim = 0)
        log_ps = model.forward(inputs)
        ps = torch.exp(log_ps)    

        top_ps, top_classes = ps.topk(topk, dim = 1)

        class_to_idx_inverted = {model.class_to_idx[c]: c for c in model.class_to_idx}
        top_mapped_classes = list()

        for label in top_classes.cpu().detach().numpy()[0]:
            top_mapped_classes.append(class_to_idx_inverted[label])

        return top_ps.cpu().detach().numpy()[0], top_mapped_classes
    
    
    # TODO: Display an image along with the top 5 classes
    def print_img_name_top_K(image_path, a_model, top_K):

        # Set up the image caption
        title_name = image_path.split('/')[2]
        flower_name = cat_to_name[title_name]

        #Make prediction on the input image and display the top K probabilities
        top_ps, top_classes = predict(image_path, a_model, top_K)
        
        print(f"Flower name: {flower_name}")
        print(f"Top {top_K} probabilities: ", top_ps)
        print("Top classes : ", top_classes)
    
    print_img_name_top_K(in_arg.image_path, model, in_arg.top_k)
    
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
