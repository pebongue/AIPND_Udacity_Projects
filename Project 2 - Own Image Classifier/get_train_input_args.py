# PROGRAMMER: Placide E.
# DATE CREATED: 24 July 2020                                  
# REVISED DATE: 24 July 2020
# PURPOSE: Create a function that defines command line inputs to train a model. 
#     Command Line Arguments:
#     1. Data Directory as data_dir with default value 'flowers/train'
#     2. CNN Model Architecture as --arch with default value 'vgg16'
#     3. Save data_dir as --save_dir
#     4. Learning rate as --learning_rate with defautl value = 0.001
#     5. Hidden units layers as --hidden_units
#     6. Number of passes or epochs as --epochs with default 5
#     7. Use GPU to train as --gpu with default the machine cpu
#
#
# Imports python modules
import argparse

# TODO 1: Define get_input_args function for training a model
# 
def get_input_args():
    """
    Retrieves and parses command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Data Directory as data_dir with default value 'flowers/train'
      2. CNN Model Architecture as --arch with default value 'vgg16'
      3. Save data_dir as --save_dir
      4. Learning rate as --learning_rate with defautl value = 0.001
      5. Hidden units layers as --hidden_units
      6. Number of passes or epochs as --epochs with default 5
      7. Use GPU to train as --gpu with default the machine cpu
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    my_parser = argparse.ArgumentParser()
    
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    my_parser.add_argument('data_dir', type = str, default = 'flowers', help = 'provide the path to the folder of flower images to train')
    my_parser.add_argument('--arch', type = str, default = 'vgg16', help = 'define the CNN Architecture Model used - vgg or densenet')
    my_parser.add_argument('--save_dir', type = str, default = '/', help = 'this is the directory to save checkpoints')
    my_parser.add_argument ('--learning_rate', type = float, default = 0.001, help = 'Learning rate, default value 0.001')
    my_parser.add_argument ('--hidden_units', type = int, default = 2048, help = 'Hidden units in Classifier. Default value is 2048')
    my_parser.add_argument ('--epochs', type = int, default = 5, help = 'Number of epochs. Deafault value is 5')
    my_parser.add_argument ('--gpu', action = "store_true", default = False, help = "Option to use GPU")
    
    # Return the parser with all command line arguments to train the model 
    return my_parser.parse_args()
