# PROGRAMMER: Placide E.
# DATE CREATED: 24 July 2020                                  
# REVISED DATE: 24 July 2020
# PURPOSE: Predict flower name from an image with predict.py along with the probability of that name. 
#         That is, you'll pass in a single image /path/to/image and return the flower name and class probability. 
#     Command Line Arguments:
#     1. Path to the image to predict as /path/to/image
#     2. Path to the saved checkpoint as checkpoint_path
#     3. Save data_dir as --save_dir for saving checkpoint
#     4. Return the top K most likely classes as --top_k, with defautl 5
#     5. This is the file path to the categories as --category_names, with default "cat_to_name.json"
#     6. Use GPU to train as --gpu with default the machine cpu
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
      1. Path to the image to predict as /path/to/image
      2. Path to the saved checkpoint as checkpoint_path
      3. Save data_dir as --save_dir for saving checkpoint
      4. Return the top K most likely classes as --top_k, with defautl 5
      5. This is the file path to the categories as --category_names, with default "cat_to_name.json"
      6. Use GPU to train as --gpu with default the machine cpu
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    my_parser = argparse.ArgumentParser()
    
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    my_parser.add_argument('--image_path', type = str, help = 'this is the path to the image to predict the classification for.')
    my_parser.add_argument('--checkpoint_path', type = str, help = 'this is the path to the checkpoint file to be loaded.')
    my_parser.add_argument('--save_dir', type = str, default = '/', help = 'this is the directory to save checkpoints')
    my_parser.add_argument ('--top_k', type = int, default = 5, help = 'return the top K most likely classes')
    my_parser.add_argument ('--category_names', type = str, default = 'cat_to_name.json', help = 'this is the file path to the categories')
    my_parser.add_argument ('--gpu', action = "store_true", default = False, help = "Option to use GPU")
   
    
    # Return the parser with all command line arguments to train the model 
    return my_parser.parse_args()
