#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images.py
#
# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: Placide E.
# DATE CREATED: 6 June 2020                                 
# REVISED DATE: 6 June 2020
# PURPOSE: Classifies pet images using a pretrained CNN model, compares these
#          classifications to the true identity of the pets in the images, and
#          summarizes how well the CNN performed on the image classification task. 
#          Note that the true identity of the pet (or object) in the image is 
#          indicated by the filename of the image. Therefore, your program must
#          first extract the pet image label from the filename before
#          classifying the images using the pretrained CNN model. With this 
#          program we will be comparing the performance of 3 different CNN model
#          architectures to determine which provides the 'best' classification.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
from time import time, sleep

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results

# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    
    # TODO 1: Define get_input_args function within the file get_input_args.py
    # This function retrieves 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg  
    def check_command_line_arguments(in_args):
        # Accesses values of Argument 1 by printing it
        print("The path to the folder with dog pictures:", in_args.dir)
        print("The CNN Architectural Model used:", in_args.arch)
        print("The dogs file name:", in_args.dogfile)
    
    check_command_line_arguments(in_arg)
    
    # TODO 2: Define get_pet_labels function within the file get_pet_labels.py
    # Once the get_pet_labels function has been defined replace 'None' 
    # in the function call with in_arg.dir  Once you have done the replacements
    # your function call should look like this: 
    #             get_pet_labels(in_arg.dir)
    # This function creates the results dictionary that contains the results, 
    # this dictionary is returned from the function call as the variable results
    results = get_pet_labels(in_arg.dir)

    # Function that checks Pet Images in the results Dictionary using results
    def check_creating_pet_image_labels(dic_results):
        print("\nThe dictionary lenght or total images are: {}\nBelow is the list of the first 10 (key, value):".format(len(dic_results)))
        
        _count = 1
        for key, value in dic_results.items():
            print("{:2d} {}: {}".format(_count, key, value))
            
            if _count == 10:
                break
            _count += 1
    
    
    check_creating_pet_image_labels(results)


    # TODO 3: Define classify_images function within the file classiy_images.py
    # Once the classify_images function has been defined replace first 'None' 
    # in the function call with in_arg.dir and replace the last 'None' in the
    # function call with in_arg.arch  Once you have done the replacements your
    # function call should look like this: 
    #             classify_images(in_arg.dir, results, in_arg.arch)
    # Creates Classifier Labels with classifier function, Compares Labels, 
    # and adds these results to the results dictionary - results
    classify_images(in_arg.dir, results, in_arg.arch)

    # Function that checks Results Dictionary using results   
    def check_classifying_images(results_dic):
        print("\nPrinting out all matches (1) between classifier and image label:")
        _count = 1
        for key, value in results_dic.items():
            if value[2] == 1:
                print("{:2d} {}: image label = {} and classifier = {}".format(_count, key, value[0], value[1]))
                _count += 1
        
        print("\nPrinting out all non-matches (0) between classifier and image label:")
        _count = 1
        for key, value in results_dic.items():
            if value[2] == 0:
                print("{:2d} {}: image label = {} and classifier = {}".format(_count, key, value[0], value[1]))
                _count += 1
    
    
    check_classifying_images(results)    

    
    # TODO 4: Define adjust_results4_isadog function within the file adjust_results4_isadog.py
    # Once the adjust_results4_isadog function has been defined replace 'None' 
    # in the function call with in_arg.dogfile  Once you have done the 
    # replacements your function call should look like this: 
    #          adjust_results4_isadog(results, in_arg.dogfile)
    # Adjusts the results dictionary to determine if classifier correctly 
    # classified images as 'a dog' or 'not a dog'. This demonstrates if 
    # model can correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(results, in_arg.dogfile)

    # Function that checks Results Dictionary for is-a-dog adjustment using results
    def check_classifying_labels_as_dogs(results_dic):
        print("\nPrinting all matches (1) between classifier and image label:")
        
        for i, (key, value) in enumerate(results_dic.items()):
            if (value[2] == 1):
                if value[3] == 1 or value[4] == 1:
                    print("IS A DOG - ", i + 1, key, value)
                else:
                    print("IS NOT A DOG - ", i + 1, key, value)
            
        
        print("\nPrinting all non-matches (0) between classifier and image label:")
        
        for i, (key, value) in enumerate(results_dic.items()):
            if (value[2] == 0):
                if value[3] == 1 or value[4] == 1:
                    print("IS A DOG - ", i + 1, key, value)
                else:
                    print("IS NOT A DOG - ", i + 1, key, value)
    
    
    check_classifying_labels_as_dogs(results)


    # TODO 5: Define calculates_results_stats function within the file calculates_results_stats.py
    # This function creates the results statistics dictionary that contains a
    # summary of the results statistics (this includes counts & percentages). This
    # dictionary is returned from the function call as the variable results_stats    
    # Calculates results of run and puts statistics in the Results Statistics
    # Dictionary - called results_stats
    results_stats = calculates_results_stats(results)

    # Function that checks Results Statistics Dictionary using results_stats
    def check_calculating_results(m_results, stats_results):
        n_images = len(m_results)
        n_isdogs_img = 0
        n_isnot_dogs_img = 0
        n_correct_notdog = 0
        n_correct_breed = 0
        n_correct_dog = 0
        
        for key, value in m_results.items():
            if value[3] == 1:
                n_isdogs_img += 1
                if value[4] == 1:
                    n_correct_dog += 1
            else:
                n_isnot_dogs_img += 1
            
            if value[4] == 0:
                n_correct_notdog +=1
            
            if value[3] == 1 and value[2] == 1:
                n_correct_breed += 1
        
        #Comparing stats values
        print("\nComparing Counts - ")
        print("Number of images in results = {} and same as {} in results_stats.".format(n_images, stats_results['n_images']))
        print("Number of dogs images in results = {} and same as {} in results_stats.".format(n_isdogs_img, stats_results['n_dogs_img']))
        print("Number of not dogs images in results = {} and same as {} in results_stats.".format(n_isnot_dogs_img, stats_results['n_notdogs_img']))
        
        #comparing percentages
        if n_isnot_dogs_img > 0:
            pct_correct_notdog = (n_correct_notdog / n_isnot_dogs_img)*100.0
        else:
            pct_correct_notdog = 0.0
        
        print("\nComparing Percentages - ")
        print("% correct dog images in results = {} and same as {} in results_stats.".format((n_correct_dog/n_isdogs_img)*100.0, stats_results['pct_correct_dogs']))
        print("% correct not a dogs images in results = {} and same as {} in results_stats.".format(pct_correct_notdog, stats_results['pct_correct_notdogs']))
        print("% correct breed of dogs images in results = {} and same as {} in results_stats.".format((n_correct_breed / n_isdogs_img)*100.0, stats_results['pct_correct_breed']))
    
    
    check_calculating_results(results, results_stats)


    # TODO 6: Define print_results function within the file print_results.py
    # Once the print_results function has been defined replace 'None' 
    # in the function call with in_arg.arch  Once you have done the 
    # replacements your function call should look like this: 
    #      print_results(results, results_stats, in_arg.arch, True, True)
    # Prints summary results, incorrect classifications of dogs (if requested)
    # and incorrectly classified breeds (if requested)
    print_results(results, results_stats, in_arg.arch, True, True)
    
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
