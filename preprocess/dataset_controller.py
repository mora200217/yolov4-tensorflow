from zipfile import ZipFile
from pathlib import Path
from tensorflow._api.v2 import data
from .utilities.process import process_zip

import os
import tensorflow as tf 
import numpy as np


def download_dataset(light = False) :
    """
        Donwload 
    """
    if light:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        folder_name = "dataset"
        
        path = os.path.join(dir_path, folder_name)
        
        try:
            os.mkdir(path)
        except FileExistsError as e: 
            print("Folder already exists!")
        except Exception:
            print("An Error has occured!")

        # path = os.path.join(dir_path, folder_name)

        path_zip = os.path.join(path, "ds.zip")
        # Download the dataset from github repo
        zip_file = tf.keras.utils.get_file(
        path_zip, 
        "https://github.com/mora200217/bioloid-yolo-light-dataset/archive/refs/heads/master.zip", 
        extract=True)

        print("Downloaded", "- " * 20)

        # unzip the donwloaded dataset 
        with ZipFile(zip_file, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
            zipObj.extractall(path = path)

        return path

    
def create_dataset(path, unzipped_folder_name = "bioloid-yolo-light-dataset-master") -> tf.data.Dataset:
    """
        Create a tf.data.Dataset object from png / txt folder.

        Parameters
        ----------
        path : os.path
                path to the folder with the png and txt files 
        [unzipped_folder_name] : str
                name of the folder that contains the files after unzipping process

        Returns 
        -------
        dataset : tf.data.Dataset 
                Tensorflow dataset with the images and labels 

    """
    # Added the unzipped folder to the path 
    path = os.path.join(path, unzipped_folder_name)
    
    # Create a Dataset 
    # print("Path to search: {}".format(path))

    file_patterns = [
        os.path.join(path, "*objects.png"),
        os.path.join(path, "*.txt"),
    ]

    images_dataset = tf.data.Dataset.list_files(file_patterns[0], shuffle=False)
    text_dataset = tf.data.Dataset.list_files(file_patterns[1],shuffle = False)

    dataset = tf.data.Dataset.zip((images_dataset, text_dataset))

    return dataset
  
def process_dataset(dataset):
    return dataset.map(process_zip)
    

