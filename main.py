# from yolo.yolo_models import YOLO
from preprocess.dataset_controller import download_dataset, create_dataset, process_dataset
from preprocess.utilities.plot import plot_with_bbox
from dotenv import load_dotenv, set_key

import os 


# load_dotenv()

DATASET_DOWNLOADED = False # bool(os.getenv('DATASET_DOWNLOADED'))
DATASET_PATH = "dataset"

# Preprocess data
path = ''
if not DATASET_DOWNLOADED:
    path = download_dataset(light = True) # Donwload Light dataset from github
    set_key('.env', 'DATASET_DOWNLOADED', 'True')

# Get Unprocessed dataset
dataset = create_dataset(path)
processed_dataset = process_dataset(dataset)

print("Unprocessed Dataset {}".format(dataset))
print("Process Dataset: {}".format(processed_dataset))

element = next(iter(processed_dataset))

# Extract image and data (For Debugging)
image = element[0].numpy()
data = element[1].numpy()


iterator = iter(processed_dataset)

element_id = 13 # Id to look for in dataset

# Get to the element id 
for i in range(element_id):
    element = next(iterator)

    # Extract info
    image = element[0].numpy()
    feature_vectors = element[1].numpy()
    new_size = element[2].numpy()

    # Plot the image with bounding box 
    plot_with_bbox(image, feature_vectors, size_correction = new_size)

