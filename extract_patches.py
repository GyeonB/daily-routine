import sys

sys.path.append("./WSI_Preprocessing/")

from WSI_Preprocessing.Preprocessing.Extarctingpatches import patch_extraction

import os
import logging
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

WHOLE_SLIDE_PATH = "../whole_slides/1-SCC/"
PATCHES_PATH = "./patches/"

logging.basicConfig(filename="progress.log", filemode='w', level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

count = 1

for WHOLE_SLIDE_FILE in os.listdir(WHOLE_SLIDE_PATH):
    if ".svs" in WHOLE_SLIDE_FILE:
        try:
            logger.info("Currently processing file#: " + str(count) + ": " + WHOLE_SLIDE_FILE)

            patch_extraction(WHOLE_SLIDE_PATH, 
                              WHOLE_SLIDE_FILE, 
                              PATCHES_PATH, 
                              magnification="20x", 
                              patch_extraction_criteria = None,
                              num_of_patches = None, 
                              filtering = None,
                              patch_size = (224, 224),
                              upperlimit = 900, 
                              lowerlimit = 300,
                              red_value = (80,220), 
                              green_value = (80,200), 
                              blue_value = (80, 170),  
                              reconstructedimagepath = None, 
                              Annotation = None, 
                              Annotatedlevel = 0, 
                              Requiredlevel = 0)

            count += 1
        except:
            logger.info("Failed: " + WHOLE_SLIDE_FILE)