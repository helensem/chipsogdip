# For testing detectron2 mask r-cnn with COCO 

import detectron2 

#Setup logger
from detectron2.utils.logger import setup_logger
setup_logger()

#Some common libraries 
import numpy as np 
import os,json,cv2,random 


#Detectron2 utilities 

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


if __name__ == "__main__": 
    im = cv2.imread(r"/Users/HeleneSemb/Documents/Master/Kode /images/input.png")
    cv2.imshow("image",im)
    cv2.waitKey(0)
  
    # closing all open windows
    cv2.destroyAllWindows()