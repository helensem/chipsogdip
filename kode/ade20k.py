"""Prepare ADE20K dataset"""
import os
import shutil
#import argparse
#import zipfile
#from gluoncv.utils import download, makedirs
from gluoncv.data import ADE20KSegmentation
import numpy as np 
import sys 
import cv2
sys.path.append("/Users/HeleneSemb/Documents/chipsogdip/kode")
#from dataset import *

#_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/ade')

def find_contours(sub_mask):
    """Generates a tuple of points where a contour was found from a binary mask 

    Args:
        sub_mask (numpy array): binary mask 
    """
    assert sub_mask is not None, "file could not be read, check with os.path.exists()"
    #imgray = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(sub_mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours)!= 0, print(contours)
    return contours[0]

def load_sky_yolo(root, subset,destination): 
    """
    Loads the images from a dataset with a dictionary of the annotations, to be loaded in detectron 
    """ 
    assert subset in ['train', 'val']
    if subset == 'train': 
        dir = 'training'
    else: 
        dir = 'validation'
    #dataset = ADE20KSegmentation(root,split=subset)
    source = os.path.join(root,"ADEChallengeData2016", "images", dir)
    mask_dir = os.path.join(root, "ADEChallengeData2016", "annotations", dir )
    mask_ids = next(os.walk(mask_dir))[2]
    #idx = 0
    for id in mask_ids:
        mask_path = os.path.join(mask_dir, id)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #print(mask)
        #print(mask.dtype)
        mask = np.where(mask==3, 255,0)
        if 255 not in mask: 
            continue
        mask = mask.astype('uint8')
        #print(mask)
        contour = find_contours(mask) 
        contours = contour.flatten().tolist()
        print(contours)
        string = "0 "
        height, width = mask.shape
        print(mask.shape)
        for i in range(1,len(contours),2): 
            string += str(round(contours[i-1]/width,6)) #x coordinate
            string += " "
            string += str(round(contours[i]/height, 6)) # y coordinate
            string += " "
       
        print(string)
        image_dest = os.path.join(destination, "images", subset, id)
        print("destination: ", image_dest)
        image_source = os.path.join(source, id)
        print("source: ", image_source)
        #shutil.copy(image_source, image_dest)
        txt_id = os.path.splitext(id)[0]+'.txt' 
        txt_path = os.path.join(destination, "labels", subset, txt_id)
        print(txt_path)
        #with open(txt_path, "w") as f: 
         # f.write(string)

        #image_dir = os.path.join(dataset_dir, image_id)
        #(_, _, file_names) = next(os.walk(image_dir))
        #file_name = file_names[0]

if __name__ == '__main__':
    root = r"/cluster/home/helensem/Master/ade"
    subset = 'train'
    destination = r"/cluster/home/helensem/Master/"


    #train_dataset = ADE20KSegmentation(root,split='train')
    #val_dataset = ADE20KSegmentation(root, split='val')
    # print('Training images:', len(train_dataset))
    # print('Validation images:', len(val_dataset))
    # print("Classes:", train_dataset.classes)
    # img, mask = val_dataset[0]
    # #mask = mask.asnumpy()
    # print(mask.shape)
    # mask = mask.asnumpy()
    # mask = np.where(mask==3, 1,0)
    # print(mask)