"""Prepare ADE20K dataset"""
import os
import shutil
#import argparse
#import zipfile
#from gluoncv.utils import download, makedirs
#from gluoncv.data import ADE20KSegmentation
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
    return contours

def load_sky_yolo(root, subset,destination): 
    """
    Loads the images from a dataset with a dictionary of the annotations, to be loaded in detectron 
    """ 
    assert subset in ['train', 'val']
    if subset == 'train': 
        dir = 'training'
    else: 
        dir = 'validation'

    source = os.path.join(root, "images", dir)
    mask_dir = os.path.join(root, "annotations", dir)
    print(mask_dir)
    mask_ids = next(os.walk(mask_dir))[2]

    for id in mask_ids:
        mask_path = os.path.join(mask_dir, id)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #print(mask)
        #print(mask.dtype)
        string = ""
        mask = np.where(mask==3, 255,0)
        if id == "ADE_val_00000370.png":
            print(mask)
        if 255 not in mask: 
            continue
        mask = mask.astype('uint8')
        #print(mask)
        height, width = mask.shape
        contours = find_contours(mask)
        for contour in contours:
            contour_list = contour.flatten().tolist()
            if len(contour_list) < 5:
                continue
            string += "0 "
        #print(mask.shape)
            for i in range(1,len(contour_list),2): 
                string += str(round(contour_list[i-1]/width,6)) #x coordinate
                string += " "
                string += str(round(contour_list[i]/height, 6)) # y coordinate
                string += " "
            string+= "\n"
        image_id = os.path.splitext(id)[0] + '.jpg'
        image_source = os.path.join(source, image_id)
        image_dest = os.path.join(destination, "images", subset, image_id)
        #print("destination: ", image_dest)
        #print("source: ", image_source)
        shutil.copy(image_source, image_dest)
        print(string)
        txt_id = os.path.splitext(id)[0]+'.txt' 
        txt_path = os.path.join(destination, "labels", subset, txt_id)
        print(txt_path)
        with open(txt_path, "w") as f: 
          f.write(string)


if __name__ == '__main__':
    
    destination = r"/cluster/home/helensem/Master/sky_data"
    root = r"/cluster/home/helensem/Master/ade/ADEChallengeData2016"

    for d in ['train', 'val']:
        load_sky_yolo(root, d, destination)
    


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