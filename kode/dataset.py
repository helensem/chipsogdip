import detectron2 

from detectron2.utils.logger import setup_logger
setup_logger("output/logger.log")

import numpy as np 
import os,json,cv2,random 
#import skimage

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil


##### Creating training images for GA ######### 

def ga_train_sets():
    os.makedirs(r"/cluster/home/helensem/Master/data/set1/val", exist_ok=True)
    os.makedirs(r"/cluster/home/helensem/Master/data/set2/val", exist_ok=True)
    os.makedirs(r"/cluster/home/helensem/Master/data/set3/val", exist_ok=True)
    
    image_ids = next(os.walk(r"/cluster/home/helensem/Master/Labeled_pictures/val"))[1]
    set1 = []
    set2 = []
    set3 = []

    for i in range(50): 
        idx = random.randint(0,len(image_ids)-1)
        set1.append(image_ids[idx])
        del image_ids[idx]
        idx = random.randint(0,len(image_ids)-1)
        set2.append(image_ids[idx])
        del image_ids[idx]
        idx = random.randint(0,len(image_ids)-1)
        set3.append(image_ids[idx])
        del image_ids[idx]
    
    print(set1)
    print(set2)
    print(set3)

    for image_id in set1: 
        image_path = os.path.join(r"/cluster/home/helensem/Master/Labeled_pictures/val", image_id)
        #image = next(os.walk(image_path))[2][0]
        #(image_path) = os.path.join(image_path, image)
        destination = os.path.join(r"/cluster/home/helensem/Master/data/set1/val", image_id)
        shutil.copytree(image_path, destination) 
    
    for image_id in set2: 
        image_path = os.path.join(r"/cluster/home/helensem/Master/Labeled_pictures/val", image_id)
        image = next(os.walk(image_path))[2][0]
        #(image_path) = os.path.join(image_path, image)
        destination = os.path.join(r"/cluster/home/helensem/Master/data/set2/val", image_id)
        shutil.copytree(image_path, destination)     

    for image_id in set3: 
        image_path = os.path.join(r"/cluster/home/helensem/Master/Labeled_pictures/val", image_id)
        image = next(os.walk(image_path))[2][0]
        #(image_path) = os.path.join(image_path, image)
        destination = os.path.join(r"/cluster/home/helensem/Master/data/set3/val", image_id)
        shutil.copytree(image_path, destination)  







####### Creating COCO-format from png masks ###########

def find_contours(sub_mask):
    """Generates a tuple of points where a contour was found from a binary mask 

    Args:
        sub_mask (numpy array): binary mask 
    """
    assert sub_mask is not None, "file could not be read, check with os.path.exists()"
    imgray = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours)!= 0, print(contours)
    return contours[0]


def create_image_annotation(file_name, width, height, image_id):
    return {
        "image_id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
    }


def create_annotation_format(contour):
    return {
        "iscrowd": 0,
        "segmentation": [contour.flatten().tolist()],
        "bbox": cv2.boundingRect(contour),
        "bbox_mode": BoxMode.XYWH_ABS,
        "category_id": 0,
    }


def load_damage_dicts(dataset_dir, subset): #? Possibly write this to a JSON-file? 
    """
    Loads the images from a dataset with a dictionary of the annotations, to be loaded in detectron 
    """ 
    dataset_dicts = []

    assert subset in ["train", "val"]
    dataset_dir = os.path.join(dataset_dir, subset)
    image_ids = next(os.walk(dataset_dir))[1]
    #idx = 0
    for image_id in image_ids:

        image_dir = os.path.join(dataset_dir, image_id)
        #print(image_dir)
        (_, _, file_names) = next(os.walk(image_dir))
        file_name = file_names[0]
        
        image_path = os.path.join(image_dir, file_name)
        #print(image_path)
        height, width = cv2.imread(image_path).shape[:2]
        record = create_image_annotation(image_path, width, height, image_id)
        #idx +=1
        mask_dir = os.path.join(image_dir, 'masks')
        objs = []
        for f in next(os.walk(mask_dir))[2][0]:
            if f.endswith('.png') and ('corrosion' or 'grov_merking' in f):
                mask_path = os.path.join(mask_dir, f)
                #print(mask_path)
                mask = cv2.imread(mask_path)
                if mask.shape[0]!=height:
                    print(image_dir)
                if mask is None: 
                    print("Couldn't retrieve mask: ", mask_path)
                    continue
                if not(255 in mask):
                    print("mask is empty: ", mask_path)
                    continue
                #if len(mask.shape) > 2: #* Some issues with certain train images 
                #    mask = mask[:,:,0]
                contour = find_contours(mask)
                if len(contour)<3: # Cant create polygons from too few coordinates
                    print("Contour too small: ", mask_path)
                    continue
                obj = create_annotation_format(contour)
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    #* For writing to JSON-file
    #json_object = json.dumps(dataset_dicts,indent=200)
    #with open(f"/cluster/home/helensem/Master/chipsogdip/damage_{subset}.json", "w") as f:
     #   f.write(json_object)
    return dataset_dicts


def load_mask(mask_dir, im_height):
    mask = []
    for f in next(os.walk(mask_dir))[2]:
        if f.endswith('.png') and ('corrosion' or 'grov_merking' in f):
            mask_path = os.path.join(mask_dir, f)
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            m = m.astype(bool)
            mask.append(m)
    mask = np.stack(mask, axis=-1)
    return mask.astype(bool)


def load_sky_dicts(path, subset): 
    dataset_dicts = []

    assert subset in ["train", "val"]
    dataset_dir = os.path.join(path, subset)
    image_ids = next(os.walk(dataset_dir))[1] #names of all directories in dir
    for image_id in image_ids:
        image_dir = os.path.join(dataset_dir, image_id)
        print(image_dir)
        (_, _, file_names) = next(os.walk(image_dir))

        for f in file_names: 
            image_path = os.path.join(image_dir, f)
            height, width = cv2.imread(image_path).shape[:2]
            record = create_image_annotation(image_path, width, height, image_id)
        

            mask_dir = os.path.join(path, 'masks')
            objs = []
            image = image_id + ".png"
            mask_path = os.path.join(mask_dir, image)
        
            mask = cv2.imread(mask_path)
            contour = find_contours(mask)
            obj = create_annotation_format(contour)
            objs.append(obj)        
            record["annotations"] = objs
            dataset_dicts.append(record)

    #* For loading JSON objects 
    #json_object = json.dumps(dataset_dicts,indent=1631)
    #with open(f"damage_{subset}.json", "w") as f:
    #    f.write(json_object)
    return dataset_dicts


def get_jason_dict(subset="train"):

    if subset == "train":
        with open(r"/cluster/home/helensem/Master/damage_train.json", "r") as f:
            data = json.load(f)
        return data
    if subset == "val":
        with open(r"/cluster/home/helensem/Master/damage_val.json", "r") as f:
            data = json.load(f)            
        return data


####### MAIN ##################

if __name__ == "__main__": 

    #ga_train_sets()

    #print(load_damage_dicts(r"/cluster/home/helensem/Master/data", "train"))
    train_dict = load_damage_dicts(r"/cluster/home/helensem/Master/Labeled_pictures", "train")
    val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/data/Labeled_pictures", "val")


    #Load data to detectron 
    # for d in ["train", "val"]:
    #     DatasetCatalog.register("damage_" + d, lambda d=d: load_damage_dicts(r"/cluster/home/helensem/Master/data", d))
    #     MetadataCatalog.get("damage_" + d).set(thing_classes=["sky"])

    # damage_metadata = MetadataCatalog.get("damage_train")
    # dataset_dicts = load_damage_dicts(r"/cluster/home/helensem/Master/data", "train")

    # #Visualization 
    # for d in random.sample(dataset_dicts, 1): 
    #    img = cv2.imread(d["file_name"])
    #    visualizer = Visualizer(img[:,:,::-1], metadata = damage_metadata, scale =0.5)
    #    out = visualizer.draw_dataset_dict(d)
    #    cv2.imshow("imageout", out.get_image()[:,:,::-1])
    #    cv2.waitKey(0)
    #    # closing all open windows
    #    cv2.destroyAllWindows()