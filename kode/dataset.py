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
np.set_printoptions(threshold=1000)


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
        "id": image_id,
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
    for image_id in image_ids:

        image_dir = os.path.join(dataset_dir, image_id)
        print(image_dir)
        (_, _, file_names) = next(os.walk(image_dir))
        file_name = file_names[0]
        
        image_path = os.path.join(image_dir, file_name)
        print(image_path)
        height, width = cv2.imread(image_path).shape[:2]
        record = create_image_annotation(image_path, width, height, image_id)

        mask_dir = os.path.join(image_dir, 'masks')
        objs = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith('.png') and ('corrosion' or 'grov_merking' in f):
                mask_path = os.path.join(mask_dir, f)
                print(mask_path)
                mask = cv2.imread(mask_path)
                if not(255 in mask):
                    continue
                #if len(mask.shape) > 2: #! Some issues with certain train images 
                #    mask = mask[:,:,0]
                contour = find_contours(mask)
                obj = create_annotation_format(contour)
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    json_object = json.dumps(dataset_dicts)
    with open(f"damage_{subset}.json", "w") as f:
        f.write(json_object)
    return dataset_dicts



####### MAIN ##################

if __name__ == "__main__": 

    #Load data to detectron 
    for d in ["train", "val"]:
        load_damage_dicts(r"/cluster/home/helensem/Master/training_all_pictures", d)
        #DatasetCatalog.register("damage_" + d, lambda d=d: load_damage_dicts(r"/cluster/home/helensem/Master/chipsogdip/Labeled_pictures", d))
        #MetadataCatalog.get("damage_" + d).set(thing_classes=["damage"])

    #damage_metadata = MetadataCatalog.get("damage_train")
    #dataset_dicts = load_damage_dicts(r"/cluster/home/helensem/Master/chipsogdip/Labeled_pictures", "train")

    #Visualization 
    #for d in random.sample(dataset_dicts, 1): 
     #   img = cv2.imread(d["file_name"])
     #   visualizer = Visualizer(img[:,:,::-1], metadata = damage_metadata, scale =0.5)
     #   out = visualizer.draw_dataset_dict(d)
     #   cv2.imshow("imageout", out.get_image()[:,:,::-1])
     #   cv2.waitKey(0)
        # closing all open windows
     #   cv2.destroyAllWindows()