
import numpy as np 
import os,json,cv2,random 
from matplotlib import pyplot as plt
#import skimage


from detectron2.structures import BoxMode 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil
import sys
sys.path.append(r"/cluster/home/helensem/Master/chipsogdip/kode")
from skyseg import remove_sky


##### Creating training images for GA ######### 

def ga_train_sets():
    os.makedirs(r"/cluster/home/helensem/Master/data/set1/val", exist_ok=True)
    #os.makedirs(r"/cluster/home/helensem/Master/data/set2/val", exist_ok=True)
    #os.makedirs(r"/cluster/home/helensem/Master/data/set3/val", exist_ok=True)
    
    image_ids = next(os.walk(r"/cluster/home/helensem/Master/damage_data/val"))[1]
    set1 = []
    set2 = []
    set3 = []

    for i in range(30): 
        idx = random.randint(0,len(image_ids)-1)
        set1.append(image_ids[idx])
        del image_ids[idx]
        idx = random.randint(0,len(image_ids)-1)
        set2.append(image_ids[idx])
        del image_ids[idx]
        idx = random.randint(0,len(image_ids)-1)
        set3.append(image_ids[idx])
        del image_ids[idx]

    for image_id in set1: 
        image_path = os.path.join(r"/cluster/home/helensem/Master/damage_data/val", image_id)
        #image = next(os.walk(image_path))[2][0]
        #(image_path) = os.path.join(image_path, image)
        destination = os.path.join(r"/cluster/home/helensem/Master/data/set1/val", image_id)
        shutil.copytree(image_path, destination) 
    
    # for image_id in set2: 
    #     image_path = os.path.join(r"/cluster/home/helensem/Master/Labeled_pictures/val", image_id)
    #     image = next(os.walk(image_path))[2][0]
    #     #(image_path) = os.path.join(image_path, image)
    #     destination = os.path.join(r"/cluster/home/helensem/Master/data/set2/val", image_id)
    #     shutil.copytree(image_path, destination)     

    # for image_id in set3: 
    #     image_path = os.path.join(r"/cluster/home/helensem/Master/Labeled_pictures/val", image_id)
    #     image = next(os.walk(image_path))[2][0]
    #     #(image_path) = os.path.join(image_path, image)
    #     destination = os.path.join(r"/cluster/home/helensem/Master/data/set3/val", image_id)
    #     shutil.copytree(image_path, destination)  
    load_damage_dicts(r"/cluster/home/helensem/Master/data/set1", "val", True)






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
    return contours


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


def load_damage_dicts(dataset_dir, subset, write_to_file=False): #? Possibly write this to a JSON-file? 
    """
    Loads the images from a dataset with a dictionary of the annotations in COCO-format  
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
        height, width = cv2.imread(image_path).shape[:2]
        record = create_image_annotation(image_path, width, height, image_id)
        #idx +=1
        mask_dir = os.path.join(image_dir, 'masks')
        objs = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith('.png') and ('corrosion' or 'grov_merking' in f):
                mask_path = os.path.join(mask_dir, f)
                #print(mask_path)
                mask = cv2.imread(mask_path)
                if mask is None: 
                    print("Couldn't retrieve mask: ", mask_path)
                    continue
                if mask.shape[0]!=height:
                    print("MISMATCH:", image_dir)
                if not(255 in mask):
                    #pr#int("mask is empty: ", mask_path)
                    continue
                contours = find_contours(mask)
                for contour in contours:
                    if len(contour) < 3:
                        #print("Contour too small: ", mask_path)
                        continue 
                    obj = create_annotation_format(contour)
                    objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    #* For writing to JSON-file
    if write_to_file:
        json_object = json.dumps(dataset_dicts)
        json_name = f"damage_{subset}.json"
        with open(os.path.join(dataset_dir, json_name), "w") as f:
            f.write(json_object)
        return
    return dataset_dicts


def load_mask(mask_dir):
    mask = []
    print(mask_dir)
    for f in next(os.walk(mask_dir))[2]:
        if f.endswith('.png') and ('corrosion' or 'grov_merking' in f):
            mask_path = os.path.join(mask_dir, f)
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            m = m.astype(bool)
            mask.append(m)
    mask = np.stack(mask, axis=-1)
    return mask.astype(bool)

def combine_masks_to_one(masks):
    combined_mask = masks[:, :, 0]
    for i in range(masks.shape[-1]):
        combined_mask += masks[:, :, i]
    return np.expand_dims(combined_mask, 2)




def load_pictures(dataset_dir, subset): #? Possibly write this to a JSON-file? 
    """
    Loads the images from a dataset with a dictionary of the annotations, to be loaded in detectron 
    """ 


    assert subset in ["train", "val"]
    dataset_dir = os.path.join(dataset_dir, subset)
    image_ids = next(os.walk(dataset_dir))[1]
    destination = r"/cluster/home/helensem/Master/damage_data"
    #idx = 0
    for image_id in image_ids:

        image_dir = os.path.join(dataset_dir, image_id)
        print(image_dir)
        (_, _, file_names) = next(os.walk(image_dir))
        file_name = file_names[0]
        
        image_path = os.path.join(image_dir, file_name)
        #destination_path = os.path.join(destination, file_name)
        id = image_id +  os.path.splitext(file_name)[1]
        destination_path = os.path.join(destination, id )
        shutil.copy(image_path, destination_path)


def histogram_percent(path, subset):
    dict = load_damage_dicts(path, subset)
    damagehisto = np.zeros((len(dict)))
    for d in dict:
        damagecount = 0
        image_dir = os.path.dirname(d["file_name"])
        mask_gt= load_mask(os.path.join(image_dir, "masks"))
        mask_gt = combine_masks_to_one(mask_gt)
        for i in range(mask_gt.shape[0]):
            for j in range(mask_gt.shape[1]):
                if mask_gt[i][j]:
                   damagecount+=1
        pixelcount = mask_gt.shape[0]*mask_gt.shape[1]
        percent = damagecount*100//pixelcount
        print("damagecount:", damagecount, " number of pixels: ", pixelcount, " percent: ", percent)
        damagehisto[id] = percent
        if percent == 100:
           print("image ",d["image_id"], "\n", mask_gt)

    return damagehisto




####### MAIN ##################
if __name__ == "__main__": 
    print("hei")
    #load_pictures(r"/cluster/home/helensem/Master/Labeled_pictures", "val")
 #   root = r"/cluster/home/helensem/Master/Labeled_pictures"
  #  destination = r"/cluster/home/helensem/Master/damage_data"

   # for d in ['train', 'val']:
    #    load_damage_yolo(root, d, destination)
    values = histogram_percent(r"/cluster/home/helensem/Master/damage_data", "train")
    plt.hist(values, bins=20)  # density=False would make counts
    print(np.mean(values))
    plt.ylabel('No. images')
    plt.xlabel('% of damaged pixels')

    plt.savefig(r"/cluster/home/helensem/Master/damage_count_train.svg", format="svg")


    #print(load_damage_dicts(r"/cluster/home/helensem/Master/data", "train"))
    #train_dict = load_damage_dicts(r"/cluster/home/helensem/Master/Labeled_pictures", "train")
    #val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/Labeled_pictures", "val")
    #load_damage_dicts(r"/cluster/home/helensem/Master/data/set1", "train", True)
    #load_damage_dicts(r"/cluster/home/helensem/Master/data/set1", "val", True)
    #im = cv2.imread(r"/cluster/home/helensem/Master/data/train/IMG_3400/1.png")
    #print(im.shape)
    #mask = cv2.imread(r"/cluster/home/helensem/Master/data/train/IMG_3400/masks/grov_merking_1.png")
    #print(mask.shape)


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


def load_damage_yolo(root, subset,destination): 
    """
    Loads the images from a dataset with a dictionary of the annotations, to be loaded in detectron 
    """ 
    assert subset in ['train', 'val']
    source = os.path.join(root, subset)

    image_ids = next(os.walk(source))[1]

    for id in image_ids:
        image_dir = os.path.join(source, id)
        (_, _, file_names) = next(os.walk(image_dir))
        file_name = file_names[0]
        
        image_path = os.path.join(image_dir, file_name)
        height, width = cv2.imread(image_path).shape[:2]
        mask_dir = os.path.join(image_dir, 'masks')
        string = ""
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith('.png') and ('corrosion' or 'grov_merking' in f):
                mask_path = os.path.join(mask_dir, f)
                #print(mask_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None: 
                    print("Couldn't retrieve mask: ", mask_path)
                    continue
                if mask.shape[0]!=height:
                    print("MISMATCH:", image_dir)
                if not(255 in mask):
                    print("mask is empty: ", mask_path)
                    continue
                contour = find_contours(mask)
                contour_list = contour.flatten().tolist()
                if len(contour_list)<5:
                    continue # Cant create polygons from too few coordinates
                string += "0 " 
                for i in range(1,len(contour_list),2): 
                    string += str(round(contour_list[i-1]/width,6)) #x coordinate
                    string += " "
                    string += str(round(contour_list[i]/height, 6)) # y coordinate
                    string += " "
                string+= "\n"
        image_id = id +  os.path.splitext(file_name)[1]
        image_dest = os.path.join(destination, "images", subset, image_id )
        print("destination: ", image_dest)
        print("source: ", image_path)
        shutil.copy(image_path, image_dest)
        #print(string)
        txt_id = id +'.txt' 
        txt_path = os.path.join(destination, "labels", subset, txt_id)
        print(txt_path)
        with open(txt_path, "w") as f: 
          f.write(string)

def get_json_dict(path, subset="train"):
    file = "damage_" + subset + ".json"
    file_path = os.path.join(path, subset, file)
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

