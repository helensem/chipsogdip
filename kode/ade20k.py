"""Prepare ADE20K dataset"""
import os
import shutil
#import argparse
#import zipfile
#from gluoncv.utils import download, makedirs
#from gluoncv.data import ADE20KSegmentation


_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/ade')

"""Pascal ADE20K Semantic Segmentation Dataset."""
import os
from PIL import Image
import numpy as np
import mxnet as mx
from gluoncv.data.segbase import SegmentationDataset

class ADE20KSegmentation(SegmentationDataset):
    """ADE20K Semantic Segmentation Dataset.
    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is '$(HOME)/mxnet/datasplits/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from mxnet.gluon.data.vision import transforms
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = gluoncv.data.ADE20KSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = gluon.data.DataLoader(
    >>>     trainset, 4, shuffle=True, last_batch='rollover',
    >>>     num_workers=4)
    """
    # pylint: disable=abstract-method
    BASE_DIR = 'ADEChallengeData2016'
    NUM_CLASS = 1
    CLASSES = ("sky")
    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/ade'),
                 split='train', mode=None, transform=None, **kwargs):
        super(ADE20KSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please setup the dataset using" + \
            "scripts/datasets/ade20k.py"
        self.images, self.masks = _get_ade20k_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _mask_transform(self, mask):
        return mx.nd.array(np.array(mask), mx.cpu(0)).astype('int32') - 1

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    @property
    def pred_offset(self):
        return 1

def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
    else:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                print('cannot find the mask:', maskpath)

    return img_paths, mask_paths

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Initialize ADE20K dataset.',
#         epilog='Example: python setup_ade20k.py',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
#     args = parser.parse_args()
#     return args

# def download_ade(path, overwrite=False):
#     _AUG_DOWNLOAD_URLS = [
#         ('http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip', '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),
#         ('http://data.csail.mit.edu/places/ADEchallenge/release_test.zip', 'e05747892219d10e9243933371a497e905a4860c'),]
#     download_dir = os.path.join(path, 'downloads')
#     makedirs(download_dir)
#     for url, checksum in _AUG_DOWNLOAD_URLS:
#         filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
#         # extract
#         with zipfile.ZipFile(filename,"r") as zip_ref:
#             zip_ref.extractall(path=path)


if __name__ == '__main__':
    root = r"/cluster/home/helensem/Master/ade"
    train_dataset = ADE20KSegmentation(root,split='train')
    val_dataset = ADE20KSegmentation(root, split='val')
    print('Training images:', len(train_dataset))
    print('Validation images:', len(val_dataset))
    print("Classes:", train_dataset.classes)
    img, mask = val_dataset[0]
    mask = mask.asnumpy()
    print(mask.shape)
