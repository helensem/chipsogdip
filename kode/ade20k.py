"""Prepare ADE20K dataset"""
import os
import shutil
import argparse
import zipfile
from gluoncv.utils import download, makedirs
from gluoncv.data import ADE20KSegmentation


_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/ade')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize ADE20K dataset.',
        epilog='Example: python setup_ade20k.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args

def download_ade(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip', '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),
        ('http://data.csail.mit.edu/places/ADEchallenge/release_test.zip', 'e05747892219d10e9243933371a497e905a4860c'),]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall(path=path)


if __name__ == '__main__':
    root = r"/cluster/home/helensem/Master/ade"
    train_dataset = ADE20KSegmentation(split='train')
    val_dataset = ADE20KSegmentation(split='val')
    train_dataset.CLASSES = {"sky"}
    val_dataset.CLASSES = {"sky"}
    train_dataset.NUM_CLASS = 1
    val_dataset.NUM_CLASS = 1
    print('Training images:', len(train_dataset))
    print('Validation images:', len(val_dataset))
    print("Classes:", train_dataset.classes)