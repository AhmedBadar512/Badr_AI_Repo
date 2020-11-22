import tensorflow as tf
import multiprocessing as mp
import numpy as np
import cv2
import glob
from citys_visualizer import display
import tqdm
import time
import os

# TODO: Add model loading
pool = mp.Pool(os.cpu_count()//2)
path = "/volumes1/Code/Badr_AI_Repo/Segmentation/runs/logs/cityscapes19_epochs-100_bs-8_Adam_lr-0.0001_bisenet_resnet18_celebamaskhq_20201115-121303012722/bisenet_resnet18_celebamaskhq/15"
imported = tf.saved_model.load(path)
imgs = glob.glob("/data/input/datasets/cityscape_processed/leftImg8bit/val/munster/*g")


def infer(im_path):
    img = tf.constant(cv2.resize(cv2.imread(im_path)[..., ::-1], (1024, 512)).astype(np.float32))
    outputs = imported(img[np.newaxis], False)
    seg = tf.argmax(outputs[0], axis=-1)
    display(img[np.newaxis], seg[..., np.newaxis], cs_19=False)


for img_path in tqdm.tqdm(imgs):
    infer(img_path)
# TODO: Inference test
