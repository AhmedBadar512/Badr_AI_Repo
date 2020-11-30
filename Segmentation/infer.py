import tensorflow as tf
import numpy as np
import cv2
import pathlib
from citys_visualizer import display
from visualization_dicts import generate_random_colors
import tqdm
import argparse
import os
from model_provider import get_model

DATASET_DICT = {"cityscapes19": 19, "cityscapes": 34}


def load_model_dynamic(pretrained_model_path, curr_model):
    if os.path.exists(os.path.join(pretrained_model_path, "saved_model.pb")):
        pretrained_model = tf.keras.models.load_model(pretrained_model_path)
        curr_model.build(input_shape=(None, None, None, 3))
        curr_model.set_weights(pretrained_model.get_weights())
        print("Model loaded from {} successfully".format(os.path.basename(pretrained_model_path)))
    else:
        print("No file found at {}".format(os.path.join(pretrained_model_path, "saved_model.pb")))


def get_model_props(pretrained_model_path):
    name = pretrained_model_path.split("/")[-2]
    dataset_name = pretrained_model_path.split("/")[-3].split("_")[0]
    return name, dataset_name


args = argparse.ArgumentParser(description="Infer with a given Checkpoint")
args.add_argument("--img_dir",
                  type=str,
                  default="/data/input/datasets/cityscape_processed/leftImg8bit/val/munster",
                  help="Path containing the png/jpg files")
args.add_argument("-m", "--model_dir",
                  type=str,
                  default="sample_model/15",
                  help="Path to TF2 saved model dir")
args.add_argument("-s", "--save_dir",
                  type=str,
                  default=None,
                  help="Path to TF2 saved model dir")
args.add_argument("--cs19",
                  action="store_true",
                  default=False,
                  help="Colorize based on CS-19 color pallete, use only if n_classes<=19")
args.add_argument("--height",
                  type=int,
                  default=512,
                  help="Path to TF2 saved model dir")
args.add_argument("--width",
                  type=int,
                  default=1024,
                  help="Path to TF2 saved model dir")
args.add_argument("-ir",
                  "--input_resize",
                  action="store_true",
                  default=False,
                  help="If true input images are not resized")
args.add_argument("-rso",
                  "--resize_original",
                  action="store_true",
                  default=False,
                  help="If true output images are resized back to input image size")
args = args.parse_args()

width, height = args.width, args.height
img_dir = args.img_dir
path = args.model_dir
save_dir = args.save_dir
# model = tf.saved_model.load(path)
# model = tf.keras.models.load_model(path)
model_name, dataset_name = get_model_props(args.model_dir)
if args.input_resize:
    model = tf.keras.models.load_model(path)
else:
    model = get_model(model_name, classes=DATASET_DICT[dataset_name])

    load_model_dynamic(args.model_dir, model)
exts = ["*jpg", "*png", "*jpeg"]
imgs = [str(img) for ext in exts for img in pathlib.Path(img_dir).rglob(ext)]


def infer(im_path, cmap):
    im = cv2.imread(im_path)[..., ::-1]
    height_old, width_old = im.shape[0:2]
    if args.input_resize:
        im = cv2.resize(im, (width, height))
    img = tf.constant(im.astype(np.float32))
    outputs = model(img[np.newaxis], False)
    if type(outputs) is list or type(outputs) is tuple:
        seg = tf.argmax(outputs[0], axis=-1)
    else:
        seg = tf.argmax(outputs, axis=-1)
    img = img[np.newaxis]
    seg = seg[..., np.newaxis]
    if args.resize_original:
        img = tf.image.resize(img, size=(height_old, width_old))
        seg = tf.image.resize(seg, size=(height_old, width_old), method="nearest")
    display(img, seg, cs_19=args.cs19, save_dir=save_dir, img_path=im_path, cmap=cmap)


if __name__ == "__main__":
    cmap = generate_random_colors()
    for img_path in tqdm.tqdm(imgs):
        infer(img_path, cmap)
