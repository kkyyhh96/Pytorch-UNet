import numpy as np
import argparse
import logging
import os
import dice_loss
import torch
import cv2
import torch.nn
import pickle
import pandas as pd
from glob import glob
from pandarallel import pandarallel

pandarallel.initialize(verbose=0)

def get_args():
    parser = argparse.ArgumentParser(description='Get comparison metrics',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default="data/imgs/",
                        help='Input folder path', dest='input_folder')
    parser.add_argument('-c', '--compare', type=str, default="data/imgs/",
                        help='Comparison folder path', dest='compare_folder')
    parser.add_argument('-p', '--img_pickles', type=str, default=False, # default="data/img_train.p",
                        help='Pickle file of image ids', dest='img_pickles')

    return parser.parse_args()

def read_img(filename):
    if filename.split('.')[-1] != "pckl":
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    else:
        with open(filename, 'rb') as img:
            img = pickle.load(img)
    return torch.from_numpy(img).type(torch.FloatTensor)

def calculate_metrix(x):
    input_img = read_img(x["input_img"])/255
    compare_img = read_img(x["compare_img"])/255

    dice_coefficient = dice_loss.dice_coeff(input_img, compare_img).item()

    criterion = torch.nn.MSELoss()
    loss = torch.sqrt(criterion(input_img, compare_img))

    return [float(loss.data.cpu().numpy()), dice_coefficient]

if __name__ == "__main__":
    args = get_args()
    
    if args.img_pickles != False:
        img_ids = pd.read_pickle(args.img_pickles)
    else:
        img_ids = []
        for file in os.listdir(args.input_folder):
            if (file.split('.')[-1] == "jpg") or (file.split('.')[-1] == "png") or (file.split('.')[-1] == "pckl") or (file.split('.')[-1] == "p"):
                img_ids.append(file.split('.')[0])
        img_ids = pd.DataFrame(img_ids, columns=["img_id"])
    
    img_ids["input_img"] = img_ids["img_id"].apply(lambda x: glob(args.input_folder + x + '.*')[0])
    img_ids["compare_img"] = img_ids["img_id"].apply(lambda x: glob(args.compare_folder + x + '.*')[0])
    img_ids["metric"] = img_ids.parallel_apply(lambda x: calculate_metrix(x), axis=1)
    
    img_ids["rmse"] = img_ids["metric"].parallel_apply(lambda x: x[0])
    img_ids["dice_coeff"] = img_ids["metric"].parallel_apply(lambda x: x[1])
    
    print(args.input_folder, "RMSE: ", img_ids.mean()["rmse"], " Dice Coefficient: ", img_ids.mean()["dice_coeff"])
