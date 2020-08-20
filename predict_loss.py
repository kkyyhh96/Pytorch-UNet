import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
import pickle
import pandas as pd
from glob import glob

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.shape[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-i', '--input', type=str, default="data/imgs/",
                        help='Input folder path', dest='input_img')
    parser.add_argument('-o', '--output', type=str, default="data/imgs/",
                        help='Output folder path', dest='output_img')
    parser.add_argument('--viz', '-z', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-v', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)
    parser.add_argument('-n', '--n_channels', type=int, default=3,
                        help='Number of channels', dest='n_channels')
    parser.add_argument('-p', '--img_pickles', type=str, default=False, # default="data/img_train.p",
                        help='Pickle file of image ids', dest='img_pickles')

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input_img
    out_files = []

    if not args.output_img:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output_img):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output_img

    return out_files


def mask_to_image(mask):
    return (mask * 255).astype(np.uint8)

if __name__ == "__main__":
    args = get_args()
    in_files = args.input_img

    net = UNet(n_channels=args.n_channels, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    
    try:
        os.mkdir(args.output_img)
    except OSError:
        pass
      
    files_all = []
    if args.img_pickles is not False:
        for id_value in pd.read_pickle(args.img_pickles).values:
            if glob(f"{in_files}" + id_value[0] + '.*')[0].split('.')[-1] == "pckl":
                files_all.append(id_value[0] + ".pckl")
            else:
                files_all.append(id_value[0] + ".png")
    else:
        files_all = os.listdir(in_files)

    for i, fn in enumerate(files_all):
        if (fn.split('.')[-1] == "pckl") or (fn.split('.')[-1] == "jpg") or (fn.split('.')[-1] == "png"):
            logging.info("\nPredicting image {} ...".format(fn))

            if fn.split('.')[-1] != "pckl":
                img = cv2.imread(f"{in_files}{fn}", cv2.IMREAD_UNCHANGED)
            else:
                with open(f"{in_files}{fn}", 'rb') as img:
                    img = pickle.load(img)

            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            if not args.no_save:
                out_fn = f"{args.output_img}{fn.split('.')[0]}.png"
                result = mask_to_image(mask)
                cv2.imwrite(out_fn, result)

                logging.info("Mask saved to {}.png".format(fn.split('.')[0]))

            if args.viz:
                logging.info("Visualizing results for image {}, close to continue ...".format(fn))
                plot_img_and_mask(img, mask)
