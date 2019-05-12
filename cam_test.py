# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode, find_recursive
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm
from torchvision import transforms

colors = loadmat('data/color150.mat')['colors']

IMGMAX_SIZE = 1000
padding_constant = 8


def visualize_display(img, pred):

    # prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)
    return im_vis


def visualize_result(data, pred, args):
    (img, info) = data

    # prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    cv2.imwrite(os.path.join(args.result,
                img_name.replace('.jpg', '.png')), im_vis)

# normalize = transforms.Normalize(
#     mean=[102.9801, 115.9465, 122.7717],
#     std=[1., 1., 1.])


def img_transform( img):
    # image to float
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = transforms.Normalize(
            mean=[102.9801, 115.9465, 122.7717],
            std=[1., 1., 1.]) (torch.from_numpy(img.copy()))
    return img

# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple( x, p):
    return ((x - 1) // p + 1) * p

def image_pre_process(image, this_short_size):
    img_resized_list = []
    for this_short_size_ in this_short_size:
        ori_height, ori_width, _ = image.shape
        scale = min(this_short_size_ / float(min(ori_height, ori_width)),
                    IMGMAX_SIZE / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_height = round2nearest_multiple(target_height, padding_constant)
        target_width = round2nearest_multiple(target_width, padding_constant)

        # resize
        img_resized = cv2.resize(image.copy(), (target_width, target_height))

        # image transform
        img_resized = img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)
    return img_resized_list



def cam_test(segmentation_module, cap, args):
    segmentation_module.eval()

    # pbar = tqdm(total=len(loader))
    # for batch_data in loader:
    while cap.isOpened():
        # process data
        # batch_data = batch_data[0]
        # segSize = (batch_data['img_ori'].shape[0],
        #            batch_data['img_ori'].shape[1])
        # img_resized_list = batch_data['img_data']

        ret, frame = cap.read()
        image = frame[:,:,::-1]
        height, width, _ = image.shape
        segSize = (height, width)

        with torch.no_grad():
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)

            img_resized_list = image_pre_process(image, [300, 400, 500])
            # feed_dict = {
            #         'img_data': feed_image
            #         }
            # feed_dict = async_copy_to(feed_dict, args.gpu)
            # pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            # scores = scores + pred_tmp 
            for img in img_resized_list:
                feed_dict = {}
                feed_dict['img_data'] = img
                feed_dict = async_copy_to(feed_dict, args.gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(args.imgSize)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        viz_res = visualize_display(image, pred)
        print(viz_res.shape)
        cv2.imshow("VIZ", viz_res)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def main(args):
    torch.cuda.set_device(args.gpu)

    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    # if len(args.test_imgs) == 1 and os.path.isdir(args.test_imgs[0]):
    #     test_imgs = find_recursive(args.test_imgs[0])
    # else:
    #     test_imgs = args.test_imgs
    # list_test = [{'fpath_img': x} for x in test_imgs]
    # dataset_test = TestDataset(
    #     list_test, args, max_sample=args.num_val)
    # loader_test = torchdata.DataLoader(
    #     dataset_test,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     collate_fn=user_scattered_collate,
    #     num_workers=5,
    #     drop_last=True)

    segmentation_module.cuda()

    # Main loop
    cap = cv2.VideoCapture(0)

    cam_test(segmentation_module, cap, args)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    # parser.add_argument('--test_imgs', required=True, nargs='+', type=str,
    #                     help='a list of image paths, or a directory name')
    parser.add_argument('--model_path', required=True,
                        help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    # Misc arguments
    parser.add_argument('--result', default='.',
                        help='folder to output visualization results')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id for evaluation')

    args = parser.parse_args()
    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.model_path,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.model_path,
                                        'decoder' + args.suffix)

    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
