import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
import random
from dataloaders import transforms
import json
# import transforms as transforms
from dataloaders.pose_estimator import get_pose_pnp
# from pose_estimator import get_pose_pnp
# input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']

# referred as DeepLidar Camera-Instrics
INSTICS = {
    "2011_09_26": [721.5377, 596.5593, 149.854],
    "2011_09_28": [707.0493, 604.0814, 162.5066],
    "2011_09_29": [718.3351, 600.3891, 159.5122],
    "2011_09_30": [707.0912, 601.8873, 165.1104],
    "2011_10_03": [718.856, 607.1928, 161.2157]
}

IMGSIZE = {
    "2011_09_26": [1242, 375],
    "2011_09_28": [1224, 370],
    "2011_09_29": [1238, 374],
    "2011_09_30": [1226, 370],
    "2011_10_03": [1241, 376]
}
# Specific Image Size for 1216, 352
ResSIZE = {
    "2011_09_26": [13., 11.5],
    "2011_09_28": [4., 9.],
    "2011_09_29": [11., 11.],
    "2011_09_30": [5., 9.],
    "2011_10_03": [12.5, 12.]
}


# Load the camera instrisics
def load_calib(substr, path=None):
    # load different camera intrinsic for train and select, test dataset.
    if path != None:
        fp = open(path, 'r')
        lines = fp.readlines()
        p_rect_line = lines[0]
        prj_str = p_rect_line.strip().split(" ")
        K = np.reshape(np.array([float(p) for p in prj_str]),
                       (3, 3)).astype(np.float32)
    else:
        param_t = INSTICS[substr]
        K = np.zeros((3, 3), dtype=np.float32)
        K[0, 0] = K[1, 1] = param_t[0]
        K[0, 2] = param_t[1] - ResSIZE[substr][0]
        K[1, 2] = param_t[2] - ResSIZE[substr][1]
        K[2, 2] = 1
    return K


#  numpy data project 2d data to 3d data
class NP2D23D:
    def __init__(self, width=1216, height=352):
        self.height, self.width = height, width
        self.U = np.arange(width).astype(np.float32)
        self.U = np.resize(self.U, (height, width))

        self.V = np.arange(height).astype(np.float32)
        self.V = np.resize(self.V, (width, height)).T

    def np2PointColud(self, npmat, instric):
        fu, fv = instric[0, 0], instric[1, 1]
        cu, cv = instric[0, 2], instric[1, 2]
        x_cam = (self.U - cu) / fu
        x_cam = np.expand_dims(x_cam, axis=-1)
        y_cam = (self.V - cv) / fv
        y_cam = np.expand_dims(y_cam, axis=-1)
        x = x_cam * npmat
        y = y_cam * npmat
        return np.concatenate((x, y, npmat), axis=2)


def get_data_path_and_transform(split, args):
    """ Args need add the data_json attr.
    """
    # assert (args.use_d or args.use_rgb
    #         or args.use_g), 'no proper input selected'
    if split == "train":
        transform = train_transform
        with open(args['split_json']) as json_file:
            json_data = json.load(json_file)
            sample_list = json_data[split]
    elif split == "val":
        transform = val_transform
        with open(args['split_json']) as json_file:
            json_data = json.load(json_file)
            sample_list = json_data[split]
    elif split == "selval":
        transform = no_transform
        with open(args['split_json']) as json_file:
            json_data = json.load(json_file)
            sample_list = json_data[split]
    elif split == "test":
        transform = no_transform
        with open(args['split_json']) as json_file:
            json_data = json.load(json_file)
            sample_list = json_data[split]

    else:
        raise ValueError("Unrecognized split " + str(split))

    paths = sample_list
    return paths, transform


def get_paths_and_transform(split, config):
    # assert (args.use_d or args.use_rgb
    #         or args.use_g), 'no proper input selected'

    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            config['data_root'],
            'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            config['data_root'],
            'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )

        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join([config['data_root']] + ['data_rgb'] + ps[-6:-4] +
                            ps[-2:-1] + ['data'] + ps[-1:])
            return pnew
    elif split == "val":
        transform = val_transform
        glob_d = os.path.join(
            config['data_root'],
            'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            config['data_root'],
            'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )

        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join(ps[:-7] + ['data_rgb'] + ps[-6:-4] + ps[-2:-1] +
                            ['data'] + ps[-1:])
            return pnew
    elif split == "select":
        transform = no_transform
        glob_d = os.path.join(
            config['data_root'],
            "depth_selection/val_selection_cropped/velodyne_raw/*.png")
        glob_gt = os.path.join(
            config['data_root'],
            "depth_selection/val_selection_cropped/groundtruth_depth/*.png")

        def get_rgb_paths(p):
            return p.replace("groundtruth_depth", "image")
    elif split == "test_completion":
        transform = no_transform
        glob_d = os.path.join(
            config['data_root'],
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        )
        glob_gt = None  # "test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            config['data_root'],
            "depth_selection/test_depth_completion_anonymous/image/*.png")
    elif split == "test_prediction":
        transform = no_transform
        glob_d = None
        glob_gt = None  # "test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            config['data_root'],
            "depth_selection/test_depth_prediction_anonymous/image/*.png")
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = sorted(glob.glob(glob_gt))
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else:
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png


def sparsedepth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(
        np.float) / 256.  # 85 denote the max depth, scale the value to 0-255
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    #assert os.path.exists(filename), "file not found: {}".format(filename)
    if filename is None:
        return None
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(
        np.float) / 256.  # 85 denote the max depth, scale the value to 0-255
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


oheight, owidth = 352, 1216


def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth


def get_params(img, output_size):
    """Get parameters for ``crop`` for a random crop.
    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    h, w = img.shape[0], img.shape[1]
    tw, th = output_size
    i, j = 0, 0
    if w == tw:
        i = 0
    else:
        i = random.randint(0, h - th - 1)
    if h == th:
        j = 0
    else:
        j = random.randint(0, w - tw - 1)
    return i, j, th, tw


def train_transform(rgb, sparse, target, rgb_near, config):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
    i, j, th, tw = get_params(
        sparse,
        (config['crop_w'], config['crop_h']))  # original input is sparse
    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s), # oheight, owidth
        transforms.Crop((j, j + tw, i, i + th)),
        transforms.HorizontalFlip(do_flip)
    ])

    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - config['jitter']),
                                       1 + config['jitter'])
        contrast = np.random.uniform(max(0, 1 - config['jitter']),
                                     1 + config['jitter'])
        saturation = np.random.uniform(max(0, 1 - config['jitter']),
                                       1 + config['jitter'])
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
        if rgb_near is not None:
            rgb_near = transform_rgb(rgb_near)
    # sparse = drop_depth_measurements(sparse, 0.9)
    return rgb, sparse, target, rgb_near


def val_transform(rgb, sparse, target, rgb_near, config):
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    if rgb_near is not None:
        rgb_near = transform(rgb_near)
    return rgb, sparse, target, rgb_near


def no_transform(rgb, sparse, target, rgb_near, config):
    return rgb, sparse, target, rgb_near


to_tensor = transforms.ToTensor()


def to_float_tensor(x):
    return to_tensor(x).float()


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """
    def __init__(self, split, config):
        self.config = config
        self.split = split
        paths, transform = get_data_path_and_transform(split, config)
        self.paths = paths
        self.transform = transform
        self.threshold_translation = 0.1
        self.prj3d = NP2D23D()

    def __getraw__(self, index):
        path_rgb = os.path.join(self.config['data_root'],
                                self.paths[index]['rgb'])
        path_depth = os.path.join(self.config['data_root'],
                                  self.paths[index]['depth'])
        path_gt = os.path.join(self.config['data_root'],
                               self.paths[index]['gt'])
        path_calib = os.path.join(self.config['data_root'],
                                  self.paths[index]['K'])
        rgb = rgb_read(path_rgb)
        depth = depth_read(path_depth)
        gt = depth_read(path_gt)

        return rgb, depth, gt, path_calib

    def __getitem__(self, index):
        rgb, sparse, target, kpath = self.__getraw__(index)
        rgb, sparse, target, _ = self.transform(rgb, sparse, target, None,
                                                self.config)
        if self.split == 'selval' or self.split == 'test':
            instric = load_calib(None, path=kpath)
        else:
            index_str = kpath.split('/')[-2][0:10]
            instric = load_calib(index_str)
        pcdata = self.prj3d.np2PointColud(sparse, instric)  # w * H * 3 (x,y,z)
        candidates = {"rgb": rgb, "pc": pcdata, "gt": target}
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }
        if self.split == 'selval' or self.split == 'test':
            filename = os.path.basename(kpath)
            return items, filename
        else:
            return items

    def __len__(self):
        return len(self.paths)