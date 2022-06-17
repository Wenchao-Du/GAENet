import numpy as np
import nyuv2_dataset.transforms as transforms
from nyuv2_dataset.dataloader import MyDataloader, iheight, iwidth
from nyuv2_dataset.dense_to_sparse import UniformSampling

# import transforms as transforms
# from dataloader import MyDataloader, iheight, iwidth
# from dense_to_sparse import UniformSampling

# import torch
# import matplotlib.pyplot as plt
# import h5py
# import open3d as o3d


class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='d'):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (224, 304)  # src 228 * 304

    def train_transform(self, rgb, depth):
        # s = np.random.uniform(1.0, 1.5)  # random scaling
        # depth_np = depth / s
        # angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(
                240.0 / iheight
            ),  # this is for computational efficiency, since rotation can be slow
            # transforms.Rotate(angle),
            # transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np


def testdataset(filepath):
    # h5f = h5py.File(filepath, "r")
    # rgb = np.array(h5f['rgb'])
    # rgb = np.transpose(rgb, (1, 2, 0))
    # depth = np.array(h5f['depth'])

    dataset = NYUDataset(filepath, 'val', UniformSampling(60000, 10))
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        sampler=None,
        worker_init_fn=lambda work_id: np.random.seed(work_id))
    for i, input in enumerate(train_loader):
        depth = input['pc'].squeeze().numpy()
        rgb = input['rgb'].squeeze().numpy()
        plt.imshow(rgb.transpose(1, 2, 0))
        plt.show()
        depth = depth.transpose(1, 2, 0)
        points = np.reshape(depth, (-1, 3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    filepath = '/mnt/90C68C19C68C01A8/3D_Ori_VisonSet/NYU_V2/nyudepthv2/val'
    testdataset(filepath)