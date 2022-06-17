import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import sys
sys.path.append('..')
from dataloaders.inverse_warp import image_to_pointcloud, Intrinsics
from dataloaders.kitti_loader import load_calib, oheight, owidth

def Init_GlobalVars(batch, point_num):
    global sample_pointNum, index_a
    sample_pointNum = point_num
    index_a = torch.Tensor(batch, point_num)
    if torch.cuda.is_available():
        index_a = index_a.cuda()
    for i in range(batch):
        index_a[i, :] = i
    index_a = index_a.flatten().long()


def PointSampling(PointColud, num, is_test=False):
    batch = PointColud.size(0)
    sampleIndex = []
    # sampling the points from pointclouds
    # becuase of different from edge points, it is necessary to sample points from each pointclouds
    for i in range(batch):
        indexsample = torch.nonzero(PointColud[i, 2, :, :], as_tuple=False)
        if not is_test:
            indexperm = torch.randperm(indexsample.size(0))
            idx = indexperm[:num]
            pointindex = indexsample[idx]
        else:
            pointindex = indexsample
            Init_GlobalVars(batch, pointindex.size(0))
        pointindex = pointindex.view(1, pointindex.size(0), pointindex.size(1))
        sampleIndex.append(pointindex)
    indexmatrix = torch.cat(sampleIndex, dim=0)
    index_wh = indexmatrix.reshape(-1, indexmatrix.size(2))
    samplepoint = PointColud[index_a, :, index_wh[:, 0].long(),
                             index_wh[:, 1].long()]
    samplepoint = samplepoint.reshape(indexmatrix.size(0), indexmatrix.size(1),
                                      3)
    samplepoint = samplepoint.permute(0, 2, 1)
    return samplepoint, indexmatrix


# project point cloud feature to the 2d map with index matrix
def PrjectMap(Pointfeature, Indexmatrix, outfeature):
    Pointfeature = Pointfeature.reshape(-1, Pointfeature.size(2))
    indexmat = Indexmatrix.reshape(-1, Indexmatrix.size(2))
    outfeature[index_a, :, indexmat[:, 0].long(),
               indexmat[:, 1].long()] += Pointfeature
    return outfeature