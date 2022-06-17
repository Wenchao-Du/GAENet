from .erfnet import FFNet
from .dgn import DGCNN, LDGCNN
from .PreProcess import Init_GlobalVars, PrjectMap, PointSampling
import torch
import torch.nn as nn
import sys

sys.path.append('..')
from Utils.utils import weights_init_kaiming, weights_init_normal, weights_init_orthogonal, weights_init_xavier


class GAENet(nn.Module):

    def __init__(self, config):
        super(GAENet, self).__init__()
        self.batch_size = config['batch_size']
        self.nsp = config['nsp']
        self.knn = config['knn']
        self.g_channel = config['gchannel']
        self.init_type = config['weight_init']
        self.backbone = FFNet(3, config['gchannel'],
                              config['out_channel'])  # point feature dim=32
        self.pointnet = DGCNN(self.knn, self.g_channel)

        self.pretrained = config['pretrained']
        self.pretrained_2 = config['pretrained_2']
        self.updataeParamters(self.batch_size, self.nsp)
        self.init_weight()
        # Consider using pretrained model for encoders
        # self.load_pretrainedmodel() 

    # update the camintrisic and point number for inference
    def updataeParamters(self, Batchsize, PointNum):
        Init_GlobalVars(Batchsize, PointNum)
        self.nsp = PointNum

    #  param: is_test is used for inference
    def PointForward(self, pointCloud, is_test=False):
        points, index = PointSampling(pointCloud, self.nsp, is_test)
        output = self.pointnet(points)
        outfeature = torch.zeros_like(pointCloud[:, 2:, :, :]).repeat(
            1, self.g_channel + 1, 1, 1)
        points = points.permute(0, 2, 1)
        outfeature = PrjectMap(torch.cat((output, points[:, :, 1:2]), dim=2),
                               index, outfeature)
        return outfeature

    def forward(self, rgb, pointCloud, is_test=False):
        pfout = self.PointForward(pointCloud, is_test)
        output = self.backbone(rgb, pfout)
        return output

    def init_weight(self):
        print('Init weights in network with [{}]'.format(self.init_type))
        if self.init_type == 'normal':
            self.backbone.apply(weights_init_normal)
            self.pointnet.apply(weights_init_normal)
        elif self.init_type == 'xavier':
            self.backbone.apply(weights_init_xavier)
            self.pointnet.apply(weights_init_xavier)
        elif self.init_type == 'kaiming':
            self.backbone.apply(weights_init_kaiming)
            self.pointnet.apply(weights_init_kaiming)
        elif self.init_type == 'orthogonal':
            self.backbone.apply(weights_init_orthogonal)
            self.pointnet.apply(weights_init_orthogonal)
        else:
            raise NotImplementedError(
                'initialization method [{}] is not implemented'.format(
                    self.init_type))

    # currently only load the pretrained encoder
    def load_pretrainedmodel(self):
        if self.pretrained is not 'None' and self.pretrained_2 is not 'None':
            erfcheck = torch.load(self.pretrained,
                                  map_location=torch.device('cpu'))
            pointcheck = torch.load(self.pretrained_2,
                                    map_location=torch.device('cpu'))
            state = self.backbone.encoder_rgb.state_dict()
            state_pf = self.backbone.encoder_point.state_dict()
            state_point = self.pointnet.state_dict()
            for name, param in erfcheck.items():
                name = name.replace('module.encoder.', '')
                if name not in state:
                    continue
                else:
                    if 'initial_block' in name:
                        continue
                    elif 'layers.0.' in name:
                        continue
                    else:
                        state[name].copy_(param)
                        state_pf[name].copy_(param)

            for name, param in pointcheck.items():
                name = name.replace('module.', '')
                if name in state_point:
                    state_point[name].copy_(param)
            print('load pretrained global network model !!')
            del erfcheck, pointcheck
        else:
            print('the pretrained model is None!')

