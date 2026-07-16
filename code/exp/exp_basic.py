import os
import torch
from models import zeroNO,zeroTN,zeroAu,zeroNO_75,zeroAu_75,zeroTN_75,\
    RAMP_Net,RAMP_Net_75,RAMP_Net_era,RAMP_Net_sta,RAMP_Net_pos_only,RAMP_Net_feat_only,\
    Kriging,MPNN,MPNN_75,LinearRegressionModel


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'zeroNO':zeroNO,
            'zeroTN':zeroTN,
            'zeroAu':zeroAu,
            'zeroNO_75': zeroNO_75,
            'zeroAu_75':zeroAu_75,
            'zeroTN_75':zeroTN_75,
            'MPNN': MPNN,
            'MPNN_75':MPNN_75,
            'RAMP_Net': RAMP_Net,
            'RAMP_Net_75': RAMP_Net_75,
            'RAMP_Net_era':RAMP_Net_era,
            'RAMP_Net_sta':RAMP_Net_sta,
            'RAMP_Net_pos_only':RAMP_Net_pos_only,
            'RAMP_Net_feat_only':RAMP_Net_feat_only,
            'Kriging':Kriging,
            'LR': LinearRegressionModel,

        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
