import pandas as pd
from matplotlib import pyplot as plt
from torch import Generator
from data_provider.data_factory_2 import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import hashlib
import warnings
import matplotlib.pyplot as plt

import numpy as np
def get_batch_hash(batch_tensor):
    batch_np = batch_tensor.detach().cpu().numpy()
    return hashlib.md5(batch_np.tobytes()).hexdigest()
def hash_state_dict(state_dict, device='cpu'):
        m = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key].detach().to('cpu').numpy()
            t_bytes = tensor.tobytes()
            m.update(t_bytes)

        return m.hexdigest()
class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
        self.loss=args.loss
        self.modelname=args.model
        self.log_dir = './parameter/'
        self.vail_loss=0

    def _build_model(self):
        print('Building model...')
        model = self.model_dict[self.args.model].Model(self.args).float()
        initial_hash = hash_state_dict(model.state_dict(), device='cpu')
        print("Model Initial Hash:", initial_hash)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.device)

    def _get_data(self, flag,train_generator=None,base_seed=None):
        data_set, data_loader = data_provider(args=self.args, flag=flag,train_generator=train_generator,base_seed=base_seed)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.loss == 'Huber':
            criterion = nn.SmoothL1Loss()
        elif self.loss == 'MSE':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee,mask) in enumerate(vali_loader):
                # random mask
                batch_x = batch_x.float().to(self.device)
                batch_x_e = batch_x_e.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                pos_s = pos_s.float().to(self.device)
                pos_e = pos_e.float().to(self.device)
                datee = datee.float().to(self.device)
                current_mask = mask.float().to(self.device)
                outputs,_= self.model(batch_x, batch_x_e, pos_s, pos_e, current_mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = current_mask[:, :, :, f_dim:]
                loss = criterion(outputs[current_mask == 0], batch_y_c[current_mask == 0])
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    def train(self, setting):
        train_generator = Generator()
        train_generator.manual_seed(self.args.seed)
        feature_dir = self.args.feature_dir
        X_train_all = np.load(os.path.join(feature_dir, 'X_train_all.npy'))
        print("Data mean =", X_train_all.mean())
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_losses = []
        vali_losses = []
        hashes = []
        train_generator = Generator()
        train_generator.manual_seed(self.args.seed)
        base_seed = train_generator.initial_seed()
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.args.train_epochs):
            train_data, train_loader = self._get_data(
                flag='train',
                train_generator=train_generator,
                base_seed=base_seed
            )
            train_steps = len(train_loader)
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee,mask) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_e = batch_x_e.float().to(self.device)
                batch_y=batch_y.float().to(self.device)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee=datee.float().to(self.device)
                mask=mask.float().to(self.device)
                outputs,_= self.model(batch_x, batch_x_e, pos_s, pos_e, mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = mask[:, :, :, f_dim:]
                loss = criterion(outputs[current_mask == 0], batch_y_c[current_mask == 0])
                loss.backward(retain_graph=True)
                model_optim.step()
                train_loss.append(loss.item())
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch , train_steps, train_loss, vali_loss))
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch, self.args)
            end_time = time.time()
            time_s = end_time - epoch_time
            print(f"epoch {epoch}: {time_s:.4f} seconds/epoch(CPU Wall Time)")
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(vali_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        filename = f"{self.modelname}_{self.args.feature}_{self.args.loss}_{self.args.learning_rate}_{self.args.d_model}_{self.args.batch_size}_{self.args.mask_rate}.png"  # 构建文件名
        picture_dir = f"./picture/{self.modelname}"
        if not os.path.exists(picture_dir):
            os.makedirs(picture_dir)
        full_path = os.path.join(picture_dir, filename)
        plt.savefig(full_path)
        plt.show()
        self.vail_loss=np.mean(vali_losses)

        return self.model,np.mean(train_losses),np.mean(vali_losses)



    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        test_steps = len(test_loader)
        if test:
            checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            mtime = os.path.getmtime(checkpoint_path)
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        final_hash = hash_state_dict(self.model.state_dict(), device='cpu')
        print("Model Final Hash:", final_hash)
        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_stamps = []
        self.model.eval()
        test_log={}
        weights_list = []
        with torch.no_grad():
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee,mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_e = batch_x_e.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                pos_s = pos_s.float().to(self.device)
                pos_e = pos_e.float().to(self.device)
                datee = datee.float().to(self.device)
                current_mask=mask.float().to(self.device)
                outputs,atten= self.model(batch_x, batch_x_e, pos_s, pos_e, current_mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = current_mask[:, :, :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true =batch_y_c.detach().cpu().numpy()
                current_mask=current_mask.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(current_mask)

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        B,N,T,C= trues.shape
        print('test shape:', preds.shape, trues.shape,masks.shape)
        if test_data.scale and self.args.inverse:
            shape = trues.shape
            if preds.shape[-1] != trues.shape[-1]:
                preds = np.tile(preds, [1, 1, int(trues.shape[-1] / preds.shape[-1])])
            preds=preds.transpose(1, 0, 2, 3)
            preds=preds.reshape(shape[1],-1,shape[-1])
            print(preds.shape)#N BT C
            preds = test_data.inverse_transform(preds)
            preds=preds.reshape(shape[1],shape[0],shape[2],shape[3])
            preds=preds.transpose(1, 0, 2, 3)
        # result save
        folder_path = './results_show/' +self.args.feature+self.args.loss+ setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(preds[0,0,:],trues[0,0,:])
        mae, mse, rmse, smape,maape,r,r2 = metric(preds[masks == 0], trues[masks == 0])
        new_preds = preds.copy()
        new_preds[masks == 1] = trues[masks == 1]
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, smape,maape,r,r2]))
        new_preds= new_preds.transpose(1, 0, 2, 3)
        trues = trues.transpose(1, 0, 2, 3)
        masks = masks.transpose(1, 0, 2, 3)
        np.save(folder_path + 'pred.npy', new_preds)#NBTC
        np.save(folder_path + 'true.npy', trues)#NBTC
        np.save(folder_path + 'mask.npy', masks)# NBTC
        N = new_preds.shape[0]
        pred_flat = new_preds.reshape(N, -1)  # (N, B*T*C)
        true_flat = trues.reshape(N, -1)
        mask_flat = masks.reshape(N, -1)
        results = [
            metric(
                pred_flat[n][mask_flat[n] == 0],
                true_flat[n][mask_flat[n] == 0]
            )
            for n in range(N)
        ]
        df = pd.DataFrame(
            results,
            columns=['mae', 'mse', 'rmse', 'smape', 'maape', 'r', 'r2']
        )
        df.index.name = 'index'
        df.to_csv(folder_path + 'station_metrics.csv')
        print(f"saved {folder_path}station_metrics.csv")
        return mae, mse, rmse, smape,maape,r,r2

