from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch
import torchaudio.transforms as T
import numpy as np

from model.fcnn import FCNN
from model.resnet_dg import ResNet_DG
from model.Resnet import ResNet
from model.utils import SemiLoss, threeclass_loss, DG_loss, random_temporal_shift, interleave, accuracy

class MInterface(pl.LightningModule):
    def __init__(self, model_name:str, lr:float, mode:str, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.args = kargs
        self.init_model()
        self.mode = mode
        self.time_mask_param = kargs['time_mask_param']
        self.freq_mask_param = kargs['freq_mask_param']
        self.T = kargs['T']
        self.alpha = kargs['alpha']
        self.alpha1 = kargs['alpha1']
        self.gamma1 = kargs['gamma1']
        self.lambda_u = kargs['lambda_u']
        self.configure_loss()

    def init_model(self):
        if self.model_name == 'resnet':
            self.model = ResNet(num_classes=10, num_filters=48, num_res_blocks=2)
        if self.model_name == 'resnet_dg':
            self.model = ResNet_DG(num_classes=10, num_filters=48, num_res_blocks=2)
        if self.model_name == 'fcnn':
            self.model = FCNN(num_classes=10, input_shape=[1, 3, 128, 423], num_filters=[48, 96, 192])

    def forward(self, x):
        return self.model(x)['logits']

    def configure_loss(self):
        self.loss_function = {}
        self.loss_function['3class']=threeclass_loss
        self.loss_function['semi']=SemiLoss()
        self.loss_function['ce']=nn.CrossEntropyLoss()
        self.loss_function['DG']=DG_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=self.args['lr_step'], gamma=self.args['lr_gamma'])
        return [optimizer], [scheduler]
        
    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        # pretraining stage, data forms as [features, labels]
        if self.mode == 'pretrain':
            x, y = batch
            batch_size = x.size(0)
            tmp = torch.zeros(batch_size, 10).cuda()
            y = tmp.scatter_(1, y.view(-1,1).long(), 1)
            prediction = self.model(x)['logits']
            loss = self.loss_function['ce'](prediction, y)
            self.log("pretrain_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss
            
        # semi-supervised training stage, data forms as {'label':[features, labels], 'unlabel': features}
        if self.mode == 'mixmatch':
            iteration = len(self.trainer.train_dataloader['unlabel'])
            inputs_x, label_x = batch['label']
            inputs_u = batch['unlabel']
            batch_size = inputs_x.size(0)
            tmp = torch.zeros(batch_size, 10).cuda()
            targets_x = tmp.scatter_(1, label_x.view(-1,1).long(), 1)
            time_masking = T.TimeMasking(time_mask_param=self.time_mask_param)
            freq_masking = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)
            u1 = time_masking(inputs_u)
            u2 = freq_masking(inputs_u)
            a = torch.rand(1)
            u3 = random_temporal_shift(inputs_u, a[0])
            with torch.no_grad():
                outputs_u1 = self.model(u1)['logits']
                outputs_u2 = self.model(u2)['logits']
                outputs_u3 = self.model(u3)['logits']
                p = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1) + torch.softmax(outputs_u3, dim=1)) / 3
                pt = p**(1/self.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()
            
            all_inputs = torch.cat([inputs_x, u1, u2, u3], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u, targets_u], dim=0)

            l = np.random.beta(self.alpha, self.alpha)
            l = max(l, 1-l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)
            logits = [self.model(mixed_input[0])['logits']]
            for input in mixed_input[1:]:
                logits.append(self.model(input)['logits'])

            # put interleaved samples back
            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)
            Lx, Lu, w = self.loss_function['semi'](logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/iteration, self.lambda_u, self.trainer.max_epochs)

            class3_loss = self.loss_function['3class'](mixed_target[:batch_size], logits_x, self.loss_function['ce'])
            if self.model_name == 'resnet_dg':
                z = self.model(inputs_x)['latent']
                CID_loss, L_decorre, L_uniform = self.loss_function['DG'](z, label_x)
                L_decorre = self.alpha1 * L_decorre
                # L_uniform = config.beta1 * L_uniform
                CID_loss = self.gamma1 * CID_loss
                loss = Lx + w * Lu + class3_loss + CID_loss + L_decorre
                log_info = {
                    'loss': loss,
                    'loss_x': Lx,
                    'loss_u': Lu,
                    'L3class':class3_loss,
                    'L_cid': CID_loss,
                    'L_dec': L_decorre 
                }
            else:
                loss = Lx + w * Lu + class3_loss
                log_info = {
                    'loss': loss,
                    'loss_x': Lx,
                    'loss_u': Lu,
                    'L3class':class3_loss
                }
            self.log_dict(log_info, on_step=False, prog_bar=True, on_epoch=True, logger=True)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)['logits']
        loss = self.loss_function['ce'](outputs, y)
        prec1, prec5 = accuracy(outputs, y, topk=(1, 3))
        # log
        log_info = {
            'loss': loss,
            'prec1': prec1,
            'prec5': prec5
        }
        self.log_dict(log_info, prog_bar=True, logger=True, on_epoch=True, on_step=False)
