import pytorch_lightning as pl
import torch
import random
import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from data.data_interface import DInterface
from model.model_interface import MInterface
import config

def main(args):
    pl.seed_everything(args.seed)
    if args.mode == 'pretrain':
        os.makedirs(config.outpath, exist_ok=True)
        pretrain_meta_csv = pd.read_csv(config.TAU_meta_csv_path, index_col=False, sep='\t')
        pretrain_meta_csv = pretrain_meta_csv.sample(frac=1)
        # prepare dataloader
        data_loader = DInterface(dataset='TAU', fea_path = config.TAU_fea_root_path, csv_file = pretrain_meta_csv, **vars(args))
        max_epochs = args.pretrain_epoch
    
    if args.mode == 'mixmatch':
    # ====================prepare data======================
        os.makedirs(config.outpath, exist_ok=True)
        dev_meta_csv = pd.read_csv(config.CAS_meta_csv_path, index_col=False)
        dev_meta_csv = dev_meta_csv.sample(frac=1)
        # divide data into labeled and unlabeled
        dev_with_label = dev_meta_csv[dev_meta_csv['scene_label'].notnull()]    # len: 3480
        dev_without_label = dev_meta_csv[dev_meta_csv['scene_label'].isnull()]  # len: 13920

        # write into new csv file
        dev_with_label.to_csv(os.path.join(config.outpath, "labeled_data.csv"), index=False)
        dev_without_label.to_csv(os.path.join(config.outpath, "unlabeled_data.csv"), index=False)
        data_loader = DInterface(dataset='CAS', fea_path = config.CAS_fea_root_path,
                                csv_file = dev_with_label, unlabel_csv = dev_without_label, **vars(args))
        max_epochs = args.max_epochs
        pretrained_model_path = args.pretrained_model
    
    model = MInterface(model_name=args.model, **vars(args))
    logger = CSVLogger('./', 'logs')
    trainer = Trainer(accelerator='cuda', devices=[1], fast_dev_run=False, max_epochs=max_epochs, logger=logger)
    if args.mode == 'mixmatch':
        trainer.fit(model, data_loader, ckpt_path=pretrained_model_path)
    else:
        trainer.fit(model, data_loader)


    

if __name__=='__main__':
    parser = ArgumentParser(description="zssm")
    # Basic Training Control
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--train_val_ratio', default=0.8, type=float)

    # training mode
    parser.add_argument('--model', choices=['resnet', 'resnet_dg', 'fcnn'], type=str)
    parser.add_argument('--mode', choices=['pretrain', 'mixmatch'], type=str)
    parser.add_argument('--pretrain_epoch', default=20, type=int)

    # lr parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_step', default=2, type=int)
    parser.add_argument('--lr_gamma', default=0.9, type=float)

    # data augmentation params and tempreture sharpening params in mixmatch
    parser.add_argument('--time_mask_param', default=200, type=int)
    parser.add_argument('--freq_mask_param', default=20, type=int)
    parser.add_argument('-T', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda_u', default=75, type=int)

    # coefficient to control representation loss
    parser.add_argument('--tou', default=0.5, type=float)
    parser.add_argument('--alpha1', default=1e-7, type=float)
    parser.add_argument('--beta1', default=1, type=float)
    parser.add_argument('--gamma1', default=1e-2, type=float)

    # load pretrained model when performing mixmatch
    parser.add_argument('--pretrained_model', default='', type=str)

    parser.set_defaults(max_epochs=20, model='resnet_dg', mode='pretrain', pretrained_model='/mnt2/yyp/ICME2024/pl_test/logs/version_0/checkpoints/epoch=19-step=23040.ckpt') # 
    args = parser.parse_args()
    if args.mode == 'pretrain':
        if os.path.exists(args.pretrained_model) is not True:
            raise FileNotFoundError('Pretrained model not found.')
    main(args)