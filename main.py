import argparse
from datetime import datetime
import multiprocessing as mp
import os
from pprint import pprint

import torch
from torch import nn
from torch.optim import Adam

from model import PMRN
from train import Trainer

import torchvision.transforms as transforms

from torch.utils import data
from image_dataset import DataSet
import utils
import torch.backends.cudnn as  cudnn
import numpy as np
import random
from test import Tester


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='data/MNIST')
    parser.add_argument('--prepro-dir', type=str, default='prepro/MNIST')

    parser.add_argument('--num-instances', type=int, default=60000)
    parser.add_argument('--image-height', type=int, default=224)
    parser.add_argument('--image-width', type=int, default=224)
    parser.add_argument('--image-channel-size', type=int, default=3)

    parser.add_argument('--train', action='store_true',default=True)
    parser.add_argument('--test', action='store_true',default=False)
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--log-dir', type=str, default='./logs-test/%s' % datetime.now().strftime('%b%d_%H-%M-%S'))
    parser.add_argument('--test-set', type=str, default='train')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--num-dataloaders', type=int, default=mp.cpu_count())

    parser.add_argument('--co-w', type=int, default=0.01)#COVID DAGN-1,6: 0.5
    parser.add_argument('--lo-w', type=int, default=0.5)
    parser.add_argument('--lc', type=int, default= 0.009)#COVID 0.001, DAGN-1,6: 0.01
    parser.add_argument('--lf', type=int, default= 0.9999))#COVID 0.009, DAGN-1,6: 0.9999

    parser.add_argument('--seed', type=int, default=2019)#2019
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)#COVID:16
    parser.add_argument('--lr', type=float, default=3e-4)#COVID 2e-4, DAGN-1:5e-4,DAGN-6:1e-4,
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--cls-loss-coef', type=float, default=0.0)
    parser.add_argument('--entropy-loss-coef', type=float, default=0.0002)#0.0002
    parser.add_argument('--condi-loss-coef', type=float, default=0.0)
    parser.add_argument('--addressing', type=str, default='sparse')

    parser.add_argument('--conv-channel-size', type=int, default=64)
    parser.add_argument('--drop-rate', type=float, default=0.2)#0.2
    cfg = parser.parse_args()
    return cfg

def preprocess_dataset(_dataset,num_instances):

        dataset = []
        instance_idx = 0
        num_instances=num_instances
        for i, (img, label) in enumerate(_dataset):
            dataset.append((img, label, instance_idx))
            instance_idx += 1
            if num_instances <= instance_idx:
                break
        max_num_instances = len(dataset)
        print('The number of instances: %s' % max_num_instances)
        return dataset


def dataset_setting(config):
    """
    setting some parameters related to dataset
    :param dataset:  dataset
    :param config:   parameters
    :return:
    """
    config.dataset ='ABCRACK500' #'RSD'
    config.img_dir = '/home/wyy/PycharmProjects/ABD1/Outlier_Dataset/ABCRACK500/'
    config.tr_data_path = '/home/wyy/PycharmProjects/ABD1/Outlier_Dataset/ABCRACK500/train.txt'
    config.ts_data_path = '/home/wyy/PycharmProjects/ABD1/Outlier_Dataset/ABCRACK500/test.txt'

    # namespace ==> dictionary
    return vars(config)


def grab_data(data_path, data_label, config, is_train):
    """
    Grab image data
    :param tr_data: list, path of train image
    :param ts_data: list, path of test image
    :param ts_label: list, path of test label
    :return:
    """
    params = {'batch_size': config['batch_size'],
              'num_workers': 4,
              'pin_memory': True,
              'sampler': None}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transforms, ts_transforms = transforms.Compose([#RSD:RAN+RESIZE128
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(224),#COVID(160,280)#DAGN-1,6 NO
        transforms.ToTensor(),
        normalize,

    ]), transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    if is_train:
        params['shuffle'] = True
        params['sampler'] = None
        dataset0=DataSet(config, is_train, data_path, data_label, tr_transforms)
        dataset0=preprocess_dataset(dataset0,num_instances=1200)

        data_set = data.DataLoader(dataset0, **params,drop_last=True)
    else:
        params['shuffle'] = False
        dataset1=DataSet(config, is_train, data_path, data_label, ts_transforms)
        dataset1=preprocess_dataset(dataset1,num_instances=1061)
        data_set = data.DataLoader(dataset1, **params,drop_last=True)

    return data_set


def main(cfg):
    if not torch.cuda.is_available():
        print('CPU mode is not supported')
        exit(1)
    device = torch.device('cuda:0')
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)


    if cfg.ckpt:
        if not os.path.exists(cfg.ckpt):
            print('Invalid ckpt path -->', cfg.ckpt)
            exit(1)
        ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc:storage)

        print(cfg.ckpt, 'loaded')
        loaded_cfg = ckpt['cfg'].__dict__
        pprint(loaded_cfg)
        del loaded_cfg['train']
        del loaded_cfg['test']
        del loaded_cfg['visualize']
        del loaded_cfg['batch_size']
        del loaded_cfg['ckpt']

        cfg.__dict__.update(loaded_cfg)
        print()
        print('Merged Config')
        pprint(cfg.__dict__)
        print()

        step = ckpt['step']
    else:
        os.makedirs(os.path.join(cfg.log_dir, 'ckpt'))
        step = 0
    #
    # Dataset = ['APSD', 'RSD', 'MT', 'CSD', 'CIFAR10']
    config = dataset_setting(cfg)


    # load data
    data_set = utils.data_load(config['tr_data_path'], config['ts_data_path'])

    # grab image data
    train_set = grab_data(data_set[0], data_set[1], config, True)
    test_set = grab_data(data_set[2], data_set[3], config, False)

    model = PMRN(cfg=cfg, device=device,)
    print()
    print(model)
    print()
    # print(cfg.save_path_ab1)
    if torch.cuda.device_count() > 1 and cfg.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = Adam(params=model.parameters(),
                     lr=cfg.lr,
                     betas=(0.5, 0.999),
                    )

    if cfg.ckpt is not None:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    if cfg.train:
        trainer = Trainer(cfg=cfg,
                          dataset_train=train_set,
                          dataset_test=test_set,
                          model=model,
                          optimizer=optimizer,
                          device=device,
                          step=step,
                         )
        trainer.train()
    elif cfg.test:
        tester = Tester(cfg=cfg,
                        dataloader=test_set,
                        dataloader_train=train_set,
                        model=model,
                        device=device,
                       )
        tester.test()
    else:
        print('Select mode')
        exit(1)


if __name__ == '__main__':
    cfg = config()
    print('Config')
    pprint(cfg.__dict__)
    print()
    main(cfg)
