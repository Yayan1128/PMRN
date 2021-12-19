import argparse
from datetime import datetime
import multiprocessing as mp
import os
from pprint import pprint

import numpy as np
import random
import torch
from torch import nn
from torch.optim import Adam

from dataset import MNIST_Dataset
from model import ICVAE
from train import Trainer
from test import Tester
from visualize import Visualizer
import torchvision.transforms as transforms
from erasing import RandomErasing
from torch.utils import data
from image_dataset import DataSet
import utils
import torch.backends.cudnn as  cudnn
# import torch
import numpy as np
import random

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='data/MNIST')
    parser.add_argument('--prepro-dir', type=str, default='prepro/MNIST')

    parser.add_argument('--num-instances', type=int, default=60000)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--num-memories', type=int, default=500)
    parser.add_argument('--image-height', type=int, default=64)
    parser.add_argument('--image-width', type=int, default=64)
    parser.add_argument('--image-channel-size', type=int, default=3)

    parser.add_argument('--train', action='store_true',default=False)
    parser.add_argument('--test', action='store_true',default=True)
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--log-dir', type=str, default='./logs-test/%s' % datetime.now().strftime('%b%d_%H-%M-%S'))
    parser.add_argument('--test-set', type=str, default='test')
    parser.add_argument('--ckpt', default='/home/wyy/PycharmProjects/ABD1/3pud/logs-test/Mar13_21-49-25/ckpt/29_model-last.ckpt')
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--num-dataloaders', type=int, default=mp.cpu_count())
    parser.add_argument('--save_path', default='/home/wyy/PycharmProjects/ABD1/3pud/results/0')
    parser.add_argument('--save_path_ab', default='/home/wyy/PycharmProjects/ABD1/3pud/results/1')



    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
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

    config.dataset = 'PIDcrack'  # 'RSD'
    config.img_dir = '/home/wyy/PycharmProjects/ABD1/Outlier_Dataset/PIDcrack/'
    config.tr_data_path = '/home/wyy/PycharmProjects/ABD1/Outlier_Dataset/PIDcrack/train.txt'
    config.ts_data_path = '/home/wyy/PycharmProjects/ABD1/Outlier_Dataset/PIDcrack/test.txt'

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

    tr_transforms, ts_transforms = transforms.Compose([
        # transforms.Resize(224),
        # transforms.Resize(160),
        # # transforms.RandomHorizontalFlip(),
        #transforms.CenterCrop(224),
        # transforms.RandomAffine(degrees=30),
        transforms.ToTensor(),
        normalize,
        # RandomErasing(),

    ]), transforms.Compose([
        # transforms.Resize(160),
        transforms.ToTensor(),
        normalize,
    ])

    # if is_train:
    params['shuffle'] = True
    params['sampler'] = None
    dataset0=DataSet(config, is_train, data_path, data_label, tr_transforms)
        # print(len(test_set))
    dataset0=preprocess_dataset(dataset0,num_instances=1200)

    data_set = data.DataLoader(dataset0, **params,drop_last=True)
        # print(len(data_set))
    # else:
    #     params['shuffle'] = False
    #     dataset1=DataSet(config, is_train, data_path, data_label, ts_transforms)
    #     dataset1=preprocess_dataset(dataset1,num_instances=1061)
    #     # print(len(dataset1))
    #     data_set = data.DataLoader(dataset1, **params,drop_last=True)

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
    # torch.manual_seed(cfg.seed)
    # torch.cuda.manual_seed_all(cfg.seed)
    # np.random.seed(cfg.seed)
    # random.seed(cfg.seed)

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
    # print(Dataset[1])
    config = dataset_setting( cfg)

    # config = fix_seeds(config)

    # load data
    data_set = utils.data_load(config['tr_data_path'], config['ts_data_path'])

    # grab image data
    train_set = grab_data(data_set[0], data_set[1], config, True)
    test_set = grab_data(data_set[2], data_set[3], config, False)



    # dataloader = MNIST_Dataset(cfg=cfg,)
    model = ICVAE(cfg=cfg, device=device,)
    print()
    print(model)
    print()

    if torch.cuda.device_count() > 1 and cfg.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = Adam(params=model.parameters(),
                     lr=cfg.lr,
                     # betas=(0.5, 0.999)
                     # weight_decay=cfg.weight_decay,
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
