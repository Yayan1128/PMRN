from tqdm import tqdm

import torch
from torch import nn

from dataset import BatchCollator

import numpy as np
import utils2
import os
from PIL import Image
from loss import l2_loss
import math
from evaluate import evaluate
import matplotlib.pyplot as plt
from matplotlib import cm


class Tester():
    def __init__(self, cfg, dataloader,dataloader_train, model, device):
        self.cfg = cfg
        self.image_height = cfg.image_height
        self.image_width = cfg.image_width
        self.image_channel_size = cfg.image_channel_size

        self.num_dataloaders = cfg.num_dataloaders
        self.device = device

        self.batch_size = cfg.batch_size

        self.model = model

        self.cls_loss_coef = cfg.cls_loss_coef
        self.entropy_loss_coef = cfg.entropy_loss_coef
        self.condi_loss_coef = cfg.condi_loss_coef
        self.addressing = cfg.addressing
        # self.num_memories = cfg.num_memories

        self.cls_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.rec_criterion = nn.MSELoss(reduction='sum')
        self.condi_criterion = nn.BCELoss(reduction='sum')

        # if cfg.test_set == 'train':
        #     self.test_set = dataloader.train_dataset
        # else:
        self.testloader  = dataloader
        self.trainloader=dataloader_train
        self.collator = BatchCollator(image_height=self.image_height,
                                      image_width=self.image_width,
                                      image_channel_size=self.image_channel_size,
                                     )
        self.la = l2_loss
        self.k=-3

    def test(self):
        scores_value = []
        m_value = []
        target = []
        denorm = utils2.utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])  # denormalization for ori images
        records = dict(loss=[],
                       rec_loss=[],
                       entropy_loss=[],
                       condi_loss=[],
                       rec_error=[],
                       cls_loss=[],
                       cls_acc=[],
                      )

        idx = 0
        for i, batch in tqdm(enumerate(self.testloader), total=len(self.testloader), desc='Test'):
            self.model.eval()

            batch = [b.to(self.device) for b in batch]
            imgs, labels, instances = batch[0], batch[1], batch[2]

            with torch.no_grad():
                rec_imgs, output2, re2 = self.model(imgs)

                for j in range(0, len(rec_imgs)):
                        scores = self.rec_criterion(imgs[j], rec_imgs[j])
                        q = output2[j].unsqueeze(0)
                        k = re2[j].unsqueeze(0)
                        scores_latent = self.rec_criterion(q, k)
                        m_value.append(scores_latent.cpu().item())
                        scores_value.append(scores.cpu().item())
                        target.append(labels[j].cpu().item())
        scores_value = (scores_value - np.min(scores_value)) / (np.max(scores_value) - np.min(scores_value))
        m_value = (m_value - np.min(m_value)) / (np.max(m_value) - np.min(m_value))
        co_img=1/(1+math.exp(-200*self.k))
        co_latent=1/(1+math.exp(200*self.k))
        score_fine=co_img*scores_value+m_value*co_latent
        np.savetxt('./result/score_RSDDS_Abandoned.txt', score_fine, fmt='%.9f',
                       delimiter='\n')
        np.savetxt('./result/label_RSDDS_Abandoned.txt', target, fmt='%.9f',
                       delimiter='\n')

        AUC = evaluate(5, target, score_fine)  # 0.01
        print( ' AUC:', np.round(AUC, 3))

