import json
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm

import numpy as np
import torch
from torch import nn


from dataset import BatchCollator

from evaluate import evaluate

import math



class Trainer():
    def __init__(self, cfg, dataset_train, dataset_test, model, optimizer, device, step=0):
        self.cfg = cfg
        self.image_height = cfg.image_height
        self.image_width = cfg.image_width
        self.image_channel_size = cfg.image_channel_size

        self.log_dir = cfg.log_dir
        self.device = device

        self.train_writer = SummaryWriter(logdir=os.path.join(cfg.log_dir, 'train'))
        self.valid_writer = SummaryWriter(logdir=os.path.join(cfg.log_dir, 'valid'))

        self.train_writer.add_text('cfg', json.dumps(cfg.__dict__))
        self.valid_writer.add_text('cfg', json.dumps(cfg.__dict__))

        self.batch_size = cfg.batch_size
        self.num_epochs = cfg.num_epochs

        self.model = model
        self.optimizer = optimizer



        self.rec_criterion = nn.MSELoss(reduction='sum')


        self.trainloader=dataset_train
        self.testloader= dataset_test



        self.collator = BatchCollator(image_height=self.image_height,
                                      image_width=self.image_width,
                                      image_channel_size=self.image_channel_size,
                                     )

        self.step = step
        self.best_cls_acc = 0.0
        self.best_rec_error = 1000000.0
        self.best=0.0
        self.e=0
        self.k=0.0


    def valid(self,epoch):

        scores_value = []
        m_value=[]
        target=[]
        epoch_i=epoch
        for i, batch in enumerate(self.testloader):

            self.model.eval()
            #
            batch = [b.to(self.device) for b in batch]
            imgs, labels, instances = batch[0], batch[1], batch[2]
            with torch.no_grad():
                rec_imgs,output2, re2 = self.model(imgs)
            for j in range(0, len(rec_imgs)):
                scores = self.rec_criterion(imgs[j], rec_imgs[j])
                q=output2[j].unsqueeze(0)
                k=re2[j].unsqueeze(0)
                scores_latent = self.rec_criterion(q,k)
                m_value.append(scores_latent.cpu().item())

                scores_value.append(scores.cpu().item())
                target.append(labels[j].cpu().item())


        scores_value = (scores_value - np.min(scores_value)) / (np.max(scores_value) - np.min(scores_value))

        m_value = ( m_value - np.min( m_value)) / (np.max( m_value) - np.min( m_value))

        co_img = 1 / (1 + math.exp(-200 * self.k))
        co_latent = 1 / (1 + math.exp(200 * self.k))
        score_fine = co_img * scores_value + m_value * co_latent
        score_fine = torch.tensor(score_fine)
        np.savetxt('./result/score_RSDDS_Abandoned_%d.txt' % epoch_i, score_fine, fmt='%.9f',
                   delimiter='\n')
        np.savetxt('./result/label_RSDDS_Abandoned_%d.txt' % epoch_i, target, fmt='%.9f',
                   delimiter='\n')

        AUC= evaluate(epoch_i, target, score_fine)
        best_v=np.round(AUC,3)
        if best_v>self.best:
            self.best=best_v
            self.e=epoch_i
            self._save_checkpoint(self.e)
        print( ' AUC:', np.round(AUC, 3),'best aUc:',np.round( self.best, 3),'bestepoch:', self.e)
        print('='*100)
        print('Valid')
        print()

    def train(self):

        for epoch in range(self.num_epochs):

            records = dict(loss=[],
                           rec_loss=[],
                           latent_loss=[],
                           V=[],
                          )


            for i, batch in tqdm(enumerate(self.trainloader), total=len(self.trainloader), desc='Epoch %d' % epoch):
                # print(i)
                # print(epoch)
                self.model.train()
                self.optimizer.zero_grad()

                batch = [b.to(self.device) for b in batch]
                imgs, labels, instances = batch[0], batch[1], batch[2]
                batch_size = imgs.size(0)
                rec_imgs,output2,re2 = self.model(imgs)
                rec_loss = self.rec_criterion(rec_imgs, imgs)
                latent_loss = self.rec_criterion(output2, re2)
                v = round(math.log10(rec_loss) - math.log10(latent_loss))
                all_loss=latent_loss+rec_loss
                all_loss /= batch_size

                loss = all_loss# + cls_loss + entropy_loss + condi_loss
                loss.backward()
                self.optimizer.step()

                self._update_tensorboard(loss=loss.item() )

                records['loss'] += [loss.cpu().item()]
                records['rec_loss'] += [rec_loss.cpu().item()]
                records['latent_loss'] += [latent_loss.cpu().item()]
                records['V'] += [v]

                self.step += 1
            #
            for k, v in records.items():
                records[k] = sum(records[k]) / len(records[k])

            if epoch==0:
                v1= records['V']
                self.k = 3.7-v1#RSD4.2
            print( self.k)


            print('='*100)
            print('Train')

            print()
            self.valid(epoch)

    def _update_tensorboard(self, loss):

        self.train_writer.add_scalar('02._Loss', loss, self.step)


    def _print_progress(self, loss, rec_loss, entropy_loss, condi_loss, rec_error, cls_loss, cls_acc):
        print('='*100)

        print('Loss: {loss:.4f}, Reconst loss: {rec_loss:.4f}' \
                  .format(loss=loss, rec_loss=rec_loss, end=' '))

        print()

    def _save_checkpoint(self, epoch_t):
        last_ckpt_path = os.path.join(self.log_dir, 'ckpt', '%d_model-last.ckpt' % epoch_t)
        torch.save(dict(
            cfg=self.cfg,
            step=self.step,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        ), last_ckpt_path)
