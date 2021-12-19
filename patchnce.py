# from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.nce_includes_all_negatives_from_minibatch=False
        self.batch_size=32
        self.nce_T=0.07
    def forward(self, f_q, f_k):

        # print(feat_q.size())
        # print(feat_k.size())
        b, c,w,h = f_q.shape
        f_q=f_q.view(b,128,-1)
        f_k=f_k.view(b,128,-1)
        B, C, S = f_q.shape

        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]

        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)

        # The diagonal entries are not negatives. Remove them.
        diagonal = torch.eye(S, device=f_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg.masked_fill_(diagonal, -10.0)
        # identity_matrix = torch.eye(S)[None, :, :].cuda()
        # l_neg.masked_fill_(identity_matrix, -float('inf'))

        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / self.nce_T

        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(device=f_q.device)
        loss = self.cross_entropy_loss(predictions, targets).mean()

        return loss
        # batchSize = feat_q.shape[0]
        # dim = feat_q.shape[1]
        # feat_k = feat_k.detach()
        #
        # # pos logit
        # l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        # # print(l_pos.size())
        # l_pos = l_pos.view(batchSize, 1)
        #
        # # neg logit
        #
        # # Should the negatives from the other samples of a minibatch be utilized?
        # # In CUT and FastCUT, we found that it's best to only include negatives
        # # from the same image. Therefore, we set
        # # --nce_includes_all_negatives_from_minibatch as False
        # # However, for single-image translation, the minibatch consists of
        # # crops from the "same" high-resolution image.
        # # Therefore, we will include the negatives from the entire minibatch.
        # if self.nce_includes_all_negatives_from_minibatch:
        #     # reshape features as if they are all negatives of minibatch of size 1.
        #     batch_dim_for_bmm = 1
        # else:
        #     batch_dim_for_bmm = self.batch_size
        #
        # # reshape features to batch size
        # feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        # feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        # npatches = feat_q.size(1)
        # l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        #
        # # diagonal entries are similarity between same features, and hence meaningless.
        # # just fill the diagonal with very small number, which is exp(-10) and almost zero
        # diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        # l_neg_curbatch.masked_fill_(diagonal, -10.0)
        # l_neg = l_neg_curbatch.view(-1, npatches)
        # print(l_pos.size(),l_neg.size())
        # out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        #
        # loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
        #                                                 device=feat_q.device))

        # return loss
