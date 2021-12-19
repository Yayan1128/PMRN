import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, precision_score,recall_score, accuracy_score


def evaluate(epoch_i,labels, scores):
    # print(labels)

    return roc(epoch_i,labels, scores)


def roc(epoch_i,labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""


    # True/False Positive Rates.
    # print(scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    # print(fpr)
    roc_auc = auc(fpr, tpr)

    # np.savetxt('./result/fpr_mae_%d.txt'% epoch_i, fpr, fmt='%.9f', delimiter=' ')
    # np.savetxt('./result/tpr_mae_%d.txt'% epoch_i, tpr, fmt='%.9f', delimiter=' ')
    # # Equal Error Rate

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f)' % (roc_auc))
        plt.plot([0, 0], [1, 1], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    return roc_auc

#
# def auprc(labels, scores):
#     """
#     Compute average precision score
#     """
#     ap = average_precision_score(labels, scores)
#     return ap