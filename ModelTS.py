import torch
import argparse
import os
import torch.nn as nn
import time
import torch.nn.functional as F
from models.criterions import dce_eviloss
from predict import tailor_and_concat
from models.lib.VNet3D import VNet
from models.lib.UNet3DZoo import Unet,AttUnet
from models.lib.TransBTS_downsample8x_skipconnection import TransBTS
# from sklearn.preprocessing import MinMaxScaler

class TSS(nn.Module):

    def __init__(self, classes, modes, model_name,modal, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param modes: Number of modes
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TSS, self).__init__()
        # ---- Net Backbone ----
        if model_name == 'AU' and modal =='four':
            self.backbone = AttUnet(in_channels=4, base_channels=16, num_classes=classes)
        elif model_name == 'AU':
            self.backbone = AttUnet(in_channels=1, base_channels=16, num_classes=classes)
        elif model_name == 'V'and modal =='four':
            self.backbone = VNet(n_channels=4, n_classes=classes, n_filters=16, normalization='gn', has_dropout=False)
        elif model_name == 'V':
            self.backbone = VNet(n_channels=1, n_classes=classes, n_filters=16, normalization='gn', has_dropout=False)
        elif model_name =='TransU':
            _, self.backbone = TransBTS(modal=modal, _conv_repr=True, _pe_type="learned")
        elif model_name == 'U'and modal =='four':
            self.backbone = Unet(in_channels=4, base_channels=16, num_classes=classes)
        else:
            self.backbone = Unet(in_channels=1, base_channels=16, num_classes=classes)
        self.backbone.cuda()
        self.modes = modes
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.modes)])

    def forward(self, X, y, global_step, mode, use_TTA=False):
        # X data
        # y target
        # global_step : epochs

        # step zero: backbone
        if mode == 'train':
            backbone_output = self.backbone(X)
        elif mode == 'val':
            backbone_output = tailor_and_concat(X, self.backbone)
        else:
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            backbone_output = tailor_and_concat(X, self.backbone)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            print('Single sample test time consumption {:.4f} minutes!'.format(elapsed_time / 60))
                # backbone_X = F.softmax(backbone_X,dim=1)
        # step one
        evidence = self.infer(backbone_output) # batch_size * class * image_size
        # step two
        alpha = evidence + 1
        if mode == 'train':
        # step three
            loss = dce_eviloss(y.to(torch.int64), alpha, self.classes, global_step, self.lambda_epochs)
            loss = torch.mean(loss)
            return evidence, loss
        else:
            return evidence

    def infer(self, input):
        """
        :param input: modal data
        :return: evidence of modal data
        """
        # evidence = (input-torch.min(input))/(torch.max(input)-torch.min(input))
        evidence = F.softplus(input)
        # evidence[m_num] = torch.exp(torch.clamp(evidence, -10, 10))
        # evidence = F.relu(evidence)
        return evidence
