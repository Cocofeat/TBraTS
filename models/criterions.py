import torch
import logging
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np

def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        xx[:, 1, :, :, :] = (x == 2)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = 1 - (nominator / (denominator + 1e-5))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        # changed by coco
        return dc

class softBCE_dice(nn.Module):
    def __init__(self, aggregate="sum",weight_ce=0.5, weight_dice=0.5):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(softBCE_dice, self).__init__()

        self.aggregate = aggregate
        # self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.Bce = torch.nn.BCELoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.sigmoid = nn.Sigmoid()
        self.dc = SoftDiceLoss()
    def forward(self, output, target):
        # ce_loss = self.ce(net_output, target)
        Bce_loss1 = self.Bce(self.sigmoid(output[:, 1, ...]), (target == 1).float())
        Bce_loss2 = self.Bce(self.sigmoid(output[:, 2, ...]), (target == 2).float())
        Bce_loss3 = self.Bce(self.sigmoid(output[:, 3, ...]), (target == 3).float())
        # Diceloss1 = Dice(output[:, 1, ...], (target == 1).float())
        # Diceloss2 = Dice(output[:, 2, ...], (target == 2).float())
        # Diceloss3 = Dice(output[:, 3, ...], (target == 4).float())
        Diceloss1 = self.dc(output[:, 1, ...], (target == 1).float())
        Diceloss2 = self.dc(output[:, 2, ...], (target == 2).float())
        Diceloss3 = self.dc(output[:, 3, ...], (target == 3).float())
        if self.aggregate == "sum":
            result1 = self.weight_ce * Bce_loss1 + self.weight_dice * Diceloss1
            result2 = self.weight_ce * Bce_loss2 + self.weight_dice * Diceloss2
            result3 = self.weight_ce * Bce_loss3 + self.weight_dice * Diceloss3
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)

        return result1+result2+result3, 1-result1.data, 1-result2.data, 1-result3.data

def softmaxBCE_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss torch.nn.BCELoss()
    '''
    Bce = torch.nn.BCELoss()
    Diceloss1 = Dice(output[:, 1, ...], (target == 1).float())
    Bceloss1 = Bce(output[:, 1, ...], (target == 1).float())
    loss1 = Diceloss1 + Bceloss1
    Diceloss2 = Dice(output[:, 2, ...], (target == 2).float())
    Bceloss2 = Bce(output[:, 2, ...], (target == 2).float())
    loss2 = Diceloss2 + Bceloss2
    Diceloss3 = Dice(output[:, 3, ...], (target == 3).float())
    Bceloss3 = Bce(output[:, 3, ...], (target == 3).float())
    loss3 = Diceloss3 + Bceloss3

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data

# loss function



def TDice(output, target,criterion_dl):
    dice = criterion_dl(output, target)
    return dice

def TFocal(output, target,criterion_fl):
    focal = criterion_fl(output, target)
    return focal

def focal_dce_eviloss(p, alpha, c, global_step, annealing_step):
    # dice focal loss
    criterion_dl = DiceLoss()
    L_dice =  TDice(alpha,p,criterion_dl)
    criterion_fl = FocalLoss(4)
    L_focal = TFocal(alpha, p, criterion_fl)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.view(-1, c)
    # digama loss
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    # log loss
    # labelK = label * (torch.log(S) -  torch.log(alpha))
    # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    L_KL = annealing_coef * KL(alp, c)

    return (L_ace + L_dice + L_focal + L_KL)

def dce_eviloss(p, alpha, c, global_step, annealing_step):
    criterion_dl = DiceLoss()
    # L_dice =  TDice(alpha,p,criterion_dl)
    L_dice,_,_,_ = softmax_dice(alpha, p)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.view(-1, c)
    # digama loss
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    # log loss
    # labelK = label * (torch.log(S) -  torch.log(alpha))
    # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    L_KL = annealing_coef * KL(alp, c)

    return (L_ace + L_dice + L_KL)


def dce_loss(p, alpha, c, global_step, annealing_step):
    criterion_dl = DiceLoss()
    L_dice =  TDice(alpha,p,criterion_dl)

    return L_dice
def ce_loss(p, alpha, c, global_step, annealing_step):
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))

    # 0.0 permute all
    # alpha = alpha.permute(0,2,3,4,1).view(-1, c)
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.view(-1, c)
    # S = S.permute(0, 2, 3, 4, 1)
    # alpha = alpha.permute(0, 2, 3, 4, 1)
    # label_K = label * (torch.digamma(S) - torch.digamma(alpha))
    # digama loss
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    # log loss
    # labelK = label * (torch.log(S) -  torch.log(alpha))
    # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    # label = label.permute(0, 4, 1, 2, 3)
    alp = E * (1 - label) + 1
    # alp = E.permute(0, 2, 3, 4, 1) * (1 - label) + 1
    L_KL = annealing_coef * KL(alp, c)

    return (L_ace  + L_KL)
    # return L_ace

def KL(alpha, c):
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones((1, c)).cuda()
    # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

class DiceLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets, weight=False):
        N = preds.size(0)
        C = preds.size(1)

        preds = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        if targets.size(1)==4:
            targets = targets.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        else:
            targets = targets.view(-1, 1)

        log_P = F.log_softmax(preds, dim=1)
        P = torch.exp(log_P)
        # P = F.softmax(preds, dim=1)
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)

        class_mask = torch.zeros(preds.shape).to(preds.device) + 1e-8
        class_mask.scatter_(1, targets, 1.)

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.sum(dim=(0)) / ((FP.sum(dim=(0)) + FN.sum(dim=(0))) + smooth)

        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8)
        #print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP, dim=(0)).float()
        den = num + self.alpha * torch.sum(FP, dim=(0)).float() + self.beta * torch.sum(FN, dim=(0)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss
        loss = 1 - dice
        if weight is not False:
            loss *= weight.squeeze(0)
        loss = loss.sum()
        if self.size_average:
            if weight is not False:
                loss /= weight.squeeze(0).sum()
            else:
                loss /= C

        return loss

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.ones(class_num, 1).cuda()
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.size_average = size_average

    def forward(self, preds, targets, weight=False):
        N = preds.size(0)
        C = preds.size(1)

        preds = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        targets = targets.view(-1, 1)

        log_P = F.log_softmax(preds, dim=1)
        P = torch.exp(log_P)
        # P = F.softmax(preds, dim=1)
        # log_P = F.log_softmax(preds, dim=1)
        # class_mask = torch.zeros(preds.shape).to(preds.device) + 1e-8
        class_mask = torch.zeros(preds.shape).to(preds.device)  # problem
        class_mask.scatter_(1, targets, 1.)
        # number = torch.unique(targets)
        alpha = self.alpha[targets.data.view(-1)] # problem alpha: weight of data
        # alpha = self.alpha.gather(0, targets.view(-1))

        probs = (P * class_mask).sum(1).view(-1, 1)  # problem
        log_probs = (log_P * class_mask).sum(1).view(-1, 1)

        # probs = P.gather(1,targets.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        # log_probs = log_P.gather(1,targets.view(-1,1))

        batch_loss = -alpha * (1-probs).pow(self.gamma)*log_probs
        if weight is not False:
            element_weight = weight.squeeze(0)[targets.squeeze(0)]
            batch_loss = batch_loss * element_weight

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

def softmax_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss torch.nn.BCELoss()
    '''

    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 3).float())

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data

def softmax_dice2(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss0 = Dice(output[:, 0, ...], (target == 0).float())
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 3).float())

    return loss1 + loss2 + loss3 + loss0, 1-loss1.data, 1-loss2.data, 1-loss3.data


def sigmoid_dice(output, target):
    '''
    The dice loss for using sigmoid activation function
    :param output: (b, num_class-1, d, h, w)
    :param target: (b, d, h, w)
    :return:
    '''
    loss1 = Dice(output[:, 0, ...], (target == 1).float())
    loss2 = Dice(output[:, 1, ...], (target == 2).float())
    loss3 = Dice(output[:, 2, ...], (target == 3).float())

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data


def Generalized_dice(output, target, eps=1e-5, weight_type='square'):
    if target.dim() == 4:  #(b, h, w, d)
        target[target == 4] = 3  #transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  #extend target from (b, h, w, d) to (b, c, h, w, d)

    output = flatten(output)[1:, ...]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)

    return 1 - 2. * intersect_sum / denominator_sum, loss1, loss2, loss3
