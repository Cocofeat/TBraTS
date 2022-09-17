import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import SimpleITK as sitk
import cv2
import math
from medpy.metric import binary
from sklearn.externals import joblib
from binary import assd
cudnn.benchmark = True
# import models.criterions as U_entropy
import numpy as np
import nibabel as nib
import imageio
from test_uncertainty import ece_binary,UncertaintyAndCorrectionEvalNumpy,Normalized_U

def cal_ueo(to_evaluate,thresholds):
    UEO = []
    for threshold in thresholds:
        results = dict()
        metric = UncertaintyAndCorrectionEvalNumpy(threshold)
        metric(to_evaluate,results)
        ueo = results['corrected_add_dice']
        UEO.append(ueo)
    max_UEO = max(UEO)
    return max_UEO

def cal_ece(logits,targets):
    # ece_total = 0
    logit = logits
    target = targets
    pred = F.softmax(logit, dim=0)
    pc = pred.cpu().detach().numpy()
    pc = pc.argmax(0)
    ece = ece_binary(pc, target)
    return ece

def cal_ece_our(preds,targets):
    # ece_total = 0
    target = targets
    pc = preds.cpu().detach().numpy()
    ece = ece_binary(pc, target)
    return ece

def Uentropy(logits,c):
    # c = 4
    # logits = torch.randn(1, 4, 240, 240,155).cuda()
    pc = F.softmax(logits, dim=1)  # 1 4 240 240 155
    logits = F.log_softmax(logits, dim=1)  # 1 4 240 240 155
    u_all = -pc * logits / math.log(c)
    NU = torch.sum(u_all[:,1:u_all.shape[1],:,:], dim=1)
    return NU

def Uentropy_our(logits,c):
    # c = 4
    # logits = torch.randn(1, 4, 240, 240,155).cuda()
    pc = logits  # 1 4 240 240 155
    logpc = torch.log(logits)  # 1 4 240 240 155
    # u_all = -pc * logpc / c
    u_all = -pc * logpc / math.log(c)
    # max_u = torch.max(u_all)
    # min_u = torch.min(u_all)
    # NU1 = torch.sum(u_all, dim=1)
    # k = u_all.shape[1]
    # NU2 = torch.sum(u_all[:, 0:u_all.shape[1]-1, :, :], dim=1)
    NU = torch.sum(u_all[:,1:u_all.shape[1],:,:], dim=1)
    return NU

def one_hot(ori, classes):

    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = (ori == j).nonzero()

        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1

    return new_gd.float()

def tailor_and_concat(x, model):
    temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    if x.shape[1] == 1:
        y = torch.cat((x.clone(), x.clone(), x.clone(), x.clone()), 1)
    elif x.shape[1] == 4:
        y = x.clone()
    else:
        y = torch.cat((x.clone(), x.clone()), 1)

    for i in range(len(temp)):
        temp[i] = model(temp[i])
    # .squeeze(0)
    # l= temp[0].unsqueeze(0)
    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y[..., :155]

def hausdorff_distance(lT,lP):
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    return hausdorffcomputer.GetAverageHausdorffDistance()#hausdorffcomputer.GetHausdorffDistance()

def hd_score(o,t, eps=1e-8):
    if (o.sum()==0) | (t.sum()==0):
        hd = eps,
    else:
        #ret += hausdorff_distance(wt_mask, wt_pb),
        hd = binary.hd95(o, t, voxelspacing=None),

    return hd

def dice_score(o, t, eps=1e-8):
    if (o.sum()==0) | (t.sum()==0):
        dice = eps
    else:
        num = 2*(o*t).sum() + eps
        den = o.sum() + t.sum() + eps
        dice = num/den
    return dice


def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den

def assd_score(o, t):
    s = assd(o, t)
    return s

def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==3)))
    return mIOU_score

def softmax_output_hd(output, target):
    ret = []

    # whole (label: 1 ,2 ,3)
    o = output > 0; t = target > 0 # ce
    ret += hd_score(o, t),
    # core (tumor core 1 and 3)
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    ret += hd_score(o, t),
    # active (enhanccing tumor region 1 )# 3
    o = (output == 3);t = (target == 3)
    ret += hd_score(o, t),

    return ret

def softmax_output_assd(output, target):
    ret = []

    # whole (label: 1 ,2 ,3)
    wt_o = output > 0; wt_t = target > 0 # ce
    ret += assd_score(wt_o, wt_t),
    # core (tumor core 1 and 3)
    tc_o = (output == 1) | (output == 3)
    tc_t = (target == 1) | (target == 3)
    ret += assd_score(tc_o, tc_t),
    # active (enhanccing tumor region 1 )# 3
    et_o = (output == 3);et_t = (target == 3)
    ret += assd_score(et_o, et_t),

    return ret

def softmax_output_dice(output, target):
    ret = []

    # whole (label: 1 ,2 ,3)
    o = output > 0; t = target > 0 # ce
    # print(o.shape)
    # print(t.shape)
    ret += dice_score(o, t),
    # core (tumor core 1 and 3)
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    ret += dice_score(o, t),
    # active (enhanccing tumor region 1 )# 3
    o = (output == 3);t = (target == 3)
    ret += dice_score(o, t),

    return ret


keys = 'whole', 'core', 'enhancing', 'loss'

def validate_softmax(
        save_dir,
        best_dice,
        current_epoch,
        end_epoch,
        save_freq,
        valid_loader,
        model,
        multimodel,
        Net_name,
        names=None,# The names of the patients orderly!
        ):

    H, W, T = 240, 240, 160

    runtimes = []
    dice_total = 0
    iou_total = 0
    num = len(valid_loader)

    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        x, target = data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        # target = target.to(device)
        target = torch.squeeze(target).cpu().numpy()
        torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
        start_time = time.time()
        logit = tailor_and_concat(x, model)

        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
        runtimes.append(elapsed_time)

        output = F.softmax(logit, dim=1)
        output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        output = output.argmax(0)
        iou_res = softmax_mIOU_score(output, target[:, :, :155])
        dice_res = softmax_output_dice(output,target[:,:,:155])
        # hd_res = softmax_output_hd(output, target[:, :, :155])
        dice_total += dice_res[1]
        iou_total += iou_res[1]
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)
        print(msg)
        print('current_dice:{}'.format(dice_res))
        # print('current_dice:{},hd_res:{}'.format(dice_res,hd_res))
    aver_dice = dice_total / num
    aver_iou = iou_total / num
    if (current_epoch + 1) % int(save_freq) == 0:
        if aver_dice > best_dice\
            or (current_epoch + 1) % int(end_epoch - 1) == 0 \
            or (current_epoch + 1) % int(end_epoch - 2) == 0 \
            or (current_epoch + 1) % int(end_epoch - 3) == 0:
            print('aver_dice:{} > best_dice:{}'.format(aver_dice, best_dice))
            logging.info('aver_dice:{} > best_dice:{}'.format(aver_dice, best_dice))
            logging.info('===========>save best model!')
            best_dice = aver_dice
            print('===========>save best model!')
            file_name = os.path.join(save_dir, Net_name +'_' + multimodel + '_epoch_{}.pth'.format(current_epoch))
            torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
                },
                file_name)
    print('runtimes:', sum(runtimes)/len(runtimes))

    return best_dice,aver_dice,aver_iou

def test_softmax(
        test_loader,
        model,
        multimodel,
        Net_name,
        Variance,
        load_file,
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly!
        verbose=False,
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        # snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        # visual='',  # the path to save visualization
        ):

    H, W, T = 240, 240, 155
    # model.eval()

    runtimes = []
    dice_total_WT = 0
    dice_total_TC = 0
    dice_total_ET = 0
    hd_total_WT = 0
    hd_total_TC = 0
    hd_total_ET = 0
    assd_total_WT = 0
    assd_total_TC = 0
    assd_total_ET = 0

    noise_dice_total_WT = 0
    noise_dice_total_TC = 0
    noise_dice_total_ET = 0
    noise_hd_total_WT = 0
    noise_hd_total_TC = 0
    noise_hd_total_ET = 0
    noise_assd_total_WT = 0
    noise_assd_total_TC = 0
    noise_assd_total_ET = 0
    mean_uncertainty_total = 0
    noise_mean_uncertainty_total = 0
    certainty_total = 0
    noise_certainty_total = 0
    num = len(test_loader)
    mne_total = 0
    noise_mne_total = 0
    ece_total = 0
    noise_ece_total = 0
    ece = 0
    noise_ece = 0
    ueo_total = 0
    noise_ueo_total = 0
    for i, data in enumerate(test_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i+1, len(test_loader))
        x, target = data
        # noise_m = torch.randn_like(x) * Variance
        # noise = torch.clamp(torch.randn_like(x) * Variance, -Variance * 2, Variance * 2)
        # noise = torch.clamp(torch.randn_like(x) * Variance, -Variance, Variance)
        # noise = torch.clamp(torch.randn_like(x) * Variance)
        # noised_x = x + noise_m

        noise_m = torch.randn_like(x) * Variance
        noised_x = x + noise_m
        # if multimodel=='both':
        #     noised_x[:, 0, ...] = x[:, 0, ...]
        x.cuda()
        noised_x.cuda()
        target = torch.squeeze(target).cpu().numpy()
        mean_uncertainty = torch.zeros(0)
        noised_mean_uncertainty = torch.zeros(0)
        # output = np.zeros((4, x.shape[2], x.shape[3], 155),dtype='float32')
        # noised_output = np.zeros((4, x.shape[2], x.shape[3], 155),dtype='float32')
        pc = np.zeros((x.shape[2], x.shape[3], 155),dtype='float32')
        noised_pc = np.zeros((x.shape[2], x.shape[3], 155),dtype='float32')
        if not use_TTA:
            # torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            # start_time = time.time()
            model_len = 2.0  # two modality or four modalityv
            if Net_name =='Udrop':
                T_drop = 2
                uncertainty = torch.zeros(1, x.shape[2], x.shape[3], 155)
                noised_uncertainty = torch.zeros(1, x.shape[2], x.shape[3], 155)
                for j in range(T_drop):
                    print('dropout time:{}'.format(j))
                    logit = tailor_and_concat(x, model)  # 1 4 240 240 155
                    logit_noise = tailor_and_concat(noised_x, model)  # 1 4 240 240 155
                    uncertainty += Uentropy(logit, 4)
                    noised_uncertainty += Uentropy(logit_noise, 4)
                    logit = F.softmax(logit, dim=1)
                    output = logit / model_len
                    output = output[0, :, :H, :W, :T].cpu().detach().numpy()
                    pc += output.argmax(0)
                    # for noise
                    logit_noise = F.softmax(logit_noise, dim=1)
                    noised_output = logit_noise / model_len
                    noised_output = noised_output[0, :, :H, :W, :T].cpu().detach().numpy()
                    noised_pc += noised_output.argmax(0)
                pc = pc / T_drop
                noised_pc = noised_pc / T_drop
                uncertainty = torch.squeeze(uncertainty) / T_drop
                noised_uncertainty = torch.squeeze(noised_uncertainty) / T_drop

                # logit = logit/T_drop
                # logit_noise= logit_noise/T_drop
                # Udropout_uncertainty=joblib.load('Udropout_uncertainty.pkl')
            else:
                logit = tailor_and_concat(x, model)
                logit = F.softmax(logit, dim=1)
                output = logit / model_len
                output = output[0, :, :H, :W, :T].cpu().detach().numpy()
                pc = output.argmax(0)
                # for input noise
                logit_noise = tailor_and_concat(noised_x, model)
                logit_noise = F.softmax(logit_noise, dim=1)
                noised_output = logit_noise / model_len
                noised_output = noised_output[0, :, :H, :W, :T].cpu().detach().numpy()
                noised_pc = noised_output.argmax(0)
                uncertainty = Uentropy(logit, 4)
                noised_uncertainty = Uentropy(logit_noise, 4)
            mean_uncertainty = torch.mean(uncertainty)
            noised_mean_uncertainty = torch.mean(noised_uncertainty)
            print('current_mean_uncertainty:{} ; current_noised_mean_uncertainty:{}'.format(mean_uncertainty, noised_mean_uncertainty))
            if Net_name == 'Udrop':
                joblib.dump({'pc': pc,
                         'noised_pc': noised_pc,'noised_uncertainty': noised_uncertainty,
                         'uncertainty': uncertainty}, 'Udropout_uncertainty_{}.pkl'.format(i))

            # Udropout_uncertainty = joblib.load('Udropout_uncertainty.pkl')

            # lnear = F.softplus(logit)
            # torch.cuda.synchronize()
            # elapsed_time = time.time() - start_time
            # logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time/60))
            # runtimes.append(elapsed_time)


            # if multimodel == 'both':
            #     model_len = 2.0 # two modality or four modality
            #     # logit = F.softmax(logit, dim=1)
            #     # output = logit / model_len
            #
            #     # for noise
            #     logit_noise = F.softmax(logit_noise, dim=1)
            #     noised_output = logit_noise / model_len

                # load_file1 = load_file.replace('7998', '7996')
                # if os.path.isfile(load_file1):
                #     checkpoint = torch.load(load_file1)
                #     model.load_state_dict(checkpoint['state_dict'])
                #     print('Successfully load checkpoint {}'.format(load_file1))
                #     logit = tailor_and_concat(x, model)
                #     logit = F.softmax(logit, dim=1)
                #     output += logit / model_len
                # load_file1 = load_file.replace('7998', '7997')
                # if os.path.isfile(load_file1):
                #     checkpoint = torch.load(load_file1)
                #     model.load_state_dict(checkpoint['state_dict'])
                #     print('Successfully load checkpoint {}'.format(load_file1))
                #     logit = tailor_and_concat(x, model)
                #     logit = F.softmax(logit, dim=1)
                #     output += logit / model_len
                # load_file1 = load_file.replace('7998', '7999')
                # if os.path.isfile(load_file1):
                #     checkpoint = torch.load(load_file1)
                #     model.load_state_dict(checkpoint['state_dict'])
                #     print('Successfully load checkpoint {}'.format(load_file1))
                #     logit = tailor_and_concat(x, model)
                #     logit = F.softmax(logit, dim=1)
                #     output += logit / model_len
            # else:
            #     # output = F.softmax(logit, dim=1)
            #     noised_output = F.softmax(logit_noise, dim=1)
        else:
            # x = x[..., :155]
            logit = F.softmax(tailor_and_concat(x, model), 1)  # no flip
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2,)), model).flip(dims=(2,)), 1)  # flip H
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3,)), model).flip(dims=(3,)), 1)  # flip W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(4,)), model).flip(dims=(4,)), 1)  # flip D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3)), model).flip(dims=(2, 3)), 1)  # flip H, W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 4)), model).flip(dims=(2, 4)), 1)  # flip H, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3, 4)), model).flip(dims=(3, 4)), 1)  # flip W, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3, 4)), model).flip(dims=(2, 3, 4)), 1)  # flip H, W, D
            # for noise x
            noised_x = noised_x[..., :155]
            noised_logit = F.softmax(tailor_and_concat(noised_x, model), 1)  # no flip
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(2,)), model).flip(dims=(2,)), 1)  # flip H
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(3,)), model).flip(dims=(3,)), 1)  # flip W
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(4,)), model).flip(dims=(4,)), 1)  # flip D
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(2, 3)), model).flip(dims=(2, 3)), 1)  # flip H, W
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(2, 4)), model).flip(dims=(2, 4)), 1)  # flip H, D
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(3, 4)), model).flip(dims=(3, 4)), 1)  # flip W, D
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(2, 3, 4)), model).flip(dims=(2, 3, 4)), 1)  # flip H, W, D
            output = logit / 8.0  # mean
            noised_output = noised_logit / 8.0  # mean
            uncertainty = Uentropy(output, 4)
            noised_uncertainty = Uentropy(noised_output, 4)
            output = output[0, :, :H, :W, :T].cpu().detach().numpy()
            pc = output.argmax(0)
            noised_output = noised_output[0, :, :H, :W, :T].cpu().detach().numpy()
            noised_pc = noised_output.argmax(0)
            mean_uncertainty = torch.mean(uncertainty)
            noised_mean_uncertainty = torch.mean(noised_uncertainty)
        output = pc
        noised_output = noised_pc
        U_output = uncertainty.cpu().detach().numpy()
        NU_output = noised_uncertainty.cpu().detach().numpy()
        certainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
        noise_certainty_total += noised_mean_uncertainty  # noised_mix_uncertainty noised_mean_uncertainty noised_mean_uncertainty_succ
        # ece
        ece_total += ece
        noise_ece_total += noise_ece
        # ueo
        # target = torch.squeeze(target).cpu().numpy()
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        to_evaluate = dict()
        to_evaluate['target'] = target[:, :, :155]
        u = torch.squeeze(uncertainty)
        U = u.cpu().detach().numpy()
        to_evaluate['prediction'] = output
        to_evaluate['uncertainty'] = U
        UEO = cal_ueo(to_evaluate, thresholds)
        ueo_total += UEO
        noise_to_evaluate = dict()
        noise_to_evaluate['target'] = target[:, :, :155]
        noise_u = torch.squeeze(noised_uncertainty)
        noise_U = noise_u.cpu().detach().numpy()
        noise_to_evaluate['prediction'] = noised_output
        noise_to_evaluate['uncertainty'] = noise_U
        noise_UEO = cal_ueo(noise_to_evaluate, thresholds)
        print('current_UEO:{};current_noise_UEO:{}; current_num:{}'.format(UEO, noise_UEO, i))
        noise_ueo_total += noise_UEO
        # print(output.shape)
        # print(target.shape)
        # output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        # output = output.argmax(0)
        # print(output.shape)
        # iou_res = softmax_mIOU_score(pc, target[:, :, :155])
        hd_res = softmax_output_hd(pc, target[:, :, :155])
        dice_res = softmax_output_dice(pc,target[:,:,:155])
        assd_res = softmax_output_assd(pc,target[:, :, :155])
        dice_total_WT += dice_res[0]
        dice_total_TC += dice_res[1]
        dice_total_ET += dice_res[2]
        hd_total_WT += hd_res[0][0]
        hd_total_TC += hd_res[1][0]
        hd_total_ET += hd_res[2][0]
        assd_total_WT += assd_res[0]
        assd_total_TC += assd_res[1]
        assd_total_ET += assd_res[2]

        # for noise_x
        noised_output = noised_pc
        # noise_iou_res = softmax_mIOU_score(noised_pc, target[:, :, :155])
        noise_hd_res = softmax_output_hd(noised_pc, target[:, :, :155])
        noise_dice_res = softmax_output_dice(noised_pc,target[:,:,:155])
        noised_assd_res = softmax_output_assd(noised_pc,
                                              target[:, :, :155])
        noise_dice_total_WT += noise_dice_res[0]
        noise_dice_total_TC += noise_dice_res[1]
        noise_dice_total_ET += noise_dice_res[2]
        noise_hd_total_WT += noise_hd_res[0][0]
        noise_hd_total_TC += noise_hd_res[1][0]
        noise_hd_total_ET += noise_hd_res[2][0]
        noise_assd_total_WT += noised_assd_res[0]
        noise_assd_total_TC += noised_assd_res[1]
        noise_assd_total_ET += noised_assd_res[2]
        mean_uncertainty_total += mean_uncertainty
        noise_mean_uncertainty_total += noised_mean_uncertainty
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        print(msg)
        snapshot= False # True
        if snapshot:
            """ --- grey figure---"""
            # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
            # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
            # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
            # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
            """ --- colorful figure--- """
            Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.float32)
            # K = [np.where(output[0,:,:,:] == 1)]
            Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
            Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
            Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255

            Noise_Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.float32)
            # K = [np.where(output[0,:,:,:] == 1)]
            Noise_Snapshot_img[:, :, 0, :][np.where(noised_output == 1)] = 255
            Noise_Snapshot_img[:, :, 1, :][np.where(noised_output == 2)] = 255
            Noise_Snapshot_img[:, :, 2, :][np.where(noised_output == 3)] = 255
            # target_img = np.zeros(shape=(H, W, 3, T), dtype=np.float32)
            # K = [np.where(output[0,:,:,:] == 1)]
            # target_img[:, :, 0, :][np.where(Otarget == 1)] = 255
            # target_img[:, :, 1, :][np.where(Otarget == 2)] = 255
            # target_img[:, :, 2, :][np.where(Otarget == 3)] = 255

            for frame in range(T):
                if not os.path.exists(os.path.join(savepath, str(Net_name), str(Variance), name)):
                    os.makedirs(os.path.join(savepath, str(Net_name), str(Variance), name))

                # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                imageio.imwrite(os.path.join(savepath, str(Net_name), str(Variance), name, str(frame) + '.png'),
                                Snapshot_img[:, :, :, frame])
                imageio.imwrite(os.path.join(savepath, str(Net_name), str(Variance), name, str(frame) + '_noised.png'),
                                Noise_Snapshot_img[:, :, :, frame])
                # imageio.imwrite(os.path.join(savepath, str(Net_name), name, str(frame) + '_gt.png'),
                #                 target_img[:, :, :, frame])
                # im0 = Image.fromarray(U_output[:, :, frame])
                # im1 = Image.fromarray(U_output[:, :, frame])
                # im2 = Image.fromarray(U_output[:, :, frame])
                # im0 = im0.convert('RGB')
                # im1 = im1.convert('RGB')
                # im2 = im2.convert('RGB')
                # im0.save(os.path.join(savepath, name, str(frame) + '_uncertainty.png'))
                # im1.save(os.path.join(savepath, name, str(frame) + '_input_T1.png'))
                # im2.save(os.path.join(savepath, name, str(frame) + '_input_T2.png'))
                # U_CV = cv2.cvtColor(U_output[:, :, frame], cv2.COLOR_GRAY2BGR)
                # U_heatmap = cv2.applyColorMap(U_CV, cv2.COLORMAP_JET)
                # cv2.imwrite(os.path.join(savepath, name, str(frame) + '_uncertainty.png'),
                #                 U_heatmap)
                # NU_CV = cv2.cvtColor(NU_output[:, :, frame], cv2.COLOR_GRAY2BGR)
                # NU_heatmap = cv2.applyColorMap(NU_CV, cv2.COLORMAP_JET)
                # cv2.imwrite(os.path.join(savepath, name, str(frame) + '_noised_uncertainty.png'),
                #                 NU_heatmap)
                imageio.imwrite(os.path.join(savepath, str(Net_name), str(Variance), name, str(frame) + '_uncertainty.png'),
                                U_output[:, :, frame])
                imageio.imwrite(
                    os.path.join(savepath, str(Net_name), str(Variance), name, str(frame) + '_noised_uncertainty.png'),
                    NU_output[:, :, frame])
                U_img = cv2.imread(os.path.join(savepath, str(Net_name), str(Variance), name, str(frame) + '_uncertainty.png'))
                U_heatmap = cv2.applyColorMap(U_img, cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(savepath, str(Net_name), str(Variance), name, str(frame) + '_colormap_uncertainty.png'),
                    U_heatmap)
                NU_img = cv2.imread(
                    os.path.join(savepath, str(Net_name), str(Variance), name, str(frame) + '_noised_uncertainty.png'))
                NU_heatmap = cv2.applyColorMap(NU_img, cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(savepath, str(Net_name), str(Variance), name, str(frame) + '_colormap_noised_uncertainty.png'),
                    NU_heatmap)
    aver_ueo = ueo_total / num
    aver_noise_ueo = noise_ueo_total/num
    aver_certainty = certainty_total  / num
    aver_noise_certainty = noise_certainty_total / num
    aver_dice_WT = dice_total_WT / num
    aver_dice_TC = dice_total_TC / num
    aver_dice_ET = dice_total_ET / num
    aver_hd_WT = hd_total_WT / num
    aver_hd_TC = hd_total_TC / num
    aver_hd_ET = hd_total_ET / num
    aver_noise_dice_WT = noise_dice_total_WT / num
    aver_noise_dice_TC = noise_dice_total_TC / num
    aver_noise_dice_ET = noise_dice_total_ET / num
    aver_noise_hd_WT = noise_hd_total_WT / num
    aver_noise_hd_TC = noise_hd_total_TC / num
    aver_noise_hd_ET = noise_hd_total_ET / num
    aver_assd_WT = assd_total_WT / num
    aver_assd_TC = assd_total_TC / num
    aver_assd_ET = assd_total_ET / num
    aver_noise_assd_WT = noise_assd_total_WT / num
    aver_noise_assd_TC = noise_assd_total_TC / num
    aver_noise_assd_ET = noise_assd_total_ET / num
    aver_noise_mean_uncertainty = noise_mean_uncertainty_total/num
    aver_mean_uncertainty = mean_uncertainty_total/num
    print('aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice_WT*100,aver_dice_TC*100, aver_dice_ET*100))
    print('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (aver_noise_dice_WT*100, aver_noise_dice_TC*100, aver_noise_dice_ET*100))
    print('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (aver_hd_WT,aver_hd_TC, aver_hd_ET))
    print('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f,aver_noise_hd_ET = %f' % (aver_noise_hd_WT, aver_noise_hd_TC, aver_noise_hd_ET))
    print('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd_WT, aver_assd_TC, aver_assd_ET))
    print('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f,aver_noise_assd_ET = %f' % (
        aver_noise_assd_WT, aver_noise_assd_TC, aver_noise_assd_ET))
    # print('aver_noise_mean_uncertainty=%f,aver_mean_uncertainty = %f' % (aver_noise_mean_uncertainty,aver_mean_uncertainty))
    print('aver_noise_mean_uncertainty=%f,aver_mean_uncertainty = %f' % (aver_noise_mean_uncertainty,aver_mean_uncertainty))
    print('aver_certainty=%f,aver_noise_certainty = %f' % (aver_certainty, aver_noise_certainty))
    # print('aver_mne=%f,aver_noise_mne = %f' % (aver_mne, aver_noise_mne))
    # print('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
    print('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
    logging.info(
        'aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice_WT*100, aver_dice_TC*100, aver_dice_ET*100))
    logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (
        aver_noise_dice_WT*100, aver_noise_dice_TC*100, aver_noise_dice_ET*100))
    logging.info('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (
        aver_hd_WT, aver_hd_TC, aver_hd_ET))
    logging.info('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f,aver_noise_hd_ET = %f' % (
        aver_noise_hd_WT, aver_noise_hd_TC, aver_noise_hd_ET))
    logging.info('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd_WT, aver_assd_TC, aver_assd_ET))
    logging.info('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f,aver_noise_assd_ET = %f' % (
        aver_noise_assd_WT, aver_noise_assd_TC, aver_noise_assd_ET))
    logging.info('aver_noise_mean_uncertainty=%f,aver_mean_uncertainty = %f' % (
    aver_noise_mean_uncertainty, aver_mean_uncertainty))
    logging.info('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
    # return [aver_dice_WT,aver_dice_TC,aver_dice_ET],[aver_noise_dice_WT,aver_noise_dice_TC,aver_noise_dice_ET]
    return [aver_dice_WT,aver_dice_TC,aver_dice_ET],[aver_noise_dice_WT,aver_noise_dice_TC,aver_noise_dice_ET],[aver_hd_WT,aver_hd_TC,aver_hd_ET],[aver_noise_hd_WT,aver_noise_hd_TC,aver_noise_hd_ET]

def testensemblemax(
        test_loader,
        model,
        multimodel,
        Net_name,
        Variance,
        load_file,
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly!
        verbose=False,
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        # snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        # visual='',  # the path to save visualization
        ):

    H, W, T = 240, 240, 160
    # model.eval()

    runtimes = []
    dice_total_WT = 0
    dice_total_TC = 0
    dice_total_ET = 0
    hd_total_WT = 0
    hd_total_TC = 0
    hd_total_ET = 0
    assd_total_WT = 0
    assd_total_TC = 0
    assd_total_ET = 0
    noise_dice_total_WT = 0
    noise_dice_total_TC = 0
    noise_dice_total_ET = 0
    noise_hd_total_WT = 0
    noise_hd_total_TC = 0
    noise_hd_total_ET = 0
    noise_assd_total_WT = 0
    noise_assd_total_TC = 0
    noise_assd_total_ET = 0
    mean_uncertainty_total = 0
    noise_mean_uncertainty_total = 0
    num = len(test_loader)

    for i, data in enumerate(test_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(test_loader))
        x, target = data
        noise_m = torch.randn_like(x) * Variance
        # noise = torch.clamp(torch.randn_like(x) * Variance, -Variance * 2, Variance * 2)
        # noise = torch.clamp(torch.randn_like(x) * Variance, -Variance, Variance)
        # noise = torch.clamp(torch.randn_like(x) * Variance)
        noised_x = x + noise_m
        x.cuda()
        noised_x.cuda()
        target = torch.squeeze(target).cpu().numpy()

        if not use_TTA:
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            logit = torch.zeros(x.shape[0], 4, x.shape[2], x.shape[3], 155)
            logit_noise = torch.zeros(x.shape[0], 4, x.shape[2], x.shape[3], 155)
            # load ensemble models
            for j in range(10):
                print('ensemble model:{}'.format(i))
                logit += tailor_and_concat(x, model[j])
                logit_noise += tailor_and_concat(noised_x, model[j])
            # calculate ensemble uncertainty by normalized entropy
            uncertainty = Uentropy(logit/10, 4)
            noised_uncertainty = Uentropy(logit_noise/10, 4)

            U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
            noised_U_output = torch.squeeze(noised_uncertainty).cpu().detach().numpy()
            joblib.dump({'logit': logit, 'logit_noise': logit_noise, 'uncertainty': uncertainty,
                         'noised_uncertainty': noised_uncertainty, 'U_output': U_output,
                         'noised_U_output': noised_U_output}, 'Uensemble_uncertainty_{}.pkl'.format(i))
            # lnear = F.softplus(logit)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time/60))
            runtimes.append(elapsed_time)
            output = F.softmax(logit/10, dim=1)
            noised_output = F.softmax(logit_noise/10, dim=1)


        else:
            x = x[..., :155]
            logit = F.softmax(tailor_and_concat(x, model), 1)  # no flip
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2,)), model).flip(dims=(2,)), 1)  # flip H
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3,)), model).flip(dims=(3,)), 1)  # flip W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(4,)), model).flip(dims=(4,)), 1)  # flip D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3)), model).flip(dims=(2, 3)), 1)  # flip H, W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 4)), model).flip(dims=(2, 4)), 1)  # flip H, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3, 4)), model).flip(dims=(3, 4)), 1)  # flip W, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3, 4)), model).flip(dims=(2, 3, 4)), 1)  # flip H, W, D
            # for noise x
            noised_x = noised_x[..., :155]
            noised_logit = F.softmax(tailor_and_concat(noised_x, model), 1)  # no flip
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(2,)), model).flip(dims=(2,)), 1)  # flip H
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(3,)), model).flip(dims=(3,)), 1)  # flip W
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(4,)), model).flip(dims=(4,)), 1)  # flip D
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(2, 3)), model).flip(dims=(2, 3)), 1)  # flip H, W
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(2, 4)), model).flip(dims=(2, 4)), 1)  # flip H, D
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(3, 4)), model).flip(dims=(3, 4)), 1)  # flip W, D
            noised_logit += F.softmax(tailor_and_concat(noised_x.flip(dims=(2, 3, 4)), model).flip(dims=(2, 3, 4)), 1)  # flip H, W, D
            output = logit / 8.0  # mean
            noised_output = noised_logit / 8.0  # mean
            uncertainty = Uentropy(logit, 4)
            noised_uncertainty = Uentropy(noised_output, 4)
        mean_uncertainty = torch.mean(uncertainty)
        noised_mean_uncertainty = torch.mean(noised_uncertainty)
        output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        output = output.argmax(0)
        # iou_res = softmax_mIOU_score(output, target[:, :, :155])
        hd_res = softmax_output_hd(output, target[:, :, :155])
        dice_res = softmax_output_dice(output, target[:, :, :155])
        assd_res = softmax_output_assd(output, target[:, :, :155])
        dice_total_WT += dice_res[0]
        dice_total_TC += dice_res[1]
        dice_total_ET += dice_res[2]
        hd_total_WT += hd_res[0][0]
        hd_total_TC += hd_res[1][0]
        hd_total_ET += hd_res[2][0]
        assd_total_WT += assd_res[0]
        assd_total_TC += assd_res[1]
        assd_total_ET += assd_res[2]
        # for noise_x
        noised_output = noised_output[0, :, :H, :W, :T].cpu().detach().numpy()
        noised_output = noised_output.argmax(0)
        noise_assd_res = softmax_output_assd(noised_output, target[:, :, :155])
        noise_hd_res = softmax_output_hd(noised_output, target[:, :, :155])
        noise_dice_res = softmax_output_dice(noised_output, target[:, :, :155])

        noise_dice_total_WT += noise_dice_res[0]
        noise_dice_total_TC += noise_dice_res[1]
        noise_dice_total_ET += noise_dice_res[2]
        noise_hd_total_WT += noise_hd_res[0][0]
        noise_hd_total_TC += noise_hd_res[1][0]
        noise_hd_total_ET += noise_hd_res[2][0]
        noise_assd_total_WT += noise_assd_res[0]
        noise_assd_total_TC += noise_assd_res[1]
        noise_assd_total_ET += noise_assd_res[2]
        mean_uncertainty_total += mean_uncertainty
        noise_mean_uncertainty_total += noised_mean_uncertainty
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        print(msg)
        print('current_dice:{} ; current_noised_dice:{}'.format(dice_res, noise_dice_res))
        print('current_uncertainty:{} ; current_noised_uncertainty:{}'.format(uncertainty, noised_uncertainty))
        # if savepath:
        #     # .npy for further model ensemble
        #     # .nii for directly model submission
        #     assert save_format in ['npy', 'nii']
        #     if save_format == 'npy':
        #         np.save(os.path.join(savepath, Net_name +'_'+ name + '_preds'), output)
        #     if save_format == 'nii':
        #         # raise NotImplementedError
        #         oname = os.path.join(savepath, Net_name + '_'+ name + '.nii.gz')
        #         seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)
        #
        #         seg_img[np.where(output == 1)] = 1
        #         seg_img[np.where(output == 2)] = 2
        #         seg_img[np.where(output == 3)] = 3
        #         if verbose:
        #             print('1:', np.sum(seg_img == 1), ' | 2:', np.sum(seg_img == 2), ' | 4:', np.sum(seg_img == 4))
        #             print('WT:', np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4)), ' | TC:',
        #                   np.sum((seg_img == 1) | (seg_img == 4)), ' | ET:', np.sum(seg_img == 4))
        #         nib.save(nib.Nifti1Image(seg_img, None), oname)
        #         print('Successfully save {}'.format(oname))

                # if snapshot:
                #     """ --- grey figure---"""
                #     # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                #     # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                #     # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                #     # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                #     """ --- colorful figure--- """
                #     Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
                #     Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
                #     Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
                #     Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255
                #
                #     for frame in range(T):
                #         if not os.path.exists(os.path.join(visual, name)):
                #             os.makedirs(os.path.join(visual, name))
                #         # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                #         imageio.imwrite(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])

    aver_dice_WT = dice_total_WT / num
    aver_dice_TC = dice_total_TC / num
    aver_dice_ET = dice_total_ET / num
    aver_hd_WT = hd_total_WT / num
    aver_hd_TC = hd_total_TC / num
    aver_hd_ET = hd_total_ET / num
    aver_assd_WT = assd_total_WT / num
    aver_assd_TC = assd_total_TC / num
    aver_assd_ET = assd_total_ET / num
    aver_noise_dice_WT = noise_dice_total_WT / num
    aver_noise_dice_TC = noise_dice_total_TC / num
    aver_noise_dice_ET = noise_dice_total_ET / num
    aver_noise_hd_WT = noise_hd_total_WT / num
    aver_noise_hd_TC = noise_hd_total_TC / num
    aver_noise_hd_ET = noise_hd_total_ET / num
    aver_noise_assd_WT = noise_assd_total_WT / num
    aver_noise_assd_TC = noise_assd_total_TC / num
    aver_noise_assd_ET = noise_assd_total_ET / num
    aver_noise_mean_uncertainty = noise_mean_uncertainty_total/num
    aver_mean_uncertainty = mean_uncertainty_total/num
    print('aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice_WT*100,aver_dice_TC*100, aver_dice_ET*100))
    print('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (aver_noise_dice_WT*100, aver_noise_dice_TC*100, aver_noise_dice_ET*100))
    print('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (aver_hd_WT,aver_hd_TC, aver_hd_ET))
    print('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f,aver_noise_hd_ET = %f' % (aver_noise_hd_WT, aver_noise_hd_TC, aver_noise_hd_ET))
    print('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd_WT,aver_assd_TC, aver_assd_ET))
    print('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f,aver_noise_assd_ET = %f' % (aver_noise_assd_WT, aver_noise_assd_TC, aver_noise_assd_ET))
    logging.info(
        'aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice_WT*100, aver_dice_TC*100, aver_dice_ET*100))
    logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (
        aver_noise_dice_WT*100, aver_noise_dice_TC*100, aver_noise_dice_ET*100))
    logging.info('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (
        aver_hd_WT, aver_hd_TC, aver_hd_ET))
    logging.info('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f,aver_noise_hd_ET = %f' % (
        aver_noise_hd_WT, aver_noise_hd_TC, aver_noise_hd_ET))
    logging.info('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd_WT, aver_assd_TC, aver_assd_ET))
    logging.info('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f,aver_noise_assd_ET = %f' % (
        aver_noise_assd_WT, aver_noise_assd_TC, aver_noise_assd_ET))
    logging.info('aver_noise_mean_uncertainty=%f,aver_mean_uncertainty = %f' % (
    aver_noise_mean_uncertainty, aver_mean_uncertainty))
    # return [aver_dice_WT,aver_dice_TC,aver_dice_ET],[aver_noise_dice_WT,aver_noise_dice_TC,aver_noise_dice_ET]
    return [aver_dice_WT,aver_dice_TC,aver_dice_ET],[aver_noise_dice_WT,aver_noise_dice_TC,aver_noise_dice_ET],[aver_hd_WT,aver_hd_TC,aver_hd_ET],[aver_noise_hd_WT,aver_noise_hd_TC,aver_noise_hd_ET]
    # return [aver_dice_WT,aver_dice_TC,aver_dice_ET],[aver_noise_dice_WT,aver_noise_dice_TC,aver_noise_dice_ET],[aver_iou_WT,aver_iou_TC,aver_iou_ET],[aver_noise_iou_WT,aver_noise_iou_TC,aver_noise_iou_ET]