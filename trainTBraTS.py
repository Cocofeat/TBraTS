import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
import time
import cv2
from torch.utils.data import DataLoader
from PIL import Image
from ModelTS import TSS
from data.BraTS2019 import BraTS
from predict import tailor_and_concat,softmax_mIOU_score,softmax_output_dice,softmax_output_hd,softmax_output_assd,Uentropy_our,cal_ueo,cal_ece_our
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import imageio

def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    import argparse
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    parser = argparse.ArgumentParser()
    # Basic Information
    parser.add_argument('--user', default='name of user', type=str)
    parser.add_argument('--experiment', default='TBraTS', type=str)
    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
    parser.add_argument('--description',
                        default='TBraTS,'
                                'training on train.txt!',
                        type=str)
    # training detalis
    parser.add_argument('--epochs', type=int, default=199, metavar='N',
                        help='number of epochs to train [default: 500]')

    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                        help='learning rate')
    # DataSet Information
    parser.add_argument('--root', default='E:/BraTSdata1/archive2019', type=str)
    parser.add_argument('--save_dir', default='./results', type=str)
    parser.add_argument('--train_dir', default='MICCAI_BraTS_2019_Data_TTraining', type=str)
    parser.add_argument('--valid_dir', default='MICCAI_BraTS_2019_Data_TValidation', type=str)
    parser.add_argument('--test_dir', default='MICCAI_BraTS_2019_Data_TTest', type=str)
    parser.add_argument('--savepath', default='./results/plot/output', type=str)
    parser.add_argument("--mode", default="train", type=str, help="train/test/train&test")
    parser.add_argument('--train_file',
                        default='E:/BraTSdata1/archive2019/MICCAI_BraTS_2019_Data_Training/Ttrain_subject.txt',
                        type=str)
    parser.add_argument('--valid_file',
                        default='E:/BraTSdata1/archive2019/MICCAI_BraTS_2019_Data_Training/Tval_subject.txt',
                        type=str)
    parser.add_argument('--test_file',
                        default='E:/BraTSdata1/archive2019/MICCAI_BraTS_2019_Data_Training/Ttest_subject.txt',
                        type=str)
    parser.add_argument('--dataset', default='BraTS', type=str)
    parser.add_argument('--input_H', default=240, type=int)
    parser.add_argument('--input_W', default=240, type=int)
    parser.add_argument('--input_D', default=160, type=int)  # 155
    parser.add_argument('--crop_H', default=128, type=int)
    parser.add_argument('--crop_W', default=128, type=int)
    parser.add_argument('--crop_D', default=128, type=int)
    parser.add_argument('--output_D', default=155, type=int)
    parser.add_argument('--batch_size', default=2, type=int, help="2/4/8")
    parser.add_argument('--noiselen', default='no', type=str, help="half/no")
    parser.add_argument('--modal', default='four', type=str, help="t1/t2/both")  # multi-modal
    parser.add_argument('--model_name', default='V', type=str)  # multi-modal V:168 AU:197
    parser.add_argument('--Variance', default=0.5, type=int)
    parser.add_argument('--test_epoch', type=int, default=186, metavar='N',
                        help='best epoch')# 159
    args = parser.parse_args()
    args.dims = [[240,240,160], [240,240,160]]
    args.modes = len(args.dims)

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = BraTS(train_list, train_root, args.mode,args.modal)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size)
    print('Samples for train = {}'.format(len(train_set)))
    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root,'valid',args.modal)
    valid_loader = DataLoader(valid_set, batch_size=1)
    print('Samples for valid = {}'.format(len(valid_set)))
    test_list = os.path.join(args.root, args.test_dir, args.test_file)
    test_root = os.path.join(args.root, args.test_dir)
    test_set = BraTS(test_list, test_root,'test',args.modal)
    test_loader = DataLoader(test_set, batch_size=1)
    print('Samples for test = {}'.format(len(test_set)))
    model = TSS(4, args.modes, args.model_name, args.modal, args.lambda_epochs)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of model's parameter: %.2fM" % (total / 1e6))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    model.cuda()

    def train(epoch):

        model.train()
        loss_meter = AverageMeter()
        step = 0
        dt_size = len(train_loader.dataset)
        for i, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)
            step += 1
            input, target = data
            x = input.cuda()  # for multi-modal combine train
            target = target.cuda()
            # refresh the optimizer

            args.mode = 'train'
            evidences, loss = model(x,target,epoch,args.mode)
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
            optimizer.zero_grad()
            loss.requires_grad_(True).backward()
            optimizer.step()
            loss_meter.update(loss.item())
        return loss_meter.avg

    def val(args,current_epoch,best_dice):
        print('===========>Validation begining!===========')
        model.eval()
        loss_meter = AverageMeter()
        num_classes = 4
        dice_total, iou_total = 0, 0
        step = 0
        # model.eval()
        for i, data in enumerate(valid_loader):
            step += 1
            input, target = data

            x = input.cuda()  # for multi-modal combine train
            target = target.cuda()

            with torch.no_grad():
                args.mode = 'val'
                evidence = model(x, target[:, :, :, :155], current_epoch,args.mode) # two modality or four modality
                ## max
                alpha = evidence + 1
                uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                _, predicted = torch.max(evidence.data, 1)
                prob = evidence / torch.sum(alpha, dim=1, keepdim=True)
                output = predicted.cpu().detach().numpy()

                target = torch.squeeze(target).cpu().numpy()
                dice_res = softmax_output_dice(output, target[:, :, :155])
                print('current_dice:{}'.format(dice_res))
                dice_total += dice_res[0]
        aver_dice = dice_total / len(valid_loader)
        if aver_dice > best_dice \
                or (current_epoch + 1) % int(args.epochs - 1) == 0 \
                or (current_epoch + 1) % int(args.epochs - 2) == 0 \
                or (current_epoch + 1) % int(args.epochs - 3) == 0:
                print('aver_dice:{} > best_dice:{}'.format(aver_dice, best_dice))
                best_dice = aver_dice
                print('===========>save best model!')
                file_name = os.path.join(args.save_dir, args.model_name+'_'+args.modal+'_epoch_{}.pth'.format(current_epoch))
                torch.save({
                    'epoch': current_epoch,
                    'state_dict': model.state_dict(),
                },
                    file_name)
        return loss_meter.avg, best_dice

    def test(args):
        print('===========>Test begining!===========')
        logging.info('===========>Test begining!===========')
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))
        num_classes = 4
        dice_total_WT = 0
        dice_total_TC = 0
        dice_total_ET = 0
        hd_total_WT = 0
        hd_total_TC = 0
        hd_total_ET = 0
        assd_total_WT = 0
        assd_total_TC = 0
        assd_total_ET = 0
        step = 0
        time_mne = 0
        time_noised_mne = 0
        certainty_total = 0
        noise_certainty_total = 0
        mne_total = 0
        noise_mne_total = 0
        ece_total = 0
        noise_ece_total = 0
        ueo_total = 0
        noise_ueo_total = 0
        H, W, T = 240, 240, 155
        snapshot = False # False True
        dt_size = len(test_loader.dataset)
        load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.model_name + '_' +args.modal +'_epoch_{}.pth'.format(args.test_epoch))

        if os.path.exists(load_file):
            checkpoint = torch.load(load_file)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            print('Successfully load checkpoint {}'.format(
                os.path.join(args.save_dir +'/'+ args.model_name + '_' +args.modal +'_epoch_' + str(args.test_epoch))))
        else:
            print('There is no resume file to load!')
        names = test_set.names

        model.eval()
        for i, data in enumerate(test_loader):

            step += 1
            input, target = data # input ground truth

            x = input.cuda()
            target = target.cuda()

            with torch.no_grad():
                evidences = model(x, target[:, :, :, :155], args.epochs, args.mode)
                # results with TTA or not

                alpha = evidences + 1
                uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                # _, predicted = torch.max(evidences.data, 1) # original
                S = torch.sum(alpha, dim=1, keepdim=True)
                _, predicted = torch.max(evidences/S, 1)
                prob = evidences / torch.sum(alpha, dim=1, keepdim=True)
                output = torch.squeeze(predicted).cpu().detach().numpy()
                target = torch.squeeze(target).cpu().numpy() # .cpu().numpy(dtype='float32')
                hd_res = softmax_output_hd(output, target[:, :, :155])
                dice_res = softmax_output_dice(output, target[:, :, :155])
                iou_res = softmax_mIOU_score(output, target[:, :, :155])
                assd_res = softmax_output_assd(output, target[:, :, :155])
                dice_total_WT += dice_res[0]
                dice_total_TC += dice_res[1]
                dice_total_ET += dice_res[2]
                hd_total_WT += hd_res[0]
                hd_total_TC += hd_res[1]
                hd_total_ET += hd_res[2]
                assd_total_WT += assd_res[0]
                assd_total_TC += assd_res[1]
                assd_total_ET += assd_res[2]

                print('current_dice:{} ; current_hd:{};current_assd'.format(dice_res,hd_res,assd_res))
                # uncertaity metircs

                mean_uncertainty = torch.mean(uncertainty)
                certainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
                Otarget = torch.squeeze(target[:, :, :, :155]).cpu().numpy()
                U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
                Oinput = torch.squeeze(input).cpu().detach().numpy()
                name = str(i)
                if names:
                    name = names[i]
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

                    target_img = np.zeros(shape=(H, W, 3, T), dtype=np.float32)
                    # K = [np.where(output[0,:,:,:] == 1)]
                    target_img[:, :, 0, :][np.where(Otarget == 1)] = 255
                    target_img[:, :, 1, :][np.where(Otarget == 2)] = 255
                    target_img[:, :, 2, :][np.where(Otarget == 3)] = 255

                    for frame in range(T):
                        if not os.path.exists(os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name)):
                            os.makedirs(os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name))
                        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name, str(frame) + '.png'), Snapshot_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name, str(frame) + '_gt.png'),
                                        target_img[:, :, :, frame])

                        imageio.imwrite(os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name,  str(frame) + '_uncertainty.png'),
                                        U_output[:, :, frame])
                        imageio.imwrite(os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name, str(frame) + '_input_FLR.png'),
                                        Oinput[0,:, :, frame])
                        imageio.imwrite(os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name, str(frame) + '_input_T1ce.png'),
                                        Oinput[1,:, :, frame])
                        imageio.imwrite(os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name, str(frame) + '_input_T1.png'),
                                        Oinput[2,:, :, frame])
                        imageio.imwrite(os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name, str(frame) + '_input_T2.png'),
                                        Oinput[3,:, :, frame])
                        U_img = cv2.imread(
                            os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name,  str(frame) + '_uncertainty.png'))
                        U_heatmap = cv2.applyColorMap(U_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(args.savepath, str(args.model_name), str(args.modal), str(args.Variance), name,  str(frame) + '_colormap_uncertainty.png'),
                            U_heatmap)

        num = len(test_loader)
        aver_dice_WT = dice_total_WT / num
        aver_dice_TC = dice_total_TC / num
        aver_dice_ET = dice_total_ET / num
        aver_certainty = certainty_total  / num
        aver_hd_WT = hd_total_WT / num
        aver_hd_TC = hd_total_TC / num
        aver_hd_ET = hd_total_ET / num
        aver_assd_WT = assd_total_WT / num
        aver_assd_TC = assd_total_TC / num
        aver_assd_ET = assd_total_ET / num

        print('aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice_WT*100, aver_dice_TC*100, aver_dice_ET*100))
        print('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (aver_hd_WT, aver_hd_TC, aver_hd_ET))
        print('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd_WT, aver_assd_TC, aver_assd_ET))
        print('aver_certainty=%f' % (aver_certainty))

        logging.info(
            'aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice_WT, aver_dice_TC, aver_dice_ET))
        logging.info('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (
            aver_hd_WT, aver_hd_TC, aver_hd_ET))
        logging.info('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd_WT, aver_assd_TC, aver_assd_ET))
        logging.info('aver_certainty=%f' % (aver_certainty))

        return [aver_dice_WT, aver_dice_TC, aver_dice_ET]


    def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
        for param_group in optimizer.param_groups:
            param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)
    epoch_loss = 0
    best_dice = 0
    for epoch in range(1, args.epochs + 1):
        print('===========Train begining!===========')
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        epoch_loss = train(epoch)
        print("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss))
        val_loss, best_dice = val(args,epoch,best_dice)
    test_dice = test(args)
