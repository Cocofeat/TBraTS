# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py

import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle
import torch
import torch.optim
from sklearn.externals import joblib
# from models import criterions
from models.lib.VNet3D import VNet
from plot import loss_plot,metrics_plot
from models.lib.UNet3DZoo import Unet,AttUnet,Unetdrop
from models.criterions import softBCE_dice,softmax_dice,FocalLoss,DiceLoss
from data.BraTS2019 import BraTS
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from predict import validate_softmax,test_softmax,testensemblemax
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='name of user', type=str)
parser.add_argument('--experiment', default='TransBTS', type=str)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='E:/BraTSdata1/archive2019', type=str) # folder_data_path
parser.add_argument('--train_dir', default='MICCAI_BraTS_2019_Data_TTraining', type=str)
parser.add_argument('--valid_dir', default='MICCAI_BraTS_2019_Data_TValidation', type=str)
parser.add_argument('--test_dir', default='MICCAI_BraTS_2019_Data_TTest', type=str)
parser.add_argument("--mode", default="train", type=str, help="train/test/train&test")
# parser.add_argument('--train_file',
#                     default='C:/Coco_file/BraTSdata/archive2019/MICCAI_BraTS_2019_Data_Training/Ttrain_subject.txt', type=str)
# parser.add_argument('--valid_file', default='C:/Coco_file/BraTSdata/archive2019/MICCAI_BraTS_2019_Data_Training/Tval_subject.txt',
#                     type=str)
# parser.add_argument('--test_file', default='C:/Coco_file/BraTSdata/archive2019/MICCAI_BraTS_2019_Data_Training/Ttest_subject.txt',
#                     type=str)
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
parser.add_argument('--input_C', default=4, type=int)
parser.add_argument('--input_H', default=240, type=int)
parser.add_argument('--input_W', default=240, type=int)
parser.add_argument('--input_D', default=160, type=int) # 155
parser.add_argument('--crop_H', default=128, type=int)
parser.add_argument('--crop_W', default=128, type=int)
parser.add_argument('--crop_D', default=128, type=int)
parser.add_argument('--output_D', default=155, type=int)
parser.add_argument('--rlt', default=-1, type=float,
                  help='relation between CE/FL and dice')
# Training Information
parser.add_argument('--lr', default=0.002, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
# parser.add_argument('--criterion', default='softmaxBCE_dice', type=str)
parser.add_argument('--submission', default='./results', type=str)
parser.add_argument('--visual', default='visualization', type=str)
parser.add_argument('--num_class', default=4, type=int)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--batch_size', default=2, type=int, help="2/4/8")
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', default=200, type=int)
parser.add_argument('--save_freq', default=5, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--load', default=True, type=bool)
parser.add_argument('--modal', default='t2', type=str) # multi-modal
parser.add_argument('--model_name', default='V', type=str, help="AU/V/U")
parser.add_argument('--Variance', default=2, type=int) # 1 2
parser.add_argument('--use_TTA', default=False, type=bool, help="True/False")
parser.add_argument('--save_format', default='nii', type=str)
parser.add_argument('--test_date', default='2022-01-04', type=str)
parser.add_argument('--test_epoch', default=184, type=int)
args = parser.parse_args()

def val(model,checkpoint_dir,epoch,best_dice):
    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root,'valid',args.modal)
    valid_loader = DataLoader(valid_set, batch_size=1)
    print('Samples for valid = {}'.format(len(valid_set)))

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        best_dice,aver_dice,aver_iou = validate_softmax(save_dir = checkpoint_dir,
                                                        best_dice = best_dice,
                                                current_epoch = epoch,
                                                save_freq = args.save_freq,
                                                end_epoch = args.end_epoch,
                                                valid_loader = valid_loader,
                                                model = model,
                                                multimodel = args.modal,
                                                Net_name = args.model_name,
                                                names = valid_set.names,
                                                )
        # dice_list.append(aver_dice)
        # iou_list.append(aver_iou)
    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))
    return best_dice,aver_dice,aver_iou

def test(model):
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))
    test_list = os.path.join(args.root, args.test_dir, args.test_file)
    test_root = os.path.join(args.root, args.test_dir)
    test_set = BraTS(test_list, test_root,'test',args.modal)
    test_loader = DataLoader(test_set, batch_size=1)
    print('Samples for test = {}'.format(len(test_set)))

    logging.info('final test........')
    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment + args.test_date, args.model_name + '_' + args.modal + '_epoch_{}.pth'.format(args.test_epoch))

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment + args.test_date, args.model_name + '_' + args.modal + '_epoch_{}.pth')))
    else:
        print('There is no resume file to load!')


    start_time = time.time()
    model.eval()
    with torch.no_grad():
        aver_dice,aver_noise_dice,aver_hd,aver_noise_hd = test_softmax( test_loader = test_loader,
                                            model = model,
                                            multimodel = args.modal,
                                            Net_name=args.model_name,
                                            Variance = args.Variance,
                                            load_file=load_file,
                                            savepath = args.submission,
                                            names = test_set.names,
                                            use_TTA = args.use_TTA,
                                            save_format = args.save_format,
                                            )
    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(test_set)
    print('{:.2f} minutes!'.format(average_time))
    logging.info('aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice[0],aver_dice[1],aver_dice[2]))
    logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (aver_noise_dice[0], aver_noise_dice[1], aver_noise_dice[2]))
    logging.info('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (aver_hd[0],aver_hd[1],aver_hd[2]))
    logging.info('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f,aver_noise_hd_ET = %f' % (aver_noise_hd[0], aver_noise_hd[1], aver_noise_hd[2]))


def test_ensemble(model):

    test_list = os.path.join(args.root, args.test_dir, args.test_file)
    test_root = os.path.join(args.root, args.test_dir)
    test_set = BraTS(test_list, test_root,'test',args.modal)
    test_loader = DataLoader(test_set, batch_size=1)
    print('Samples for test = {}'.format(len(test_set)))

    logging.info('final test........')
    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment + args.test_date, args.model_name + '_' + args.modal + '_epoch_{}.pth'.format(args.test_epoch))

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment + args.test_date, args.model_name + '_' + args.modal + '_epoch_{}.pth')))
    else:
        print('There is no resume file to load!')

    # load ensemble models
    load_model=[]
    for i in range(10):
        save_name1 = args.model_name + '_' + args.modal + '_epoch_' +'199' + 'e' + str(i) + '.pth'
        load_model[i] = torch.load(save_name1)
        model[i] = load_model[i]['state_dict']

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        aver_dice, aver_noise_dice, aver_hd, aver_noise_hd = testensemblemax(test_loader=test_loader,
                                                                             model=model,
                                                                             multimodel=args.modal,
                                                                             Net_name=args.model_name,
                                                                             Variance=args.Variance,
                                                                             load_file=load_file,
                                                                             savepath=args.submission,
                                                                             names=test_set.names,
                                                                             use_TTA=args.use_TTA,
                                                                             save_format=args.save_format,
                                                                             )
    end_time = time.time()
    full_test_time = (end_time - start_time) / 60
    average_time = full_test_time / len(test_set)
    print('{:.2f} minutes!'.format(average_time))
    logging.info('aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice[0], aver_dice[1], aver_dice[2]))
    logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (
    aver_noise_dice[0], aver_noise_dice[1], aver_noise_dice[2]))
    logging.info('aver_iou_WT=%f,aver_iou_TC = %f,aver_iou_ET = %f' % (aver_hd[0], aver_hd[1], aver_hd[2]))
    logging.info('aver_noise_iou_WT=%f,aver_noise_iou_TC = %f,aver_noise_iou_ET = %f' % (
    aver_noise_hd[0], aver_noise_hd[1], aver_noise_hd[2]))

def train(criterion,model,criterion_fl,criterion_dl):
    # dataset
    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = BraTS(train_list, train_root, args.mode,args.modal)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size)
    print('Samples for train = {}'.format(len(train_set)))

    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    # criterion = getattr(criterions, args.criterion)

    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    resume = ''

    writer = SummaryWriter()

    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    start_time = time.time()

    torch.set_grad_enabled(True)
    loss_list = []
    dice_list = []
    iou_list = []
    best_dice =0
    for epoch in range(args.start_epoch, args.end_epoch):
        epoch_loss = 0
        loss = 0
        # loss1 = 0
        # loss2 = 0
        # loss3 = 0
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        start_epoch = time.time()
        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target = data
            x = x.cuda()
            target = target.cuda()
            output = model(x)

            if args.rlt > 0:
                loss = criterion_fl(output, target) + args.rlt * criterion_dl(output, target)
            else:
                loss = criterion_dl(output, target)

            # loss, loss1, loss2, loss3 = criterion(output, target)
            # loss1.requires_grad_(True)
            # loss2.requires_grad_(True)
            # loss3.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            # loss1.backward()
            # loss2.backward()
            # loss3.backward()
            optimizer.step()
            reduce_loss = loss.data.cpu().numpy()
            # reduce_loss1 = loss1.data.cpu().numpy()
            # reduce_loss2 = loss2.data.cpu().numpy()
            # reduce_loss3 = loss3.data.cpu().numpy()
            # logging.info('Epoch: {}_Iter:{}  loss: {:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f} ||'
            #             .format(epoch, i, reduce_loss, reduce_loss1, reduce_loss2, reduce_loss3))
            logging.info('Epoch: {}_Iter:{}  loss: {:.5f}'
                        .format(epoch, i, reduce_loss))

            epoch_loss += reduce_loss
        end_epoch = time.time()
        loss_list.append(epoch_loss)

        writer.add_scalar('lr', optimizer.defaults['lr'], epoch)
        writer.add_scalar('loss', loss, epoch)
        # writer.add_scalar('loss1', loss1, epoch)
        # writer.add_scalar('loss2', loss2, epoch)
        # writer.add_scalar('loss3', loss3, epoch)

        epoch_time_minute = (end_epoch-start_epoch)/60
        remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
        logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))
        best_dice,aver_dice,aver_iou = val(model,checkpoint_dir,epoch,best_dice)
        dice_list.append(aver_dice)
        iou_list.append(aver_iou)
    writer.close()
    # validation

    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))
    logging.info('----------------------------------The training process finished!-----------------------------------')

    loss_plot(args, loss_list)
    metrics_plot(args, 'dice',dice_list)

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


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

if __name__ == '__main__':
    # criterion = softBCE_dice(aggregate="sum")
    criterion = softmax_dice
    criterion_fl = FocalLoss(4)
    criterion_dl = DiceLoss()
    num = 2
    # _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    # x = [i for i in range(num)]
    # l = [i*random.random() for i in range(num)]
    # plt.figure()
    # plt.plot(x, l, label='dice')
    # log
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)
    # Net model choose
    if args.model_name == 'AU' and args.modal == 'both':
        model = AttUnet(in_channels=2, base_channels=16, num_classes=4)
    elif args.model_name == 'AU':
        model = AttUnet(in_channels=1, base_channels=16, num_classes=4)
    elif args.model_name == 'V' and args.modal == 'both':
        model = VNet(n_channels=2, n_classes=4, n_filters=16, normalization='gn', has_dropout=False)
    elif args.model_name == 'V' :
        model = VNet(n_channels=1, n_classes=4, n_filters=16, normalization='gn', has_dropout=False)
    elif args.model_name == 'Udrop'and args.modal == 'both':
        model = Unetdrop(in_channels=2, base_channels=16, num_classes=4)
    elif args.model_name == 'Udrop':
        model = Unetdrop(in_channels=1, base_channels=16, num_classes=4)
    elif args.model_name == 'U' and args.modal == 'both':
        model = Unet(in_channels=2, base_channels=16, num_classes=4)
    else:
        model = Unet(in_channels=1, base_channels=16, num_classes=4)
    # if 'train' in args.mode:
    #     train(criterion,model,criterion_fl,criterion_dl)
    args.mode = 'test'
    # Udropout_uncertainty = joblib.load('Udropout_uncertainty.pkl')
    test(model)
    # test_ensemble(model)
