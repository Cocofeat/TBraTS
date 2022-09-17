import torch.nn as nn
import torch.nn.functional as F
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m



class InitConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi

class De_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(De_Cat, self).__init__()
        # self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        # self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, prev):
        # x1 = self.conv1(x)
        # y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((x, prev), dim=1)
        y = self.conv1(y)
        return y

class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1

class AttUnet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, num_classes=4):
        super(AttUnet, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)
        self.Att4 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att3 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Att2 = Attention_block(F_g=16, F_l=16, F_int=16)

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

        self.DeUp4 = DeUp(in_channels=base_channels*8, out_channels=base_channels*4)
        self.DeUpCat4 = De_Cat(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.DeBlock4 = DeBlock(in_channels=base_channels*4)

        self.DeUp3 = DeUp(in_channels=base_channels*4, out_channels=base_channels*2)
        self.DeUpCat3 = De_Cat(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.DeBlock3 = DeBlock(in_channels=base_channels*2)

        self.DeUp2 = DeUp(in_channels=base_channels*2, out_channels=base_channels)
        self.DeUpCat2 = De_Cat(in_channels=base_channels * 2, out_channels=base_channels)
        self.DeBlock2 = DeBlock(in_channels=base_channels)
        self.endconv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.InitConv(x)       # (1, 16, 128, 128, 128)

        x1_1 = self.EnBlock1(x)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 64, 64, 64)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 32, 32, 32)

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2 = self.EnDown3(x3_1)  # (1, 128, 16, 16, 16)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)
        x4_4 = self.EnBlock4_4(x4_3)  # (1, 128, 16, 16, 16)

        y4 = self.DeUp4(x4_4)  # (1, 64, 32, 32, 32)
        x3_1 = self.Att4(g=y4, x=x3_1)
        y4 = self.DeUpCat4(x3_1, y4)
        y4 = self.DeBlock4(y4) # (1, 64, 32, 32, 32)

        y3 = self.DeUp3(y4) # (1, 32, 64, 64, 64)
        x2_1 = self.Att3(g=y3, x=x2_1)
        y3 = self.DeUpCat3(y3, x2_1)
        y3 = self.DeBlock3(y3) # (1, 32, 64, 64, 64)

        y2 = self.DeUp2(y3)
        x1_1 = self.Att2(g=y2, x=x1_1)
        y2 = self.DeUpCat2(y2, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)
        y = self.endconv(y2)

        return y

class Unet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, num_classes=4):
        super(Unet, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

        self.DeUpCat4 = DeUp_Cat(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.DeBlock4 = DeBlock(in_channels=base_channels*4)

        self.DeUpCat3 = DeUp_Cat(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.DeBlock3 = DeBlock(in_channels=base_channels*2)

        self.DeUpCat2 = DeUp_Cat(in_channels=base_channels * 2, out_channels=base_channels)
        self.DeBlock2 = DeBlock(in_channels=base_channels)
        self.endconv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.InitConv(x)       # (1, 16, 128, 128, 128)

        x1_1 = self.EnBlock1(x)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 64, 64, 64)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 32, 32, 32)

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2 = self.EnDown3(x3_1)  # (1, 128, 16, 16, 16)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)
        x4_4 = self.EnBlock4_4(x4_3)  # (1, 128, 16, 16, 16)

        y4 = self.DeUpCat4(x4_4, x3_1)
        y4 = self.DeBlock4(y4) # (1, 64, 32, 32, 32)

        y3 = self.DeUpCat3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)

        y2 = self.DeUpCat2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)
        y = self.endconv(y2)

        return y
class Unetdrop(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, num_classes=4):
        super(Unetdrop, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.dropoutd1 = nn.Dropout(p=0.5)
        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)
        self.dropoutd2 = nn.Dropout(p=0.5)
        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)
        self.dropoutu1 = nn.Dropout(p=0.5)
        self.DeUpCat4 = DeUp_Cat(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.DeBlock4 = DeBlock(in_channels=base_channels*4)
        self.dropoutu2 = nn.Dropout(p=0.5)
        self.DeUpCat3 = DeUp_Cat(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.DeBlock3 = DeBlock(in_channels=base_channels*2)

        self.DeUpCat2 = DeUp_Cat(in_channels=base_channels * 2, out_channels=base_channels)
        self.DeBlock2 = DeBlock(in_channels=base_channels)
        self.endconv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.InitConv(x)       # (1, 16, 128, 128, 128)

        x1_1 = self.EnBlock1(x)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 64, 64, 64)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 32, 32, 32)

        x3_1 = self.EnBlock3_1(x2_2) # (1, 64, 32, 32, 32)
        x3_1drop = self.EnBlock3_2(x3_1)
        x3_1drop = self.dropoutd1(x3_1drop)
        x3_2 = self.EnDown3(x3_1drop)  # (1, 128, 16, 16, 16)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)
        x4_3 = self.dropoutd2(x4_3)
        x4_4 = self.EnBlock4_4(x4_3)  # (1, 128, 16, 16, 16)

        y4 = self.DeUpCat4(x4_4, x3_1)
        y4 = self.dropoutu1(y4)
        y4 = self.DeBlock4(y4) # (1, 64, 32, 32, 32)

        y3 = self.DeUpCat3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.dropoutu2(y3)
        y3 = self.DeBlock3(y3)

        y2 = self.DeUpCat2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)
        y = self.endconv(y2)
        return y
if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 1, 128, 128, 128), device=cuda0)
        model = AttUnet(in_channels=1, base_channels=16, num_classes=4)
        # model = Unet(in_channels=1, base_channels=16, num_classes=4)
        model.cuda()
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of model's parameter: %.2fM" % (total / 1e6))
        output = model(x)
        print('output:', output.shape)
