import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch.nn.functional as F
from torch.autograd import Variable
import torch


__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class MFSAN(nn.Module):

    def __init__(self, num_classes=2):
        super(MFSAN, self).__init__()
        self.sharedNet = resnet50(True)
        self.sonnet1 = ADDneck(2048, 256)
        self.sonnet2 = ADDneck(2048, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.domain_fc_son2 = nn.Linear(256, num_classes)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, data_src, data_tgt = 0, label_src = 0, mark = 1):
        mmd_loss = 0
        if self.training == True:
            if mark == 1:
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)

                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)

                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son1)

                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2)
                l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1))
                l1_loss = torch.mean(l1_loss)
                pred_src = self.cls_fc_son1(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss

            if mark == 2:
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)

                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son2)

                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2)

                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1)
                l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1))
                l1_loss = torch.mean(l1_loss)

                #l1_loss = F.l1_loss(torch.nn.functional.softmax(data_tgt_son1, dim=1), torch.nn.functional.softmax(data_tgt_son2, dim=1))

                pred_src = self.cls_fc_son2(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss

        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)

            return pred1, pred2
        
class MADA(nn.Module):

    def __init__(self, num_classes=2, domain_classes=2):
        super(MADA, self).__init__()
        self.sharedNet = resnet50(True)
        self.sonnet1 = ADDneck(2048, 256)
        self.sonnet2 = ADDneck(2048, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son11 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.cls_fc_son22 = nn.Linear(256, num_classes)
        self.domain_fc = nn.Linear(256, domain_classes)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, data_src, data_tgt = 0, label_src = 0, mark = 1, batch_size = 16):
        mmd_loss = 0
        if self.training == True:
            domain1_label = torch.zeros(batch_size).long().cuda()
            domain2_label = torch.ones(batch_size).long().cuda()
            #mark用来区分数据是来自于哪个源中心
            if mark == 0:
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)#目标域数据

                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)

                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                #mmd损失
                mmd_loss += mmd.mmd(data_src, data_tgt_son1)

                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2)
                #l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1))
                #l1_loss = torch.mean(l1_loss)
                
                #预测源域的分类输出
                pred_src = self.cls_fc_son1(data_src)
                #预测现在这个源域数据来自于哪个源域的输出
                pred_dm = self.domain_fc(data_src)
                #计算领域判别损失函数dm，和分类损失函数cls
                dm_loss = F.nll_loss(F.log_softmax(pred_dm, dim=1), domain1_label)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, dm_loss

            if mark == 1:
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)

                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son2)

                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2)

                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1)
                #l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1))
                #l1_loss = torch.mean(l1_loss)
                
                #l1_loss = F.l1_loss(torch.nn.functional.softmax(data_tgt_son1, dim=1), torch.nn.functional.softmax(data_tgt_son2, dim=1))

                pred_src = self.cls_fc_son2(data_src)
                pred_dm = self.domain_fc(data_src)
                dm_loss = F.nll_loss(F.log_softmax(pred_dm, dim=1), domain2_label)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, dm_loss

        else:
            #测试部分
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)

        
            pred_dm1 = self.domain_fc(fea_son1)
            pred_dm1 = self.relu(pred_dm1)
            pred_dm2 = self.domain_fc(fea_son2)
            pred_dm2 = self.relu(pred_dm2)
            pred_dm = (pred_dm1+pred_dm2) / 2
            return pred1, pred2, pred_dm

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model