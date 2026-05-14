import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from datetime import datetime
from model.SFINet_V3 import SFINet
# from model.SFINet_V2 import SFINet
from utils.data import get_loader
from utils.func import AvgMeter, clip_gradient, adjust_lr
import pytorch_iou
import pytorch_fm

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 1e-4
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--trainset', type=str, default='ORSSD', help='training  dataset')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models
model = SFINet()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = 'E:/Datasets/' + opt.trainset + '/train-images/'
gt_root = 'E:/Datasets/' + opt.trainset + '/train-labels/'


train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)
floss = pytorch_fm.FLoss()
size_rates = [0.75, 1, 1.25]  # multi-scale training
train_losses=[]

def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_record1, loss_record2, loss_record3 = AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
           optimizer.zero_grad()
           images, gts = pack
           images = Variable(images).cuda()
           gts = Variable(gts).cuda()

           # multi-scale training samples
           trainsize = int(round(opt.trainsize * rate / 32) * 32)

           if rate != 1:
               images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
               gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

           sal, sal_sig = model(images)

           # bce+iou+fmloss
           loss1 = CE(sal[0], gts) + IOU(sal_sig[0], gts) + floss(sal_sig[0], gts)
           loss2 = CE(sal[1], gts) + IOU(sal_sig[1], gts) + floss(sal_sig[1], gts)
           loss3 = CE(sal[2], gts) + IOU(sal_sig[2], gts) + floss(sal_sig[2], gts)

           loss = loss1 + loss2 + loss3

           loss.backward()

           clip_gradient(optimizer, opt.clip)
           optimizer.step()

           if rate == 1:
               loss_record1.update(loss1.data, opt.batchsize)
               loss_record2.update(loss2.data, opt.batchsize)
               loss_record3.update(loss3.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                           loss2.data))
        if i == total_step:
            train_losses.append(loss.data)

    save_path = 'models/SFINet/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) > 40:
        torch.save(model.state_dict(), save_path + 'SFINet_' + opt.trainset + '.pth' + '.%d' % epoch)


print("Let's go!")
if __name__ == '__main__':
 for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
