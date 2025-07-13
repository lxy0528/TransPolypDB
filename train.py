import time
import torch
from torch.autograd import Variable
import os
import re
import argparse
from datetime import datetime
import sys
import logging

import torch.nn.functional as F
import matplotlib.pyplot as plt

from lib.pvt import PolypPVT
from lib.PraNet_Res2Net import PraNet
from lib.ACSNet import ACSNet
from lib.CaraNet import caranet
from lib.CFANet import CFANet
from lib.LDNet import LDNet
from lib.CCBANet import CCBANet
from lib.BDM_PVT import BDMPVT_Net, BDM_Net, BDM_PVT_Net, BDM_CSA_Net, BDM_PVT_CSA_Net
from lib.Unet import UNet, UNetPlusPlus, AttentionUNet
from lib.vision_transformer import SwinUnet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from utils.loss import Forward_after
from utils.mics import get_metrics, parse_metrics, print_metrics

def print_trainable_parameters(model, detail=False):
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total += param_count
        if param.requires_grad:
            trainable += param_count
            if detail:
                print(f"[Trainable] {name:<60} ({param_count:,} parameters)")
        else:
            if detail:
                print(f"[Frozen   ] {name:<60} ({param_count:,} parameters)")

    print(f"\n✅ Total parameters: {total:,}")
    print(f"✅ Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    return f"✅ Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)"

def get_model(opt, device):
    name = opt.model_name
    if name == 'PolypPVT':
        model = PolypPVT()
    elif name == 'PraNet':
        model = PraNet()
    elif name == 'ACSNet':
        model = ACSNet(num_classes=1)
    elif name == 'CaraNet':
        model = caranet()
    elif name == 'CFANet':
        model = CFANet()
    elif name == 'CCBANet':
        model = CCBANet()
    elif name == 'LDNet':
        model = LDNet()
    elif name == 'BDMPVTNet':
        model = BDMPVT_Net(nclass=1)
    elif name == 'UNetPlusPlus':
        model = UNetPlusPlus(1, True)
    elif name == 'UNet':
        model = UNet(3, 1)
    elif name == 'AttentionUNet':
        model = AttentionUNet(1)
    elif name == 'SwinUNet':
        model = SwinUnet(num_classes=1, img_size=opt.testsize, window_size=11 if opt.testsize == 352 else 7)
    elif name == 'BDMNet':
        model = BDM_Net(nclass=1)
    elif name == 'BDM_DS':
        model = BDM_Net(nclass=1, deep_supervise=True)
    elif name == 'BDM_CS':
        model = BDM_CSA_Net(nclass=1)
    elif name == 'BDM_PVT':
        model = BDM_PVT_Net(nclass=1)
    elif name == 'BDM_PVT_DS':
        model = BDM_PVT_Net(nclass=1, deep_supervise=True)
    elif name == 'BDM_PVT_CS':
        model = BDM_PVT_CSA_Net(nclass=1)
    elif name == 'BDM_CS_DS':
        model = BDM_CSA_Net(nclass=1, deep_supervise=True)
    elif name == 'BDM_PVT_CS_DS':
        model = BDM_PVT_CSA_Net(nclass=1, deep_supervise=True)
    else:
        raise ValueError("Invalid model name")
    return model.to(device)

def test(model, path, dataset,device):
    data_path = os.path.join(path, dataset)
    model.eval()
    test_loader = get_loader(data_path, batchsize=opt.batchsize, trainsize=opt.testsize,shuffle=False,
                                augmentation=False, edge=opt.edge, bdm=opt.bdm, num_workers=opt.workers)
    frd_after = Forward_after(opt.model_name)
    Dice, IoU, Acc, Recall, Precision, F2, MAE = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for i, pack in enumerate(test_loader, start=1):
            # ---- data prepare ----
            images, gts = pack
            images = images.to(device)
            gts = gts.to(device)
            preds = model(images)
            losses,loss = frd_after.forward_loss(preds,gts)
            frd_after.loss_update(loss=loss, bs=opt.batchsize, losses=losses)
            # print(f'preds:{[p.shape for p in preds]}, gts:{gts.shape}')
            pred = frd_after.forward_pred(preds,gts)
            # print(f'preds:{[p.shape for p in preds]},pred:{pred.shape} gts:{gts.shape}')
            dice, iou, acc, recall, precision, f2, mae  = get_metrics(pred[:,0].cpu(),gts[:,0].cpu()) #pred[:,0]表示nc=1输出，gts[:,0]表示mask
            Dice, IoU, Acc, Recall, Precision, F2, MAE = Dice + dice, IoU + iou, Acc + acc, Recall + recall, Precision + precision, F2 + f2, MAE + mae

    num = len(test_loader.dataset)
    misc = Dice/num, IoU/num, Acc/num, Recall/num, Precision/num, F2/num, MAE/num
    loss = frd_after.loss.show()
    losses = [los.show() for i, los in enumerate(frd_after.losses)]
    return misc,loss,losses

def train(train_loader, model, optimizer, epoch, test_path, model_name,device):
    model.train()
    global best
    if model_name in {'SwinUNet'}:
        size_rates = [1]
    else:
        size_rates = [0.75, 1, 1.25]
    frd_after = Forward_after(model_name)
    for i, pack in enumerate(train_loader, start=1):
        # if i>101:
        #     break
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = images.to(device)
            gts = gts.to(device)
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)


            # print(images.device,gts.device,model.device,'*************')
            preds = model(images)
            losses,loss = frd_after.forward_loss(preds,gts)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                frd_after.loss_update(loss, opt.batchsize,losses)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '.
                  format(datetime.now(), epoch, opt.epoch, i, total_step),end='')
            print('Total_Loss: %.4f, '%frd_after.loss.show(),end='')
            print(', '.join(['loss%d: %.4f'%(i+1,los.show()) for i,los in enumerate(frd_after.losses)]))

            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '.format(datetime.now(), epoch, opt.epoch, i, total_step) +
                         'Total_Loss: %.4f, ' % frd_after.loss.show() +
                         ', '.join(['loss%d: %.4f'%(i+1,los.show()) for i,los in enumerate(frd_after.losses)])
                )

    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # torch.save(model.state_dict(), save_path +'/ckpt%d'%epoch+ '.pth')
    # choose the best model

    if epoch % 1 == 0:
        mics,loss,losses = test(model, test_path, 'WLI',device=device)
        mics = parse_metrics(mics)
        str_res = print_metrics(phase='{} epoch: {}, dataset: {:^25},'.format(datetime.now(),epoch, 'test'),loss=loss, metrics=mics, losses=losses)
        logging.info(str_res)
        print(str_res)

        meandice = mics['Dice']
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + '/best.pth')
            torch.save(model.state_dict(), save_path +'/ckpt%d'%epoch+ '-best.pth')
            print('############################################################################## best', best)
            logging.info('##############################################################################best:{}'.format(best))

    if epoch == opt.epoch:
        print('Test in best')
        model.load_state_dict(torch.load(save_path + '/best.pth'))
        for _dataset in dataset_names:
            if not os.path.exists(os.path.join(opt.test_path,_dataset)):
                continue
            mics, loss, losses = test(model, opt.test_path, _dataset,device=device)
            mics = parse_metrics(mics)
            str_res = print_metrics(phase='dataset: {:^25},'.format( _dataset), loss=loss, metrics=mics, losses=losses)
            logging.info(str_res)
            print(str_res)


def extract_continuous_numbers(s):
    # 使用正则表达式提取所有连续的数字
    numbers = re.findall(r'\d+', s)
    # 转换为整数列表
    if numbers:
        return int(numbers[0])
    else:
        return 0

def load_model(model):
    ns = os.listdir(opt.train_save)
    if len(ns)<=1:
        return False
    d = dict((extract_continuous_numbers(k),k) for k in ns)
    # print(d)
    ep = max(list(d.keys()))
    ls = torch.load(os.path.join(opt.train_save,d[ep]))
    model.load_state_dict(ls)
    print(f'model load from epoch {ep}')
    mics, loss, losses = test(model, opt.test_path, 'WLI', device=device)
    best = mics[0]
    return ep+1,best





if __name__ == '__main__':
    dataset_names = ['BLI', 'FICE', 'LCI', 'NBI', 'WLI']

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='PolypPVT')
    parser.add_argument('--resume', action = 'store_true', help='continue train')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--augmentation', default=True)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=30)
    parser.add_argument('--train_path', nargs='+', default=['../datasets/PolypDB/Split/train/WLI/'])
    parser.add_argument('--val_path', type=str, default='../datasets/PolypDB/Split/val/')
    parser.add_argument('--test_path', type=str, default='../datasets/PolypDB/Split/test/')
    parser.add_argument('--edge', type=str, default=None)
    parser.add_argument('--bdm', type=str, default=None)
    parser.add_argument('--train_save', type=str, default='../model_pth/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--log_dir', type=str, default='../log')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    opt = parser.parse_args()

    device = torch.device(opt.device if torch.cuda.is_available() and opt.device == 'cuda' else 'cpu')
    print(f'✅ Using device: {device}')

    opt.train_save = os.path.join(opt.train_save, opt.model_name)
    os.makedirs(opt.train_save, exist_ok=True)
    os.makedirs(os.path.join(opt.log_dir, opt.model_name), exist_ok=True)

    log_filename = os.path.join(opt.log_dir, opt.model_name, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_train.log')
    logging.basicConfig(
        filename=log_filename,
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )

    logging.info(str(opt))
    print(opt)

    torch.cuda.set_device(opt.gpu)
    model = get_model(opt, device)
    if opt.resume:
        resume = load_model(model)
    else:
        resume = False

    if resume:
        start_ep,best = resume
    else:
        start_ep = 1
        best = 0

    print_trainable_parameters(model)
    torch.backends.cudnn.benchmark = True
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=0.05) if opt.optimizer == 'AdamW' \
                else torch.optim.SGD(params, opt.lr, weight_decay=0.05, momentum=0.9)

    print(optimizer)

    train_loader = get_loader(opt.train_path, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation, edge=opt.edge, bdm=opt.bdm, num_workers=opt.workers)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    for epoch in range(start_ep, opt.epoch + 1):
        st = time.time()
        adjust_lr(optimizer, opt.lr, epoch, 0.1, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.test_path, opt.model_name,device=device)
        et = time.time()
        print(f'Each epoch spends {et - st:.2f}s')
