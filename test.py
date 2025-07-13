import os
import argparse
from datetime import datetime
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
from utils.dataloader import get_loader
from utils.loss import Forward_after
from utils.mics import get_metrics, parse_metrics, print_metrics
import logging
import numpy as np
import cv2
import torch
from thop import profile
import time

def calculate_flops_and_fps(model, input_size=(1, 3, 224, 224), batch_size=1, device='cpu'):
    """
    自动计算模型的 FLOPs 和推理速度（FPS）

    参数:
        model (torch.nn.Module): 待计算的 PyTorch 模型
        input_size (tuple): 输入张量的尺寸 (batch, channel, height, width)
        batch_size (int): 批次大小
        device (str): 'cpu' 或 'cuda'

    返回:
        flops (float): 模型的 FLOPs（单位：GFLOPs）
        fps (float): 推理速度（帧/秒）
    """
    # 设置模型为评估模式
    model.eval()

    # 构造输入张量
    input_tensor = torch.randn(batch_size, *input_size[1:])  # (batch, channel, height, width)
    input_tensor = input_tensor.to(device)

    # 计算 FLOPs
    flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    flops = flops / 1e9  # 转换为 GFLOPs

    # 计算推理速度（FPS）
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # 多次推理取平均
            _ = model(input_tensor)
    end_time = time.time()

    # 计算平均推理时间（秒/次）
    avg_inference_time = (end_time - start_time) / 100
    fps = 1 / avg_inference_time if avg_inference_time > 0 else 0

    # 打印结果
    print(f"模型 FLOPs: {flops:.4f} GFLOPs")
    print(f"模型推理速度 (FPS): {fps:.2f} FPS")

    return flops, fps


def print_trainable_parameters(model, detail=False):
    # print(model)
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

    print(f"\n�� Total parameters: {total:,}")
    print(f"✅ Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    return f"✅ Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)"

def get_model(opt):
    if opt.model_name=='PolypPVT':
        model = PolypPVT().cuda()
    elif opt.model_name=='PraNet':
        model = PraNet().cuda()
    elif opt.model_name=='ACSNet':
        model = ACSNet(num_classes=1).cuda()
    elif opt.model_name=='CaraNet':
        model = caranet().cuda()
    elif opt.model_name=='CFANet':
        model = CFANet().cuda()
    elif opt.model_name=='CCBANet':
        model = CCBANet().cuda()
    elif opt.model_name=='LDNet':
        model = LDNet().cuda()
    elif opt.model_name=='BDMPVTNet':
        model = BDMPVT_Net(nclass=1).cuda()
    elif opt.model_name=='UNetPlusPlus':
        model = UNetPlusPlus(1,True).cuda()
    elif opt.model_name=='UNet':
        model = UNet(3, 1).cuda()
    elif opt.model_name=='AttentionUNet':
        model = AttentionUNet(1).cuda()
    elif opt.model_name == 'SwinUNet':
        if opt.testsize == 352:
            model = SwinUnet(num_classes=1,img_size=opt.testsize,window_size=11).cuda()
        else:
            model = SwinUnet(num_classes=1).cuda()
    elif opt.model_name=='BDMNet':
        model = BDM_Net(nclass=1).cuda()
    elif opt.model_name == 'BDM_DS':
        model = BDM_Net(nclass=1,deep_supervise=True).cuda()
    elif opt.model_name == 'BDM_CS':
        model = BDM_CSA_Net(nclass=1).cuda()
    elif opt.model_name == 'BDM_PVT':
        model = BDM_PVT_Net(nclass=1).cuda()
    elif opt.model_name == 'BDM_PVT_DS':
        model = BDM_PVT_Net(nclass=1,deep_supervise=True).cuda()
    elif opt.model_name == 'BDM_PVT_CS':
        model = BDM_PVT_CSA_Net(nclass=1).cuda()
    elif opt.model_name == 'BDM_CS_DS':
        model = BDM_CSA_Net(nclass=1,deep_supervise=True).cuda()
    elif opt.model_name == 'BDM_PVT_CS_DS':
        model = BDM_PVT_CSA_Net(nclass=1,deep_supervise=True).cuda()
    else:
        raise ValueError("Invalid model name")
    return model

def test(model, path, dataset,save_dir):
    data_path = os.path.join(path, dataset)
    model.eval()
    test_loader = get_loader(data_path, batchsize=opt.batchsize, trainsize=opt.testsize, shuffle=False,
                             augmentation=False, edge=opt.edge, bdm=opt.bdm, num_workers=opt.workers,return_imgurl=True)
    frd_after = Forward_after(opt.model_name)
    Dice, IoU, Acc, Recall, Precision, F2, MAE = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for i, pack in enumerate(test_loader, start=1):
            # ---- data prepare ----
            images, gts, img_urls = pack
            images = images.cuda()
            gts = gts.cuda()
            preds = model(images)
            losses, loss = frd_after.forward_loss(preds, gts)
            frd_after.loss_update(loss=loss, bs=opt.batchsize, losses=losses)
            pred = frd_after.forward_pred(preds, gts)

            # 保存预测结果
            for j, (img_url, pred_img) in enumerate(zip(img_urls, pred)):
                # 将预测结果转换为numpy数组
                pred_np = pred_img[0].cpu().numpy()  # 取第一个通道
                # 归一化到0-255
                pred_np = (pred_np * 255).astype(np.uint8)

                # 保存预测结果
                save_path = os.path.join(save_dir, os.path.basename(img_url))
                # print(save_path)
                # print(pred_np.dtype)
                cv2.imwrite(os.path.join(save_dir, os.path.basename(img_url)), pred_np)


            dice, iou, acc, recall, precision, f2, mae = get_metrics(pred[:, 0].cpu(), gts[:,0].cpu())  # pred[:,0]表示nc=1输出，gts[:,0]表示mask
            Dice, IoU, Acc, Recall, Precision, F2, MAE = Dice + dice, IoU + iou, Acc + acc, Recall + recall, Precision + precision, F2 + f2, MAE + mae

    num = len(test_loader.dataset)
    misc = Dice / num, IoU / num, Acc / num, Recall / num, Precision / num, F2 / num, MAE / num
    loss = frd_after.loss.show()
    losses = [los.show() for i, los in enumerate(frd_after.losses)]
    return misc, loss, losses



if __name__ == '__main__':
    dataset_names = ['BLI', 'FICE', 'LCI', 'NBI', 'WLI']
    ##################config#############################

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='PolypPVT',help='model_name')

    parser.add_argument('--model_path', type=str, default='./model_pth/', help='pretrained_pth model dir')

    parser.add_argument('--model_load_name', type=str, default='best.pth', help='pretrained_pth model weights name')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--testsize', type=int,
                        default=352, help='testing dataset size')

    parser.add_argument('--test_path', type=str,
                        default='/media/mirukj/6e204f04-b878-4d6e-8c5a-ad25aa98eff4/HBP/datasets/MyPolypDataset/TestDataset',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--edge', type=str,
                        default=None,
                        help='sobel_edge,canny_edge,erosion_edge or None')

    parser.add_argument('--bdm', type=str,
                        default=None,
                        help='true or None')

    parser.add_argument('--save_dir', type=str,
                        default='../result/Test')

    parser.add_argument('--log_name', type=str,
                        default=None)
    parser.add_argument('--log_dir', type=str,
                        default='../log',
                        help='log dir')

    parser.add_argument('--gpu', type=int,
                        default=0, help='gpu id')

    parser.add_argument('--workers', type=int,
                        default=12, help='num workers')

    ###############################################

    opt = parser.parse_args()

    log_dir = os.path.join(opt.log_dir, opt.model_name)
    os.makedirs(log_dir, exist_ok=True)
    # 获取当前时间并格式化
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if opt.log_name is None:
        log_filename = os.path.join(log_dir, f'{current_time}_test.log')
    else:
        log_filename = os.path.join(log_dir, f'{opt.log_name}.log')
    # 配置日志
    logging.basicConfig(
        filename=log_filename,
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )


    logging.info(str(opt))
    print(opt)

    # ---- build models ----
    torch.cuda.set_device(opt.gpu)  # set your gpu device
    print('use gpu device: %d' % opt.gpu)

    model = get_model(opt)
    s = print_trainable_parameters(model)
    logging.info(s)
    torch.backends.cudnn.benchmark = True


    load_path = os.path.join(opt.model_path,opt.model_name,opt.model_load_name)
    model.load_state_dict(torch.load(load_path,map_location='cpu'))
    print(f'model load weights from {load_path}')
    logging.info(f'model load weights from {load_path}')

    for _dataset in dataset_names:
        if not os.path.exists(os.path.join(opt.test_path, _dataset)):
            print('%s not exist'%os.path.join(opt.test_path, _dataset))
            continue
        savedir = os.path.join(opt.save_dir,opt.model_name,_dataset)
        os.makedirs(savedir,exist_ok=True)
        mics, loss, losses = test(model, opt.test_path, _dataset, savedir)
        mics = parse_metrics(mics)
        str_res = print_metrics(phase='model: {}, dataset: {:^25},'.format(opt.model_name, _dataset), loss=loss,
                                metrics=mics, losses=losses)
        logging.info(str_res)
        print(str_res)
