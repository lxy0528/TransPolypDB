import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.utils import AvgMeter


"""BCE loss"""
class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, reduction='mean')

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss

"""Dice loss"""
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

"""BCE + DICE Loss"""
class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss


LogitsBCE = torch.nn.BCEWithLogitsLoss()
criterion = BceDiceLoss()

""" Deep Supervision Loss"""
def DeepSupervisionLoss(pred, gt):
    d0, d1, d2, d3, d4 = pred[0:]

    loss0 = criterion(d0, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss1 = criterion(d1, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion(d2, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion(d3, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = criterion(d4, gt)

    return loss0 + loss1 + loss2 + loss3 + loss4


def bdm_loss(pred, target, thresh=0.002, min_ratio=0.1):
    # print(pred.shape,target.shape)
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    # print(pred.shape, target.shape)
    loss = F.mse_loss(pred, target, reduction='none')
    _, index = loss.sort()  # 从小到大排序
    threshold_index = index[-round(min_ratio * len(index))]  # 找到min_kept数量的hardexample的阈值
    if loss[threshold_index] < thresh:  # 为了保证参与loss的比例不少于min_ratio
        thresh = loss[threshold_index].item()
    loss[loss < thresh] = 0
    loss = loss.mean()
    return loss


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def polyppvt_loss(pred, gts):
    # pred 应该包含 P1 和 P2
    P1, P2 = pred
    # ---- loss function ----
    loss_P1 = structure_loss(P1, gts)
    loss_P2 = structure_loss(P2, gts)
    loss = loss_P1 + loss_P2
    return (loss_P1, loss_P2), loss

def polyppvt_pred(pred, gts):
    # pred 应该包含 res 和 res1
    res, res1 = pred
    # eval Dice
    res = F.interpolate(res + res1, size=gts.shape[2:], mode='bilinear', align_corners=False)
    res = res.sigmoid()  # .data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res

def pranet_loss(pred, gts):
    # pred 应该包含 lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2
    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = pred
    # ---- loss function ----
    loss5 = structure_loss(lateral_map_5, gts)
    loss4 = structure_loss(lateral_map_4, gts)
    loss3 = structure_loss(lateral_map_3, gts)
    loss2 = structure_loss(lateral_map_2, gts)
    loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
    return (loss5.cpu(), loss4.cpu(), loss3.cpu(), loss2.cpu()), loss

def pranet_pred(pred, gts):
    # pred 应该包含 res5, res4, res3, res2
    res5, res4, res3, res2 = pred
    res = res2
    res = F.interpolate(res, size=gts.shape[2:], mode='bilinear', align_corners=False)
    res = res.sigmoid()  # .data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res

def acsnet_loss(pred, gts):
    # pred 应该包含 d0, d1, d2, d3, d4
    d0, d1, d2, d3, d4 = pred
    loss0 = criterion(d0, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss1 = criterion(d1, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion(d2, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion(d3, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = criterion(d4, gts)
    loss = loss0 + loss1 + loss2 + loss3 + loss4
    return (loss0, loss1, loss2, loss3, loss4), loss

def acsnet_pred(pred, gts):
    # pred 应该包含 res
    res = pred[0]
    res = F.interpolate(res, size=gts.shape[2:], mode='bilinear', align_corners=False)
    res = res.sigmoid()  # .data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res

def caranet_pred(pred, gts):
    # pred 应该包含 res5, res4, res2, res1
    res5, res4, res2, res1 = pred
    res = res5
    res = F.interpolate(res, size=gts.shape[2:], mode='bilinear', align_corners=False)
    res = res.sigmoid()  # .data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res

def cfanet_loss(pred, gts):
    masks = gts[:,0:1]
    edges = gts[:,1:2]
    cam_edge, sal_out1, sal_out2, sal_out3 = pred
    loss_edge = LogitsBCE(cam_edge, edges)
    loss_sal1 = structure_loss(sal_out1, masks)
    loss_sal2 = structure_loss(sal_out2, masks)
    loss_sal3 = structure_loss(sal_out3, masks)

    loss_total = loss_edge + loss_sal1 + loss_sal2 + loss_sal3
    return (loss_edge, loss_sal3,loss_sal2, loss_sal1), loss_total

def cfanet_pred(pred, gts):
    # pred 应该包含 res
    res = pred[3]
    res = F.interpolate(res, size=gts.shape[2:], mode='bilinear', align_corners=False)
    res = res.sigmoid()  # .data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res


def bdmnet_loss(pred, gts):
    y = gts[:,0:1]
    ibdm = gts[:,1:2]
    y_hat, bdm = pred
    loss_seg = structure_loss(y_hat, y)
    loss_bdm = bdm_loss(bdm, ibdm)
    loss_total = loss_seg + loss_bdm
    return (loss_seg, loss_bdm), loss_total

def bdmds_loss(pred, gts):
    y = gts[:,0:1]
    ibdm = gts[:,1:2]
    y_hat, bdm, res1,res2,res3= pred
    loss_seg = structure_loss(y_hat, y)
    loss_bdm = bdm_loss(bdm, ibdm)
    loss3 = structure_loss(res3, y)
    loss2 = structure_loss(res2, y)
    loss1 = structure_loss(res1, y)

    loss_total = loss_seg + loss_bdm + loss1 + loss2 + loss3
    return (loss_seg, loss_bdm, loss1, loss2, loss3), loss_total

def bdmnet_pred(pred, gts):
    # pred 应该包含 res
    res = pred[0]
    res = F.interpolate(res, size=gts.shape[2:], mode='bilinear', align_corners=False)
    res = res.sigmoid()  # .data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res

# --------- UNet ---------
def unet_loss(pred, gts):
    """UNet单输出损失"""
    loss = structure_loss(pred, gts)
    return [loss],loss

# --------- UNet++ ---------
def unet_plusplus_loss(preds, gts):
    """UNet++深度监督损失"""
    # preds应包含 [out1, out2, out3, out4] 四个层级输出
    losses = [structure_loss(p, gts) for p in preds]
    total_loss = sum(w * l for w, l in zip([0.4, 0.3, 0.2, 0.1], losses))  # 加权求和
    return losses, total_loss

def unet_pred(pred, gts):
    """通用预测处理（适用于所有变体）"""
    if isinstance(pred, list):  # 处理UNet++多输出
        pred = pred[-1]  # 取最终输出
    
    # 上采样至GT尺寸并归一化
    # print(pred.shape,gts.shape)
    pred = F.interpolate(pred, size=gts.shape[2:], mode='bilinear', align_corners=False)
    pred = torch.sigmoid(pred)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return pred


class Forward_after(object):
    def __init__(self,name):
        self.name = name
        loss_len = {
            'PolypPVT': 2,
            'PraNet': 4,
            'CaraNet': 4,
            'ACSNet': 5,
            'CFANet': 4,
            'UNet': 1,
            'UNetPlusPlus': 4,
            'AttentionUNet': 1,
            'SwinUNet': 1,
            'BDMPVTNet': 5,
            'BDMNet': 2,
            "BDM_CS": 2,
            "BDM_PVT_CS": 2,
            'BDM_DS': 2,
            'BDM_PVT': 2,
            'BDM_PVT_DS': 5,
            'BDM_CS_DS': 5,
            'BDM_PVT_CS_DS': 5,
        }
        loss_fn = {
            'PolypPVT': polyppvt_loss,
            'PraNet': pranet_loss,
            'CaraNet': pranet_loss, #equl pranet_loss
            'ACSNet': acsnet_loss,
            'CFANet': cfanet_loss,
            'BDMPVTNet': bdmds_loss,
            'UNet': unet_loss,
            'UNetPlusPlus': unet_plusplus_loss,
            'AttentionUNet': unet_loss,
            'SwinUNet': unet_loss,
            'BDMNet': bdmnet_loss,
            'BDM_DS': bdmds_loss,
            "BDM_CS": bdmnet_loss,
            "BDM_PVT_CS": bdmnet_loss,
            'BDM_PVT':bdmnet_loss,
            'BDM_PVT_DS': bdmds_loss,
            'BDM_CS_DS': bdmds_loss,
            'BDM_PVT_CS_DS': bdmds_loss,
        }
        pred_fn = {
            'PolypPVT': polyppvt_pred,
            'PraNet': pranet_pred,
            'CaraNet': caranet_pred,
            'ACSNet':acsnet_pred,
            'CFANet': cfanet_pred,
            'BDMPVTNet': bdmnet_pred,
            'UNet': unet_pred,
            'UNetPlusPlus': unet_pred,
            'AttentionUNet': unet_pred,
            'SwinUNet': unet_pred,
            'BDMNet': bdmnet_pred,
            'BDM_DS': bdmnet_pred,
            "BDM_CS": bdmnet_pred,
            "BDM_PVT_CS": bdmnet_pred,
            'BDM_PVT':bdmnet_pred,
            'BDM_PVT_DS': bdmnet_pred,
            'BDM_CS_DS': bdmnet_pred,
            'BDM_PVT_CS_DS': bdmnet_pred,
        }
        self.loss = AvgMeter()
        self.losses = [AvgMeter() for _ in range(loss_len[self.name])]
        self.loss_fn = loss_fn[self.name]
        self.predictor = pred_fn[self.name]

    def forward_loss(self,pred,gts):
        losses,loss = self.loss_fn(pred,gts)
        return losses,loss

    def forward_pred(self,pred,gts):
        pred = self.predictor(pred,gts)
        return pred

    def loss_update(self,loss,bs,losses=None):
        self.loss.update(loss.data,bs)
        if losses is not None:
            for Los,los in zip(self.losses,losses):
                Los.update(los.data,bs)

































