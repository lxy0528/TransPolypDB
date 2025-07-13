import torch

def get_metrics(pred, mask, reduce='sum',th=0.5):
    pred = (pred>th).float()
    pred_positives = pred.sum(dim=(1, 2))
    # print(pred_positives.shape)
    mask_positives = mask.sum(dim=(1, 2))
    # print(mask_positives.shape)
    inter = (pred * mask).sum(dim=(1, 2))
    # print(pred.shape,mask.shape,inter.shape)
    union = pred_positives + mask_positives
    dice = (2 * inter) / (union + 1e-6)
    iou = inter / (union - inter + 1e-6)
    acc = (pred == mask).float().mean(dim=(1, 2))
    recall = inter / (mask_positives + 1e-6)
    precision = inter / (pred_positives + 1e-6)
    f2 = (5 * inter) / (4 * mask_positives + pred_positives + 1e-6)
    mae = (torch.abs(pred - mask)).mean(dim=(1, 2))
    # print(dice.shape, iou.shape, acc.shape, recall.shape, precision.shape, f2.shape, mae.shape)
    if reduce=='sum':
        return dice.sum(), iou.sum(), acc.sum(), recall.sum(), precision.sum(), f2.sum(), mae.sum()
    elif reduce == 'mean':
        return dice.mean(), iou.mean(), acc.mean(), recall.mean(), precision.mean(), f2.mean(), mae.mean()
    else:
        RuntimeError('Invalid reduce')

def print_metrics(phase, loss, metrics,losses=None):
    """格式化打印指标"""
    s = f"{phase} Total_Loss: {loss:.4f}"
    if losses is not None:
        s+=', '+', '.join(['loss%d: %.4f'%(i+1,los) for i,los in enumerate(losses)])
    metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    s+=f", {metrics_str}"
    return s

def parse_metrics(acc_tuple):
    """将指标元组转换为字典"""
    metrics = ['Dice', 'IoU', 'Acc', 'Recall', 'Precision', 'F2', 'MAE']
    return {k: v.item() for k, v in zip(metrics, acc_tuple)}
