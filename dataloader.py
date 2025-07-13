import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import RandomRotation, InterpolationMode
from scipy.ndimage.morphology import distance_transform_edt
import torch.nn.functional as F
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2



def binary_erosion(tensor, kernel_size=3):
    """
    二值化Tensor的腐蚀操作
    :param tensor: 输入Tensor (B,C,H,W)或(H,W)，值为0.0或1.0
    :param kernel_size: 腐蚀核大小（奇数）
    :return: 腐蚀后的Tensor
    """
    # 创建矩形核
    kernel = torch.ones((kernel_size, kernel_size),
                        dtype=torch.float32,
                        device=tensor.device)

    # 归一化核
    kernel = kernel / kernel.sum()

    # 调整输入维度（确保是4D）
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)

    # 腐蚀 = 最小值滤波
    pad_size = kernel_size // 2
    eroded = -F.max_pool2d(-tensor,
                           kernel_size=kernel_size,
                           stride=1,
                           padding=pad_size)

    # 恢复原始维度
    return eroded.squeeze() if tensor.dim() == 4 else eroded

def binary_dilation(tensor, kernel_size=3):
    """
    二值化Tensor的膨胀操作
    :param tensor: 输入Tensor (B,C,H,W)或(H,W)，值为0.0或1.0
    :param kernel_size: 膨胀核大小（奇数）
    :return: 膨胀后的Tensor
    """
    # 创建矩形核
    kernel = torch.ones((kernel_size, kernel_size),
                        dtype=torch.float32,
                        device=tensor.device)

    # 调整输入维度
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)

    # 膨胀 = 最大值滤波
    pad_size = kernel_size // 2
    dilated = F.max_pool2d(tensor,
                           kernel_size=kernel_size,
                           stride=1,
                           padding=pad_size)

    # 恢复原始维度
    return dilated.squeeze() if tensor.dim() == 4 else dilated

def distribution_map(mask, sigma=5):
    # 先腐蚀后膨胀 = 开运算
    mask = binary_dilation(binary_erosion(mask))
    dist1 = distance_transform_edt(mask)
    dist2 = distance_transform_edt(1 - mask)
    dist = dist1 + dist2
    dist = torch.tensor(dist - 1,dtype=torch.float32)
    f = lambda x, sigma: (1 / (torch.sqrt(torch.tensor(2 * np.pi)) * sigma)) * torch.exp(-x ** 2 / (2 * sigma ** 2))
    # 应用高斯函数
    bdm = f(dist, sigma)
    # 将负值置为0
    bdm[bdm < 0] = 0
    # 返回结果乘以sigma^2
    return bdm * (sigma ** 2)

class PolypDataset(Dataset):
    def __init__(self, dataset_dir, trainsize=352, augmentations=True, edge=None, bdm=None, return_imgurl=False):
        """
        Args:
            dataset_dir (str or list): 一个目录路径或路径列表，每个路径下应包含 images/ 和 masks/ 文件夹
            edge (str): 可选子文件夹名，例如 'edges'
            bdm (str): 可选子文件夹名，例如 'bdm_maps'
        """
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.use_edge = edge is not None
        self.use_bdm = bdm is not None
        self.return_imgurl = return_imgurl

        # 确保 dataset_dir 是列表
        if isinstance(dataset_dir, str):
            dataset_dir = [dataset_dir]

        self.image_paths, self.gt_paths = [], []
        self.edge_paths = [] if self.use_edge else None
        self.bdm_paths = [] if self.use_bdm else None

        for ddir in dataset_dir:
            image_root = os.path.join(ddir, 'images')
            gt_root = os.path.join(ddir, 'masks')
            self.image_paths.extend(sorted([
                os.path.join(image_root, f) for f in os.listdir(image_root)
                if f.lower().endswith(('.jpg', '.png'))
            ]))
            self.gt_paths.extend(sorted([
                os.path.join(gt_root, f) for f in os.listdir(gt_root)
                if f.lower().endswith(('.jpg', '.png'))
            ]))

            if self.use_edge:
                edge_root = os.path.join(ddir, edge)
                self.edge_paths.extend(sorted([
                    os.path.join(edge_root, f) for f in os.listdir(edge_root)
                    if f.lower().endswith(('.jpg', '.png'))
                ]))

            if self.use_bdm:
                bdm_root = os.path.join(ddir, bdm)
                self.bdm_paths.extend(sorted([
                    os.path.join(bdm_root, f) for f in os.listdir(bdm_root)
                    if f.lower().endswith(('.jpg', '.png'))
                ]))

        # 检查数量一致性
        assert len(self.image_paths) == len(self.gt_paths), "Mismatched image and mask counts"
        if self.use_edge:
            assert len(self.image_paths) == len(self.edge_paths), "Mismatched edge count"
        if self.use_bdm:
            assert len(self.image_paths) == len(self.bdm_paths), "Mismatched BDM count"

        self.filter_files()
        self.size = len(self.image_paths)
        self.transform = self.build_transforms()


    def build_transforms(self):
        # 基础变换
        transforms = [
            A.Resize(height=self.trainsize, width=self.trainsize),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
        # 数据增强
        if self.augmentations:
            transforms = [
                A.Rotate(limit=90, interpolation=cv2.INTER_LINEAR, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                *transforms
            ]

        # 多目标配置
        additional_targets = {'gt': 'mask'}
        if self.use_edge:
            additional_targets['edge'] = 'mask'

        return A.Compose(transforms, additional_targets=additional_targets)


    def __getitem__(self, index):
        # 加载图像和GT
        img_url = self.image_paths[index]
        image = cv2.imread(img_url)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gt_paths[index], cv2.IMREAD_GRAYSCALE)

        # 加载边缘（如果启用）
        edge = None
        if self.use_edge:
            edge = cv2.imread(self.edge_paths[index], cv2.IMREAD_GRAYSCALE)
            transformed = self.transform(image=image, gt=gt, edge=edge)
            edge = transformed['edge']
        else:
            transformed = self.transform(image=image, gt=gt)


        # 获取变换后的数据
        image = transformed['image']
        gt = transformed['gt']
        gt = (gt == 255).to(dtype=torch.float32)
        # print(gt.unsqueeze(0).shape)
        if self.use_bdm and self.use_edge:
            bdm = distribution_map(gt,5)
            edge = (edge == 255).to(dtype=torch.float32)
            mask = torch.stack([gt, edge, bdm], dim=0)
        elif self.use_bdm and (not self.use_edge):
            bdm = distribution_map(gt, 5)
            mask = torch.stack([gt, bdm], dim=0)
        elif self.use_edge and (not self.use_bdm):
            edge = (edge == 255).to(dtype=torch.float32)
            mask = torch.stack([gt, edge], dim=0)
        else:
            # 单通道GT [1, H, W]
            mask = gt.unsqueeze(0)
        # print(mask.shape)
        if self.return_imgurl:
            return image, mask, img_url
        else:
            return image, mask

    def filter_files(self):
        """过滤尺寸不匹配的文件"""
        valid_pairs = []
        for i in range(len(self.image_paths)):
            img = Image.open(self.image_paths[i])
            gt = Image.open(self.gt_paths[i])

            if self.use_edge:
                edge = Image.open(self.edge_paths[i])
                if img.size == gt.size == edge.size:
                    assert img.size == gt.size == edge.size, 'img:{} gt:{} edge:{}'.format(img,gt,edge)
                    valid_pairs.append((self.image_paths[i], self.gt_paths[i], self.edge_paths[i]))
            else:
                if img.size == gt.size:
                    assert img.size == gt.size, 'img:{} gt:{}'.format(img, gt)
                    valid_pairs.append((self.image_paths[i], self.gt_paths[i]))

        # 更新有效路径
        if self.use_edge:
            self.image_paths, self.gt_paths, self.edge_paths = zip(*valid_pairs)
        else:
            self.image_paths, self.gt_paths = zip(*valid_pairs)

    def __len__(self):
        return self.size

def get_loader(datapath, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True, augmentation=False,edge=None,bdm=None,return_imgurl=False):
    dataset = PolypDataset(datapath, trainsize, augmentation,edge=edge,bdm=bdm,return_imgurl=return_imgurl)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize, edge=None,bdm=None):
        """
        Args:
            edge_root: 边缘图路径，None表示不启用edge
            testsize: 统一缩放尺寸
        """
        self.testsize = testsize
        self.use_edge = edge is not None
        self.use_bdm = bdm is not None
        # 加载文件路径
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) 
                            if f.endswith(('.jpg', '.png'))])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) 
                         if f.endswith(('.tif', '.png', '.jpg'))])
        
        if self.use_edge:
            edge_root = gt_root.replace('masks', edge)
            self.edge_paths = sorted([os.path.join(edge_root, f) for f in os.listdir(edge_root) 
                                     if f.endswith(('.png', '.jpg'))])
            assert len(self.images) == len(self.gts) == len(self.edge_paths), "文件数量不匹配"
        else:
            assert len(self.images) == len(self.gts), "文件数量不匹配"
            
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
        
        # GT/Edge变换（保持原始值）
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), 
                             interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        # 加载图像和GT
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)  # [1, C, H, W]
        
        gt = self.binary_loader(self.gts[self.index])
        gt = self.gt_transform(gt)  # [1, H, W]
        # print(gt.shape)
        # 加载边缘（如果启用）
        if self.use_edge:
            edge = self.binary_loader(self.edge_paths[self.index])
            edge = self.gt_transform(edge)  # [1, H, W]

        if self.use_bdm and self.use_edge:
            bdm = distribution_map(gt,5)
            edge = (edge == 255).to(dtype=torch.float32)
            mask = torch.cat([gt, edge, bdm], dim=0)
        elif self.use_bdm and (not self.use_edge):
            bdm = distribution_map(gt, 5)
            mask = torch.cat([gt, bdm], dim=0)
        elif self.use_edge and (not self.use_bdm):
            edge = (edge == 255).to(dtype=torch.float32)
            mask = torch.cat([gt, edge], dim=0)
        else:
            # 单通道GT [1, H, W]
            mask = gt
        # print(mask.shape)
        # 处理文件名
        name = os.path.basename(self.images[self.index])
        if name.endswith('.jpg'):
            name = name.replace('.jpg', '.png')
        
        self.index += 1
        return image, mask, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

