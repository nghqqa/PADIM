import os

import torch
from PIL import Image
from numpy.ma.core import shape
from torch.utils.data import Dataset
from torchvision import transforms as T

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
# 数据集分类
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


# 数据集类定义
class MVTecDataset(Dataset):

    def __init__(self, dataset_path='D:/dataset/mvtec_anomaly_detection', class_name='bottle', is_train=True,
                 resize=256, cropsize=224):
        # 判断class_name是否属于CLASS_NAMES中
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        # 数据集地址
        self.dataset_path = dataset_path
        # 分类名
        self.class_name = class_name
        # 是否训练
        self.is_train = is_train
        # 调整图像大小
        self.resize = resize
        # 中心裁剪的大小
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        # 创建transform_x 变换器 对于数据集进行变化 调整大小 中心裁切，转换为tensor格式，进行归一化操作
        self.transform_x = T.Compose([T.Resize(resize, Image.LANCZOS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        # 创建transform_mask掩膜变换器 对于图像调整大小 中心裁切，转换为tensor格式
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    # 获取单个样本
    def __getitem__(self, idx):
        # 得到索引值对应的数据
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        # 通过RGB打开图片
        x = Image.open(x).convert('RGB')
        # 通过transform加工
        x = self.transform_x(x)
        # 如果标签为0则 直接使用对应0来填充掩码
        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        # 如果非0 则表示当前存在异常存在真实的掩码文件 掩码需要进行处理
        else:
            #mask为真实的掩膜图片 进行transform转换为tensor
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    # 得到图像的总数量
    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        # 加载图片子集下的train或者test数据 train下仅有good文件夹 test下有多种异常图片和good文件夹
        phase = 'train' if self.is_train else 'test'
        # x,y,mask的存放
        x, y, mask = [], [], []
        # phase的图片地址（train或者test文件夹下的）
        # D:/dataset/mvtec_anomaly_detection\bottle\train
        # D:/dataset/mvtec_anomaly_detection\bottle\test
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        # 测试集的缺陷掩码地址 用于评估模型效果 eg：D:/dataset/mvtec_anomaly_detection\bottle\ground_truth
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        # 获取img_dir底下的文件夹名称 eg：good
        img_types = sorted(os.listdir(img_dir))

        for img_type in img_types:
            # load images
            # 得到完整的数据集文件夹路径 eg：D:/dataset/mvtec_anomaly_detection\bottle\train\good
            img_type_dir = os.path.join(img_dir, img_type)
            # 路径不存在时，直接跳出
            # 获得.png图片路径加载
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            # 加入x列表中
            x.extend(img_fpath_list)

            # load gt labels
            # 加载图像掩码
            if img_type == 'good':
                # 如果图像是good文件夹下 则添加标签 0 和掩码为 None
                #eg: bottle类 则list(y)=[0,0.........] 209个0
                y.extend([0] * len(img_fpath_list))
                # print(y)
                mask.extend([None] * len(img_fpath_list))
                # print(mask)
            else:
                # 如果图像路径不是train下 则表示当前路径为test集
                # 添加标签为1
                #len(img_fpath_list)是指test或者train文件夹下所有的图片数量 eg bottle train的图片数量有209张
                #test下有83张图片
                y.extend([1] * len(img_fpath_list))
                # 掩码地址
                gt_type_dir = os.path.join(gt_dir, img_type)
                # 从图像路径中得到原始的图像标号 eg：D:/dataset/mvtec_anomaly_detection/bottle/train/good/image1.png
                # 得到image1
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                # 从img_fname_list中得到的图像标号中加上_mask.png扩展名，得到掩码名称列表
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                # 添加入mask列表中
                mask.extend(gt_fpath_list)
        # 断言x长度==y 如果不相同跳出提示句
        #len(x)=32 len(Y)=32
        assert len(x) == len(y), 'number of x and y should be same'
        # 得到trian集的图像路径 ,标签，掩码地址
        return list(x), list(y), list(mask)


#     def download(self):
#         """Download dataset if not exist"""

#         if not os.path.exists(self.mvtec_folder_path):
#             tar_file_path = self.mvtec_folder_path + '.tar.xz'
#             if not os.path.exists(tar_file_path):
#                 download_url(URL, tar_file_path)
#             print('unzip downloaded dataset: %s' % tar_file_path)
#             tar = tarfile.open(tar_file_path, 'r:xz')
#             tar.extractall(self.mvtec_folder_path)
#             tar.close()

#         return


# class DownloadProgressBar(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)


# def download_url(url, output_path):
#     with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
#         urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
if __name__ == '__main__':
    pass
    # data = MVTecDataset()
    # #bottle训练集中总有209张训练图片
    # sample = data[0]
    # x, y, mask = sample
    # print(x.shape)  # torch.Size([3, 224, 224])
    # #y是通过文件夹路径来判断是否存在异常的 数据集默认为train集中的无异常图片 则y永远为0
    # print(y)
    # #y=0 则mask为0填充的torch.Size([1, 224, 224])
    # print(mask.shape)

    # data = MVTecDataset(is_train=False)
    # x,y,mask = data[0]
    # print(x.shape)
    # print(y)
    # print(mask.shape)

