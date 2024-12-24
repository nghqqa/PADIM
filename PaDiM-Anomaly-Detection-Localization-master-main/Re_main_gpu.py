# -*- encoding: utf-8 -*-
'''
@File    :   Re_main.py   
@Contact :   2801511808@qq.com
@License :   (C)Copyright 2018-2024
  
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2024/11/19 15:06   ngh      1.0         None
'''
import argparse
import json
import os
import pickle
import random
import warnings
from collections import OrderedDict
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from tqdm import tqdm
import time
from datasets import mvtec
from metrics import compute_pro

warnings.filterwarnings("ignore")


def mahalanobis_gpu(x, mean, cov_inv):
    diff = x - mean
    return torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=-1))


def parse_args():
    parser = argparse.ArgumentParser("Re-Padim")
    parser.add_argument("--data_path", type=str, default="D:/dataset/mvtec")
    parser.add_argument("--save_path", type=str, default="./my_mvtec_result")
    parser.add_argument("--arch", type=str, choices=["resnet18", "wide_resnet50_2"], default="wide_resnet50_2")
    return parser.parse_args()


def emedding_upsample(vertors1, vertors2):
    outsize = vertors1.shape[2:]
    vertors2 = F.interpolate(vertors2, size=outsize, mode='bicubic', align_corners=True)
    vertors1 = torch.cat((vertors1, vertors2), dim=1)
    return vertors1


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.arch == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        t_d = 448
        d = 100
    elif args.arch == "wide_resnet50_2":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # 中间三层通道数
        t_d = 1792
        # 随机降维成550个通道
        d = 550
    model.to(device)
    model.eval()
    random.seed(0)
    torch.manual_seed(0)
    if device == "cuda:0":
        torch.cuda.manual_seed(0)
    # 从1792里面选550
    idx = torch.tensor(sample(range(t_d), d))
    # 存放中间特征
    outputs = []

    def hook(model, input, output):
        outputs.append(output)

    model.layer1.register_forward_hook(hook)
    model.layer2.register_forward_hook(hook)
    model.layer3.register_forward_hook(hook)
    # 创建暂存文件夹
    os.makedirs(os.path.join(args.save_path, "temp_{}".format(args.arch)), exist_ok=True)
    # 绘制完整roc曲线
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_ROCAUC = ax[0]
    fig_pixel_ROCAUC = ax[1]
    total_img_ROCAUC = []
    total_pixel_ROCAUC = []

    for class_name in mvtec.CLASS_NAMES:
        train_set = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_load = DataLoader(train_set, batch_size=8, shuffle=True,  pin_memory=True)

        test_set = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_load = DataLoader(test_set, batch_size=8, shuffle=True,  pin_memory=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        train_feature_path = os.path.join(args.save_path, "temp_{}".format(args.arch),
                                          "train_feature_{}.pth".format(class_name))
        if not os.path.exists(train_feature_path):
            for (x, _, _) in tqdm(train_load, '| feature extraction | train | {} |'.format(class_name)):
                with torch.no_grad():
                    _ = model(x.to(device))
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v)
                outputs = []
            for k, v in train_outputs.items():
                # 按通道拼接
                train_outputs[k] = torch.cat(train_outputs[k], dim=0)
            embedding_vector = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vector = emedding_upsample(embedding_vector, train_outputs[layer_name])
            # 在通道维度随机选择维度
            embedding_vectors = torch.index_select(embedding_vector, 1, idx.to(device))
            B, C, H, W = embedding_vectors.shape
            # 转换通道 方便计算
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            # 按批次取平均值 得到一个（550，xxx)的均值矩阵
            mean = torch.mean(embedding_vectors, dim=0)
            # 存放550 x 550的协方差矩阵值，形成协方差矩阵空间
            cov = torch.zeros(C, C, H * W)
            # 生成单位矩阵
            I = np.identity(C)
            I_gpu = torch.tensor(I, dtype=torch.float32).cuda()
            # 通过遍历每个图像块 来计算单个图像块的协方差 存入对应位置 为放置出现异常添加一个正则项
            for i in range(H * W):
                # cov按列计算 需要转置 不支持rowvar参数 np.cov支持
                cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I_gpu
            train_outputs = [mean, cov]
            with open(train_feature_path, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: {}'.format(train_feature_path))
            with open(train_feature_path, 'rb') as f:
                train_outputs = pickle.load(f)

        test_imgs = []
        gt_list = []
        gt_list_mask = []

        for (x, y, mask) in tqdm(test_load, '| feature extraction | test | {} |'.format(class_name)):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_list_mask.extend(mask.cpu().detach().numpy())
            with torch.no_grad():
                _ = model(x.to(device))
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v)
            outputs = []

        for k, v in test_outputs.items():
            # 按批次拼接
            test_outputs[k] = torch.cat(test_outputs[k], dim=0)
        # 上采样
        embedding_vector = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vector = emedding_upsample(embedding_vector, test_outputs[layer_name])
        # 随机选择550个通道
        embedding_vectors = torch.index_select(embedding_vector, 1, idx.to(device))
        B, C, H, W = embedding_vectors.shape
        # 转换尺寸
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        # 距离矩阵存放
        dist_list = []
        # 遍历像素点总数
        for i in range(H * W):
            # 取得每个像素点的均值
            mean = train_outputs[0][:, i]
            # 计算逆协方差
            cov_inv = torch.linalg.inv(train_outputs[1][:, :, i]).cuda()
            # cov_inv=torch.tensor(cov_inv).cuda()
            # 计算马氏距离 sample[:,i]是第 i个位置的测试样本特征向量
            # mean是第 i个位置的均值
            # cov_inv是第 i个位置的逆协方差矩阵
            dist = [mahalanobis_gpu(sample[:, i], mean, cov_inv) for sample in embedding_vectors]
            dist_list.append(dist)
        # 这样最后生成了一个（ H*W,batch）的矩阵
        # 理解起来就是有一个像素位置上的所有图片的马氏距离（1，83）
        # 一共有h*w个像素位置

        # 转换维度后出现reshape 方便后续计算指标
        dist_list = torch.tensor(dist_list).cuda().transpose(1, 0).reshape(B, H, W)

        # 进行插值 保证与测试集数据维度一致 而不是用特征图维度
        # bicubic需要四维 所以先利用unsqueeze生成一个维度，后压缩回去
        score_map = F.interpolate(dist_list.unsqueeze(1), size=mask.size(2), mode="bicubic",
                                  align_corners=False).squeeze().cpu().numpy()
        # (B,224,224)

        # 高斯平滑
        for i in range(score_map.shape[0]):
            # 对于每一张图片得到的距离值进行高斯平滑 论文中的核为4
            score_map[i] = gaussian_filter(score_map[i], 4)

        # 平滑结束后 进行归一化处理
        max_score = score_map.max()
        min_score = score_map.min()

        # 归一化公式
        # 压缩值到[0-1]保证检测的鲁棒性和可解释性
        score = (score_map - min_score) / (max_score - min_score)
        # 按列取最大值 得到一个83个最大值 后续进行计算
        img_score = score.reshape(score.shape[0], -1).max(axis=1)
        # int 转ndarray
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_score)

        Img_ROCAUC = roc_auc_score(gt_list, img_score)

        Img_AP = average_precision_score(gt_list, img_score)
        total_img_ROCAUC.append(Img_ROCAUC)
        print('{} Image ROCAUC: {:.3f},Image AP: {:.3f}'.format(class_name, Img_ROCAUC, Img_AP))
        fig_img_ROCAUC.plot(fpr, tpr, label=f'ROC curve (area = {Img_ROCAUC:.2f},class_name={class_name})')

        gt_mask = np.asarray(gt_list_mask)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), score.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1_scores = np.divide(a, b, out=np.zeros_like(a), where=(b) != 0)
        thresholds = thresholds[np.argmax(f1_scores)]

        fpr, tpr, _ = roc_curve(gt_mask.flatten(), score.flatten())

        pixel_ROCAUC = roc_auc_score(gt_mask.flatten(), score.flatten())
        pixel_AP = average_precision_score(gt_mask.flatten(), score.flatten())
        pixel_PRO = compute_pro(gt_mask, score)
        print(
            '{} pixel ROCAUC: {:.3f},pixel AP: {:.3f},pixel pro: {:.3f}'.format(class_name, pixel_ROCAUC, pixel_AP,
                                                                                pixel_PRO))
        total_pixel_ROCAUC.append(pixel_ROCAUC)
        fig_pixel_ROCAUC.plot(fpr, tpr, label='%s pixel_ROCAUC: %.3f' % (class_name, pixel_ROCAUC))

        metrics = {
            "class_name": class_name,
            "ROCAUC": Img_ROCAUC,
            "AP": Img_AP,
            "pixel_ROCAUC": pixel_ROCAUC,
            "pixel_AP": pixel_AP,
            "pixel_pro": pixel_PRO
        }
        file_path = "{}.json_resnet18".format(class_name)

        with open(os.path.join(args.save_path, "json_resnet18", file_path), "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"The data has been saved to {file_path}")

    # 计算imgage级别的平均ROCAUC
    print("Averge Image ROCAUC{:.3f}".format(np.mean(total_img_ROCAUC)))
    fig_img_ROCAUC.set_title("Averge ROCAUC::{:.3f}".format(np.mean(total_img_ROCAUC)))
    fig_img_ROCAUC.legend(loc='lower right')
    # 计算pixel级别的平均ROCAUC
    print('Average Pixel ROCUAC:{:.3f}'.format(np.mean(total_pixel_ROCAUC)))
    fig_pixel_ROCAUC.set_title('Average pixel ROCAUC::{:.3f}'.format(np.mean(total_pixel_ROCAUC)))
    fig_pixel_ROCAUC.legend(loc="lower right")
    # 优化布局
    fig.tight_layout()
    # 保存图片
    fig.savefig(f"rocauc_{args.arch}.png", dpi=100)


if __name__ == '__main__':
    start_time=time.time()
    print("start time={}".format(start_time))
    main()
    end_time=time.time()
    print("end time={}".format(start_time))
    elapsed_time = end_time - start_time
    print("processing time={}".format(elapsed_time))
