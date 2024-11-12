# -*- encoding: utf-8 -*-
'''
@File    :   exploded_data.py   
@Contact :   2801511808@qq.com
@License :   (C)Copyright 2018-2024
  
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2024/11/9 14:00   ngh      1.0         None
'''
import argparse
import os
import pickle
import random
from sklearn.metrics import roc_curve, roc_auc_score
from collections import OrderedDict
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from tqdm import tqdm

import datasets.mvtec as mvtec

# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embedding_concat(x, y):
    # x的h1 w1大于y的尺寸
    B, C1, H1, W1 = x.size()
    B, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    # 展开x 将x转换为三维张量
    x = F.unfold(x, kernel_size=s, stride=s)
    # 修改为5维度
    x = x.view(B, C1, -1, H2, W2)
    # 初始化z张量形状 x.size(2) 是为了后续按这个维度进行拼接
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        # 按通道拼接
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), dim=1)
    # Z从5维修改为3维
    z = z.view(B, -1, H2 * W2)
    # 折叠为原来的尺寸大小 output_size=(H1,W1)
    z = F.fold(z, kernel_size=s, stride=s, output_size=(H1, W1))
    return z


def embedding_concat_upsample(x, y):
    y = F.interpolate(y, size=(56, 56), mode='bicubic', align_corners=False)
    result = torch.cat((x, y), dim=1)
    return result


# 传入参数
def parse_args():
    # 创建一个参数解析器 名称为PaDiM
    parser = argparse.ArgumentParser('PaDiM')
    # 为参数解析器添加参数选项
    # 数据加载路径
    parser.add_argument('--data_path', type=str, default='D:/dataset/mvtec_anomaly_detection')
    # 输出路径
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    # 预训练模型架构选择
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    # 返回解析器对象
    return parser.parse_args()


def main():
    # 创建传参器
    args = parse_args()
    # load model
    # 选择加载预训练模型 resnet18或者resnet50
    if args.arch == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        # 下载预训练模型
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        #拼接后的最终通道数 256+512+1024=1792
        t_d = 1792
        d = 550
    # 模型使用gpu
    model.to(device)
    # 评估模式
    model.eval()
    # 固定各种随机数种子
    random.seed(1024)
    torch.manual_seed(1024)
    if device == "cuda":
        torch.cuda.manual_seed_all(1024)
    # 在0~t_d的范围内 挑选d个样本 形成idx索引  随机选取
    idx = torch.tensor(sample(range(0, t_d), d))
    # set model's intermediate outputs
    # 存放layer的中间特征
    outputs = []

    # 通过hook函数将模型中间输出的中间特征添加到outputs内
    def hook(module, input, output):
        outputs.append(output)

    # 将模型的指定层的输出，利用hook函数写入output列表中
    # 这个是为了取得resnet中三个不同layer的feature map 后续进行拼接得到embedding向量
    # layer1[-1]得到layer1层的最后一层中间输出
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    # 创建暂存模型参数的文件夹
    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    # 绘制评估指标曲线 生成一个1x2的子图画布
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]
    # 指标暂存 使用AUROC指标 在整张图片和像素级上进行计算
    total_roc_auc = []
    total_pixel_roc_auc = []

    # for class_name in mvtec.CLASS_NAMES:
    # 加载训练数据集
    train_dataset = mvtec.MVTecDataset(args.data_path, is_train=True)
    # for x,y,mask in train_dataset:
    #     print(x)#[3, 224, 224] 对应一张图片
    #     print(y)#图片对应的标签 如果是good 则为0
    #     print(mask)#[1, 224, 224] 对应的mask掩码 如果是good 则[1,224,224] 全为0
    #     break
    # DataLoader对于dataset数据集进行批量处理 同时固定数据集内存位置可加快读写速度
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    # for idx,data in enumerate(train_dataloader):
    #     x,y,mask = data
    #     print(x.shape)#[32, 3, 224, 224] 图片进行分块后得到bchw的数据格式
    #     print(y.shape)#[32]
    #     print(mask.shape)#[32, 1, 224, 224]
    #     print(len(x))#32
    #     break
    # break
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    test_dataset = mvtec.MVTecDataset(args.data_path, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
    # extract train set features
    # 创建训练特征存储路径
    train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % "bottle")
    # 当训练特征存储路径不存在时 代表我们还未提取特征 需要开始提取
    if not os.path.exists(train_feature_filepath):
        # 得到x,y，mask 但是不接收y，mask 仅需要x
        # 循环次数取决于total/batch_size
        for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % "bottle"):
            x = x.to(device)

            with torch.no_grad():
                # x输入模型进行预测 ，不接收返回值 我们需要的是模型中间三层的中间特征 而不是最后输出
                # 在模型预测时 会捕获中间三层的中间特征
                _ = model(x)
            # 按layer作为key 存放中间的输出values
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            #添加完成后需要清空outputs中捕获的中间特征 否则无法捕获下一批次的数据特征
            outputs
        for k, v in train_outputs.items():
            # 将三层中的张量拼接为一个 之前layer1中可能存在7个 （32，256，56，56）的 cat后得到（224，256，56，56）的一个
            train_outputs[k] = torch.cat(v)
            # Embedding concat
            # 嵌入向量拼接
        # 得到第一层的向量
        embedding_vectors = train_outputs['layer1']

        for layer_name in ['layer2', 'layer3']:
            # 将其另外两层的向量拼接成为一个
            embedding_vectors = embedding_concat_upsample(embedding_vectors, train_outputs[layer_name])
        # embedding_vectors最终得到[224, 1792, 56, 56]

        # 根据idx 按维度1 随机返回嵌入向量 idx中有550个随机值  随机选择其中550个通道 [224, 550, 56, 56]
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # 通过维度计算高斯分布
        # [224, 550, 56, 56]
        B, C, H, W = embedding_vectors.size()
        # 将向量维度转换为B C H*W 方便后续计算协方差 (B C H*W)可以认为 是B 个局部块的特征，其中每个局部块包含 C 个通道的特征
        #embedding_vectors.view(B, C, H * W) 每一列(B*C)都可以看作一个局部块的特征
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        # 按批次计算平均值 SUM/B  这样返回的是(C, H * W) C=550 所以得到的是(550, 3136)的均值矩阵
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        # 存放每个位置的协方差矩阵 (550, 550, 3136)
        cov = torch.zeros(C, C, H * W).numpy()
        # 创建正则项 也就是一个单位矩阵 (550,550)
        I = np.identity(C)
        # 遍历每个位置
        for i in range(H * W):
            # 计算协方差矩阵 rowvar=false 则每列都是一个变量进行计算
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        # save learned distribution
        # 计算完 将均值和协方差存放到列表中
        train_outputs = [mean, cov]
        with open(train_feature_filepath, 'wb') as f:
            # 保存为pkl文件
            pickle.dump(train_outputs, f)
    else:
        # 文件已经存在时候 加载特征
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            train_outputs = pickle.load(f)

    # 存放测试集图片路径 异常标签（全1） 掩膜mask地址
    gt_list = []
    gt_mask_list = []
    test_imgs = []
    # x为[32, 3, 224, 224] y与x的批次数一致[32] mask为[32, 1, 224, 224] 这是异常图片掩膜的tensor格式
    for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % "bottle"):
        # 将测试图x转换np格式后添加到列表内
        test_imgs.extend(x.cpu().detach().numpy())
        # y为图片的标签 类同于label  good文件夹下的一张图片 对应的y为0 非good文件夹下 一张图片 对应y为1
        gt_list.extend(y.cpu().detach().numpy())
        # 加载mask信息为[1,224,224]格式的张量 一个mask对应test下的一张异常图片
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # 模型预测
        with torch.no_grad():
            _ = model(x.to(device))
        # 预测完成后 获得中间三层的中间输出特征 加载到test_outputs 存放 后续进行拼接
        for k, v in zip(test_outputs.keys(), outputs):
            # k 为对应层layer1 or layer2 or layer3
            # v为output捕获的模型不同的三层得到的中间特征输出
           test_outputs[k].append(v.cpu().detach())
        # output捕获的中间特征添加到test_outputs后清空
        outputs = []
     #之前的v是一个layer下面有好几个中间层特征 因为for 循环要跑n轮 跑一轮 对应的layer下面就会多出一个中间层特征
    for k, v in test_outputs.items():
        #将一个layer下面的多个中间层特征拼接为1个 按批次拼接
        #比如有82张图片 这样batch_size=32 for循环走3轮 这样一个layer下面会有3个中间特征
        #例如 （32，256，56，56）*3 cat后等于(96 ，256，56， 56)
        test_outputs[k] = torch.cat(v,0)
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        #将3层不同的中间特征拼接为一个大的嵌入向量 统一维度尺寸
        embedding_vectors = embedding_concat_upsample(embedding_vectors, test_outputs[layer_name])
    #bottle总一共有83张图
    #随机选择其中550个通道 [83, 550, 56, 56]
    embedding_vectors=torch.index_select(embedding_vectors, 1, idx)

    #计算距离矩阵
    #得到各类尺寸
    B,C,H,W=embedding_vectors.size()
    # 将向量维度转换为B C H*W 方便后续计算协方差 (83, 550, 3136)
    embedding_vectors=embedding_vectors.view(B, C, H * W).numpy()
    dist_list=[]
    #一共循环H*W次
    for i in range(H * W):
        #取得训练集建立的[mean,cov]列表值
        #(550, 3136)->(550,)
        mean=train_outputs[0][:,i]
        #计算协方差矩阵的逆矩阵
        # cov_inv(550,550,3136)->(550, 550)
        #利用训练集的协方差矩阵 计算逆协方差矩阵 作为后续计算测试集的马氏距离的参数
        cov_inv=np.linalg.inv(train_outputs[1][:,:,i])
        #sample维度是（550，3136）
        #从 sample[:,i]->(550,) mean(550,) cov_inv(550, 550)
        #对于每个sample样本即x_ij 计算其马氏距离 dist列表存放83个距离值 因为sample共有83个值
        # 每个测试样本与训练样本整体数据分布可以用马氏距离都代表 这样后续可以通过距离大小来计算异常分数
        dist=[mahalanobis(sample[:,i],mean,cov_inv) for sample in  embedding_vectors]
        dist_list.append(dist)
    #列表中存放了H*W个值 每个值是一个83大小的列表 这样就形成了一个类似于（H*W,B）的马氏距离矩阵
    #其中H*W是局部块的数量 而83是测试样本的数量
    #转换为array 后利用np的transpose转置行列维度 从（3136，550）变成（550，3136）后reshape为（550，56，56）

    dist_list=np.array(dist_list).transpose(1,0).reshape(B,H,W)
    #转换为tensor格式
    dist_list=torch.tensor(dist_list)

    #进行插值 bicubic需要四维才能进行插值 利用unsqueeze 将input格式转换为(B,1,H,W) 尺寸为 x.size(2)=224 224为测试集图片的大小
    #插值后为(550，1,224，224)
    #后压缩维度为(550，224，224) 后转换为numpy格式
    # score_map 是根据马氏距离矩阵插值得到的
    # 代表了每个局部块（patch）或像素位置相对于训练集分布的异常程度
    score_map=F.interpolate(dist_list.unsqueeze(1),size=x.size(2),mode='bicubic',align_corners=False).squeeze().numpy()

    #使用高斯滤波器 平滑化分布
    for i in range(score_map.shape[0]):
        #score_map[i]为（224，224）的大小 利用gaussian_filter 卷积核=4 来进行高斯滤波
        score_map[i]=gaussian_filter(score_map[i],sigma=4)

    #对于score_map进行归一化操作
    max_score=score_map.max()
    min_score=score_map.min()
    #归一化score_map 将数据缩放到[0,1]之间
    scores=(score_map-min_score)/(max_score-min_score)

    #返回一个torch.return_types.max类型 包括values和indices values是每行的最大值
    #indices是每行最大值所在的索引
    #scores.reshpe(scores.shape[0],-1) 后的大小是83，50176（224*224）
    #索引的范围在（0-50175)之间
    img_scores=scores.reshape(scores.shape[0],-1).max(axis=1)
    #修改格式而不创建新的副本 减少内存开销
    #gt_list是图片标签一般为0 or 1 一张图片对应一个值
    #gt_list(83,)
    gt_list=np.asarray(gt_list)
    #gt_list 为真实label img_scores为预测的分数 (83,)

    #roc_curve用于评估二分类模型性能
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    #返回三个参数 ：假正例率 真正例率 阈值

    #计算auc值 值越大 模型效果越好
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    #添加到结果存储列表中 为后续求平均roc做准备
    total_roc_auc.append(img_roc_auc)
    print('image ROCAUC: %.3f' % (img_roc_auc))
    fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % ("bottle", img_roc_auc))






if __name__ == '__main__':
    main()
