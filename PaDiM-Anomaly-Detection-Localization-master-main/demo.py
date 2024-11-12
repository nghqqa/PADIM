# -*- encoding: utf-8 -*-
'''
@File    :   demo.py   
@Contact :   2801511808@qq.com
@License :   (C)Copyright 2018-2024
  
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2024/11/11 14:25   ngh      1.0         None
'''
# -*- encoding: utf-8 -*-
'''
@File    :   res_demo.py   
@Contact :   2801511808@qq.com
@License :   (C)Copyright 2018-2024
  
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2024/11/10 23:53   ngh      1.0         None
'''
import random
from random import sample
import argparse
import numpy as np
import os
import pickle

from torch.cuda import device
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
from torchvision import models
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights,ResNet18_Weights
import datasets.mvtec as mvtec

# if __name__ == '__main__':
#     input_tensor = torch.randn(550,224,224)
#     print(input_tensor.reshape(input_tensor.shape[0],-1).shape)
#     #550 50176
#     img_scores = input_tensor.reshape(input_tensor.shape[0], -1).max(axis=1)
#     # print(img_scores.values)
#     print(img_scores.indices)
#     # print(len(img_scores.values))

from sklearn.metrics import roc_auc_score


# 假设 y_true 是真实标签，y_scores 是模型预测分数
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

# 计算 AUC-ROC 分数
auc_score = roc_auc_score(y_true, y_scores)
print("AUC-ROC Score:", auc_score)  # 输出大约为 0.75