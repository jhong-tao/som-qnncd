#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：som-qnncd -> utils.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2022/9/20
@Desc   ：
==================================================
"""
import os

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame

from rpy2.robjects import ListVector
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def data4R2csv(data_name: str, dir: str ='./data/', re=False):
    importr('NPCD')
    importr('CDM')
    importr('GDINA')
    file_dir = dir+data_name+'.csv'
    data = pd.DataFrame(pandas2ri.rpy2py(r(data_name)))
    if re:
        return data
    else:
        if not os.path.exists(dir):
            dir = os.mkdir(dir)
        data.to_csv(file_dir)


def get_data4R(path='./data/', **kwargs):
    importr('NPCD')
    importr('CDM')
    'Data.DINA',
    'fraction.subtraction.data',
    'data.fraction1',
    'data.timss07.G4.lee',
    'data.timss07.G4.py'
    if kwargs:
        kwargs = kwargs
    else:
        kwargs={
            'Data.DINA': ['$Q', '$response', '$true.alpha', '$true.par$slip', '$true.par$guess'],
            'frac20': ['$Q', '$dat'],
            'data.fraction1': ['$q.matrix', '$data'],
            'data.timss07.G4.lee': ['$q.matrix', '$data'],
            'data.timss07.G4.py': ['$q.matrix', '$data']
        }
    for k, v in kwargs.items():
        dir = path + f'{k}/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        for d in v:
            data4R2csv(k+d, dir)


def get_resp4timss25_7():
    importr('CDM')
    resp = pandas2ri.rpy2py(r('data.timss07.G4.lee$data'))
    q_15 = pandas2ri.rpy2py(r('data.timss07.G4.lee$q.matrix'))
    q_7 = pandas2ri.rpy2py(r('data.timss07.G4.py$q.matrix'))


def get_one_skill4q(q) -> dict:
    """
    根据q矩阵获取每个技能被单独的一道题考察的技能和题
    技能为key, 题标号构成的列表为value
    ex:k=0, v=[1, 3] 表示第0个技能，在第1题和第3题中被单独考察了
    Returns:kw
    """
    kw = {}
    for i in range(q.shape[1]):
        row_i = np.zeros(q.shape[1])
        row_i[i] = 1
        index = np.argwhere(np.apply_along_axis(lambda x: (x == row_i).all(), axis=1, arr=q) == True)
        if index.shape[0] > 0:
            kw[i] = index
    return kw


def get_amp2binary(q, smp:ndarray) -> ndarray:
    """
    将带有小数的学生技能掌握模式转换为只包含0,1的形式
    Args:
        smp:

    Returns:smp，表示未掌握，1表示掌握

    """
    smp = np.where(smp > 0.5, 1, smp)
    smp = np.where(smp < 0.5, 0, smp)
    for j in range(q.shape[1]):
        s = pd.DataFrame(smp[:, j]).value_counts(normalize=True)
        if 1.0 in s.index and 0.0 in s.index:
            if s[1.0] > s[0.0]:
                smp[:, j] = np.where(smp[:, j] > 0, 1, smp[:, j])
            else:
                smp[:, j] = np.where(smp[:, j] < 1, 0, smp[:, j])
        elif 1.0 in s.index and 0.0 not in s.index:
            smp[:, j] = np.where(smp[:, j] > 0, 1, smp[:, j])
        elif 1.0 not in s.index and 0.0 in s.index:
            smp[:, j] = np.where(smp[:, j] < 1, 0, smp[:, j])
    return smp


def data2csv(data, dir, file='label.csv'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    pd.DataFrame(data).to_csv(dir+file)


def get_labels(q_file, resp_file, y_dir):
    q = pd.read_csv(q_file).values[:, 1:]
    orp = pd.read_csv(resp_file).values[:, 1:]
    kw = get_one_skill4q(q)
    amp = np.zeros((0, q.shape[1]))
    for i in range(orp.shape[0]):
        row = np.zeros([1, q.shape[1]])
        if (orp[i]==1).all():  # 学生全部答对
            row = np.ones([1, q.shape[1]])
        elif (orp[i]==0).all():  # 学生全部答错
            row = row
        else:  # 部分答对
            for k, v in kw.items():  # 先判断只考察单个知识点的题是否答对
                v = v.flatten()
                num = 0
                for t in v:
                    if orp[i][t]==1:
                        num += 1
                if num / v.shape[0] > 0.5:
                    row[:, k] = 1
                else:
                    if np.sum(orp[i][q[:, k]==1]) / np.sum(q[:, k]) > 0.5:
                        row[:, k] = 1
                    else:
                        row[:, k] =0
            for col in range(q.shape[1]):  # 混合知识点的题
                if row[:, col]==0:
                    if np.sum(orp[i][q[:, col]==1]) / np.sum(q[:, col]) > 0.5:
                        row[:, col] = 1
                    elif np.sum(orp[i][q[:, col]==1]) / np.sum(q[:, col]) == 0.5:
                        row[:, col] = 0.5
                    else:

                        row[:, col] = 0
        amp = np.append(amp, row, axis=0)
    amp = get_amp2binary(q, amp)  # 如果该技能的正误率为0.5，则看全部的人在该题上的作答情况，大多数人对，则为对
    data2csv(amp, y_dir)


if __name__ == '__main__':
    get_labels('./data/real/timss07/25_15/q.csv', './data/real/timss07/25_15/resp.csv', './data/real/timss07/25_15/')