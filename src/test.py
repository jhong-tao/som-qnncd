#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：som-qnncd -> test.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2022/9/24
@Desc   ：
==================================================
"""
import os
import shutil

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame

from rpy2.robjects import ListVector
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

from utils import read_csv2numpy, metric_par, metric_aar, run_r_models, get_labels4expert, del_files2
from model import Test as t
from somnn import Eval


def acc_label_expert(path):
    for root, dirs, files in os.walk(path):
        if 'label_expert.csv' in files and 'label.csv' in files:
            y = read_csv2numpy(os.path.join(root, 'label.csv'))
            y_e = read_csv2numpy(os.path.join(root, 'label_expert.csv'))
            aar = metric_aar(y_e, y)
            par = metric_par(y_e, y)
            print(f'------{root}---y-y_e--aar:{aar}----par:{par}------')


def acc_model_expert(path):
    for root, dirs, files in os.walk(path):
        if 'label_expert.csv' in files:
            y = read_csv2numpy(os.path.join(root, 'label_expert.csv'))
            for label in files:
                if label != 'label_expert.csv' and 'label' in label:
                    y_e = read_csv2numpy(os.path.join(root, label))
                    aar = metric_aar(y_e, y)
                    par = metric_par(y_e, y)
                    print(f'------{root}---expert_{label}--aar:{aar}----par:{par}------')


def acc_model_label(path):
    for root, dirs, files in os.walk(path):
        if 'label.csv' in files:
            y = read_csv2numpy(os.path.join(root, 'label.csv'))
            for label in files:
                if label != 'label.csv' and 'label' in label:
                    y_e = read_csv2numpy(os.path.join(root, label))
                    aar = metric_aar(y_e, y)
                    par = metric_par(y_e, y)
                    print(f'------{root}---label_{label}--aar:{aar}----par:{par}------')


def get_real_label4R(path):
    for root, dirs, files in os.walk(path):
        if 'q.csv' in files and 'resp.csv' in files:
            q = os.path.join(root, 'q.csv')
            resp = os.path.join(root, 'resp.csv')
            run_r_models(q, resp, root)


def get_real_label_expert4dir(path):
    for root, dirs, files in os.walk(path):
        if 'q.csv' in files and 'resp.csv' in files:
            q = os.path.join(root, 'q.csv')
            resp = os.path.join(root, 'resp.csv')
            y = get_labels4expert(q, resp)
            pd.DataFrame(y).to_csv(os.path.join(root, 'label_expert.csv'))


def get_sim_label4R(path):
    # todo 后续删除
    for root, dirs, files in os.walk(path):
        if 'resp.csv' in files:
            q = os.path.join(root, os.pardir, os.pardir, 'q.csv')
            resp = os.path.join(root, 'resp.csv')
            run_r_models(q, resp, root)
            print(root)


def test_all_nn4all_data(path, label_name=['label', 'label_expert']):
    for root, dirs, files in os.walk(path):
        if 'resp.csv' in files:
            resp = read_csv2numpy(os.path.join(root, 'resp.csv'))
            if 'q.csv' in files:
                q = read_csv2numpy(os.path.join(root, 'q.csv'))
            else:
                try:
                    q = read_csv2numpy(os.path.join(root, os.pardir, os.pardir, 'q.csv'))
                except Exception as e:
                    q = read_csv2numpy(os.path.join(root, os.pardir, 'q.csv'))
            for label in label_name:
                if f'{label}.csv' in files:
                    y = read_csv2numpy(os.path.join(root, f'{label}.csv'))
                    dir = os.path.join(root, label)
                    if not os.path.exists(dir):
                        os.mkdir(dir)
                    else:

                        del_files2(dir)
                t.test_all(dir, q, resp, y, 500, True, True, False, True)
            print(root)
        # break




if __name__ == '__main__':
    # path = '../data/real/'
    # path = '../data/real/dina/'
    # path = '../data/ex/compare4nn/'

    # path = '../data/sim/high/10_3/acdm/100/'  # 可以
    # path = '../data/ex/compare4nn/high/10_3/300/'
    # path = '../data/sim/low/20_3/gdina/1000/'
    #
    # test_all_nn4all_data(path)

    path = '../data/ex/compare4som-qnn/real'
    eval = Eval(path)
    eval.somnn_run4path()





