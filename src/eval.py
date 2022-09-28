#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：som-qnncd -> eval.py.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2022/9/23
@Desc   ：
==================================================
"""
import os
import warnings

from options import som_parm
from options import dir_data_real
from options import dir_data
from utils import run_r_models
from utils import read_csv2numpy
from models import PNN
from models import SomQnnCD


def get_labels4R2real():
    for dir in dir_data_real:
        q_file = f'{dir}q.csv'
        resp_file = f'{dir}resp.csv'
        run_r_models(q_file=q_file, resp_file=resp_file, save_dir=dir)


def get_labels4R2sim():
    pass


def run():
    warnings.filterwarnings(action='ignore')
    for path in dir_data:
        for root, dirs, files in os.walk(path):
            if 'resp.csv' in files:
                try:
                    if 'label_expert.csv' not in files:
                        raise ValueError('the file of label_expert.csv dose not exist')
                except Exception as e:
                    print(e)
                resp = read_csv2numpy(os.path.join(root, 'resp.csv'))
                if 'q.csv' in files:
                    q = read_csv2numpy(os.path.join(root, 'q.csv'))
                else:
                    q = read_csv2numpy(os.path.join(os.path.dirname(root), 'q.csv'))

                print(f'==================={root}===================')

                # label_expert = read_csv2numpy(os.path.join(root, 'label_expert.csv'))
                label_expert = read_csv2numpy(os.path.join(root, 'label.csv'))
                som_parm['input_len'] = q.shape[1] + q.shape[0]

                model = SomQnnCD(Q_Matrix=q, X=resp, net=PNN, is_x_y=False, **som_parm)

                model.train_unsupervised(threshold=0.9)
                y_hat_unsupervised = model.predicate(train_type='unsupervised', y=label_expert)

                model.train_semi_supervised()
                y_hat_semi_supervised = model.predicate(train_type='semi_supervised', y=label_expert)

                model.train_supervised(label_expert)
                y_hat_supervised = model.predicate(train_type='supervised', y=label_expert)

                model.run_classical_model(q, resp, label_expert)


def rum_all(is_save=True):
    warnings.filterwarnings(action='ignore')
    result={}
    for path in dir_data:
        for root, dirs, files in os.walk(path):
            if 'resp.csv' in files:
                try:
                    if 'label_expert.csv' not in files:
                        raise ValueError('the file of label_expert.csv dose not exist')
                except Exception as e:
                    print(e)
                resp = read_csv2numpy(os.path.join(root, 'resp.csv'))
                if 'q.csv' in files:
                    q = read_csv2numpy(os.path.join(root, 'q.csv'))
                else:
                    q = read_csv2numpy(os.path.join(os.path.dirname(root), 'q.csv'))

                print(f'==================={root}===================')

                label_expert = read_csv2numpy(os.path.join(root, 'label_expert.csv'))
                som_parm['input_len'] = q.shape[1] + q.shape[0]

                model = SomQnnCD(Q_Matrix=q, X=resp, net=PNN, is_x_y=False, **som_parm)

                model.train_unsupervised(threshold=0.92)
                y_hat_unsupervised = model.predicate(train_type='unsupervised', y=label_expert)

                model.train_semi_supervised()
                y_hat_semi_supervised = model.predicate(train_type='semi_supervised', y=label_expert)

                model.train_supervised(label_expert)
                y_hat_supervised = model.predicate(train_type='supervised', y=label_expert)

                model.run_classical_model(q, resp, label_expert)


def test():
    get_labels4R2real()


if __name__ == '__main__':
    run()
