#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：som-qnncd -> som-nn.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2022/11/12
@Desc   ：
==================================================
"""
import os
from itertools import product
import time

import numpy as np
from numpy import ndarray
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

from minisom import MiniSom

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from utils import run_r_model, read_csv2numpy, metric_par, metric_aar
from utils import device
from configs import best_baseline


class BaseNet(nn.Module):
    def __init__(self, q: ndarray):
        super(BaseNet, self).__init__()
        self._device = device
        self.q = self._to_Tensor(q)
        self.q_star = self.__get_q_stars_matrix()

    # q_stars
    def __get_q_stars_matrix(self, is_tensor=True) -> any:

        """
        生成交互式Q矩阵
        Args:
            is_tensor: 是否返回Tensor

        Returns:返回交互式Q矩阵

        """
        poly = PolynomialFeatures(degree=self.q.shape[1], include_bias=False, interaction_only=True)
        Q_Q_stars = poly.fit_transform(self.q.detach().cpu().numpy())
        Q_stars = Q_Q_stars[:, self.q.shape[1]:]
        if is_tensor:
            return torch.tensor(Q_stars, device=self._device).float()
        else:
            return Q_stars

    # loss
    def loss_func(self, y_pred, y_true):
        loss_f = nn.MSELoss()
        return loss_f(y_pred, y_true)

    # par
    def metric_par(self, y_pred, y_true):
        y_pred = y_pred if type(y_pred) == Tensor else self._to_Tensor(y_pred)
        y_true = y_true if type(y_true) == Tensor else self._to_Tensor(y_true)
        y_pred = torch.where(y_pred > 0.5,
                             torch.ones_like(y_pred, dtype=torch.float),
                             torch.zeros_like(y_pred, dtype=torch.float))

        par = torch.mean(torch.prod(torch.where((y_pred == y_true) == True,
                                                torch.ones_like(y_pred, dtype=torch.float),
                                                torch.zeros_like(y_pred, dtype=torch.float)), axis=1))  # par
        return par

    # acc
    def metric_acc(self, y_pred, y_true):
        y_pred = y_pred if type(y_pred) == Tensor else self._to_Tensor(y_pred)
        y_true = y_true if type(y_true) == Tensor else self._to_Tensor(y_true)
        y_pred = torch.where(y_pred > 0.5,
                             torch.ones_like(y_pred, dtype=torch.float),
                             torch.zeros_like(y_pred, dtype=torch.float))
        acc = torch.mean(1 - torch.abs(y_true - y_pred))  # aar
        return acc

    # optimizer
    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def _to_Tensor(self, data: ndarray) -> Tensor:
        """
        将ndarray快速转换为Tensor
        """
        data = pd.DataFrame(data).fillna(0).values

        data = data if type(data) == Tensor else torch.tensor(data=data, device=self._device, dtype=torch.float32)
        return data

    def _to_ndarray(self, data: Tensor) -> ndarray:
        """
        tensor to numpy
        Args:
            data:

        Returns:

        """
        data = data if type(data) == ndarray else data.detach().cpu().numpy()
        return data

    def _get_data_loader(self, X, Y, batch_size=64, shuffle=True, num_workers=0, verbose=False):
        """
        训练数据加载器
        Args:
            batch_size: 批量大小，默认64
            shuffle: 是否打乱顺序，默认为True
            num_workers: 是否开启多线程加载数据，默认不开启 workers =0

        Returns:dl 数据加载器

        """
        if verbose:
            print(f'训练样本大小：{X.shape[0]}')
        X = X if type(X) == Tensor else self._to_Tensor(X)
        Y = Y if type(Y) == Tensor else self._to_Tensor(Y)
        dl = DataLoader(TensorDataset(X, Y), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
        return dl

    def _train_net_step(self, features, labels):
        """
        网络一个批量的单步训练
        Args:
            features: 训练特征
            labels: 标签

        Returns:loss, metric  损失和准确率

        """
        # forword
        pre = self.forward(features)
        loss = self.loss_func(pre, labels)

        # Zero gradient
        self.optimizer.zero_grad()

        # backword
        loss.backward()

        # updated parameter
        self.optimizer.step()

    def train_net(self, X, y, epochs=1000, verbose=False):
        """
        网络模型训练函数
        Args:
            dl: 训练数据加载器
            epochs: 迭代次数

        Returns: None

        """
        dl = self._get_data_loader(X, y)
        for epoch in range(0, epochs):
            # train
            for features, labels in dl:
                self._train_net_step(features, labels)


class QNN(BaseNet):
    def __init__(self, q):
        super(QNN, self).__init__(q)
        self.mc = nn.Linear(self.q.shape[0], self.q.shape[1])
        self.lc1 = nn.Linear(self.q.shape[0], self.q.shape[1] + self.q_star.shape[1])
        self.lc2 = nn.Linear(self.q.shape[1] + self.q_star.shape[1], self.q.shape[1])
        self.sc1 = nn.Linear(self.q.shape[0], self.q_star.shape[1])
        self.sc2 = nn.Linear(self.q_star.shape[1], self.q.shape[1])

    def forward(self, x):
        x = x if type(x) == Tensor else self._to_Tensor(x)
        m = torch.relu(self.mc(x))
        l = torch.sigmoid(self.lc1(x))
        l = torch.sigmoid(self.lc2(l))
        s = torch.tanh(self.sc1(x))
        s = torch.tanh(self.sc2(s))
        y = torch.sigmoid(l * m + (1 - l) * s)
        return y

    def predictive(self, features):
        self.eval()
        features = features if type(features) == Tensor else self._to_Tensor(features)
        return self._to_ndarray(self.forward(features))


class PQNN(QNN):
    def __init__(self, q):
        super(PQNN, self).__init__(q)
        self._cons_mc = torch.t(self.q)
        self._cons_lc1 = torch.t(torch.cat((self.q, self.q_star), 1))
        self._cons_lc2 = torch.where((torch.t(self.q) @ torch.cat((self.q, self.q_star), 1)) > 0, 1, 0)
        self._cons_sc1 = torch.t(self.q_star)
        self._cons_sc2 = torch.where((torch.t(self.q) @ self.q_star) > 0, 1, 0)

    def _train_net_step(self, features, labels):
        """
        网络一个批量的单步训练
        Args:
            features: 训练特征
            labels: 标签

        Returns:loss, metric  损失和准确率

        """
        # forword
        pre = self.forward(features)
        loss = self.loss_func(pre, labels)

        # Zero gradient
        self.optimizer.zero_grad()

        # backword
        loss.backward()

        # updated parameter
        self.optimizer.step()
        self.mc.weight.data = self.mc.weight.data * self._cons_mc
        self.lc1.weight.data = self.lc1.weight.data * self._cons_lc1
        self.lc2.weight.data = self.lc2.weight.data * self._cons_lc2
        self.sc1.weight.data = self.sc1.weight.data * self._cons_sc1
        self.sc2.weight.data = self.sc2.weight.data * self._cons_sc2


class SOMNN():
    def __init__(self, q: ndarray, resp: ndarray, net: BaseNet, **somPars):
        self.q = q
        self.X = resp
        self.som_pars = somPars
        self.som = MiniSom(**somPars)
        self.net = net(self.q).to(device)
        self.acc_pre = None

    def __get_imp_irp(self, is_tensor=False)-> tuple:
        """
        计算理想掌握模式和理想反应模式
        Args:
            is_tensor: 是否返回tensor类型的数据， 默认为False返回ndarray类型的数据

        Returns:irp and imp

        """
        k = self.q.shape[1]
        imp = np.array(list(product([0, 1], repeat=self.q.shape[1]))).astype(float)
        irp = np.empty(shape=[0, self.q.shape[0]])
        for i in range(imp.shape[0]):
            row = np.apply_along_axis(lambda x: np.prod(np.power(imp[i, :], x)), axis=1, arr=self.q)
            irp = np.row_stack((irp, row))

        irp = irp.astype(float)
        x_one_zero, y_one_zero = self.__get_one_zero4resp()
        irp = np.append(irp, x_one_zero, axis=0)
        imp = np.append(imp, y_one_zero, axis=0)
        return irp, imp

    def _som_train(self, X, epochs=500):
        self.som.random_weights_init(X)
        self.som.train(X, epochs, random_order=True)

    def _get_x4som(self, y: ndarray)-> ndarray:
        x_y = np.append(self.X, y, axis=1)
        irp, imp = self.__get_imp_irp()
        irp_imp = np.append(irp, imp, axis=1)
        x4som = np.append(x_y, irp_imp, axis=0)
        return x4som

    def _get_x4nn(self, x_som, neighbor_d=1, is_tensor=False):
        """
        筛选训练数据通过som映射后的获胜神经元周围的具有代表性的训练数据
        Args:
            neighbor_d: 领域大小
            is_tensor: 是否返回tensor类型的数据

        Returns:返回data_som，具有代表性的训练数据(x:ndarray, y: ndarray)

        """
        data = x_som
        y = x_som[:, self.X.shape[1]:]
        xyp = pd.DataFrame(columns=['x', 'x_norm', 'l', 'y', 'p'])
        for d, x, l in zip(data, x_som, y):
            position = self.som.winner(d)
            xyp = xyp.append({'x': x, 'x_norm': d, 'l':l, 'y': tuple(l), 'p': position}, ignore_index=True)

        y_p_group = xyp['p'].groupby(xyp['y']).value_counts()

        y_p_df = pd.DataFrame(zip(y_p_group.index.to_frame().values[:, 0],
                                  y_p_group.index.to_frame().values[:, 1],
                                  y_p_group.values), columns=['y', 'p', 'n'])
        y_p_sort = y_p_df.sort_values(by='n', ascending=False)

        y_p_n = pd.DataFrame(columns=xyp.columns)
        for y, p, n in y_p_sort.values:
            if y not in y_p_n.y.values.tolist():
                if p not in y_p_n['p'].values.tolist():
                    for i in range(-neighbor_d, neighbor_d + 1):
                        for j in range(-neighbor_d, neighbor_d + 1):
                            position = (p[0] + i, p[1] + j)
                            df = xyp.loc[xyp.y==y].loc[xyp.p==position]
                            if not df.empty:
                                y_p_n = y_p_n.append(df, ignore_index=True)
                    continue
                elif p in y_p_n['p'].values.tolist():
                    for i in range(-neighbor_d, neighbor_d + 1):
                        for j in range(-neighbor_d, neighbor_d + 1):
                            position = (p[0] + i, p[1] + j)
                            df = xyp.loc[xyp.y == y].loc[xyp.p == position]
                            if not df.empty:
                                y_p_n = y_p_n.append(df, ignore_index=True)
        # data_som = pd.concat([DataFrame(y_p_n.x.to_list()),
        #                       DataFrame(y_p_n.y.to_list()),
        #                       DataFrame(y_p_n.p.to_list())
        #                       ], axis=1)
        ypn_x = pd.DataFrame(y_p_n.x.to_dict()).T.values[:, 0:self.X.shape[1]]
        # if self.__is_x_y:
        #     ypn_x = pd.DataFrame(y_p_n.x.to_dict()).T.values
        ypn_y = pd.DataFrame(y_p_n.y.to_dict()).T.values

        if is_tensor:
            return self.__to_Tensor(ypn_x), self.__to_Tensor(ypn_y)
        else:
            return ypn_x, ypn_y

    def __is_continue(self, threshold, y, **kwargs):
        y_hat = self.net.predictive(self.X)
        acc = self.net.metric_acc(y, y_hat)

        self.acc_pre = self.acc_pre if self.acc_pre else acc/2
        if kwargs.get('verbose'):
            print(f'som-nn train process epochs:{kwargs.get("epochs")}----aar:{acc}')
        if threshold > abs(acc-self.acc_pre):
            self.acc_pre = None
            return True
        self.acc_pre = acc
        return False

    def __get_amp2binary(self, smp:ndarray) -> ndarray:
        """
        将带有小数的学生技能掌握模式转换为只包含0,1的形式
        Args:
            smp:

        Returns:smp，表示未掌握，1表示掌握

        """
        smp = np.where(smp > 0.5, 1, smp)
        smp = np.where(smp < 0.5, 0, smp)
        for j in range(self.q.shape[1]):
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

    def __get_one_skill4q(self) -> dict:
        """
        根据q矩阵获取每个技能被单独的一道题考察的技能和题
        技能为key, 题标号构成的列表为value
        ex:k=0, v=[1, 3] 表示第0个技能，在第1题和第3题中被单独考察了
        Returns:kw
        """
        kw = {}
        for i in range(self.q.shape[1]):
            row_i = np.zeros(self.q.shape[1])
            row_i[i] = 1
            index = np.argwhere(np.apply_along_axis(lambda x: (x == row_i).all(), axis=1, arr=self.q) == True)
            if index.shape[0] > 0:
                kw[i] = index
        return kw

    def __get_amp4orp(self, orp: ndarray) -> ndarray:
        kw = self.__get_one_skill4q()
        amp = np.zeros((0, self.q.shape[1]))
        for i in range(orp.shape[0]):
            row = np.zeros([1, self.q.shape[1]])
            if (orp[i]==1).all():  # 学生全部答对
                row = np.ones([1, self.q.shape[1]])
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
                        if np.sum(orp[i][self.q[:, k]==1]) / np.sum(self.q[:, k]) > 0.5:
                            row[:, k] = 1
                        else:
                            row[:, k] =0
                for col in range(self.q.shape[1]):  # 混合知识点的题
                    if row[:, col]==0:
                        if np.sum(orp[i][self.q[:, col]==1]) / np.sum(self.q[:, col]) > 0.5:
                            row[:, col] = 1
                        elif np.sum(orp[i][self.q[:, col]==1]) / np.sum(self.q[:, col]) == 0.5:
                            row[:, col] = 0.5
                        else:
                            row[:, col] = 0
            amp = np.append(amp, row, axis=0)
        amp = self.__get_amp2binary(amp)  # 如果该技能的正误率为0.5，则看全部的人在该题上的作答情况，大多数人对，则为对
        return amp

    def __get_one_zero4resp(self) -> tuple:
        y = np.ones([1, self.q.shape[1]])
        x = np.ones([1, self.q.shape[0]])
        for i in range(self.X.shape[0]):
            if (self.X[i] == 1).all():
                y_i = np.ones([1, self.q.shape[1]])
                x_i = np.ones([1, self.q.shape[0]])
                x = np.append(x, x_i, axis=0)
                y = np.append(y, y_i, axis=0)
            elif (self.X[i] == 0).all():
                y_i = np.zeros([1, self.q.shape[1]])
                x_i = np.zeros([1, self.q.shape[0]])
                x = np.append(x, x_i, axis=0)
                y = np.append(y, y_i, axis=0)

        return x, y

    def _supervised(self, target, threshold=0.005, epochs=10, verbose=True):
        x4som = self._get_x4som(target)
        self._som_train(x4som)
        X = self.X
        y = target
        while epochs > 0:
            X4som, y4som = self._get_x4nn(x4som)
            X = np.append(X, X4som, axis=0)
            y = np.append(y, y4som, axis=0)
            self.net.train_net(X, y)
            if self.__is_continue(threshold, epochs=epochs, verbose=verbose, y=target):
                break
            epochs -= 1

    def _unsupervised(self, threshold=0.005, epochs=10, verbose=True):
        y_aux = self.__get_amp4orp(self.X)
        x4som = self._get_x4som(y_aux)
        self._som_train(x4som)
        x, y = self._get_x4nn(x4som)
        x_one_zero, y_one_zero = self.__get_one_zero4resp()
        x = np.append(x, x_one_zero, axis=0)
        y = np.append(y, y_one_zero, axis=0)
        irp, imp = self.__get_imp_irp()
        while epochs > 0:
            x = np.append(x, irp, axis=0)
            y = np.append(y, imp, axis=0)
            self.net.train_net(x, y)
            if self.__is_continue(threshold, epochs=epochs, verbose=verbose, y=y_aux):
                break
            y_aux = (self.net.predictive(self.X)+y_aux) / 2
            x4som = self._get_x4som(y_aux)
            self._som_train(x4som)
            x, y = self._get_x4nn(x4som)
            x_one_zero, y_one_zero = self.__get_one_zero4resp()
            x = np.append(x, x_one_zero, axis=0)
            y = np.append(y, y_one_zero, axis=0)
            irp = np.append(irp, irp, axis=0)
            imp = np.append(imp, imp, axis=0)
            epochs -= 1

    def predictive(self, X, is_train_net=False, type: Literal['supervised', 'unsupervised'] = 'supervised', **kwargs):
        assert type in ("supervised", "unsupervised"), "invalid est_type: %r" % (type)
        if is_train_net:
            if type == 'supervised':
                self._supervised(y=kwargs.get('y'))
            else:
                self._unsupervised()
        y_hat = self.net.predictive(X)
        return y_hat


class Test:
    q = pd.read_csv('../data/ex/compare4som-qnn/sim/SD1H/50/q.csv').values[:, 1:].astype(int)
    resp = pd.read_csv('../data/ex/compare4som-qnn/sim/SD1H/50/resp.csv').values[:, 1:].astype(int)
    target = pd.read_csv('../data/ex/compare4som-qnn/sim/SD1H/50/target.csv').values[:, 1:].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(resp, target, test_size=0.25, random_state=0)
    som_parm = {
        'x': 10,
        'y': 10,
        'input_len': q.shape[1] + q.shape[0],
        'sigma': 1.0,
        'learning_rate': 0.01,
        'neighborhood_function': 'gaussian'}
    somqnn = SOMNN(q, X_train, PQNN, **som_parm)

    @staticmethod
    def test_qnn():
        qnn = PQNN(Test.q).to(device)
        qnn.train_net(Test.X_train, Test.y_train)
        y_hat = qnn.predictive(Test.resp)
        y_hat_test = qnn.predictive(Test.X_test)
        y_hat_dina = run_r_model(Test.q, Test.resp)
        print(f'y_hat_aar:{qnn.metric_acc(Test.target, y_hat)}')
        print(f'y_hat_test_aar:{qnn.metric_acc(Test.y_test, y_hat_test)}')
        print(f'y_hat_dina_aar:{qnn.metric_acc(Test.target, y_hat_dina)}')

    @staticmethod
    def test_somnn(type: Literal['supervised', 'unsupervised'] = 'supervised'):
        assert type in ("supervised", "unsupervised"), "invalid est_type: %r" % (type)
        if type == 'supervised':
            Test.somqnn._supervised(target=Test.y_train)
        else:
            Test.somqnn._unsupervised()
        y_hat = Test.somqnn.predictive(Test.resp)
        y_hat_test = Test.somqnn.predictive(Test.X_test)
        y_hat_dina = run_r_model(Test.q, Test.resp)
        print(f'y_hat_aar:{Test.somqnn.net.metric_acc(Test.target, y_hat)}')
        print(f'y_hat_aar_test:{Test.somqnn.net.metric_acc(Test.y_test, y_hat_test)}')
        print(f'y_hat_aar_dina:{Test.somqnn.net.metric_acc(Test.target, y_hat_dina)}')


class Eval:
    def __init__(self, path, net=PQNN, som_x=10, som_y=10, som_sigma=1.0, som_lr=0.01, nei_fun='gaussian'):
        self.path = path
        self.net = net
        self.som_pars = {
                        'x': som_x,
                        'y': som_y,
                        # 'input_len': q.shape[1] + q.shape[0],
                        'sigma': som_sigma,
                        'learning_rate': som_lr,
                        'neighborhood_function': nei_fun}

    def somnn_run4one(self, q: ndarray, x_train: ndarray, type: Literal['supervised', 'unsupervised'] = 'supervised', **kwargs):
        assert type in ("supervised", "unsupervised"), "invalid est_type: %r" % (type)
        try:
            kwargs.get('som_pars')
        except Exception as e:
            print("please input som_pars")
        somnn = SOMNN(q, x_train, self.net, **kwargs.get('som_pars'))
        if type == 'supervised':
            try:
                kwargs.get('target')
            except Exception as e:
                print("supervised methods require input target")
            somnn._supervised(target=kwargs.get('target'), verbose=False)
        else:
            somnn._unsupervised(verbose=False)
        return somnn

    def get_acc4nn(self, x_pre, y_target, som_nn: SOMNN, unsom_nn: SOMNN, qnn: QNN):
        y_hat_sup = som_nn.predictive(x_pre)
        y_hat_unsup = unsom_nn.predictive(x_pre)
        y_hat_qnn = qnn.predictive(x_pre)

        aar_sup = metric_aar(y_hat_sup, y_target)
        aar_unsup = metric_aar(y_hat_unsup, y_target)
        aar_qnn = metric_aar(y_hat_qnn, y_target)

        par_sup = metric_par(y_hat_sup, y_target)
        par_unsup = metric_par(y_hat_unsup, y_target)
        par_qnn = metric_par(y_hat_qnn, y_target)
        return round(aar_qnn, 3), round(par_qnn, 3), round(aar_sup, 3), round(par_sup, 3), round(aar_unsup, 3), round(par_unsup, 3)

    def qnn_rum4one(self, q, X_train, y_train):
        qnn = PQNN(q).to(device)
        qnn.train_net(X_train, y_train)
        return qnn

    def get_baseline_acc(self, q, x, y, mod_name):
        y_hat_r = run_r_model(q, x, mod_name=mod_name)
        y_hat_r_np = run_r_model(q, x, mod_name='AlphaNP')

        base_aar = metric_aar(y_hat_r, y)
        np_aar = metric_aar(y_hat_r_np, y)

        base_par = metric_par(y_hat_r, y)
        np_par = metric_par(y_hat_r_np, y)

        return round(base_aar, 3), round(base_par, 3), round(np_aar, 3), round(np_par, 3)

    def somnn_run4path(self, epochs=5, label='target_3.csv', verbose=True, is_saive=True):
        record_acc = pd.DataFrame(columns=['data_name', 'base_aar', 'base_par', 'np_aar', 'np_par', 'qnn_aar',
                                           'qnn_par', 'nn_sup_aar', 'nn_sup_par', 'nn_unsup_aar', 'nn_unsup_par'])
        somnn_supervised = None
        somnn_unsupervised = None
        if verbose:
            print(record_acc.columns)
        for root, dirs, files in os.walk(self.path):
            if label in files and 'q.csv' in files and 'resp.csv' in files:
                iters = epochs
                print(root)
                os.path.basename(root)
                os.path.split(root)
                q = read_csv2numpy(os.path.join(root, 'q.csv'))
                resp = read_csv2numpy(os.path.join(root, 'resp.csv'))
                target = read_csv2numpy(os.path.join(root, label))
                X_train, X_test, y_train, y_test = train_test_split(resp, target, test_size=0.25)
                self.som_pars['input_len'] = q.shape[1] + q.shape[0]
                data_name = os.path.basename(os.path.split(root)[0]) + os.path.basename(root)
                baseline = best_baseline.get(data_name)
                base_acc = list(self.get_baseline_acc(q, resp, target, baseline))

                while iters > 0:
                    somnn_supervised = self.somnn_run4one(q, X_train, som_pars=self.som_pars, target=y_train)
                    somnn_unsupervised = self.somnn_run4one(q, X_train, type='unsupervised', som_pars=self.som_pars)
                    qnn = self.qnn_rum4one(q, X_train, y_train)
                    nn_acc = list(self.get_acc4nn(resp, target, somnn_supervised, somnn_unsupervised, qnn))
                    columns = [f'{data_name}_'+str(iters)] + base_acc + nn_acc
                    record_acc.loc[len(record_acc)] = columns
                    iters -= 1
                    if verbose:
                        print(columns)
                #     break
                # break

        if is_saive:
            record_acc.to_csv(os.path.join(self.path, f'aar_par_result_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv'), index=False)
        else:
            print(record_acc)


if __name__ == '__main__':
    # Test.test_qnn()
    # Test.test_somnn()
    # Test.test_somnn(type='unsupervised')
    path = '../data/ex/compare4som-qnn/sim/SD1H/50'
    eval = Eval(path)
    sup, unsup = eval.somnn_run4path(is_saive=False)




