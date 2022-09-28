#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
==================================================
@Project -> File   ：CognitiveDiagnosis -> somqnncd.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2022/8/9
@Desc   ：
==================================================
"""
from itertools import product
import datetime
from collections import Counter
import random
import warnings
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchinfo import summary

from minisom import MiniSom

from functools import singledispatch

from rpy2.robjects import ListVector
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri


class Utils:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_q_matrix(file: str)-> ndarray:
        df_q = pd.read_csv(file).values[:, 1:].astype(float)
        return df_q

    @staticmethod
    def get_q_stars_matrix(q_matrix: ndarray)-> ndarray:
        poly = PolynomialFeatures(degree=q_matrix.shape[1], include_bias=False, interaction_only=True)
        Q_Q_stars = poly.fit_transform(q_matrix).astype(float)
        Q_stars = Q_Q_stars[:, q_matrix.shape[1]:].astype(float)
        return Q_stars

    @staticmethod
    def get_imp_irp(q_matrix: ndarray)-> tuple:
        k = q_matrix.shape[1]
        imp = np.array(list(product([0, 1], repeat=q_matrix.shape[1]))).astype(float)
        irp = np.empty(shape=[0, q_matrix.shape[0]])
        for i in range(imp.shape[0]):
            row = np.apply_along_axis(lambda x: np.prod(np.power(imp[i, :], x)), axis=1, arr=q_matrix)
            irp = np.row_stack((irp, row))

        irp = irp.astype(float)
        return imp, irp

    @staticmethod
    def get_orp_smp(file: str)-> ndarray:
        orp = pd.read_csv(file).values[:, 1:].astype(float)
        return orp

    @staticmethod
    def get_device()-> torch.device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device

    @staticmethod
    def get_data_loader(x, y, batch_size=64, shuffle=True, num_workers=0):
        dl = DataLoader(TensorDataset(x, y), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
        return dl

    @staticmethod
    def train_step(model, features, labels):
        # forword
        pre = model(features)
        loss = model.loss_func(pre, labels)
        metric = model.metric_func(pre, labels, metric='aar')

        # Zero gradient
        model.optimizer.zero_grad()

        # backword
        loss.backward()

        # updated parameter
        model.optimizer.step()

        # constraint of Q and Q_stars
        # model.mc.weight.data = model.mc.weight.data * model.q.T
        # model.wm.data =  model.wm.data * model.q
        # # model.sc1.weight.data = model.sc1.weight.data * model.q_star.T
        # model.ws1.data = model.ws1.data * model.q_star
        # # model.sc2.weight.data = model.sc2.weight.data * (model.q.T @ model.q_star)
        # model.ws2.data = model.ws2.data * (model.q_star.T @ model.q)

        return loss, metric

    @staticmethod
    def printbar():
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)

    @staticmethod
    def train_model(model, epochs, dl: DataLoader):
        for epoch in range(1, epochs+1):
            loss_list, metric_list = [], []
            for features, labels in dl:
                lossi, metrici = Utils.train_step(model, features, labels)
                loss_list.append(lossi)
                metric_list.append(metrici)
            loss = torch.mean(torch.tensor(loss_list, device=Utils.device))
            metric = torch.mean(torch.tensor(metric_list, device=Utils.device))

            if epoch%50==0:
                Utils.printbar()
                print("epoch =", epoch, "loss = ", loss, "metric = ", metric)

    @staticmethod
    def get_aux_smp(q_matrix, orp, type='numpy')-> any:
        smp = np.empty([0, q_matrix.shape[1]])
        for index_row in range(orp.shape[0]):
            row = (((orp[index_row]*q_matrix.T) / q_matrix.sum(axis=1)).sum(axis=1))
            if 0.5 in row and 1.0 in row:
                counter = Counter(row)
                if counter.get(0.5) > counter.get(1.0):
                    row = np.where(row==0.5, 0.5-random.random()/4, row)
                elif counter.get(0.5) < counter.get(1.0):
                    row = np.where(row==0.5, 0.5+random.random()/4, row)
            smp = np.append(smp, row.reshape([1, row.shape[0]]), axis=0)
            smp = np.where(smp>1., 1.0, smp)
        if type == 'numpy':
            return smp
        elif type == 'tensor':
            return torch.tensor(data=smp, device=Utils.device,).float()
        return smp, torch.tensor(data=smp, device=Utils.device,).float()

    @staticmethod
    def get_neighbor_position(self, position: tuple, neighbor_d: int) -> list:
        """
        根据位置 position 获取距离为neighbor_d的领域位置
        Args:
            position: som 竞争层位置，e.g.:(3, 4)
            neighbor_d: position的领域距离

        Returns:p_neighbor， 距离position 为neighbor_d的领域位置

        """
        p_neighbor = list()
        for i in range(-neighbor_d, neighbor_d+1):
            for j in range(-neighbor_d, neighbor_d+1):
                p = (position[0]+i, position[1]+j)
                p_neighbor.append(p)
        return p_neighbor

    @staticmethod
    def get_quality_X(x_train: ndarray, y_train: ndarray, som: MiniSom, neighbor_d=1):
        """
        筛选训练数据通过som映射后的获胜神经元周围的具有代表性的训练数据
        Args:
            x_train: 训练特征
            y_train: 训练标签
            som: SOM网络
            neighbor_d: 领域大小

        Returns:返回data_som，具有代表性的训练数据

        """
        # data = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
        data = x_train
        xyp = pd.DataFrame(columns=['x', 'x_norm', 'l', 'y', 'p'])
        for d, x, l in zip(data, x_train, y_train):
            position = som.winner(d)
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
        data_som = pd.concat([DataFrame(y_p_n.x.to_list()),
                              DataFrame(y_p_n.y.to_list()),
                              DataFrame(y_p_n.p.to_list())
                              ], axis=1)
        return data_som, xyp

    @staticmethod
    def get_data4r(package: str, q: str, *args, **kwargs) -> tuple:
        """
        从R语言包中获取数据
        Args:
            package: NPCD, CDM, GDNIA 从这三个中选择一个
            q: 学生作答数据对应的Q矩阵名称
            *args:
            **kwargs: 学生作答数据名称

        Returns:

        """
        importr(package)
        q = numpy2ri.rpy2py(r(q))
        if not(kwargs.get('orp') is None):
            orp = numpy2ri.rpy2py(r(kwargs.get('orp')))
        if not(kwargs.get('amp') is None):
            amp = numpy2ri.rpy2py(r(kwargs.get('amp')))
            return q, orp, amp
        return q, orp

    @staticmethod
    def run_r_model(mod_name: str, q: ndarray, orp: ndarray, est_type='MLE', package='CDM') -> ndarray:
        """
        运行R NPCD CDM  GDINA 中的模型
        Args:
            mod_name:
            q:q矩阵
            orp:学生作答数据
            est_type:模型评估方法

        Returns: amp 学生技能掌握模式

        """
        importr('NPCD')
        importr('CDM')
        importr('GDINA')

        q = numpy2ri.py2rpy(q)
        orp = numpy2ri.py2rpy(orp)

        if package.upper() == 'NPCD':
            amp = numpy2ri.rpy2py(r(mod_name)(orp, q).rx2('alpha.est'))  # NPCD
        else:
            if package.upper() == 'CDM':
                # DINA, GDINA, RUM 等
                amp = numpy2ri.rpy2py(r('IRT.factor.scores')(r(mod_name)(orp, q, progress=False), est_type))
            else:
                amp = numpy2ri.rpy2py(r('IRT.factor.scores')(r(mod_name)(orp, q, verbose=0), est_type))
        return amp

    @staticmethod
    def get_x4som(q, data, **kwargs):
        imp, irp = Utils.get_imp_irp(q)
        data = np.append(data, irp, axis=0)
        som = MiniSom(2**q.shape[1], 1, q.shape[0])
        som.pca_weights_init(data)
        som.train(data, 200)
        p_c = som.win_map(data)
        x = np.empty((0, q.shape[0]+q.shape[1]))
        for i in range(irp.shape[0]):
            row = irp[i]
            p = som.winner(row)
            p_map = np.array(p_c.get(p))
            mean_dit = Utils.cost(row, p_map)
            d = np.apply_along_axis(lambda x: np.linalg.norm(row-x), axis=1, arr=p_map)
            index = np.where(d > mean_dit, True, False)
            x_p = p_map[index]
            y_p = imp[i]
            y_p = np.tile(y_p, (x_p.shape[0], 1))
            x_y_p = np.append(x_p, y_p, axis=1)
            x = np.append(x, x_y_p, axis=0)
        return x[:, 0:q.shape[0]], x[:, q.shape[0]:]

    @staticmethod
    def cost(c, all_points):
        """
        # c指定点，all_points:为集合类的所有点
        Args:
            c:
            all_points:

        Returns:

        """
        d = np.apply_along_axis(lambda x: np.linalg.norm(c-x), axis=1, arr=all_points)
        d[np.isinf(d)] = np.nan
        d = pd.DataFrame(d)
        d.fillna(d.mean(), inplace=True)
        return np.mean(d.values)

    @staticmethod
    def check_data(data):
        pass




class BaseNet(nn.Module):
    def __init__(self, q, is_x_y=False):
        super(BaseNet, self).__init__()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_x_y = is_x_y
        self.q = q if type(q) == Tensor else torch.tensor(q, device=self.__device).float()
        self.q_star = self.__get_q_stars_matrix()

    # loss
    def loss_func(self, y_pred, y_true):
        loss_f = nn.MSELoss()
        return loss_f(y_pred, y_true)

    # acc
    def metric_aar(self, y_pred, y_true):
        y_pred = torch.where(y_pred > 0.5,
                             torch.ones_like(y_pred, dtype=torch.float),
                             torch.zeros_like(y_pred, dtype=torch.float))
        aar = torch.mean(1 - torch.abs(y_true - y_pred))  # aar
        return aar

    def metric_par(self, y_pred, y_true):
        y_pred = torch.where(y_pred > 0.5,
                             torch.ones_like(y_pred, dtype=torch.float),
                             torch.zeros_like(y_pred, dtype=torch.float))
        par = torch.mean(torch.prod(torch.where((y_pred==y_true)==True,
                                         torch.ones_like(y_pred, dtype=torch.float),
                                         torch.zeros_like(y_pred, dtype=torch.float)), axis=1)) # par
        return par

    def metric_func(self, y_pred, y_true, metric='aar'):
        y_pred = torch.where(y_pred > 0.5,
                             torch.ones_like(y_pred, dtype=torch.float),
                             torch.zeros_like(y_pred, dtype=torch.float))
        if metric == 'aar':
            acc = torch.mean(1-torch.abs(y_true-y_pred)) # aar
        else:
            acc = torch.mean(torch.prod(torch.where((y_pred==y_true)==True,
                                         torch.ones_like(y_pred, dtype=torch.float),
                                         torch.zeros_like(y_pred, dtype=torch.float)), axis=1)) # par
        return acc

    # optimizer
    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

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
            return torch.tensor(Q_stars, device=self.__device).float()
        else:
            return Q_stars


class QNN(BaseNet):
    def __init__(self, q, is_x_y=False):
        super(QNN, self).__init__(q, is_x_y)
        if is_x_y:
            self.mc = nn.Linear(self.q.shape[0]+q.shape[1], self.q.shape[1])
            self.lc1 = nn.Linear(self.q.shape[0]+q.shape[1], self.q.shape[1] + self.q_star.shape[1])
            self.sc1 = nn.Linear(self.q.shape[0]+q.shape[1], self.q_star.shape[1])
        else:
            self.mc = nn.Linear(self.q.shape[0], self.q.shape[1])
            self.lc1 = nn.Linear(self.q.shape[0], self.q.shape[1] + self.q_star.shape[1])
            self.sc1 = nn.Linear(self.q.shape[0], self.q_star.shape[1])
        # self.mc = nn.Linear(self.q.shape[0], self.q.shape[1])
        # self.lc1 = nn.Linear(self.q.shape[0], self.q.shape[1] + self.q_star.shape[1])
        self.lc2 = nn.Linear(self.q.shape[1] + self.q_star.shape[1], self.q.shape[1])
        # self.sc1 = nn.Linear(self.q.shape[0], self.q_star.shape[1])
        self.sc2 = nn.Linear(self.q_star.shape[1], self.q.shape[1])

    # forword
    def forward(self, x):
        m = torch.relu(self.mc(x))
        l = torch.sigmoid(self.lc1(x))
        l = torch.sigmoid(self.lc2(l))
        s = torch.tanh(self.sc1(x))
        s = torch.tanh(self.sc2(s))
        y = torch.sigmoid(l * m + (1 - l) * s)
        return y


class QPNN(BaseNet):
    def __init__(self, q, is_x_y=False):
        super(QPNN, self).__init__(q, is_x_y)
        self.wm = nn.Parameter(torch.randn(self.q.shape[0], self.q.shape[1]))
        self.bm = nn.Parameter(torch.randn(1, self.q.shape[1]))
        self.wl1 = nn.Parameter(torch.randn(self.q.shape[0], self.q_star.shape[1] + self.q.shape[1]))
        self.bl1 = nn.Parameter(torch.randn(1, self.q_star.shape[1] + self.q.shape[1]))
        self.wl2 = nn.Parameter(torch.randn(self.q_star.shape[1] + self.q.shape[1], self.q.shape[1]))
        self.bl2 = nn.Parameter(torch.randn(1, self.q.shape[1]))
        self.ws1 = nn.Parameter(torch.randn(self.q.shape[0], self.q_star.shape[1]))
        self.bs1 = nn.Parameter(torch.randn(1, self.q_star.shape[1]))
        self.ws2 = nn.Parameter(torch.randn(self.q_star.shape[1], self.q.shape[1]))
        self.bs2 = nn.Parameter(torch.randn(1, self.q.shape[1]))

    def forward(self, x):
        m = torch.relu(x@(self.wm*self.q) + self.bm)
        l = torch.sigmoid(x@self.wl1 + self.bl1)
        l = torch.sigmoid(l@self.wl2 + self.bl2)
        s = torch.tanh(x@(self.ws1*self.q_star) + self.bs1)
        s = torch.tanh(s@(self.ws2*(self.q_star.T@self.q)) + self.bs2)
        y = torch.sigmoid(l*m+(1-l)*s)
        return y


class PNN(BaseNet):
    def __init__(self, q, is_x_y=False):
        super(PNN, self).__init__(q, is_x_y)
        """
        is_x_y=False, 表示默认不将标签也作为特征
        """
        if self.is_x_y:
            self.l1 = nn.Linear(self.q.shape[0]+self.q.shape[1], self.q.shape[1]+self.q_star.shape[1])
        else:
            self.l1 = nn.Linear(self.q.shape[0], self.q.shape[1] + self.q_star.shape[1])
        self.l2 = nn.Linear(self.q.shape[1] + self.q_star.shape[1], self.q.shape[1])

    def forward(self, x):
        # x = torch.sigmoid(self.l1(x))
        x = torch.relu(self.l1(x))
        y = torch.sigmoid(self.l2(x))
        return y


class SomQnnCD:
    def __init__(self, Q_Matrix: ndarray, X: ndarray, net: BaseNet, is_x_y=False, *args, **kwargs):
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__is_x_y = is_x_y
        self.Q = Q_Matrix
        self.X: ndarray = X
        # self.Y : ndarray = Y
        self.som = MiniSom(**kwargs)
        self.net = net(self.Q, self.__is_x_y).to(self.__device)

    def __get_imp_irp(self, is_tensor=False)-> tuple:
        """
        计算理想掌握模式和理想反应模式
        Args:
            is_tensor: 是否返回tensor类型的数据， 默认为False返回ndarray类型的数据

        Returns:irp and imp

        """
        k = self.Q.shape[1]
        imp = np.array(list(product([0, 1], repeat=self.Q.shape[1]))).astype(float)
        irp = np.empty(shape=[0, self.Q.shape[0]])
        for i in range(imp.shape[0]):
            row = np.apply_along_axis(lambda x: np.prod(np.power(imp[i, :], x)), axis=1, arr=self.Q)
            irp = np.row_stack((irp, row))

        irp = irp.astype(float)
        if is_tensor:
            return self.__to_Tensor(irp), self.__to_Tensor(imp)
        else:
            return irp, imp

    def __train_som(self, x, epochs=500):
        self.som.random_weights_init(x)
        self.som.train(x, epochs, random_order=True)

    def __get_q_stars_matrix(self, is_tensor=False, is_q_qstar=False) -> any:
        """
        生成交互式Q矩阵
        Args:
            is_tensor: 是否返回Tensor,默认为False, 默认返回ndarray类型的交互式q矩阵

        Returns:返回交互式Q矩阵

        """
        poly = PolynomialFeatures(degree=self.Q.shape[1], include_bias=False, interaction_only=True)
        Q_Q_stars = poly.fit_transform(self.Q).astype(float)
        Q_stars = Q_Q_stars[:, self.Q.shape[1]:].astype(float)
        if is_q_qstar:
            if is_tensor:
                return self.__to_Tensor(Q_stars), self.__to_Tensor(Q_Q_stars)
            else:
                return Q_stars, Q_Q_stars
        else:
            if is_tensor:
                return self.__to_Tensor(Q_stars)
            else:
                return Q_stars

    def __to_Tensor(self, data: ndarray) -> Tensor:
        """
        将ndarray快速转换为Tensor
        """
        data = data if type(data) == Tensor else torch.tensor(data=data, device=self.__device).float()
        return data

    def __to_ndarray(self, data: Tensor) -> ndarray:
        """
        tensor to numpy
        Args:
            data:

        Returns:

        """
        data = data if type(data)==ndarray else data.detach().cpu().numpy()
        return data

    def __get_quality_X(self, x_som, neighbor_d=1, is_tensor=False):
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
        if self.__is_x_y:
            ypn_x = pd.DataFrame(y_p_n.x.to_dict()).T.values
        ypn_y = pd.DataFrame(y_p_n.y.to_dict()).T.values

        if is_tensor:
            return self.__to_Tensor(ypn_x), self.__to_Tensor(ypn_y)
        else:
            return ypn_x, ypn_y

    def __get_aux_smp(self, is_tensor=False, is_ndarray_tensor=False) -> any:
        """
        __get_aux_smp 根据学生的观察反应来生成辅助的技能掌握模式smp
        该方法可以保证准确生成只考察单个技能的试题所对应的技能模式，同时也能辅助生成多技能的模式
        Args:
            is_tensor: 用来控制生成ndarray类型的返回值，还是tensor类型的返回值，默认生成ndarray
            is_ndarray_tensor：是否同时返回ndarray和tensor类型的数据

        Returns:
            smp：辅助技能掌握模式
        """
        smp = np.empty([0, self.Q.shape[1]])
        for index_row in range(self.X.shape[0]):
            row = (((self.X[index_row]*self.Q.T) / self.Q.sum(axis=1)).sum(axis=1))
            if 0.5 in row and 1.0 in row:
                counter = Counter(row)
                if counter.get(0.5) > counter.get(1.0):
                    row = np.where(row==0.5, 0.5-random.random()/4, row)
                elif counter.get(0.5) < counter.get(1.0):
                    row = np.where(row==0.5, 0.5+random.random()/4, row)
            smp = np.append(smp, row.reshape([1, row.shape[0]]), axis=0)
            smp = np.where(smp>1., 1.0, smp)
        if is_ndarray_tensor:
            return smp, self.__to_Tensor(smp)
        else:
            if is_tensor:
                return self.__to_Tensor(smp)
            else:
                return smp

    def __get_x4som(self, iter=200, dist=2, **kwargs):
        irp, imp = self.__get_imp_irp()
        data = np.append(self.X, irp, axis=0)
        som = MiniSom(2**self.Q.shape[1], 1, self.Q.shape[0])
        try:
            som.pca_weights_init(data)
        except Exception as e:
            som.random_weights_init(data)
        som.train(data, iter)
        p_c = som.win_map(data)
        x = np.empty((0, self.Q.shape[0]+self.Q.shape[1]))
        for i in range(irp.shape[0]):
            row = irp[i]
            p = som.winner(row)
            p_1 = (p[0]-1, p[1])
            p_2 = (p[0]+1, p[1])
            # p_map = np.append(np.append(np.array(p_c.get(p)), np.array(p_c.get(p_1, np.empty((0, self.Q.shape[0])))), axis=0),
            #                   np.array(p_c.get(p_2, np.empty((0, self.Q.shape[0])))), axis=0)
            p_map = np.array(p_c.get(p))
            mean_dit = Utils.cost(row, p_map) / dist
            d = np.apply_along_axis(lambda x: np.linalg.norm(row-x), axis=1, arr=p_map)
            index = np.where(d > mean_dit, True, False)
            x_p = p_map[index]
            y_p = imp[i]
            y_p = np.tile(y_p, (x_p.shape[0], 1))
            x_y_p = np.append(x_p, y_p, axis=1)
            x = np.append(x, x_y_p, axis=0)
        if self.__is_x_y:
            return x, x[:, self.Q.shape[0]:]
        else:
            return x[:, 0:self.Q.shape[0]], x[:, self.Q.shape[0]:]

    def __get_aux_y(self):
        pass

    def __get_data_loader(self, X, Y, batch_size=64, shuffle=True, num_workers=0, verbose=False):
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
        X = X if type(X) == Tensor else self.__to_Tensor(X)
        Y = Y if type(Y) == Tensor else self.__to_Tensor(Y)
        dl = DataLoader(TensorDataset(X, Y), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
        return dl

    def __train_net_step(self, features, labels):
        """
        网络一个批量的单步训练
        Args:
            features: 训练特征
            labels: 标签

        Returns:loss, metric  损失和准确率

        """
        # forword
        pre = self.net(features)
        loss = self.net.loss_func(pre, labels)
        metric = self.net.metric_func(pre, labels, metric='aar')

        # Zero gradient
        self.net.optimizer.zero_grad()

        # backword
        loss.backward()

        # updated parameter
        self.net.optimizer.step()

        return loss, metric

    def __printbar(self):
        """
        时间打印函数
        Returns:

        """
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)

    def __train_net(self, dl: DataLoader, epochs=500, verbose=False):
        """
        网络模型训练函数
        Args:
            dl: 训练数据加载器
            epochs: 迭代次数

        Returns: None

        """
        for epoch in range(1, epochs+1):
            loss_list, metric_list = [], []
            for features, labels in dl:
                lossi, metrici = self.__train_net_step(features, labels)
                loss_list.append(lossi)
                metric_list.append(metrici)
            loss = torch.mean(torch.tensor(loss_list, device=self.__device))
            metric = torch.mean(torch.tensor(metric_list, device=self.__device))

            if verbose:
                if epoch%(epochs // 20) == 0:
                    self.__printbar()
                    print("epoch =", epoch, "loss = ", loss, "metric = ", metric)

    def __is_continue(self, y_hat, y, threshold=0.90, verbose=False) -> bool:
        """
        判断是否继续训练
        Args:
            y_hat: 预测值
            y: 真实值
            threshold:阈值

        Returns:是否继续

        """
        acc = self.net.metric_func(self.__to_Tensor(y_hat), self.__to_Tensor(y))
        if verbose:
            print(f'-------con-acc------:{acc}')
        return self.__to_ndarray(acc) > threshold

    def __predicate(self, X) -> Tensor:
        """
        预测知识技能掌握程度
        Args:
            X: 学生作答反应

        Returns:预测结果 学生技能掌握情况

        """
        return self.net(self.__to_Tensor(X))

    def __get_x_som(self, y_aux) -> ndarray:
        """
        生成 som训练数据
        Args:
            y_aux: 辅助标签

        Returns:som训练数据

        """
        x_y = np.append(self.X, self.__to_ndarray(y_aux), axis=1)
        irp, imp = self.__get_imp_irp()
        irp_imp = np.append(irp, imp, axis=1)
        x_som = np.append(x_y, irp_imp, axis=0)
        return x_som

    def train_semi_supervised(self, epochs_som=500, epochs_net=500):
        x, y = self.__get_x4som(iter=epochs_som)
        for i in range(self.X.shape[0]):
            if (self.X[i] == 1).all():
                y_i = np.ones([1, self.Q.shape[1]])
                x_i = np.ones([1, self.Q.shape[0]])
                x = np.append(x, x_i, axis=0)
                y = np.append(y, y_i, axis=0)

        dl = self.__get_data_loader(x, y)
        self.__train_net(dl, epochs=epochs_net)

    def train_unsupervised(self, epochs=50, epochs_som=500, epochs_net=500, threshold=0.9, verbose=False,is_x_y=True):
        """
        训练模型
        Args:
            epochs: som-pnnCD迭代次数
            epochs_som: som网络迭代次数
            epochs_net: pnn迭代次数

        Returns:

        """
        # y_aux = self.__get_aux_smp()
        y_aux = self.__get_amp4orp(self.X)
        x_som = self.__get_x_som(y_aux)
        while epochs > 0:
            self.__train_som(x_som, epochs_som)
            X, Y = self.__get_quality_X(x_som)
            dl = self.__get_data_loader(X, Y)
            self.__train_net(dl, epochs_net)
            if self.__is_x_y:
                y_hat = self.__predicate(x_som)
            else:
                y_hat = self.__predicate(x_som[:, 0:self.X.shape[1]])
            y = x_som[:, self.X.shape[1]:]
            if self.__is_continue(y_hat, y, threshold=threshold):
                break
            else:
                if self.__is_x_y:
                    y_aux = self.__get_amp4orp(self.X)
                    y_aux = self.__predicate(np.append(self.X, y_aux, axis=1))
                else:
                    y_aux = self.__predicate(self.X)
                # y_aux = self.__get_aux_smp()
                x_som = self.__get_x_som(y_aux)
            if verbose:
                if epochs % 5 == 0:
                    print('----'*5 + f'epoch:{epochs}' + '----'*5)
                epochs -= 1

    def train_supervised(self, y: ndarray, epochs_som=500, epochs_net=500):
        """
        有监督训练网络
        Args:
            y: 学生真实的技能掌握模式
            epochs_som: som 迭代次数
            epochs_net: pnn迭代次数

        Returns:

        """
        x_som = self.__get_x_som(y)
        self.__train_som(x_som, epochs_som)
        X, Y = self.__get_quality_X(x_som)
        dl = self.__get_data_loader(X, Y)
        self.__train_net(dl, epochs_net)

    def predicate(self, train_type: Literal["supervised", "semi_supervised", "unsupervised"], verbose=True, **kwargs) -> ndarray:
        """
        模型预测
        Args:
            kwargs: kwargs中可以存放y表示学生的真实技能掌握模式
            train_type：可以取supervised，unsupervised和semi_supervised
            verbose: 是否打印acc 默认为True

        Returns:返回y_hat  模型预测的学生技能掌握模式

        """
        assert train_type in ("supervised", "semi_supervised", "unsupervised"), "invalid train_type: %r" % (train_type,)

        if self.__is_x_y:
            if train_type == 'supervised':
                try:
                    if kwargs.get('y') is None:
                        raise RuntimeError('需要输入学生的真实技能掌握模式:y')
                except Exception as e:
                    print(e)
                y = kwargs.get('y')
                y_hat = self.__predicate(np.append(self.X, y, axis=1))
            elif train_type == 'unsupervised':
                y_aux = self.__get_aux_smp()
                y_hat = self.__predicate(np.append(self.X, y_aux, axis=1))
            else:
                x_semi, y_semi = self.__get_x4som()
                y_hat = self.__predicate(x_semi)
        else:
            y_hat = self.__predicate(self.X)

        if verbose:
            if kwargs.get('y') is None:
                print(f'-------{train_type}-acc-----:没有输入学生的真实技能掌握模式，无法计算预测准确率，不影响预测结果，若要计算预测准确率，请输入学生真实技能掌握模式')
            else:
                y = kwargs.get('y')
                acc = self.net.metric_func(self.__to_Tensor(y_hat), self.__to_Tensor(y))
                par = self.net.metric_par(self.__to_Tensor(y_hat), self.__to_Tensor(y))
                print(f'-------{train_type}-aar:{acc}-----par:{par}')
        return self.__to_ndarray(y_hat)

    def get_par(self, y_true: ndarray, y_hat: ndarray):
        acc = self.net.metric_par(self.__to_Tensor(y_hat), self.__to_Tensor(y_true))
        return self.__to_ndarray(acc)

    def get_acc(self, y_true: ndarray, y_hat: ndarray):
        """
        计算准确率
        Args:
            y_true: 学生真实技能掌握模式
            y_hat: 模型估计的技能掌握模式

        Returns:精确度

        """
        acc = self.net.metric_func(self.__to_Tensor(y_hat), self.__to_Tensor(y_true))
        return self.__to_ndarray(acc)

    def __get_amp2binary(self, smp:ndarray) -> ndarray:
        """
        将带有小数的学生技能掌握模式转换为只包含0,1的形式
        Args:
            smp:

        Returns:smp，表示未掌握，1表示掌握

        """
        smp = np.where(smp > 0.5, 1, smp)
        smp = np.where(smp < 0.5, 0, smp)
        for j in range(self.Q.shape[1]):
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

    def get_amp2binary(self, smp:ndarray) -> ndarray:
        """
        将学生技能掌握模式中存在概率的情况，装换为0,1确定值
        Args:
            smp: 学生技能掌握模式

        Returns: amp 确定的只包含0,1形式的技能掌握模式

        """
        return self.__get_amp2binary(smp=smp)

    def __get_one_skill4q(self) -> dict:
        """
        根据q矩阵获取每个技能被单独的一道题考察的技能和题
        技能为key, 题标号构成的列表为value
        ex:k=0, v=[1, 3] 表示第0个技能，在第1题和第3题中被单独考察了
        Returns:kw
        """
        kw = {}
        for i in range(self.Q.shape[1]):
            row_i = np.zeros(self.Q.shape[1])
            row_i[i] = 1
            index = np.argwhere(np.apply_along_axis(lambda x: (x == row_i).all(), axis=1, arr=self.Q) == True)
            if index.shape[0] > 0:
                kw[i] = index
        return kw

    def __get_amp4orp(self, orp: ndarray) -> ndarray:
        kw = self.__get_one_skill4q()
        amp = np.zeros((0, self.Q.shape[1]))
        for i in range(orp.shape[0]):
            row = np.zeros([1, self.Q.shape[1]])
            if (orp[i]==1).all():  # 学生全部答对
                row = np.ones([1, self.Q.shape[1]])
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
                        if np.sum(orp[i][self.Q[:, k]==1]) / np.sum(self.Q[:, k]) > 0.5:
                            row[:, k] = 1
                        else:
                            row[:, k] =0
                for col in range(self.Q.shape[1]):  # 混合知识点的题
                    if row[:, col]==0:
                        if np.sum(orp[i][self.Q[:, col]==1]) / np.sum(self.Q[:, col]) > 0.5:
                            row[:, col] = 1
                        elif np.sum(orp[i][self.Q[:, col]==1]) / np.sum(self.Q[:, col]) == 0.5:
                            row[:, col] = 0.5
                        else:
                            row[:, col] = 0
            amp = np.append(amp, row, axis=0)
        amp = self.__get_amp2binary(amp)  # 如果该技能的正误率为0.5，则看全部的人在该题上的作答情况，大多数人对，则为对
        return amp

    def get_amp4expert(self, orp: ndarray) -> ndarray:
        """
        通过专家来根据学生的作答情况来判断学生的技能掌握模式
        Args:
            orp: 学生作答数据

        Returns: amp 学生技能掌握模式

        """
        return self.__get_amp4orp(orp=orp)

    def _run_r_model(self, q: ndarray, orp: ndarray, smp: ndarray,
                    package: Literal["CDM", "GDINA", "NPCD"]='GDINA',
                    mod_name: Literal["GDINA","DINA","DINO","ACDM","LLM", "RRUM", "MSDINA", "AlphaNP"]='DINA',
                    est_type: Literal['MLE', 'MAP', 'EAP']='MLE') -> ndarray:
        """
        运行R NPCD CDM  GDINA 中的模型
        Args:
            mod_name: "GDINA","DINA","DINO","ACDM","LLM", "RRUM", "MSDINA" and "UDF"
            q:q矩阵
            orp:学生作答数据
            est_type:模型评估方法

        Returns: amp 学生技能掌握模式

        """
        assert est_type in ("MLE", "MAP", "EAP"), "invalid est_type: %r" % (est_type,)
        assert package in ("CDM", "GDINA", "NPCD"), "invalid package: %r" % (package,)
        importr('NPCD')
        importr('CDM')
        importr('GDINA')

        q = numpy2ri.py2rpy(q)
        orp = numpy2ri.py2rpy(orp)

        if package.upper() == 'NPCD':
            amp = numpy2ri.rpy2py(r(mod_name)(orp, q).rx2('alpha.est'))  # NPCD
        else:
            if package.upper() == 'CDM':
                # DINA, GDINA, RUM 等
                amp = numpy2ri.rpy2py(r('IRT.factor.scores')(r(mod_name)(orp, q, progress=False), est_type))
            else:
                # GDINA(dat = dat, Q = Q, model = "ACDM")
                amp = numpy2ri.rpy2py(r('personparm')(r('GDINA')(orp, q, mod_name, verbose=0)))  # GDINA
        acc = self.get_acc(smp, amp)
        par = self.get_par(smp, amp)
        print(f'-------{mod_name}-aar:{acc}------par:{par}')
        return amp

    def run_classical_model(self, q: ndarray, orp: ndarray, smp: ndarray,
                            kwargs={'GDINA':["DINA","GDINA","ACDM","LLM", "RRUM"],
                                    'NPCD':['AlphaNP']
                                    }):
        try:
            if kwargs:
                for k, v in kwargs.items():
                    for m in v:
                        y_hat = self._run_r_model(q, orp, smp, k, m)
            else:
                raise ValueError('kwargs为空')
        except Exception as e:
            print(e)



if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')

    # orp = Utils.get_orp_smp('./data/input/simulation15x5/gdina/orp_1000_uniform_0.2_gdina.csv')
    # smp = Utils.get_orp_smp('./data/input/simulation15x5/gdina/smp_1000_uniform_0.2_gdina.csv')
    # Q = Utils.get_q_matrix(file='./data/input/simulation15x5/q.csv')

    Q, orp, smp= Utils.get_data4r('NPCD', 'dina$Q', orp='dina$response', amp='dina$true.alpha')

    # Q = pd.read_csv('./data/frac_sub_15_5/q_15_5.csv').iloc[:,1:].values
    # orp = pd.read_csv('./data/frac_sub_15_5/orp.csv').iloc[:,1:16].values

    # Q = pd.read_csv('../data/real/timss07/25_15/q.csv').iloc[:,1:].values
    # orp = pd.read_csv('../data/real/timss07/25_15/resp.csv').iloc[:,1:].values
    # smp = pd.read_csv('../data/real/timss07/25_15/label_expert.csv').iloc[:,1:].values


    som_parm = {
        'x': 10,
        'y': 10,
        'input_len': Q.shape[1]+Q.shape[0],
        'sigma': 1.0,
        'learning_rate': 0.01,
        'neighborhood_function':  'gaussian'}

    model = SomQnnCD(Q_Matrix=Q, X=orp, net=PNN, is_x_y=False, **som_parm)
    # smp = model.get_amp4expert(orp)

    model.train_unsupervised(threshold=0.92)
    y_hat_unsupervised = model.predicate(train_type='unsupervised', y=smp)

    model.train_semi_supervised()
    y_hat_semi_supervised = model.predicate(train_type='semi_supervised', y=smp)

    model.train_supervised(smp)
    y_hat_supervised = model.predicate(train_type='supervised', y=smp)

    model.run_classical_model(Q, orp, smp)
























