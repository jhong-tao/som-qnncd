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

import datetime
import os.path
import random
import time

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchinfo import summary

from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri


class BaseNet(nn.Module):
    def __init__(self, q: ndarray, data: ndarray, target: ndarray):
        super(BaseNet, self).__init__()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.__device = torch.device("cpu")
        self.q = self._to_Tensor(q)
        self.q_star = self.__get_q_stars_matrix()
        self.X = self._to_Tensor(data)
        self.y = self._to_Tensor(target)
        # 9 3 4 6 7
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=0.25, random_state=0)
        self.dl = self.__get_data_loader(self.X_train, self.y_train)
        self.dl_test = self.__get_data_loader(self.X_test, self.y_test)

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
            return torch.tensor(Q_stars, device=self.__device).float()
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

        par = torch.mean(torch.prod(torch.where((y_pred==y_true)==True,
                                        torch.ones_like(y_pred, dtype=torch.float),
                                        torch.zeros_like(y_pred, dtype=torch.float)), axis=1)) # par
        return par

    # acc
    def metric_acc(self, y_pred, y_true):
        y_pred = y_pred if type(y_pred) == Tensor else self._to_Tensor(y_pred)
        y_true = y_true if type(y_true) == Tensor else self._to_Tensor(y_true)
        y_pred = torch.where(y_pred > 0.5,
                             torch.ones_like(y_pred, dtype=torch.float),
                             torch.zeros_like(y_pred, dtype=torch.float))
        acc = torch.mean(1-torch.abs(y_true-y_pred)) # aar
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

        data = data if type(data) == Tensor else torch.tensor(data=data, device=self.__device, dtype=torch.float32)
        return data

    def _to_ndarray(self, data: Tensor) -> ndarray:
        """
        tensor to numpy
        Args:
            data:

        Returns:

        """
        data = data if type(data)==ndarray else data.detach().cpu().numpy()
        return data
    
    def __get_data_loader(self, X, Y, batch_size=32, shuffle=True, num_workers=0, verbose=False):
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
        metric_acc = self.metric_acc(pre, labels)
        metric_par = self.metric_par(pre, labels)

        # Zero gradient
        self.optimizer.zero_grad()

        # backword
        loss.backward()

        # updated parameter
        self.optimizer.step()

        return loss, metric_acc, metric_par

    def __eval_net_step(self, dl):
        self.eval()
        val_loss_list, val_aar_list, val_par_list = [], [], []
        for features, labels in dl:
            pre = self.forward(features)
            loss = self.loss_func(pre, labels)
            metric_acc = self.metric_acc(pre, labels)
            metric_par = self.metric_par(pre, labels)
            val_loss_list.append(loss)
            val_aar_list.append(metric_acc)
            val_par_list.append(metric_par)

        val_loss = torch.mean(torch.tensor(val_loss_list, device=self.__device))
        val_aar = torch.mean(torch.tensor(val_aar_list, device=self.__device))
        val_par = torch.mean(torch.tensor(val_par_list, device=self.__device))

        return val_loss, val_aar, val_par

    def __printbar(self):
        """
        时间打印函数
        Returns:

        """
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)

    def _train_net(self, epochs=500, verbose=False):
        """
        网络模型训练函数
        Args:
            dl: 训练数据加载器
            epochs: 迭代次数

        Returns: None

        """
        history = pd.DataFrame(columns=["epoch", "loss", "aar", 'par',
                                        "val_loss", "val_aar", 'val_par',
                                        "x_loss", "x_aar", 'x_par'])
        for epoch in range(1, epochs+1):
            # train
            loss_list, aar_list, par_list = [], [], []
            for features, labels in self.dl:
                lossi, aar, par = self._train_net_step(features, labels)
                loss_list.append(lossi)
                aar_list.append(aar)
                par_list.append(par)
            loss = torch.mean(torch.tensor(loss_list, device=self.__device))
            aar = torch.mean(torch.tensor(aar_list, device=self.__device))
            par = torch.mean(torch.tensor(par_list, device=self.__device))

            if verbose:
                if epoch%(epochs // 20) == 0:
                    self.__printbar()
                    print("epoch =", epoch, "loss = ", loss, "metric_aar = ", aar, "metric_par = ", par)

            # eval test

            val_loss, val_aar, val_par = self.__eval_net_step(self.dl_test)

            # eval all
            x_loss, x_aar, x_par = self.__eval_net_step(self.__get_data_loader(self.X, self.y))

            info = (epoch, loss.cpu().detach().numpy(), aar.cpu().detach().numpy(), par.cpu().detach().numpy(),
                    val_loss.cpu().numpy(), val_aar.cpu().numpy(), val_par.cpu().numpy(),
                    x_loss.cpu().numpy(), x_aar.cpu().numpy(), x_par.cpu().numpy())
            history.loc[epoch - 1] = info
        return history

    def _plot_metric(self, dfhistory, metric):
        index = np.linspace(0, dfhistory.shape[0]-1, 25).astype(np.int64)
        train_metrics = dfhistory[metric].iloc[index]
        val_metrics = dfhistory['val_' + metric].iloc[index]
        # epochs = range(1, len(train_metrics) + 1)
        epochs = index+1
        plt.plot(epochs, train_metrics, 'bp--', linewidth=0.5)
        plt.plot(epochs, val_metrics, 'r*-', linewidth=0.5)
        plt.title(f'{self.__class__.__name__} Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        plt.show()
        plt.close()

    def _plots_metrics(self, history, metrics=['loss', 'aar', 'par'], data_name_list={'':'bp--', 'val_':'r*-', 'x_':'g^.'}):
        index = np.linspace(0, history.shape[0] - 1, 25).astype(np.int64)
        epochs = index + 1
        for metric in metrics:
            legends = []
            for name, line in data_name_list.items():
                name_metric = history[f'{name}{metric}'].iloc[index]
                plt.plot(epochs, name_metric, line, linewidth=0.5)
                if name == '':
                    legends.append(f'train_{metric}')
                else:
                    legends.append(f'{name}{metric}')
            plt.title(f'{self.__class__.__name__} Training and validation ' + metric)
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.legend(legends)
            plt.show()
            plt.close()

    def predictive(self, data=None, is_show=False, epochs_train=500, verbose_train = False, verbose_pre=True):
        time_str = datetime.datetime.now()
        history = self._train_net(epochs=epochs_train, verbose=verbose_train)
        if verbose_pre:
            time_end = datetime.datetime.now()
            time = (time_end - time_str).seconds
            len_his = history.shape[0]
            index = random.sample(range(len_his // 10, len_his), len_his//50)

            acc_test = history.val_aar.iloc[index].mean()
            acc_train = history.aar.iloc[index].mean()
            acc_x = history.x_aar.iloc[index].mean()
            par_test = history.val_par.iloc[index].mean()
            par_train = history.par.iloc[index].mean()
            par_x = history.x_par.iloc[index].mean()

            print(f'{data}----{self.__class__.__name__}----acc_train:{acc_train}-----par_train:{par_train}------time_cost:{time}s------')
            print(f'{data}----{self.__class__.__name__}----acc_test:{acc_test}-----par_test:{par_test}------time_cost:{time}s------')
            print(f'{data}----{self.__class__.__name__}----acc_X:{acc_x}-----par_X:{par_x}------time_cost:{time}s------')

        if is_show:
            # self._plot_metric(history, 'aar')
            # self._plot_metric(history, 'par')
            # self._plot_metric(history, 'loss')
            self._plots_metrics(history)
        return history


class MlP(BaseNet):
    def __init__(self, q, data, target):
        super(MlP, self).__init__(q, data, target)
        self.l1 = nn.Linear(self.q.shape[0], self.q.shape[1])

    def forward(self, x):
        y = torch.sigmoid(self.l1(x))
        return y


class PMLP(MlP):
    def __init__(self, q, data, target):
        super(PMLP, self).__init__(q, data, target)
        self.__cons_l1 = torch.t(self.q)

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
        metric_acc = self.metric_acc(pre, labels)
        metric_par = self.metric_par(pre, labels)

        # Zero gradient
        self.optimizer.zero_grad()

        # backword
        loss.backward()

        # updated parameter
        self.optimizer.step()
        self.l1.weight.data = self.l1.weight.data * self.__cons_l1
        return loss, metric_acc, metric_par


class ANN(BaseNet):
    def __init__(self, q, data, target):
        super().__init__(q, data, target)
        self.l1 = nn.Linear(self.q.shape[0], self.q.shape[1] + self.q_star.shape[1])
        self.l2 = nn.Linear(self.q.shape[1] + self.q_star.shape[1], self.q.shape[1])

    def forward(self, x):
        # x = torch.sigmoid(self.l1(x))
        x = torch.relu(self.l1(x))
        y = torch.sigmoid(self.l2(x))
        return y


class PNN(ANN):
    def __init__(self, q, data, target):
        super(PNN, self).__init__(q, data, target)
        self.__cons_l1 = torch.t(torch.cat((self.q, self.q_star), 1))
        self.__cons_l2 = torch.where((torch.t(self.q) @ torch.cat((self.q, self.q_star), 1)) > 0, 1, 0)

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
        metric_acc = self.metric_acc(pre, labels)
        metric_par = self.metric_par(pre, labels)

        # Zero gradient
        self.optimizer.zero_grad()

        # backword
        loss.backward()

        # updated parameter
        self.optimizer.step()
        self.l1.weight.data = self.l1.weight.data * self.__cons_l1  # torch.t(torch.cat((self.q, self.q_star), 1))
        self.l2.weight.data = self.l2.weight.data * self.__cons_l2

        return loss, metric_acc, metric_par


class QNN(BaseNet):
    def __init__(self, q, data, target):
        super(QNN, self).__init__(q, data, target)
        self.mc = nn.Linear(self.q.shape[0], self.q.shape[1])
        self.lc1 = nn.Linear(self.q.shape[0], self.q.shape[1] + self.q_star.shape[1])
        self.lc2 = nn.Linear(self.q.shape[1] + self.q_star.shape[1], self.q.shape[1])
        self.sc1 = nn.Linear(self.q.shape[0], self.q_star.shape[1])
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


class PQNN(QNN):
    def __init__(self, q, data, target):
        super(PQNN, self).__init__(q, data, target)
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
        metric_acc = self.metric_acc(pre, labels)
        metric_par = self.metric_par(pre, labels)

        # Zero gradient
        self.optimizer.zero_grad()

        # backword
        loss.backward()

        # updated parameter
        self.optimizer.step()
        self.mc.weight.data = self.mc.weight.data * self._cons_mc
        # self.lc1.weight.data = self.lc1.weight.data * self._cons_lc1
        # self.lc2.weight.data = self.lc2.weight.data * self._cons_lc2
        self.sc1.weight.data = self.sc1.weight.data * self._cons_sc1
        self.sc2.weight.data = self.sc2.weight.data * self._cons_sc2

        return loss, metric_acc, metric_par


class PWQNN(PQNN):
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
        metric_acc = self.metric_acc(pre, labels)
        metric_par = self.metric_par(pre, labels)

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

        return loss, metric_acc, metric_par


class Utils:
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


class Test:
    Q, orp, smp = Utils.get_data4r('NPCD', 'Data.DINA$Q', orp='Data.DINA$response', amp='Data.DINA$true.alpha')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    epochs_train = 1000

    @staticmethod
    def test_mlp(q=Q, x=orp, y=smp):
        net = MlP(q, x, y).to(Test.device)
        net.predictive(epochs_train=Test.epochs_train)

    @staticmethod
    def test_pmlp(q=Q, x=orp, y=smp):
        net = PMLP(q, x, y).to(Test.device)
        net.predictive()

    @staticmethod
    def test_ann(q=Q, x=orp, y=smp):
        net = ANN(q, x, y).to(Test.device)
        net.predictive()

    @staticmethod
    def test_pnn(q=Q, x=orp, y=smp):
        net = PNN(q, x, y).to(Test.device)
        net.predictive(epochs_train=500)

    @staticmethod
    def test_qnn(q=Q, x=orp, y=smp):
        net = QNN(q, x, y).to(Test.device)
        net.predictive(epochs_train=500)

    @staticmethod
    def test_pqnn(q=Q, x=orp, y=smp):
        net = PQNN(q, x, y).to(Test.device)
        net.predictive(epochs_train=500)

    @staticmethod
    def test_all(data, q=Q, x=orp, y=smp, e=epochs_train, is_sive=False, is_show=True, is_show_train=False, verbose_pre=True):
        mlp = MlP(q, x, y).to(Test.device)
        history_mlp = mlp.predictive(data, epochs_train=e, verbose_pre=verbose_pre)
        if os.path.exists(data):
            history_mlp.to_csv(os.path.join(data, f'train_history_{mlp.__class__.__name__}.csv'))
        #
        # pmlp = PMLP(q, x, y).to(Test.device)
        # history_pmlp = pmlp.predictive(data, epochs_train=e)

        # ann = ANN(q, x, y).to(Test.device)
        # history_ann = ann.predictive(data, epochs_train=e, verbose_pre=verbose_pre)

        pnn = PNN(q, x, y).to(Test.device)
        history_pnn = pnn.predictive(data, epochs_train=e, verbose_pre=verbose_pre)
        if os.path.exists(data):
            history_pnn.to_csv(os.path.join(data, f'train_history_{pnn.__class__.__name__}.csv'))


        # qnn = QNN(q, x, y).to(Test.device)
        # history_qnn = qnn.predictive(data, epochs_train=e, verbose_pre=verbose_pre)

        # pqnn = PQNN(q, x, y).to(Test.device)
        # history_pqnn = pqnn.predictive(data, epochs_train=e, verbose_pre=verbose_pre)

        pwqnn = PWQNN(q, x, y).to(Test.device)
        history_pwqnn = pwqnn.predictive(data, epochs_train=e, verbose_pre=verbose_pre)
        if os.path.exists(data):
            history_pwqnn.to_csv(os.path.join(data, f'train_history_{pwqnn.__class__.__name__}.csv'))

        Test.show_all(data, is_show_train, is_sive, is_show, *('loss', 'aar', 'par'), **{
            'ANN': {'his': history_mlp, 'line': 'y'},
            # 'PMLP': {'his': history_pmlp, 'line': 'g'},
            # 'ANN': {'his': history_ann, 'line': 'r'},
            'PNN': {'his': history_pnn, 'line': 'm'},
            # 'QNN': {'his': history_qnn, 'line': 'k'},
            # 'PQNN': {'his': history_pqnn, 'line': 'g'},
            'QNN': {'his': history_pwqnn, 'line': 'b'},
        })


        # Test.show_all(data, not is_show_train, is_sive, is_show, *('loss', 'aar', 'par'), **{
        #     'ANN': {'his': history_mlp, 'line': 'b'},
        #     # 'PMLP': {'his': history_pmlp, 'line': 'g'},
        #     # 'ANN': {'his': history_ann, 'line': 'r'},
        #     'PNN': {'his': history_pnn, 'line': 'k'},
        #     # 'QNN': {'his': history_qnn, 'line': 'm'},
        #     # 'PQNN': {'his': history_pqnn, 'line': 'y'},
        #     'QNN': {'his': history_pwqnn, 'line': 'g'},
        # })

    @staticmethod
    def show_all(data_name=None, is_show_train=True, is_sive=False, is_show=True, *args, **kwargs):
        plt.rcParams['figure.figsize'] = (4.8, 4)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        for metric in args:
            legend_tuple = []
            for net_name, dfhistory in kwargs.items():
                index = np.linspace(0, dfhistory.get('his').shape[0] - 1, 50).astype(np.int64)
                train_metrics = dfhistory.get('his')[metric].iloc[index]
                val_metrics = dfhistory.get('his')['val_' + metric].iloc[index]
                x_metrics = dfhistory.get('his')['x_' + metric].iloc[index]
                epochs = index + 1
                if is_show_train:
                    plt.plot(epochs, train_metrics, dfhistory.get('line')+'p-', linewidth=1, markersize=4)
                    legend_tuple.append(f'{net_name}_train_{metric}')
                plt.plot(epochs, val_metrics, dfhistory.get('line')+'*--', linewidth=1, markersize=4)
                legend_tuple.append(f'{net_name}_val_{metric}')
                plt.plot(epochs, x_metrics, color=dfhistory.get('line'), linestyle=(0, (1, 1)), marker='x', linewidth=1, markersize=4)
                legend_tuple.append(f'{net_name}_all_{metric}')
            # plt.legend(legend_tuple, bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
            plt.legend(legend_tuple)
            if is_show_train:
                plt.title(f'{data_name} Training and validation ' + metric)
            else:
                plt.title(f'{data_name} validation ' + metric)
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            # plt.gcf().subplots_adjust(right=0.7)
            if is_sive:
                if is_show_train:
                    img_file = os.path.join(data_name, f'train_val_{metric}')
                    plt.savefig(f'{img_file}.pdf', dpi=600, format='pdf')
                    plt.savefig(f'{img_file}.jpg', dpi=600, format='jpg')
                else:
                    img_file = os.path.join(data_name, f'val_{metric}')
                    plt.savefig(f'{img_file}.pdf', dpi=600, format='pdf')
                    plt.savefig(f'{img_file}.jpg', dpi=600, format='jpg')
            if is_show:
                plt.show()
            plt.close()


if __name__ == '__main__':
    path = ''
    Test.test_all(data='30_20_3', verbose_pre=True, is_show_train=False)