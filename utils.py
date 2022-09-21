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















if __name__ == '__main__':
    get_data4R('./data/test/')