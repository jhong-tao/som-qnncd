#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：som-qnncd -> options.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2022/9/23
@Desc   ：
==================================================
"""

som_parm = {
        'x': 10,
        'y': 10,
        # 'input_len': Q.shape[1]+Q.shape[0],
        'sigma': 1.0,
        'learning_rate': 0.01,
        'neighborhood_function':  'gaussian'}

dir_data_real = [
        '../data/real/frac/15_5/',
        '../data/real/frac/20_8/',
        '../data/real/timss07/25_7/',
        '../data/real/timss07/25_15/',
]

dir_data_sim = [
        '../data/sim/10_3/',
        '../data/sim/20_3/',
        '../data/sim/20_5/',
        '../data/sim/30_5/',
]

dir_data_num = ['50', '100', '300', '1000']

dir_data = [
        # '../data/real/',
        '../data/sim/',
]