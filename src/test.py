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
import glob

path = '../data/sim/10_3/'

for root, dirs, files in os.walk(path):
    print(root)


