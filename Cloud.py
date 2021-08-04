'''
Author: 娄炯
Date: 2021-08-04 11:14:39
LastEditors: loujiong
LastEditTime: 2021-08-04 11:14:40
Description: 
Email:  413012592@qq.com
'''
import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import numpy as np
                
# the speed of the cloud is fastest
class Cloud():
    def __init__(self, cost_per_mip = 4,data_rate = 15):
        self.process_data_rate = 2
        self.data_rate = data_rate
        self.cost_per_mip = cost_per_mip