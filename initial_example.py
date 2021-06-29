'''
Author: 娄炯
Date: 2021-05-13 15:37:25
LastEditors: loujiong
LastEditTime: 2021-05-14 13:41:05
Description: initial edge network and task graph
Email:  413012592@qq.com
'''

import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import utils
import draw
import random
import time
import math
import numpy as np
import pqdict


if __name__ == '__main__':
    edge_number = 10
    edge_list = [
        utils.Edge(task_concurrent_capacity=1,
                   process_data_rate=rd(7, 10),
                   upload_data_rate=rd(2, 4)) for i in range(edge_number)
    ]
    edge_transmission_data_rate = [[edge_list[i].upload_data_rate]*edge_number for i in range(edge_number)]
    for i in range(edge_number):
        edge_transmission_data_rate[i][i] = 0
    app = utils.Application(release_time=0,
                          task_num=rd(5, 15),
                          release_node=rd(0, edge_number - 1))
    draw.draw(app.task_graph)


