'''
Author: 娄炯
Date: 2021-09-11 08:59:30
LastEditors: loujiong
LastEditTime: 2021-09-11 09:07:49
Description: 
Email:  413012592@qq.com
'''

'''
Author: 娄炯
Date: 2021-09-07 16:14:30
LastEditors: loujiong
LastEditTime: 2021-09-10 15:15:06
Description: 
Email:  413012592@qq.com
'''
import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import utils_backup as utils
import utils_backup2 as utils2
import draw
import random
import time
import math
import numpy as np
import pqdict
import baseline
import sys, getopt
import dasa

np.set_printoptions(suppress=True)



if __name__ == '__main__':
    resultfile = ''
    try:
      opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
    for opt, arg in opts:
        if opt in ("-o", "--ofile"):
            resultfile = arg
    if resultfile != '':
        with open(resultfile,mode="w") as f:
            f.write("")
         
    is_draw = False
    is_annotation = True
    is_draw_task_graph = False
    application_num = 100
    application_average_interval = 120
    edge_number = 20
    random_seed = 1.2
    is_multiple = False
    deadline_alpha = 0.2
    ccr = 0.1
    for ccr in [0.5]:
        for application_average_interval in range(150,550,400):
            for deadline_alpha in range(10):
                deadline_alpha = 0 + 0.05*deadline_alpha
                exp_num = 1
                a_list = [0,0]
                c_list = [0,0]
                a_w_list = [0,0]
                for edge_number in [10,20,30,40,50]:
                    random_seed = 1+0.1
                    print("edge_number",edge_number)
                    _a, _c, application_list, a_w = baseline.re_scheduling(
                        is_draw=is_draw,
                        is_annotation=is_annotation,
                        application_num=application_num,
                        application_average_interval=application_average_interval,
                        edge_number=edge_number,
                        scheduler=utils2.get_node_with_earliest_finish_time,
                        random_seed=random_seed,
                        is_draw_task_graph=is_draw_task_graph,
                        is_multiple=False,
                        deadline_alpha=deadline_alpha,
                        base_deadline = [],
                        ccr = ccr)
                    a_list[0]+=_a
                    c_list[0]+=_c
                    a_w_list[0] += a_w
                    print()

                    base_deadline = []
                    for a in application_list:
                        _sink = a.task_graph.number_of_nodes()-1
                        base_deadline.append(a.task_graph.nodes[_sink]["finish_time"]-a.release_time)
                        
                    _a, _c, _, a_w = dasa.re_scheduling(
                            is_draw=is_draw,
                            is_annotation=is_annotation,
                            application_num=application_num,
                            application_average_interval=application_average_interval,
                            edge_number=edge_number,
                            scheduler=utils.
                            get_node_with_least_cost_constrained_by_start_subdeadline_without_cloud,
                            random_seed=random_seed,
                            is_draw_task_graph=is_draw_task_graph,
                            is_multiple=is_multiple,
                            deadline_alpha=deadline_alpha,
                            base_deadline = base_deadline,
                            ccr = ccr)
                    print()
                    baseline.re_scheduling(
                        is_draw=is_draw,
                        is_annotation=is_annotation,
                        application_num=application_num,
                        application_average_interval=application_average_interval,
                        edge_number=edge_number,
                        scheduler=utils2.get_node_with_earliest_finish_time,
                        random_seed=random_seed,
                        is_draw_task_graph=is_draw_task_graph,
                        is_multiple=False,
                        deadline_alpha=deadline_alpha,
                        base_deadline = [],
                        ddmethod ="pcp",
                        ccr = ccr)
                    print()
                    baseline.re_scheduling(
                        is_draw=is_draw,
                        is_annotation=is_annotation,
                        application_num=application_num,
                        application_average_interval=application_average_interval,
                        edge_number=edge_number,
                        scheduler=utils2.get_node_with_earliest_finish_time,
                        random_seed=random_seed,
                        is_draw_task_graph=is_draw_task_graph,
                        is_multiple=False,
                        deadline_alpha=deadline_alpha,
                        base_deadline = [],
                        ddmethod ="prolis",
                        ccr = ccr)
                    print()
                    baseline.re_scheduling(
                        is_draw=is_draw,
                        is_annotation=is_annotation,
                        application_num=application_num,
                        application_average_interval=application_average_interval,
                        edge_number=edge_number,
                        scheduler=utils2.get_node_with_earliest_finish_time,
                        random_seed=random_seed,
                        is_draw_task_graph=is_draw_task_graph,
                        is_multiple=False,
                        deadline_alpha=deadline_alpha,
                        base_deadline = [],
                        ddmethod ="bdas",
                        ccr = ccr)
                    print()
                    a_list[1]+=_a
                    c_list[1]+=_c
                    a_w_list[1] += a_w
                quit()
                print("deadline_alpha",deadline_alpha)
                print("application_average_interval",application_average_interval)
                print([i/exp_num for i in a_list])
                print([i/exp_num for i in c_list])
                print([c_list[i]/(a_w_list[i]+0.00000000001) for i in range(len(c_list))])
                # quit()
                print()
                if resultfile != '':
                    with open(resultfile,mode="a") as f:
                        f.write("ccr:{0}\n".format(ccr))
                        f.write("deadline_alpha:{0}\n".format(deadline_alpha))
                        f.write("application_average_interval:{0}\n".format(application_average_interval))
                        f.write("success_number:"+str([i/exp_num for i in a_list])+"\n")
                        f.write("total_cost:"+str([i/exp_num for i in c_list])+"\n")
                        f.write("normalized_cost:"+str([c_list[i]/(a_w_list[i]+0.00000000001) for i in range(len(c_list))])+"\n")


