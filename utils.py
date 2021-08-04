'''
Author: 娄炯
Date: 2021-04-16 13:18:37
LastEditors: loujiong
LastEditTime: 2021-07-25 17:40:11
Description: utils file
Email:  413012592@qq.com
'''
import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import random
import draw
import math
import numpy as np
import pqdict
                

def get_remain_length(G,edge_weight=1,node_weight=1):
    remain_length_list = [0] * G.number_of_nodes()
    for v in list(reversed(list(nx.topological_sort(G)))):
        for u, v in G.in_edges(v):
            remain_length_list[u] = max(
                remain_length_list[u], remain_length_list[v] + G.nodes[u]["w"]*node_weight + G.edges[u,v]["e"]*edge_weight)
    return (remain_length_list)

def get_sub_deadline_list(G,remain_length_list,deadline = 10,edge_weight=1,node_weight=1):
    sub_deadline_list = [0] * G.number_of_nodes()
    for i in range(G.number_of_nodes()):
        sub_deadline_list[i] = deadline*(remain_length_list[0]-remain_length_list[i]+G.nodes[i]["w"]*node_weight)/remain_length_list[0]
    return sub_deadline_list

def get_start_sub_deadline_list(G,remain_length_list,deadline = 10):
    sub_deadline_list = [0] * G.number_of_nodes()
    for i in range(G.number_of_nodes()):
        sub_deadline_list[i] = deadline*(remain_length_list[0]-remain_length_list[i])/remain_length_list[0]
    return sub_deadline_list

def set_tmax(_app,edge_list,cloud):
    total_workload = sum([_app.task_graph.nodes()[_n]["w"] for _n in _app.task_graph.nodes()])
    slowest_process_date_rate = max([_e.process_data_rate for _e in edge_list])
    slowest_process_date_rate = max([slowest_process_date_rate,cloud.process_data_rate])
    _app.tmax = total_workload * slowest_process_date_rate
    
def get_node_with_least_cost_constrained_by_subdeadline(selected_task_index, _application,
                                   edge_list, cloud, _release_time):
    edge_number = len(edge_list)
    finish_time_list = []
    
    # estimate the cloud start time
    precedence_task_finish_time = []
    for u, v in _application.task_graph.in_edges(selected_task_index):
        precedence_task_node = _application.task_graph.nodes[u][
            "selected_node"]
        if precedence_task_node == edge_number:
            precedence_task_finish_time.append(
                max(_application.task_graph.nodes[u]["finish_time"],_release_time))
        else:
            precedence_task_finish_time.append(
                _application.task_graph.edges[u, v]["e"] * cloud.data_rate +
                max(_application.task_graph.nodes[u]["finish_time"],_release_time))
    # globally earliest start time is _application.release_time
    cloud_earliest_start_time = max(precedence_task_finish_time) if len(
        precedence_task_finish_time) > 0 else _release_time
    cloud_estimated_finish_time = cloud_earliest_start_time + _application.task_graph.nodes[selected_task_index]["w"] * cloud.process_data_rate
    actual_start_time_list = []
    for _selected_node in range(edge_number):
        precedence_task_finish_time = []
        for u, v in _application.task_graph.in_edges(selected_task_index):
            precedence_task_node = _application.task_graph.nodes[u][
                "selected_node"]
            if precedence_task_node == edge_number:
                # from the cloud
                precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    cloud.data_rate +
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))
            elif precedence_task_node != _selected_node:
                # not same edge node
                precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    edge_list[precedence_task_node].upload_data_rate +
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))
            else:
                # same ege node
                precedence_task_finish_time.append(
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))

        # globally earliest start time is _release_time
        earliest_start_time = _release_time if len(
            precedence_task_finish_time) == 0 else max(
                precedence_task_finish_time)

        # run time
        estimated_runtime = _application.task_graph.nodes[selected_task_index][
            "w"] * edge_list[_selected_node].process_data_rate

        # actual start time and _cpu
        actual_start_time, _, _ = edge_list[_selected_node].find_actual_earliest_start_time_by_planed(earliest_start_time, estimated_runtime,_release_time)

        # set start time and node for each task
        _selected_node_finish_time = actual_start_time + estimated_runtime
        finish_time_list.append(_selected_node_finish_time)
        actual_start_time_list.append(actual_start_time)
        
    actual_start_time_list.append(cloud_earliest_start_time)    
    finish_time_list.append(cloud_estimated_finish_time)
    sub_deadline = _application.task_graph.nodes[selected_task_index]["sub_deadline"]+_application.release_time
    cost_per_mip_list = [i.cost_per_mip for i in edge_list]
    cost_per_mip_list.append(cloud.cost_per_mip)
    selected_node = -1
    min_cost = 10000
    ft = 100000000000
    for i in range(edge_number+1):
        if cost_per_mip_list[i] < min_cost and finish_time_list[i] < sub_deadline:
            selected_node = i
            min_cost = cost_per_mip_list[i]
            ft = finish_time_list[i]
        if cost_per_mip_list[i] == min_cost and finish_time_list[i] < sub_deadline and finish_time_list[i] < ft:
            selected_node = i
            min_cost = cost_per_mip_list[i]
            ft = finish_time_list[i]

    # if no node satisfy the sub_deadline
    if selected_node < 0:
        selected_node = np.argmin(np.array(finish_time_list))
        # print("unsatisfied deadline")
        # if selected_node == edge_number:
            # print("unsatisfied and to the cloud")
        # print("selected_task_index:{0},selected_node:{1}".format(selected_task_index,selected_node))
        # print("actual_start_time_list:{0}".format(actual_start_time_list))
        # print("finish_time_list:{0}".format(finish_time_list))
        
    return selected_node

def get_node_with_earliest_finish_time(selected_task_index, _application,
                                   edge_list, cloud, _release_time):
    edge_number = len(edge_list)
    finish_time_list = []

    # estimate the cloud start time
    precedence_task_finish_time = []
    for u, v in _application.task_graph.in_edges(selected_task_index):
        precedence_task_node = _application.task_graph.nodes[u][
            "selected_node"]
        if precedence_task_node == edge_number:
            precedence_task_finish_time.append(
                max(_application.task_graph.nodes[u]["finish_time"],_release_time))
        else:
            precedence_task_finish_time.append(
                _application.task_graph.edges[u, v]["e"] * cloud.data_rate +
                max(_application.task_graph.nodes[u]["finish_time"],_release_time))
    # globally earliest start time is _application.release_time
    cloud_earliest_start_time = max(precedence_task_finish_time) if len(
        precedence_task_finish_time) > 0 else _application.release_time
    cloud_estimated_finish_time = cloud_earliest_start_time + _application.task_graph.nodes[selected_task_index]["w"] * cloud.process_data_rate

    for _selected_node in range(edge_number):
        precedence_task_finish_time = []
        for u, v in _application.task_graph.in_edges(selected_task_index):
            precedence_task_node = _application.task_graph.nodes[u][
                "selected_node"]
            if precedence_task_node == edge_number:
                # from the cloud
                precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    cloud.data_rate +
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))
            elif precedence_task_node != _selected_node:
                # not same edge node
                precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    edge_list[precedence_task_node].upload_data_rate +
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))
            else:
                # same ege node
                precedence_task_finish_time.append(
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))

        # globally earliest start time is _application.release_time
        earliest_start_time = _application.release_time if len(
            precedence_task_finish_time) == 0 else max(
                precedence_task_finish_time)

        # run time
        estimated_runtime = _application.task_graph.nodes[selected_task_index][
            "w"] * edge_list[_selected_node].process_data_rate

        # actual start time and _cpu
        actual_start_time, _cpu, selected_interval_key = edge_list[_selected_node].find_actual_earliest_start_time_by_planed(earliest_start_time, estimated_runtime,_release_time)

        # set start time and node for each task
        _selected_node_finish_time = actual_start_time + estimated_runtime
        finish_time_list.append(_selected_node_finish_time)
    finish_time_list.append(cloud_estimated_finish_time)
    selected_node = np.argmin(np.array(finish_time_list))

    return selected_node

def get_node_with_earliest_finish_time_without_cloud(selected_task_index, _application,
                                   edge_list, cloud, _release_time):
    edge_number = len(edge_list)
    finish_time_list = []

    # estimate the cloud start time
    precedence_task_finish_time = []
    for u, v in _application.task_graph.in_edges(selected_task_index):
        precedence_task_node = _application.task_graph.nodes[u][
            "selected_node"]
        if precedence_task_node == edge_number:
            precedence_task_finish_time.append(
                max(_application.task_graph.nodes[u]["finish_time"],_release_time))
        else:
            precedence_task_finish_time.append(
                _application.task_graph.edges[u, v]["e"] * cloud.data_rate +
                max(_application.task_graph.nodes[u]["finish_time"],_release_time))
    # globally earliest start time is _application.release_time
    cloud_earliest_start_time = max(precedence_task_finish_time) if len(
        precedence_task_finish_time) > 0 else _application.release_time
    cloud_estimated_finish_time = cloud_earliest_start_time + _application.task_graph.nodes[selected_task_index]["w"] * cloud.process_data_rate

    for _selected_node in range(edge_number):
        precedence_task_finish_time = []
        for u, v in _application.task_graph.in_edges(selected_task_index):
            precedence_task_node = _application.task_graph.nodes[u][
                "selected_node"]
            if precedence_task_node == edge_number:
                # from the cloud
                precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    cloud.data_rate +
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))
            elif precedence_task_node != _selected_node:
                # not same edge node
                precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    edge_list[precedence_task_node].upload_data_rate +
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))
            else:
                # same ege node
                precedence_task_finish_time.append(
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))

        # globally earliest start time is _application.release_time
        earliest_start_time = _application.release_time if len(
            precedence_task_finish_time) == 0 else max(
                precedence_task_finish_time)

        # run time
        estimated_runtime = _application.task_graph.nodes[selected_task_index][
            "w"] * edge_list[_selected_node].process_data_rate

        # actual start time and _cpu
        actual_start_time, _cpu, selected_interval_key = edge_list[_selected_node].find_actual_earliest_start_time_by_planed(earliest_start_time, estimated_runtime,_release_time)

        # set start time and node for each task
        _selected_node_finish_time = actual_start_time + estimated_runtime
        finish_time_list.append(_selected_node_finish_time)
    finish_time_list.append(cloud_estimated_finish_time)
    selected_node = np.argmin(np.array(finish_time_list[:-1]))
    is_in_deadline = False
    return is_in_deadline,selected_node
    
def check(application_list):
    d = dict()
    for application_index,application in enumerate(application_list):
        if application.is_accept:
            for task_index in application.task_graph.nodes():
                if application.task_graph.nodes()[task_index]["selected_node"] != -1 and application.task_graph.nodes()[task_index]["cpu"] != -1:
                    start_time = application.task_graph.nodes()[task_index]["start_time"]
                    finish_time = application.task_graph.nodes()[task_index]["finish_time"]
                    node_cpu = "{0}-{1}".format(application.task_graph.nodes()[task_index]["selected_node"],application.task_graph.nodes()[task_index]["cpu"])
                    if node_cpu not in d:
                        d[node_cpu] = pqdict.pqdict()
                    d[node_cpu][(application_index,task_index)] = (start_time,finish_time,application_index,task_index)
    for node_cpu in d:
        keys = pqdict.nsmallest(len(d[node_cpu]),d[node_cpu])
        st = -1
        fi = -1
        _a_i = -1
        _t_i = -1
        for application_index,task_index in keys:
            st_time,fi_time,a_i,t_i = d[node_cpu][(application_index,task_index)]
            if st_time != fi_time:
                if st_time > fi_time:
                    # print("same task fi_time st time error")
                    # print(st_time,fi_time)
                    quit()
                if st_time < fi:
                    # print("cross task fi_time st time error")
                    # print(fi,st_time,fi_time)
                    # print(node_cpu)
                    # print("last application-task: {0}-{1}".format(_a_i,_t_i))
                    # print("current application-task: {0}-{1}".format(application_index,task_index))
                    quit()
                st,fi = st_time,fi_time
                _a_i,_t_i = a_i,t_i


def get_node_with_least_cost_constrained_by_subdeadline_without_cloud(selected_task_index, _application,
                                   edge_list, cloud, _release_time):
    edge_number = len(edge_list)
    finish_time_list = []
    
    # estimate the cloud start time
    precedence_task_finish_time = []
    for u, v in _application.task_graph.in_edges(selected_task_index):
        precedence_task_node = _application.task_graph.nodes[u][
            "selected_node"]
        if precedence_task_node == edge_number:
            precedence_task_finish_time.append(
                max(_application.task_graph.nodes[u]["finish_time"],_release_time))
        else:
            precedence_task_finish_time.append(
                _application.task_graph.edges[u, v]["e"] * cloud.data_rate +
                max(_application.task_graph.nodes[u]["finish_time"],_release_time))
    # globally earliest start time is _application.release_time
    cloud_earliest_start_time = max(precedence_task_finish_time) if len(
        precedence_task_finish_time) > 0 else _release_time
    cloud_estimated_finish_time = cloud_earliest_start_time + _application.task_graph.nodes[selected_task_index]["w"] * cloud.process_data_rate
    actual_start_time_list = []
    for _selected_node in range(edge_number):
        precedence_task_finish_time = []
        for u, v in _application.task_graph.in_edges(selected_task_index):
            precedence_task_node = _application.task_graph.nodes[u][
                "selected_node"]
            if precedence_task_node == edge_number:
                # from the cloud
                precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    cloud.data_rate +
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))
            elif precedence_task_node != _selected_node:
                # not same edge node
                precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    edge_list[precedence_task_node].upload_data_rate +
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))
            else:
                # same ege node
                precedence_task_finish_time.append(
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))

        # globally earliest start time is _release_time
        earliest_start_time = _release_time if len(
            precedence_task_finish_time) == 0 else max(
                precedence_task_finish_time)

        # run time
        estimated_runtime = _application.task_graph.nodes[selected_task_index][
            "w"] * edge_list[_selected_node].process_data_rate

        # actual start time and _cpu
        actual_start_time, _, _ = edge_list[_selected_node].find_actual_earliest_start_time_by_planed(earliest_start_time, estimated_runtime,_release_time)

        # set start time and node for each task
        _selected_node_finish_time = actual_start_time + estimated_runtime
        finish_time_list.append(_selected_node_finish_time)
        actual_start_time_list.append(actual_start_time)
        
    actual_start_time_list.append(cloud_earliest_start_time)    
    finish_time_list.append(cloud_estimated_finish_time)
    sub_deadline = _application.task_graph.nodes[selected_task_index]["sub_deadline"]+_application.release_time
    cost_per_mip_list = [i.cost_per_mip for i in edge_list]
    cost_per_mip_list.append(cloud.cost_per_mip)
    selected_node = -1
    min_cost = 10000
    ft = 100000000000
    for i in range(edge_number):
        if cost_per_mip_list[i] < min_cost and finish_time_list[i] < sub_deadline:
            selected_node = i
            min_cost = cost_per_mip_list[i]
            ft = finish_time_list[i]
        if cost_per_mip_list[i] == min_cost and finish_time_list[i] < sub_deadline and finish_time_list[i] < ft:
            selected_node = i
            min_cost = cost_per_mip_list[i]
            ft = finish_time_list[i]

    is_in_deadline = selected_node >= 0 
    # if no node satisfy the sub_deadline
    if selected_node < 0:
        selected_node = np.argmin(np.array(finish_time_list[:-1]))
        # print("unsatisfied deadline")
        # if selected_node == edge_number:
            # print("unsatisfied and to the cloud")
        # print("selected_task_index:{0},selected_node:{1}".format(selected_task_index,selected_node))
        # print("actual_start_time_list:{0}".format(actual_start_time_list))
        # print("finish_time_list:{0}".format(finish_time_list))

      
    return is_in_deadline,selected_node

def get_node_with_least_cost_constrained_by_start_subdeadline_without_cloud(selected_task_index, _application,
                                   edge_list, cloud, _release_time):
    edge_number = len(edge_list)
    finish_time_list = []
    
    actual_start_time_list = []
    is_in_deadline  = []
    overdue_start_deadline = []
    for _selected_node in range(edge_number):
        precedence_task_finish_time = []
        for u, v in _application.task_graph.in_edges(selected_task_index):
            precedence_task_node = _application.task_graph.nodes[u][
                "selected_node"]
            if precedence_task_node == edge_number:
                # from the cloud
                precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    cloud.data_rate +
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))
            elif precedence_task_node != _selected_node:
                # not same edge node
                precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    edge_list[precedence_task_node].upload_data_rate +
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))
            else:
                # same ege node
                precedence_task_finish_time.append(
                    max(_application.task_graph.nodes[u]["finish_time"],_release_time))

        # globally earliest start time is _release_time
        earliest_start_time = _release_time if len(
            precedence_task_finish_time) == 0 else max(
                precedence_task_finish_time)

        # run time
        estimated_runtime = _application.task_graph.nodes[selected_task_index][
            "w"] * edge_list[_selected_node].process_data_rate

        # actual start time and _cpu
        actual_start_time, _, _ = edge_list[_selected_node].find_actual_earliest_start_time_by_planed(earliest_start_time, estimated_runtime,_release_time)
        # actual_start_time, _, _ = edge_list[_selected_node].find_actual_earliest_start_time_by_planed_modify(earliest_start_time, estimated_runtime,_release_time)

        # set start time and node for each task
        _selected_node_finish_time = actual_start_time + estimated_runtime
        finish_time_list.append(_selected_node_finish_time)
        actual_start_time_list.append(actual_start_time)

        _is_in_deadline = True
        _overdue_start_deadline = 0
        for u,v in _application.task_graph.out_edges(selected_task_index):
            if _selected_node_finish_time + _application.task_graph.edges[u,v]["e"]*edge_list[_selected_node].upload_data_rate > _application.task_graph.nodes()[v]["start_sub_deadline"]+_application.release_time:
                _is_in_deadline = False
                _overdue_start_deadline = max(_overdue_start_deadline,_selected_node_finish_time + _application.task_graph.edges[u,v]["e"]*edge_list[_selected_node].upload_data_rate - _application.task_graph.nodes()[v]["start_sub_deadline"]+_application.release_time)
        is_in_deadline.append(_is_in_deadline)
        overdue_start_deadline.append(_overdue_start_deadline)

    # print("is_in_deadline:{0}".format(is_in_deadline))
    cost_per_mip_list = [i.cost_per_mip for i in edge_list]
    selected_node = -1
    min_cost = 10000
    ft = 100000000000
    for i in range(edge_number):
        if cost_per_mip_list[i] < min_cost and is_in_deadline[i]:
            selected_node = i
            min_cost = cost_per_mip_list[i]
            ft = finish_time_list[i]
        if cost_per_mip_list[i] == min_cost and is_in_deadline[i] and finish_time_list[i] < ft:
            selected_node = i
            min_cost = cost_per_mip_list[i]
            ft = finish_time_list[i]

    is_in_deadline =  selected_node >= 0 
    # if no node satisfy the sub_deadline
    if selected_node < 0:
        selected_node = np.argmin(np.array(overdue_start_deadline))
        # selected_node = np.argmin(np.array(finish_time_list))
        # print("unsatisfied deadline")
        # if selected_node == edge_number:
            # print("unsatisfied and to the cloud")
        # print("selected_task_index:{0},selected_node:{1}".format(selected_task_index,selected_node))
        # print("actual_start_time_list:{0}".format(actual_start_time_list))
        # print("finish_time_list:{0}".format(finish_time_list))

     
    return is_in_deadline,selected_node
    
def get_node_by_random(selected_task_index, _application, edge_list, cloud, _release_time):
    return rd(0, len(edge_list))



if __name__ == '__main__':
    pass