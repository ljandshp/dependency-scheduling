'''
Author: 娄炯
Date: 2021-04-16 13:18:37
LastEditors: loujiong
LastEditTime: 2021-08-19 16:19:08
Description: utils file
Email:  413012592@qq.com
'''
import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import random
import draw
import time
import math
import numpy as np
import pqdict
import read_in_task_graph

class Edge():
    def __init__(self, task_concurrent_capacity, process_data_rate,
                 upload_data_rate, cost_per_mip = 1):
        # set 10000 time slots first
        self.task_concurrent_capacity = task_concurrent_capacity
        self.planed_start_finish = [np.array([[0.0,10000000000000]]) for i in range(self.task_concurrent_capacity)]
        self.start_finish = [np.array([[0.0,10000000000000]]) for i in range(self.task_concurrent_capacity)]
        self.process_data_rate = process_data_rate
        self.upload_data_rate = upload_data_rate
        self.cost_per_mip = cost_per_mip

    def find_actual_earliest_start_time_by_planed(self, start_time, runtime, _release_time):
        min_start_time = 1000000000000000
        selected_cpu = -1
        selected_interval_key = -1
        for _cpu in range(self.task_concurrent_capacity):
            # earliest start time for virtual task is its start_time
            if runtime == 0:
                return (start_time,0, 0)
            is_in_interval = self.planed_start_finish[_cpu][:,1] - np.maximum(start_time,self.planed_start_finish[_cpu][:,0]) >= runtime
            st_time = (1 - is_in_interval)* 1000000000000000 + np.maximum(start_time,self.planed_start_finish[_cpu][:,0])
            interval_key = np.argmin(st_time)
            _st = max(start_time,self.planed_start_finish[_cpu][interval_key][0])
            if min_start_time > _st:
                min_start_time = _st
                selected_interval_key = interval_key
                selected_cpu = _cpu
        return (min_start_time, selected_cpu, selected_interval_key)

    def set_cpu_state_by_planed(self, _cpu, actual_start_time,
                                estimated_runtime, _application_index,
                                selected_index, selected_interval_key):
        # do not set for virtual task
        if estimated_runtime != 0:   
            _start = self.planed_start_finish[_cpu][selected_interval_key][0]
            _end = self.planed_start_finish[_cpu][selected_interval_key][1]
            actual_end_time = actual_start_time + estimated_runtime
            self.planed_start_finish[_cpu] = np.delete(self.planed_start_finish[_cpu], selected_interval_key, 0)
            if _end == 10000000000000:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([actual_end_time, _end])])
            elif _end > actual_end_time:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([actual_end_time, _end])])
            if _start < actual_start_time:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([_start, actual_start_time])])

    def update_plan_to_actural(self, new_finish_task_set, application_list):
        for item in new_finish_task_set:
            _ap_index, _ta_index, start_time, finish_time = [i for i in item]
            _ap_index = int(_ap_index)
            _ta_index = int(_ta_index)
            cpu = application_list[_ap_index].task_graph.nodes()[_ta_index]["cpu"]
            
            if start_time == finish_time:
                continue
            
            # first find the item in self.start_finish[cpu]
            is_in_interval = (self.start_finish[cpu][:,0] <= start_time) * 1 + (self.start_finish[cpu][:,1] >= finish_time) * 1
            interval_key = np.argmax(is_in_interval)
            
            # then try to split the time interval 
            _start = self.start_finish[cpu][interval_key][0]
            _end = self.start_finish[cpu][interval_key][1]
            # delete time interval
            self.start_finish[cpu] = np.delete(self.start_finish[cpu], interval_key, 0)
            if _end == 10000000000000:
                self.start_finish[cpu] = np.vstack([self.start_finish[cpu], np.array([finish_time, _end])])
            elif _end > finish_time:
                self.start_finish[cpu] = np.vstack([self.start_finish[cpu], np.array([finish_time, _end])])
            if _start < start_time:
                self.start_finish[cpu] = np.vstack([self.start_finish[cpu], np.array([_start, start_time])])

    def generate_plan(self, _release_time):
        self.planed_start_finish = [0 for i in range(self.task_concurrent_capacity)]
        for _cpu in range(self.task_concurrent_capacity):
            self.planed_start_finish[_cpu] = self.start_finish[_cpu].copy()
            self.planed_start_finish[_cpu] = self.planed_start_finish[_cpu][self.planed_start_finish[_cpu][:,1]>=_release_time,:] 

    def interval_statistical(self):
        interval_number_list = []
        total_length_list = []
        interval_sum_list = []
        for _cpu in range(self.task_concurrent_capacity):
            interval_number = self.start_finish[_cpu].shape[0]
            interval_number_list.append(interval_number - 1)
            total_length = self.start_finish[_cpu][interval_number-1][0]
            total_length_list.append(total_length)
            interval_sum = 0
            for i in self.start_finish[_cpu][:-1]:
                interval_sum += i[1] - i[0] + 1
            interval_sum_list.append(interval_sum)
        return([interval_number_list,total_length_list,interval_sum_list])

                
# the speed of the cloud is fastest
class Cloud():
    def __init__(self, cost_per_mip = 4,data_rate = 15):
        self.process_data_rate = 2
        self.data_rate = data_rate
        self.cost_per_mip = cost_per_mip

class Application():
    def __init__(
        self,
        release_time,
        release_node,
        task_num=5,
        application_index = -1
    ):
        self.release_time = release_time
        self.finish_time = -1
        self.release_node = release_node
        # self.generate_application_by_random(task_num = task_num)
        self.generate_task_graph_by_random_by_level(task_num = task_num,level_num=rd(3,5),jump_num=rd(2,3))
        self.dynamic_longest_remain_length = 0
        self.tmax = 0
        self.deadline = 0
        self.application_index = application_index
        self.is_accept = False

    def generate_task_graph_by_random_by_level(self, task_num,level_num=4,jump_num=2):
        task_graph_source = 0
        if task_graph_source == 0:
            self.task_graph = read_in_task_graph.get_workflow()
            task_num = self.task_graph.number_of_nodes()
        else:
            edge_num = rd(task_num, min(task_num * task_num, math.ceil(task_num * 1.5)))
            level_num = min(level_num,task_num)

            node_for_level = self.generate_node_for_level(task_num, level_num)
            self.task_graph = nx.DiGraph()
            for i in range(task_num):
                self.task_graph.add_node(i + 1)

    # ----------------------- connect the next level---------------------------
            # current_edge_num = 0
            # for i in range(level_num - 1):
            #     for j in node_for_level[i]:
            #         self.task_graph.add_edge(j + 1, random.choice(node_for_level[i + 1]) + 1)
            #         current_edge_num += 1

    # ----------------------- connect the previous level---------------------------        
            current_edge_num = 0
            for i in range(level_num - 1):
                for j in node_for_level[i+1]:
                    self.task_graph.add_edge(random.choice(node_for_level[i]) + 1,j + 1)
                    current_edge_num += 1

            
    # ----------------------- connect the previous and the next level---------------------------     
            
            # current_edge_num = 0
            # for i in range(level_num - 1):
            #     for j in node_for_level[i]:
            #         self.task_graph.add_edge(j + 1, random.choice(node_for_level[i + 1]) + 1)
            #         current_edge_num += 1
            # print("first try, edge_num:{0},current_edge_num:{1}".format(edge_num,current_edge_num))
            # for i in range(1,level_num):
            #     for j in node_for_level[i]:
            #         if self.task_graph.in_degree(j + 1) == 0:
            #             self.task_graph.add_edge(random.choice(node_for_level[i-1]) + 1 ,j + 1)
            #             current_edge_num += 1
            
            # print("node_for_level:{0}".format(node_for_level))

            # add sink and source edges to current_edge_num
            current_edge_num += len(node_for_level[0])
            current_edge_num += len(node_for_level[-1])

            # print("edge_num:{0},current_edge_num:{1}".format(edge_num,current_edge_num))
            # print("edge list:{0}".format([(u - 1,v - 1) for u,v in self.task_graph.edges()]))

            test_num = 0
            while (current_edge_num < edge_num):
                sink_level = random.randint(1, level_num - 1)
                source_level = sink_level - random.randint(1, jump_num)
                source_level = max(0, source_level)
                sink_node = random.choice(node_for_level[sink_level]) + 1
                source_node = random.choice(node_for_level[source_level]) + 1

                if (source_node, sink_node) not in self.task_graph.edges:
                    self.task_graph.add_edge(source_node, sink_node)
                    current_edge_num += 1

                test_num += 1
                if test_num > 2000:
                    break

        source_node = 0
        sink_node = task_num + 1

        for i in range(1, task_num + 1):
            if self.task_graph.in_degree(i) == 0:
                self.task_graph.add_edge(source_node, i)
            if self.task_graph.out_degree(i) == 0:
                self.task_graph.add_edge(i, sink_node)

        # add weight
        for i in range(1, task_num + 1):
            self.task_graph.nodes[i]["w"] = 7+14*random.random() #rd(3, 20) #rd(10, 15)
            self.task_graph.nodes[i]["latest_change_time"] = self.release_time
            self.task_graph.nodes[i]["is_scheduled"] = 0
            self.task_graph.nodes[i]["selected_node"] = -1
        self.task_graph.nodes[source_node]["w"] = 0
        self.task_graph.nodes[source_node][
            "latest_change_time"] = self.release_time
        self.task_graph.nodes[source_node]["is_scheduled"] = 0
        self.task_graph.nodes[source_node]["selected_node"] = -1
        self.task_graph.nodes[sink_node]["w"] = 0
        self.task_graph.nodes[sink_node][
            "latest_change_time"] = self.release_time
        self.task_graph.nodes[sink_node]["is_scheduled"] = 0
        self.task_graph.nodes[sink_node]["selected_node"] = -1

        for u, v in self.task_graph.edges():
            if u == source_node or v == sink_node:
                self.task_graph.edges[u, v]["e"] = 2 + 4*random.random() #rd(2, 7)
            else:
                self.task_graph.edges[u, v]["e"] = 2 + 4*random.random() #rd(2, 7)

    def generate_node_for_level(self, node_num, level_num):
        node_number_for_level = [[1] for i in range(level_num)]
        _node_num = 0
        node_for_level = []
        for i in range(node_num - level_num):
            random.choice(node_number_for_level).append(1)
        for i in range(level_num):
            node_for_level.append([
                j for j in range(_node_num, _node_num +
                                len(node_number_for_level[i]))
            ])
            _node_num += len(node_number_for_level[i])
        return node_for_level
        
    def set_start_time(self, selected_task, selected_node, start_time, _cpu):
        self.task_graph.nodes[selected_task]["selected_node"] = selected_node
        self.task_graph.nodes[selected_task]["cpu"] = _cpu
        self.task_graph.nodes[selected_task]["start_time"] = start_time

    def generate_application_by_random(self, task_num):
        self.generate_task_graph_by_random(task_num)

    def generate_task_graph_by_random(self, task_num):
        node = range(1, task_num + 1)
        edge_num = rd(task_num - 1, min(task_num * task_num, task_num * 2))
        source_node = 0
        sink_node = task_num + 1
        self.task_graph = nx.DiGraph()
        for i in range(task_num + 2):
            self.task_graph.add_node(i)

        while self.task_graph.number_of_edges() < edge_num:
            p1 = rd(0, task_num - 2)
            p2 = rd(p1 + 1, task_num - 1)
            if (p1, p2) not in self.task_graph.edges:
                self.task_graph.add_edge(p1, p2)
        
        for i in range(1, task_num + 1):
            if self.task_graph.in_degree(i) == 0:
                self.task_graph.add_edge(source_node, i)
            if self.task_graph.out_degree(i) == 0:
                self.task_graph.add_edge(i, sink_node)

        # add weight
        for i in range(1, task_num + 1):
            self.task_graph.nodes[i]["w"] = rd(6, 10)
            self.task_graph.nodes[i]["latest_change_time"] = self.release_time
            self.task_graph.nodes[i]["is_scheduled"] = 0
            self.task_graph.nodes[i]["selected_node"] = -1
        self.task_graph.nodes[source_node]["w"] = 0
        self.task_graph.nodes[source_node][
            "latest_change_time"] = self.release_time
        self.task_graph.nodes[source_node]["is_scheduled"] = 0
        self.task_graph.nodes[source_node]["selected_node"] = -1
        self.task_graph.nodes[sink_node]["w"] = 0
        self.task_graph.nodes[sink_node][
            "latest_change_time"] = self.release_time
        self.task_graph.nodes[sink_node]["is_scheduled"] = 0
        self.task_graph.nodes[sink_node]["selected_node"] = -1

        for u, v in self.task_graph.edges():
            if u == source_node or v == sink_node:
                self.task_graph.edges[u, v]["e"] = rd(4, 8)
            else:
                self.task_graph.edges[u, v]["e"] = rd(4, 8)

    def set_latest_change_time(self, selected_node, selected_task_index,
                               edge_list, cloud, earliest_start_time):
        if selected_task_index == 0:
            self.task_graph.nodes[selected_task_index][
                "latest_change_time"] = self.release_time
        else:
            # earliest_state_time+ transimmission time + precedence job finish time
            precedence_latest_transmission_time = []
            for u, v in self.task_graph.in_edges(selected_task_index):
                precedence_task_node = self.task_graph.nodes[u][
                    "selected_node"]
                bandwidth = get_bandwidth(precedence_task_node,selected_node,edge_list,cloud)
                precedence_latest_transmission_time.append(
                        earliest_start_time -
                        self.task_graph.edges[u, v]["e"] *bandwidth)
                
            # globally earliest start time is _application.release_time
            latest_change_time = min(
                precedence_latest_transmission_time
            ) if len(precedence_latest_transmission_time
                        ) > 0 else self.release_time
            self.task_graph.nodes[selected_task_index][
                "latest_change_time"] = latest_change_time

def get_remain_length(G,edge_weight=1,node_weight=1):
    remain_length_list = [0] * G.number_of_nodes()
    for v in list(reversed(list(nx.topological_sort(G)))):
        for u, _ in G.in_edges(v):
            remain_length_list[u] = max(
                remain_length_list[u], remain_length_list[v] + G.nodes[u]["w"]*node_weight + G.edges[u,v]["e"]*edge_weight)
    return (remain_length_list)

def get_sub_deadline_list(G,remain_length_list,deadline = 10,edge_weight=1,node_weight=1):
    sub_deadline_list = [0] * G.number_of_nodes()
    for i in range(G.number_of_nodes()):
        sub_deadline_list[i] = deadline*(remain_length_list[0]-remain_length_list[i]+G.nodes[i]["w"]*node_weight)/remain_length_list[0]
    return sub_deadline_list

def get_sub_deadline_list_BDAS(G,remain_length_list,edge_list,deadline = 10,edge_weight=1,node_weight=1):
    process_data_rate_list = [edge.process_data_rate for edge in edge_list]
    upload_data_rate = [edge.upload_data_rate for edge in edge_list]
    level_list = [0]*G.number_of_nodes()
    for _n in range(G.number_of_nodes()-1,-1,-1):
        if _n == G.number_of_nodes()-1:
            level_list[_n] = 1
        else:
            level_list[_n] = max([level_list[v] for _,v in G.out_edges(_n)]) + 1

    level_dict = dict()
    for index,item in enumerate(level_list):
        level_dict.setdefault(item, []).append(index)
    
    ECT_list = [0]*G.number_of_nodes()
    ECT_list[0] = 0

    EST_list = [0]*G.number_of_nodes()
    for _n in range(G.number_of_nodes()):
        est = 0
        for v,_ in G.in_edges(_n):
            est = max(est,EST_list[v]+min(process_data_rate_list)*G.nodes[v]["w"]+G.edges[v,_n]["e"]*min(upload_data_rate))
        EST_list[_n] = est

    subD_list = [0]*(level_list[0]+1)
    for level in range(level_list[0],0,-1):
        if level == level_list[0]:
            ECT_list[0] = 0
        else:
            for _n in level_dict[level]:
                pre_subD = 0
                for v,_ in G.in_edges(_n):
                    pre_subD = max(pre_subD,subD_list[level_list[v]])
                ECT_list[_n] = max(pre_subD, EST_list[_n])+ min(process_data_rate_list)*G.nodes[_n]["w"]
        subD_list[level] = max([ECT_list[i] for i in level_dict[level]])
    
    alpha = 0
    for level in range(1,level_list[0]):
        alpha += (subD_list[level]-subD_list[level+1])*len(level_dict[level])
    
    df = (deadline - subD_list[1])/alpha

    subd_final_list = [0]*(level_list[0]+1)
   
    alpha = 0
    for level in range(level_list[0],0,-1):
        if level == level_list[0]:
            subd_final_list[level] = subD_list[level]
        else:
            alpha += (subD_list[level]-subD_list[level+1])*len(level_dict[level])*df
            subd_final_list[level] = alpha + subD_list[level]

    sub_deadline_list = [0] * G.number_of_nodes()
    for _n in range(G.number_of_nodes()):
        sub_deadline_list[_n] = subd_final_list[level_list[_n]]
    
    return sub_deadline_list

def get_start_sub_deadline_list(G,remain_length_list,deadline = 10):
    sub_deadline_list = [0] * G.number_of_nodes()
    for i in range(G.number_of_nodes()):
        sub_deadline_list[i] = deadline*(remain_length_list[0]-remain_length_list[i])/remain_length_list[0]
    return sub_deadline_list
    
def get_node_with_least_cost_constrained_by_subdeadline(selected_task_index, _application:Application,
                                   edge_list, cloud, _release_time):
    edge_number = len(edge_list)
    finish_time_list = []
    
    # estimate the cloud start time
    precedence_task_finish_time = []
    for u, v in _application.task_graph.in_edges(selected_task_index):
        precedence_task_node = _application.task_graph.nodes[u][
            "selected_node"]
        bandwidth = get_bandwidth(precedence_task_node,len(edge_list),edge_list,cloud)
        precedence_task_finish_time.append(
                _application.task_graph.edges[u, v]["e"] * bandwidth +
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
            bandwidth = get_bandwidth(precedence_task_node,_selected_node,edge_list,cloud)
            precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *bandwidth +
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
        if cost_per_mip_list[i] < min_cost and finish_time_list[i] <= sub_deadline:
            selected_node = i
            min_cost = cost_per_mip_list[i]
            ft = finish_time_list[i]
        if cost_per_mip_list[i] == min_cost and finish_time_list[i] <= sub_deadline and finish_time_list[i] < ft:
            selected_node = i
            min_cost = cost_per_mip_list[i]
            ft = finish_time_list[i]
    
    
    
    is_in_deadline = selected_node > -1
    # if no node satisfy the sub_deadline
    if selected_node < 0:
        selected_node = np.argmin(np.array(finish_time_list))
    
    # print("application:{0}, task:{1}".format(_application.application_index,selected_task_index))
    # print(actual_start_time_list)
    # print(finish_time_list)
    # print("selection:{0},cloud:{1}".format(selected_node,edge_number))
    # print("sub_deadline:{0}".format(sub_deadline))
    # print("release_time:{0}".format(_application.release_time))
    # print()
    
    return is_in_deadline,selected_node

def get_node_with_earliest_finish_time(selected_task_index, _application,
                                   edge_list, cloud, _release_time):
    edge_number = len(edge_list)
    finish_time_list = []

    # estimate the cloud start time
    precedence_task_finish_time = []
    for u, v in _application.task_graph.in_edges(selected_task_index):
        precedence_task_node = _application.task_graph.nodes[u][
            "selected_node"]
        bandwidth = get_bandwidth(precedence_task_node,len(edge_list),edge_list,cloud)
        precedence_task_finish_time.append(
                _application.task_graph.edges[u, v]["e"] * bandwidth +
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
            bandwidth = get_bandwidth(precedence_task_node,_selected_node,edge_list,cloud)
            precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    bandwidth +
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

    is_in_deadline = True
    return is_in_deadline,selected_node

def get_node_with_earliest_finish_time_without_cloud(selected_task_index, _application,
                                   edge_list, cloud, _release_time):
    edge_number = len(edge_list)
    finish_time_list = []

    # estimate the cloud start time
    precedence_task_finish_time = []
    for u, v in _application.task_graph.in_edges(selected_task_index):
        precedence_task_node = _application.task_graph.nodes[u][
            "selected_node"]
        bandwidth = get_bandwidth(precedence_task_node,len(edge_list),edge_list,cloud)
        precedence_task_finish_time.append(
                _application.task_graph.edges[u, v]["e"] * bandwidth +
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
            bandwidth = get_bandwidth(precedence_task_node,_selected_node,edge_list,cloud)
            precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *bandwidth +
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
    
def check(is_multiple:bool, application_list, edge_list, cloud):
    # first check the precedence constraint
    for application_index, application in enumerate(application_list):
        for _n in application.task_graph.nodes():
            for u,_ in application.task_graph.in_edges(_n):
                bandwidth = get_bandwidth(application.task_graph.nodes[u]["selected_node"],application.task_graph.nodes[_n]["selected_node"], edge_list, cloud)
                if application.task_graph.nodes[u]["finish_time"]+bandwidth*application.task_graph.edges[u,_n]["e"]>application.task_graph.nodes[_n]["start_time"]:
                    print("precedence error")
                    exit(1)

    if is_multiple:          
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
                        print("same task fi_time st time error")
                        print(st_time,fi_time)
                        quit()
                    if st_time < fi:
                        print("cross task fi_time st time error")
                        print(fi,st_time,fi_time)
                        print(a_i,t_i)
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
        bandwidth = get_bandwidth(precedence_task_node,len(edge_list),edge_list,cloud)
        precedence_task_finish_time.append(
                _application.task_graph.edges[u, v]["e"] * bandwidth +
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
            bandwidth = get_bandwidth(precedence_task_node,_selected_node,edge_list,cloud)
            precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] *
                    bandwidth +
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
        if cost_per_mip_list[i] < min_cost and finish_time_list[i] <= sub_deadline:
            selected_node = i
            min_cost = cost_per_mip_list[i]
            ft = finish_time_list[i]
        if cost_per_mip_list[i] == min_cost and finish_time_list[i] <= sub_deadline and finish_time_list[i] < ft:
            selected_node = i
            min_cost = cost_per_mip_list[i]
            ft = finish_time_list[i]

    is_in_deadline = selected_node >= 0 
    # if no node satisfy the sub_deadline
    if selected_node < 0:
        selected_node = np.argmin(np.array(finish_time_list[:-1]))
      
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
            bandwidth = get_bandwidth(precedence_task_node,_selected_node,edge_list,cloud)
            precedence_task_finish_time.append(
                    _application.task_graph.edges[u, v]["e"] * bandwidth +
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

        _is_in_deadline = True
        _overdue_start_deadline = 0
        for u,v in _application.task_graph.out_edges(selected_task_index):
            if v == _application.task_graph.number_of_nodes()-1:
                bandwidth = get_bandwidth(_selected_node,_application.release_node,edge_list,cloud)
            else:
                bandwidth = edge_list[_selected_node].upload_data_rate

            if _selected_node_finish_time + _application.task_graph.edges[u,v]["e"]*bandwidth > _application.task_graph.nodes()[v]["start_sub_deadline"]+_application.release_time:
                _is_in_deadline = False
                _overdue_start_deadline = max(_overdue_start_deadline,_selected_node_finish_time + _application.task_graph.edges[u,v]["e"]*bandwidth - _application.task_graph.nodes()[v]["start_sub_deadline"]-_application.release_time)
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

def set_tmax(_app,edge_list,cloud):
    total_workload = sum([_app.task_graph.nodes()[_n]["w"] for _n in _app.task_graph.nodes()])
    slowest_process_date_rate = max([_e.process_data_rate for _e in edge_list])
    slowest_process_date_rate = max([slowest_process_date_rate,cloud.process_data_rate])
    _app.tmax = total_workload * slowest_process_date_rate

def get_bandwidth(node1,node2, edge_list, cloud):
    cloud_number = len(edge_list)
    if node1 == node2:
        return 0
    elif node1 == cloud_number or node2 == cloud_number:
        bandwidth = cloud.data_rate
        return bandwidth
    else:
        bandwidth = edge_list[node1].upload_data_rate
        return bandwidth

if __name__ == '__main__':
    pass