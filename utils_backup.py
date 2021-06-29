'''
Author: 娄炯
Date: 2021-04-18 15:22:05
LastEditors: loujiong
LastEditTime: 2021-04-18 15:22:19
Description: utils backup
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

class Edge():
    def __init__(self,task_concurrent_capacity,process_data_rate,upload_data_rate):
        # set 10000 time slots first
        self.task_concurrent_capacity = task_concurrent_capacity
        self.is_task_for_each_time = [[-1]*1000 for i in range(self.task_concurrent_capacity)]
        self.task_for_each_time = [[-1]*1000 for i in range(self.task_concurrent_capacity)] 
        self.planed_is_task_for_each_time = [[-1]*1000 for i in range(self.task_concurrent_capacity)]
        self.planed_task_for_each_time = [[-1]*1000 for i in range(self.task_concurrent_capacity)] 
        self.process_data_rate = process_data_rate
        self.upload_data_rate = upload_data_rate
        
    def find_actual_earliest_start_time(self,start_time,runtime):
        min_start_time = 10000000000000
        selected_cpu = -1
        for _cpu in range(self.task_concurrent_capacity):
            _start_time = int(start_time)
            _runtime = int(runtime)
            while(True):
                while len(self.is_task_for_each_time[_cpu]) <= _start_time+ _runtime:
                    self.is_task_for_each_time[_cpu].extend([-1]*1000)
                    self.task_for_each_time[_cpu].extend([-1]*1000)
                    # print("cpu-{1},length:{1}".format(_cpu,len(self.is_task_for_each_time[_cpu])))
                if sum(self.is_task_for_each_time[_cpu][_start_time:_start_time+ _runtime]) < 0.5-_runtime:
                    if min_start_time > _start_time:
                        min_start_time = _start_time
                        selected_cpu = _cpu
                    break
                else:
                    _start_time += 1
        return (min_start_time,selected_cpu)
    
    def find_actual_earliest_start_time_by_planed(self,start_time,runtime):
        min_start_time = 10000000000000
        selected_cpu = -1
        cnt = 0
        for _cpu in range(self.task_concurrent_capacity):
            _start_time = int(start_time)
            _runtime = int(runtime)
            while(True):
                while len(self.planed_is_task_for_each_time[_cpu]) <= _start_time+ _runtime:
                    self.planed_is_task_for_each_time[_cpu].extend([-1]*1000)
                    self.planed_task_for_each_time[_cpu].extend([-1]*1000)
                    cnt += 1
                inteval = self.planed_is_task_for_each_time[_cpu][_start_time:_start_time+ _runtime]
                if len(inteval) == 0 or max(inteval) < 0:
                    if min_start_time > _start_time:
                        min_start_time = _start_time
                        selected_cpu = _cpu
                    break
                else:
                    _start_time += 1
                   
        return (min_start_time,selected_cpu,cnt)

    def set_cpu_state(self,_cpu,actual_start_time,estimated_runtime,_application_index,selected_index):
        actual_start_time = int(actual_start_time)
        estimated_runtime = int(estimated_runtime)
        for i in range(actual_start_time,actual_start_time+estimated_runtime):
            self.is_task_for_each_time[_cpu][i] = 1
            self.task_for_each_time[_cpu][i] = (_application_index,selected_index)
    
    def set_cpu_state_by_planed(self,_cpu,actual_start_time,estimated_runtime,_application_index,selected_index):
        actual_start_time = int(actual_start_time)
        estimated_runtime = int(estimated_runtime)
        
        self.planed_is_task_for_each_time[_cpu][actual_start_time:actual_start_time+estimated_runtime] = [1]*(estimated_runtime)
        self.planed_task_for_each_time[_cpu][actual_start_time:actual_start_time+estimated_runtime] = [(_application_index,selected_index)]* (estimated_runtime)
    
    def update_plan_to_actural(self,_release_time):
        # 先补全actural
        if len(self.planed_is_task_for_each_time) > len(self.is_task_for_each_time):
            self.is_task_for_each_time.extend([-1]*(len(self.planed_is_task_for_each_time)-len(self.is_task_for_each_time)))
            self.task_for_each_time.extend([-1]*(len(self.planed_is_task_for_each_time)-len(self.is_task_for_each_time)))

        # update actural
        self.is_task_for_each_time[0:_release_time] = self.planed_is_task_for_each_time[0:_release_time]
        self.task_for_each_time[0:_release_time] = self.planed_task_for_each_time[0:_release_time]

    def generate_plan(self,_release_time):
        self.planed_is_task_for_each_time = []
        self.planed_task_for_each_time = []

        self.planed_is_task_for_each_time.extend(self.is_task_for_each_time)
        self.planed_task_for_each_time.extend(self.task_for_each_time)

        
# the speed of the cloud is fastest
class Cloud():
    def __init__(self):
        self.process_data_rate = 1
        self.data_rate = 20

class Application():
    def __init__(self,release_time,release_node,task_num= 5,):
        self.release_time = release_time
        self.finish_time = -1
        self.release_node = release_node
        self.generate_application_by_random(task_num)
    
    def set_start_time(self,selected_task,selected_node,start_time,_cpu):
        self.task_graph.nodes[selected_task]["selected_node"] = selected_node
        self.task_graph.nodes[selected_task]["cpu"] = _cpu
        self.task_graph.nodes[selected_task]["start_time"] = start_time
    
    def generate_application_by_random(self,task_num):
        self.generate_task_graph_by_random(task_num)

    def generate_task_graph_by_random(self,task_num):
        node = range(1,task_num+1)
        edge_num = rd(1,min(task_num*task_num,task_num*2))
        source_node = 0
        sink_node = task_num + 1
        self.task_graph = nx.DiGraph()
        for i in range(task_num+2):
            self.task_graph.add_node(i)

        while self.task_graph.number_of_edges() < edge_num:
            p1 = rd (0,task_num-2)
            p2 = rd (p1+1,task_num-1)
            if (p1, p2) not in self.task_graph.edges:
                self.task_graph.add_edge(p1, p2)
        for i in range(1,task_num+1):
            if self.task_graph.in_degree(i) == 0:
                self.task_graph.add_edge(source_node,i)
            if self.task_graph.out_degree(i) == 0:
                self.task_graph.add_edge(i,sink_node)

        # add weight
        for i in range(1,task_num+1):
            self.task_graph.nodes[i]["w"] = rd(1,10)
            self.task_graph.nodes[i]["latest_change_time"] = self.release_time
            self.task_graph.nodes[i]["is_scheduled"] = 0
        self.task_graph.nodes[source_node]["w"] = 0
        self.task_graph.nodes[source_node]["latest_change_time"] = self.release_time
        self.task_graph.nodes[source_node]["is_scheduled"] = 0
        self.task_graph.nodes[sink_node]["w"] = 0
        self.task_graph.nodes[sink_node]["latest_change_time"] = self.release_time
        self.task_graph.nodes[sink_node]["is_scheduled"] = 0
        
        for u,v in self.task_graph.edges():
            self.task_graph.edges[u,v]["e"] = rd(1,10)
    
    def set_latest_change_time(self,selected_node,selected_task_index,edge_list,cloud,earliest_start_time):
        if selected_task_index == 0:
            self.task_graph.nodes[selected_task_index]["latest_change_time"] = self.release_time
        else:
            edge_number = len(edge_list)
            if selected_node == edge_number:
                # schedule to the cloud
                # earliest_state_time+ transimmission time + precedence job finish time
                precedence_latest_transmission_time = []
                for u, v in self.task_graph.in_edges(
                        selected_task_index):
                    precedence_task_node = self.task_graph.nodes[u][
                        "selected_node"]
                    if precedence_task_node == edge_number:
                        precedence_latest_transmission_time.append(earliest_start_time)
                    else:
                        precedence_latest_transmission_time.append(earliest_start_time-self.task_graph.edges[u, v]["e"] *
                            cloud.data_rate)
                # globally earliest start time is _application.release_time
                latest_change_time = max(precedence_latest_transmission_time) if len(
                    precedence_latest_transmission_time
                ) > 0 else self.release_time
                self.task_graph.nodes[selected_task_index]["latest_change_time"] = latest_change_time
            else:
                precedence_latest_transmission_time = []
                for u, v in self.task_graph.in_edges(
                        selected_task_index):
                    # print(u, v,selected_task_index, self.release_time)
                    precedence_task_node = self.task_graph.nodes[u][
                        "selected_node"]
                    if precedence_task_node == edge_number:
                        # from the cloud
                        precedence_latest_transmission_time.append(earliest_start_time-
                            self.task_graph.edges[u, v]["e"] *
                            cloud.data_rate)
                    elif precedence_task_node != selected_node:
                        # not same edge node
                        precedence_latest_transmission_time.append(earliest_start_time-
                            self.task_graph.edges[u, v]["e"] *
                            edge_list[precedence_task_node].upload_data_rate)
                    else:
                        # same ege node
                        precedence_latest_transmission_time.append(
                            earliest_start_time)

                # globally earliest start time is _application.release_time
                latest_change_time = self.release_time if len(
                    precedence_latest_transmission_time) == 0 else max(
                        precedence_latest_transmission_time)
                self.task_graph.nodes[selected_task_index]["latest_change_time"] = latest_change_time


def get_remain_length(G):
    remain_length_list = [0] * G.number_of_nodes()
    for i in range(G.number_of_nodes()):
        v = G.number_of_nodes() - 1 - i
        for u,v in G.in_edges(v):
           remain_length_list[u]  =  max(remain_length_list[u],remain_length_list[v] + G.nodes[u]["w"])
    return(remain_length_list)

def get_node_with_least_start_time(selected_task_index,_application,edge_list,cloud):
    edge_number = len(edge_list)
    finish_time_list = []

    # estimate the cloud start time
    precedence_task_finish_time = []
    for u,v in _application.task_graph.in_edges(selected_task_index):
        precedence_task_node = _application.task_graph.nodes[u]["selected_node"]
        if precedence_task_node == edge_number:
            precedence_task_finish_time.append(_application.task_graph.nodes[u]["finish_time"])
        else:
            precedence_task_finish_time.append(_application.task_graph.edges[u,v]["e"]*cloud.data_rate+_application.task_graph.nodes[u]["finish_time"])
    # globally earliest start time is _application.release_time
    cloud_earliest_start_time = max(precedence_task_finish_time) if len(precedence_task_finish_time) > 0 else _application.release_time
    cloud_estimated_finish_time  = cloud_earliest_start_time+_application.task_graph.nodes[selected_task_index]["w"] * cloud.process_data_rate

    for _selected_node in range(edge_number):
        precedence_task_finish_time = []
        for u,v in _application.task_graph.in_edges(selected_task_index):
            precedence_task_node = _application.task_graph.nodes[u]["selected_node"]
            if precedence_task_node == edge_number:
                # from the cloud
                precedence_task_finish_time.append(_application.task_graph.edges[u,v]["e"]*cloud.data_rate+_application.task_graph.nodes[u]["finish_time"])
            elif precedence_task_node != _selected_node:
                # not same edge node
                precedence_task_finish_time.append(_application.task_graph.edges[u,v]["e"]*edge_list[precedence_task_node].upload_data_rate+_application.task_graph.nodes[u]["finish_time"])
            else:
                # same ege node
                precedence_task_finish_time.append(_application.task_graph.nodes[u]["finish_time"])
                
        # globally earliest start time is _application.release_time
        earliest_start_time = _application.release_time if len(precedence_task_finish_time) == 0 else max(precedence_task_finish_time)
        
        # run time
        estimated_runtime = _application.task_graph.nodes[selected_task_index]["w"] * edge_list[_selected_node].process_data_rate

        # actual start time and _cpu
        actual_start_time,_cpu = edge_list[_selected_node].find_actual_earliest_start_time(earliest_start_time,estimated_runtime)

        # set start time and node for each task 
        _selected_node_finish_time = actual_start_time + estimated_runtime
        finish_time_list.append(_selected_node_finish_time)
    finish_time_list.append(cloud_estimated_finish_time)

    selected_node = np.argmin(np.array(finish_time_list))

    return selected_node


    
def get_node_by_random(selected_task_index,_application,edge_list,cloud):
    return rd(0,len(edge_list))