'''
Author: 娄炯
Date: 2021-04-25 13:22:26
LastEditors: loujiong
LastEditTime: 2021-04-25 13:22:26
Description: 
Email:  413012592@qq.com
'''
'''
Author: 娄炯
Date: 2021-04-16 13:18:37
LastEditors: loujiong
LastEditTime: 2021-04-25 13:21:50
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
from interval import Interval
import pqdict

class Edge():
    def __init__(self, task_concurrent_capacity, process_data_rate,
                 upload_data_rate):
        # set 10000 time slots first
        self.task_concurrent_capacity = task_concurrent_capacity
        self.planed_start_finish = [np.array([[0,10000000000000]]) for i in range(self.task_concurrent_capacity)]
        self.planed_start_finish_filtered = [np.array([[0,10000000000000]]) for i in range(self.task_concurrent_capacity)]
        self.start_finish = [np.array([[0,10000000000000]]) for i in range(self.task_concurrent_capacity)]
        self.process_data_rate = process_data_rate
        self.upload_data_rate = upload_data_rate

    def find_actual_earliest_start_time_by_planed(self, start_time, runtime, _release_time):
        min_start_time = 1000000000000000
        selected_cpu = -1
        selected_interval_key = -1
        for _cpu in range(self.task_concurrent_capacity):
            if runtime == 0:
                return (start_time, -1, 0)
            is_in_interval = self.planed_start_finish[_cpu][:,1] - np.maximum(start_time,self.planed_start_finish[_cpu][:,0]) >= runtime - 1
            st_time = (1 - is_in_interval)* 1000000000000000 + np.maximum(start_time,self.planed_start_finish[_cpu][:,0])
            interval_key = np.argmin(st_time)

            is_in_interval_filtered = self.planed_start_finish_filtered[_cpu][:,1] - np.maximum(start_time,self.planed_start_finish_filtered[_cpu][:,0]) >= runtime - 1
            st_time_filtered = (1 - is_in_interval_filtered)* 1000000000000000 + np.maximum(start_time,self.planed_start_finish_filtered[_cpu][:,0])
            interval_key_filtered = np.argmin(st_time_filtered)
            
            if st_time[interval_key] != st_time_filtered[interval_key_filtered]:
                print(_release_time)
                print(st_time[interval_key])
                print(st_time_filtered[interval_key_filtered])
                print(self.planed_start_finish[_cpu],self.planed_start_finish_filtered[_cpu])
                print(start_time, runtime)
                quit()
            if min_start_time > st_time[interval_key]:
                min_start_time = st_time[interval_key]
                selected_interval_key = interval_key
                selected_cpu = _cpu
        return (min_start_time, selected_cpu, selected_interval_key)

    def set_cpu_state_by_planed(self, _cpu, actual_start_time,
                                estimated_runtime, _application_index,
                                selected_index, selected_interval_key):
        if estimated_runtime != 0:
            _start = self.planed_start_finish[_cpu][selected_interval_key][0]
            _end = self.planed_start_finish[_cpu][selected_interval_key][1]
            min_end_time = actual_start_time + estimated_runtime - 1
            self.planed_start_finish[_cpu] = np.delete(self.planed_start_finish[_cpu], selected_interval_key, 0)
            if _end == 10000000000000:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([min_end_time + 1, _end])])
            elif _end > min_end_time:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([min_end_time + 1, _end])])
            if _start < actual_start_time:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([_start, actual_start_time - 1])])

        if estimated_runtime != 0:
            selected_interval_key_filtered = np.where(( self.planed_start_finish_filtered[_cpu]== np.array([_start,_end])).all(1))[0][0]
            _start = self.planed_start_finish_filtered[_cpu][selected_interval_key_filtered][0]
            _end = self.planed_start_finish_filtered[_cpu][selected_interval_key_filtered][1]
            min_end_time = actual_start_time + estimated_runtime - 1
            self.planed_start_finish_filtered[_cpu] = np.delete(self.planed_start_finish_filtered[_cpu], selected_interval_key_filtered, 0)
            if _end == 10000000000000:
                self.planed_start_finish_filtered[_cpu] = np.vstack([self.planed_start_finish_filtered[_cpu], np.array([min_end_time + 1, _end])])
            elif _end > min_end_time:
                self.planed_start_finish_filtered[_cpu] = np.vstack([self.planed_start_finish_filtered[_cpu], np.array([min_end_time + 1, _end])])
            if _start < actual_start_time:
                self.planed_start_finish_filtered[_cpu] = np.vstack([self.planed_start_finish_filtered[_cpu], np.array([_start, actual_start_time - 1])])

    def update_plan_to_actural(self, _release_time, finish_task_set,
                               application_list):
        finish_time_list = [0 for i in range(self.task_concurrent_capacity)]
        task_num = len(finish_task_set)

        # sort finished task_set by the start time
        ids = finish_task_set[:,2].argsort()
        
        self.start_finish = [[] for cpu in range(self.task_concurrent_capacity)]
        for k in ids:
            _ap_index, _ta_index, start_time, finish_time = [int(i) for i in finish_task_set[k]]
            cpu = application_list[_ap_index].task_graph.nodes()[_ta_index]["cpu"]
            if start_time == finish_time:
                continue
            if finish_time_list[cpu] < start_time:
                self.start_finish[cpu].append([finish_time_list[cpu],start_time - 1])
                finish_time_list[cpu] = finish_time
            elif finish_time_list[cpu] == start_time:
                finish_time_list[cpu] = finish_time
        for cpu, f in enumerate(finish_time_list):
            self.start_finish[cpu].append([f, 10000000000000])
            self.start_finish[cpu] = np.array(self.start_finish[cpu])

    def generate_plan(self, _release_time):
        self.planed_start_finish = [0 for i in range(self.task_concurrent_capacity)]
        self.planed_start_finish_filtered = [0 for i in range(self.task_concurrent_capacity)]
        for _cpu in range(self.task_concurrent_capacity):
            self.planed_start_finish[_cpu] = self.start_finish[_cpu].copy()
            # print()
            # print("Release time:{0}".format(_release_time))
            # print("Before filtering")
            # print(self.planed_start_finish[_cpu])
            self.planed_start_finish_filtered[_cpu] = self.planed_start_finish[_cpu][self.planed_start_finish[_cpu][:,1]>=_release_time,:] 
            # print("After filtering")
            # print(self.planed_start_finish[_cpu])
# the speed of the cloud is fastest
class Cloud():
    def __init__(self):
        self.process_data_rate = 1
        self.data_rate = 20

class Application():
    def __init__(
        self,
        release_time,
        release_node,
        task_num=5,
    ):
        self.release_time = release_time
        self.finish_time = -1
        self.release_node = release_node
        self.generate_application_by_random(task_num)
        self.dynamic_longest_remain_length = 0

    def set_start_time(self, selected_task, selected_node, start_time, _cpu):
        self.task_graph.nodes[selected_task]["selected_node"] = selected_node
        self.task_graph.nodes[selected_task]["cpu"] = _cpu
        self.task_graph.nodes[selected_task]["start_time"] = start_time

    def generate_application_by_random(self, task_num):
        self.generate_task_graph_by_random(task_num)

    def generate_task_graph_by_random(self, task_num):
        node = range(1, task_num + 1)
        edge_num = rd(1, min(task_num * task_num, task_num * 2))
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
            self.task_graph.nodes[i]["w"] = rd(1, 10)
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
            self.task_graph.edges[u, v]["e"] = rd(1, 10)

    def set_latest_change_time(self, selected_node, selected_task_index,
                               edge_list, cloud, earliest_start_time):
        if selected_task_index == 0:
            self.task_graph.nodes[selected_task_index][
                "latest_change_time"] = self.release_time
        else:
            edge_number = len(edge_list)
            if selected_node == edge_number:
                # schedule to the cloud
                # earliest_state_time+ transimmission time + precedence job finish time
                precedence_latest_transmission_time = []
                for u, v in self.task_graph.in_edges(selected_task_index):
                    precedence_task_node = self.task_graph.nodes[u][
                        "selected_node"]
                    if precedence_task_node == edge_number:
                        precedence_latest_transmission_time.append(
                            earliest_start_time)
                    else:
                        precedence_latest_transmission_time.append(
                            earliest_start_time -
                            self.task_graph.edges[u, v]["e"] * cloud.data_rate)
                # globally earliest start time is _application.release_time
                latest_change_time = min(
                    precedence_latest_transmission_time
                ) if len(precedence_latest_transmission_time
                         ) > 0 else self.release_time
                self.task_graph.nodes[selected_task_index][
                    "latest_change_time"] = latest_change_time
            else:
                precedence_latest_transmission_time = []
                for u, v in self.task_graph.in_edges(selected_task_index):
                    precedence_task_node = self.task_graph.nodes[u][
                        "selected_node"]
                    if precedence_task_node == edge_number:
                        # from the cloud
                        precedence_latest_transmission_time.append(
                            earliest_start_time -
                            self.task_graph.edges[u, v]["e"] * cloud.data_rate)
                    elif precedence_task_node != selected_node:
                        # not same edge node
                        precedence_latest_transmission_time.append(
                            earliest_start_time -
                            self.task_graph.edges[u, v]["e"] *
                            edge_list[precedence_task_node].upload_data_rate)
                    else:
                        # same ege node
                        precedence_latest_transmission_time.append(
                            earliest_start_time)

                # globally earliest start time is _application.release_time
                latest_change_time = self.release_time if len(
                    precedence_latest_transmission_time) == 0 else min(
                        precedence_latest_transmission_time)
                self.task_graph.nodes[selected_task_index][
                    "latest_change_time"] = latest_change_time

def get_remain_length(G):
    remain_length_list = [0] * G.number_of_nodes()
    for i in range(G.number_of_nodes()):
        v = G.number_of_nodes() - 1 - i
        for u, v in G.in_edges(v):
            remain_length_list[u] = max(
                remain_length_list[u], remain_length_list[v] + G.nodes[u]["w"])
    return (remain_length_list)

def get_node_with_least_start_time(selected_task_index, _application,
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

def check(application_list):
    d = dict()
    for application_index,application in enumerate(application_list):
        for task_index in application.task_graph.nodes():
            if application.task_graph.nodes()[task_index]["selected_node"] != -1 and application.task_graph.nodes()[task_index]["cpu"] != -1:
                start_time = application.task_graph.nodes()[task_index]["start_time"]
                finish_time = application.task_graph.nodes()[task_index]["finish_time"]
                node_cpu = "{0}-{1}".format(application.task_graph.nodes()[task_index]["selected_node"],application.task_graph.nodes()[task_index]["cpu"])
                if node_cpu not in d:
                    d[node_cpu] = pqdict.pqdict()
                d[node_cpu][(application_index,task_index)] = (start_time,finish_time)
    for node_cpu in d:
        keys = pqdict.nsmallest(len(d[node_cpu]),d[node_cpu])
        st = -1
        fi = -1
        for application_index,task_index in keys:
            st_time,fi_time = d[node_cpu][(application_index,task_index)]
            if st_time > fi_time:
                print("same task fi_time st time error")
                print(st_time,fi_time)
                quit()
            if st_time < fi:
                print("cross task fi_time st time error")
                print(fi,st_time)
                print(node_cpu)
                quit()
            st,fi = st_time,fi_time

def get_node_by_random(selected_task_index, _application, edge_list, cloud, _release_time):
    return rd(0, len(edge_list))

if __name__ == '__main__':
    pass