'''
Author: 娄炯
Date: 2021-04-16 13:18:37
LastEditors: loujiong
LastEditTime: 2021-07-01 15:06:55
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

class Edge():
    def __init__(self, task_concurrent_capacity, process_data_rate,
                 upload_data_rate, cost_per_mip = 1):
        # set 10000 time slots first
        self.task_concurrent_capacity = task_concurrent_capacity
        self.planed_start_finish = [np.array([[0,10000000000000]]) for i in range(self.task_concurrent_capacity)]
        self.start_finish = [np.array([[0,10000000000000]]) for i in range(self.task_concurrent_capacity)]
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
            is_in_interval = self.planed_start_finish[_cpu][:,1] - np.maximum(start_time,self.planed_start_finish[_cpu][:,0]) >= runtime - 1
            st_time = (1 - is_in_interval)* 1000000000000000 + np.maximum(start_time,self.planed_start_finish[_cpu][:,0])
            interval_key = np.argmin(st_time)
            if min_start_time > st_time[interval_key]:
                min_start_time = st_time[interval_key]
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
            min_end_time = actual_start_time + estimated_runtime - 1
            self.planed_start_finish[_cpu] = np.delete(self.planed_start_finish[_cpu], selected_interval_key, 0)
            if _end == 10000000000000:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([min_end_time + 1, _end])])
            elif _end > min_end_time:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([min_end_time + 1, _end])])
            if _start < actual_start_time:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([_start, actual_start_time - 1])])

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
        for _cpu in range(self.task_concurrent_capacity):
            self.planed_start_finish[_cpu] = self.start_finish[_cpu].copy()
            # print()
            # print("Release time:{0}".format(_release_time))
            # print("Before filtering")
            # print(self.planed_start_finish[_cpu])
            self.planed_start_finish[_cpu] = self.planed_start_finish[_cpu][self.planed_start_finish[_cpu][:,1]>=_release_time,:] 
            # print("After filtering")
            # print(self.planed_start_finish[_cpu])

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
    def __init__(self, cost_per_mip = 4):
        self.process_data_rate = 2
        self.data_rate = 30
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

    def generate_task_graph_by_random_by_level(self, task_num,level_num=4,jump_num=2):
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
        print("node_for_level:{0}".format(node_for_level))
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
        
        print("edge_num:{0},current_edge_num:{1}".format(edge_num,current_edge_num))
        print("edge list:{0}".format([(u - 1,v - 1) for u,v in self.task_graph.edges()]))

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
        
        print("task list:{0}, task_num:{1}, edge_num:{2}".format(self.task_graph.nodes(),task_num,edge_num))
        print("-"*100)

        source_node = 0
        sink_node = task_num + 1

        for i in range(1, task_num + 1):
            if self.task_graph.in_degree(i) == 0:
                self.task_graph.add_edge(source_node, i)
            if self.task_graph.out_degree(i) == 0:
                self.task_graph.add_edge(i, sink_node)

        # add weight
        for i in range(1, task_num + 1):
            self.task_graph.nodes[i]["w"] = rd(10, 15)
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
                self.task_graph.edges[u, v]["e"] = rd(6, 9)
            else:
                self.task_graph.edges[u, v]["e"] = rd(6, 9)

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

def get_remain_length(G,edge_weight=1,node_weight=1):
    remain_length_list = [0] * G.number_of_nodes()
    for i in range(G.number_of_nodes()):
        v = G.number_of_nodes() - 1 - i
        # print("node:{0}".format(v))
        for u, v in G.in_edges(v):
            remain_length_list[u] = max(
                remain_length_list[u], remain_length_list[v] + G.nodes[u]["w"]*node_weight + G.edges[u,v]["e"]*edge_weight)
    return (remain_length_list)

def get_sub_deadline_list(G,remain_length_list,deadline = 10,edge_weight=1,node_weight=1):
    sub_deadline_list = [0] * G.number_of_nodes()
    for i in range(G.number_of_nodes()):
        sub_deadline_list[i] = deadline*(remain_length_list[0]-remain_length_list[i]+G.nodes[i]["w"]*node_weight)/remain_length_list[0]
    return sub_deadline_list
    
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
        precedence_task_finish_time) > 0 else _application.release_time
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

        # globally earliest start time is _application.release_time
        earliest_start_time = _application.release_time if len(
            precedence_task_finish_time) == 0 else max(
                precedence_task_finish_time)

        # run time
        estimated_runtime = _application.task_graph.nodes[selected_task_index][
            "w"] * edge_list[_selected_node].process_data_rate

        # actual start time and _cpu
        # print()
        # print(earliest_start_time, estimated_runtime,_release_time)
        actual_start_time, _cpu, selected_interval_key = edge_list[_selected_node].find_actual_earliest_start_time_by_planed(earliest_start_time, estimated_runtime,_release_time)

        # set start time and node for each task
        _selected_node_finish_time = actual_start_time + estimated_runtime
        finish_time_list.append(_selected_node_finish_time)
        actual_start_time_list.append(actual_start_time)
        # print("edge:{0}, earliest_start_time:{1}".format(_selected_node,earliest_start_time))
    actual_start_time_list.append(cloud_earliest_start_time)    
    finish_time_list.append(cloud_estimated_finish_time)
    sub_deadline = _application.task_graph.nodes[selected_task_index]["sub_deadline"]+_release_time
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
        print("unsatisfied deadline")
        print("selected_task_index:{0},selected_node:{1}".format(selected_task_index,selected_node))
        print("actual_start_time_list:{0}".format(actual_start_time_list))
        print("finish_time_list:{0}".format(finish_time_list))
        
    # if _application.application_index == 8 and selected_task_index == 1:
    #     quit()
        
    return selected_node

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

def set_tmax(_app,edge_list,cloud):
    total_workload = sum([_app.task_graph.nodes()[_n]["w"] for _n in _app.task_graph.nodes()])
    slowest_process_date_rate = max([_e.process_data_rate for _e in edge_list])
    slowest_process_date_rate = max([slowest_process_date_rate,cloud.process_data_rate])
    _app.tmax = total_workload * slowest_process_date_rate


if __name__ == '__main__':
    pass