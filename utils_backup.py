'''
Author: 娄炯
Date: 2021-04-16 13:18:37
LastEditors: loujiong
LastEditTime: 2021-09-09 11:40:50
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
from collections import deque
import read_in_task_graph

min_runtime = 0.00001

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
        self.min_runtime = 0.00001

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


    def delete_task(self,start_time,finish_time,cpu):
        #find the interval before
        is_interval_before = (start_time - self.planed_start_finish[cpu][:,1] < self.min_runtime) & (start_time - self.planed_start_finish[cpu][:,1] > 0- self.min_runtime)
        interval_before_key = np.argmax(is_interval_before)

        if is_interval_before[interval_before_key]:
            start = self.planed_start_finish[cpu][interval_before_key][0]
            self.planed_start_finish[cpu] = np.delete(self.planed_start_finish[cpu], interval_before_key, 0)
        else:
            start = start_time

        #find the interval after
        is_interval_after = (self.planed_start_finish[cpu][:,0]-finish_time < self.min_runtime) & (self.planed_start_finish[cpu][:,0]-finish_time > 0 - self.min_runtime)
        interval_after_key = np.argmax(is_interval_after)

        if is_interval_after[interval_after_key]:
            finish = self.planed_start_finish[cpu][interval_after_key][1]
            self.planed_start_finish[cpu] = np.delete(self.planed_start_finish[cpu], interval_after_key, 0)
        else:
            finish = finish_time
    
        self.planed_start_finish[cpu] = np.vstack([self.planed_start_finish[cpu], np.array([start, finish])])
        

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

    def update_plan_to_actural(self, _release_time, new_finish_task_set,
                               application_list):
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
        # print("start initial application {0}".format(application_index))
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
                self.task_graph.edges[u, v]["e"] = 2 + 4*random.random()#rd(2, 7)
            else:
                self.task_graph.edges[u, v]["e"] = 2 + 4*random.random()#rd(2, 7)

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
                        self.task_graph.edges[u, v]["e"] * bandwidth)
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

def get_start_sub_deadline_list(G,remain_length_list,deadline = 10):
    sub_deadline_list = [0] * G.number_of_nodes()
    for i in range(G.number_of_nodes()):
        sub_deadline_list[i] = deadline*(remain_length_list[0]-remain_length_list[i])/remain_length_list[0]
    return sub_deadline_list

def get_start_sub_deadline_list_pcp(G,remain_length_list,deadline = 10,edge_weight=1,node_weight=1):
    node_deque = deque()
    sub_start_deadline_list = [-1]*G.number_of_nodes()
    sub_start_deadline_list[0] = 0
    sub_start_deadline_list[-1] = deadline
    eft_list = [0]*G.number_of_nodes()
    for _n in range(G.number_of_nodes()):
        _eft = 0+node_weight*G.nodes[_n]['w']
        for u,_ in G.in_edges(_n):
            _eft = min(_eft,eft_list[u]+node_weight*G.nodes[_n]['w']+edge_weight*G.edges[u,_n]['e'])
        eft_list[_n] = _eft
    node_deque.append(G.number_of_nodes()-1)
    while(len(node_deque)>0):
        _n = node_deque.pop()
        # print(_n)
        #find critical path
        critical_path = deque()
        critical_path.append(_n)
        _current_n = _n
        while(True):
            has_unassigned_parent = False
            critical_parent = -1
            _lst = -1
            for u,_ in G.in_edges(_current_n):
                if sub_start_deadline_list[u] < 0:
                    has_unassigned_parent = True
                    if eft_list[u]+edge_weight*G.edges[u,_current_n]['e'] > _lst:
                        _lst = eft_list[u]+edge_weight*G.edges[u,_current_n]['e']
                        critical_parent = u
            if has_unassigned_parent:
                critical_path.append(critical_parent)
                _current_n = critical_parent
            if not has_unassigned_parent:
                break
        
        if len(critical_path) < 2:
            continue
        
        #find the first node
        _current_n = critical_path[-1]
        first_node = -1
        _lst = -1
        for u,_ in G.in_edges(_current_n):
            if sub_start_deadline_list[u] + node_weight*G.nodes[u]['w']+ edge_weight*G.edges[u,_current_n]['e'] > _lst:
                _lst = sub_start_deadline_list[u] + node_weight*G.nodes[u]['w']+ edge_weight*G.edges[u,_current_n]['e']
                first_node = u
        critical_path.append(first_node)

        # print('critical_path',critical_path)
        #assign sub deadline
        total_time = 0
        total_time_list = []
        sub_start_deadline_s = sub_start_deadline_list[critical_path[-1]]
        sub_start_deadline_f = sub_start_deadline_list[critical_path[0]]
        for i in range(0,len(critical_path)-1):
            _n = critical_path[i]
            total_time += node_weight*G.nodes[critical_path[i+1]]['w']
            total_time += edge_weight*G.edges[critical_path[i+1],_n]['e']
            total_time_list.append(total_time)
        for i in range(len(critical_path)-2):
            _n = critical_path[i+1]
            sub_start_deadline_list[_n] = sub_start_deadline_f-(sub_start_deadline_f-sub_start_deadline_s)*total_time_list[i]/total_time
        
        #add to node_deque
        for i in range(len(critical_path)-2,-1,-1):
            node_deque.append(critical_path[i])

        # node_deque.append(critical_path[0])
        # for i in range(1,len(critical_path)-1):
        #     node_deque.appendleft(critical_path[i])

        # print('node_deque',node_deque)
    # print(sub_deadline_list)
    # print(deadline)
    # quit()
    return(sub_start_deadline_list)

def get_node_with_least_cost_constrained_by_subdeadline(selected_task_index, _application,
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
                    bandwidth +max(_application.task_graph.nodes[u]["finish_time"],_release_time))
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
                    bandwidth +max(_application.task_graph.nodes[u]["finish_time"],_release_time))

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

    return False,selected_node

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
                    _application.task_graph.edges[u, v]["e"] *
                    bandwidth + max(_application.task_graph.nodes[u]["finish_time"],_release_time))


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
            if application.task_graph.nodes[_n]["selected_node"] == len(edge_list):
                runtime = application.task_graph.nodes[_n]["w"]*cloud.process_data_rate
            else:
                runtime = application.task_graph.nodes[_n]["w"]*edge_list[application.task_graph.nodes[_n]["selected_node"]].process_data_rate
            if application.task_graph.nodes[_n]["start_time"] + runtime > application.task_graph.nodes[_n]["finish_time"]:
                print("runtime length error")
                exit(1)
            if application.task_graph.nodes[_n]["start_time"] < application.release_time:
                print("release time error")
                exit(1)
            for u,_ in application.task_graph.in_edges(_n):
                bandwidth = get_bandwidth(application.task_graph.nodes[u]["selected_node"],application.task_graph.nodes[_n]["selected_node"], edge_list, cloud)
                if application.task_graph.nodes[u]["finish_time"]+bandwidth*application.task_graph.edges[u,_n]["e"]-application.task_graph.nodes[_n]["start_time"]>0.00000001:
                    print("precedence error")
                    print(application.task_graph.nodes[u]["finish_time"])
                    print(bandwidth*application.task_graph.edges[u,_n]["e"])
                    print(application.task_graph.nodes[_n]["start_time"])
                    exit(1)

    if is_multiple:
        cloud_number= len(edge_list)
        d = dict()
        for application_index,application in enumerate(application_list):
            if application.is_accept:
                for task_index in application.task_graph.nodes():
                    if application.task_graph.nodes()[task_index]["selected_node"] != -1 and application.task_graph.nodes()[task_index]["cpu"] != -1 and application.task_graph.nodes()[task_index]["selected_node"] != cloud_number:
                        start_time = application.task_graph.nodes()[task_index]["start_time"]
                        finish_time = application.task_graph.nodes()[task_index]["finish_time"]
                        node_cpu = "{0}-{1}".format(application.task_graph.nodes()[task_index]["selected_node"],application.task_graph.nodes()[task_index]["cpu"])
                        if node_cpu not in d:
                            d[node_cpu] = pqdict.pqdict()
                        d[node_cpu][(application_index,task_index)] = (start_time,finish_time,application_index,task_index)
                    if application.task_graph.nodes()[task_index]["selected_node"] == -1:
                        print("node -1 error")
                        quit()
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
                    if st_time+ 0.00000001< fi:
                        print("cross task fi_time st time error")
                        print(st, fi,st_time,fi_time)
                        print(node_cpu)
                        print("last application-task: {0}-{1}".format(_a_i,_t_i))
                        print("current application-task: {0}-{1}".format(application_index,task_index))
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


def get_node_with_least_cost_constrained_by_start_subdeadline_without_cloud(selected_task_index, _application:Application,
                                   edge_list, cloud, _release_time):
    edge_number = len(edge_list)
    finish_time_list = []
    
    is_in_deadline  = []
    overdue_start_deadline = []
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

    node_shuffle_index = list(range(edge_number))
    # random.shuffle(node_shuffle_index)
    cost_per_mip_list = [i.cost_per_mip for i in edge_list]
    selected_node = -1
    min_cost = 10000
    ft = 100000000000
    for i in node_shuffle_index:
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
    overdue_start_deadline = [overdue_start_deadline[i] for i in node_shuffle_index]

    # print("application-{0},task-{1}".format(_application.application_index,selected_task_index))
    # print("is_in_deadline:{0}".format(is_in_deadline))
    # print("minimal sub lateness:{0}".format(min(overdue_start_deadline)))
    
    if selected_node < 0:
        selected_node = node_shuffle_index[np.argmin(np.array(overdue_start_deadline))]
     
    return is_in_deadline,selected_node
    
def get_node_by_random(selected_task_index, _application, edge_list, cloud, _release_time):
    return rd(0, len(edge_list))

def set_tmax(_app,edge_list,cloud):
    total_workload = sum([_app.task_graph.nodes()[_n]["w"] for _n in _app.task_graph.nodes()])
    slowest_process_date_rate = max([_e.process_data_rate for _e in edge_list])
    slowest_process_date_rate = max([slowest_process_date_rate,cloud.process_data_rate])
    _app.tmax = total_workload * slowest_process_date_rate


def schedule_with_cloud(application,edge_list,cloud):
    sink = application.task_graph.number_of_nodes()-1

    check_start_offloading_task_list = deque()
    is_in_edge = [True for i in application.task_graph.nodes()]
    has_been_checked_list = [False for i in application.task_graph.nodes()]
    
    # calculate the offloading time for each task
    offloading_time_list = [0 for i in application.task_graph.nodes()]
    for _n in range(sink,-1,-1):
        if _n == sink or _n == 0:
            continue
        for _,v in application.task_graph.out_edges(_n):
            if v == sink:
                offloading_time_list[_n] = max(offloading_time_list[_n],application.task_graph.edges[_n,v]["e"]*cloud.data_rate+application.task_graph.nodes[_n]["w"]*cloud.process_data_rate)
            else:
                offloading_time_list[_n] = max(offloading_time_list[_n],offloading_time_list[v]+application.task_graph.nodes[_n]["w"]*cloud.process_data_rate)

    delay_node_list =[]
    # find the delay reason
    for u,_ in application.task_graph.in_edges(sink):
        bandwidth = 0 if application.task_graph.nodes[u]["selected_node"] == application.task_graph.nodes[sink]["selected_node"] else edge_list[application.task_graph.nodes[u]["selected_node"]].upload_data_rate
        if application.task_graph.nodes[u]["finish_time"]+application.task_graph.edges[u,sink]["e"]*bandwidth -application.release_time > application.deadline:
            check_start_offloading_task_list.append(u)
            is_in_edge[u] = False
            delay_node_list.append(u)
    
    
    while len(check_start_offloading_task_list) > 0:
        check_task = check_start_offloading_task_list.popleft()
        
        #check task is able to be the offloading task
        for u,_ in application.task_graph.in_edges(check_task):
            if is_in_edge[u]:
                if application.task_graph.nodes[u]["finish_time"]+application.task_graph.edges[u,check_task]["e"]*cloud.data_rate+offloading_time_list[check_task] - application.release_time > application.deadline:
                    check_start_offloading_task_list.append(u)
                    is_in_edge[u] = False
                    if u == 0:
                        # print("schedule failed")
                        return False

        #add Descendant tasks
        Descendant_list = deque()
        Descendant_list.append(check_task)
        while len(Descendant_list) > 0:
            _descendant_task = Descendant_list.popleft()
            for _,v in application.task_graph.out_edges(_descendant_task):
                if v != sink:
                    if not has_been_checked_list[v] and v not in check_start_offloading_task_list:
                        check_start_offloading_task_list.append(v)
                        is_in_edge[v] = False
                        Descendant_list.append(v)
        
        has_been_checked_list[check_task] = True
    
    cloud_number = len(edge_list)
    # set start_time finish_time and set_cpu_state_by_planed
    for _node in range(len(edge_list)):
        edge_list[_node].generate_plan(application.release_time)

    for _n in application.task_graph.nodes():
        if is_in_edge[_n] and _n != 0 and _n != sink:
            selected_node = application.task_graph.nodes[_n]["selected_node"]
            _cpu = application.task_graph.nodes[_n]["cpu"]
            actual_start_time = application.task_graph.nodes[_n]["start_time"]
            estimated_runtime = application.task_graph.nodes[_n]["finish_time"] - application.task_graph.nodes[_n]["start_time"]
            _ap_index = application.application_index
            selected_task_index = _n

            is_in_interval = (edge_list[selected_node].planed_start_finish[_cpu][:,0] <= actual_start_time) & (edge_list[selected_node].planed_start_finish[_cpu][:,1] >= application.task_graph.nodes[_n]["finish_time"])
            st_time = 1-is_in_interval
            selected_interval_key = np.argmin(st_time)
            
            edge_list[selected_node].set_cpu_state_by_planed(_cpu, actual_start_time,
                                                            estimated_runtime,
                                                            _ap_index,
                                                            selected_task_index,
                                                            selected_interval_key)

    for _n in range(application.task_graph.number_of_nodes()):
        if not is_in_edge[_n]:
            application.task_graph.nodes[_n]["selected_node"] = cloud_number
            application.task_graph.nodes[_n]["cpu"] = 0
            # _start_time = -1
            # for u,_ in application.task_graph.in_edges(_n):
            #     if is_in_edge[u]:
            #         _start_time = max(_start_time,application.task_graph.nodes[u]["finish_time"]+cloud.data_rate*application.task_graph.edges[u,_n]["e"])
            #     else:
            #         _start_time = max(_start_time,application.task_graph.nodes[u]["finish_time"])
            # application.task_graph.nodes[_n]["start_time"] = _start_time
            # application.task_graph.nodes[_n]["finish_time"] = _start_time + application.task_graph.nodes[_n]["w"] * cloud.process_data_rate
            application.task_graph.nodes[_n]["start_time"] = application.release_time + application.deadline - offloading_time_list[_n]
            application.task_graph.nodes[_n]["finish_time"] = application.task_graph.nodes[_n]["start_time"] + application.task_graph.nodes[_n]["w"] * cloud.process_data_rate

    _sink_start_time = 0
    for u,_ in application.task_graph.in_edges(sink):
        bandwidth = get_bandwidth(application.task_graph.nodes[u]["selected_node"],application.task_graph.nodes[sink]["selected_node"],edge_list,cloud)
        _sink_start_time = max(_sink_start_time,bandwidth*application.task_graph.edges[u,sink]["e"]+application.task_graph.nodes[u]["finish_time"])
    
    for _n in range(application.task_graph.number_of_nodes()):
        if not is_in_edge[_n]:
            _is_right = True
            for u in nx.descendants(application.task_graph,_n):
                if is_in_edge[u] and u != sink:
                    print("{0} descendants {1} error".format(_n,u))
                    quit()
                    
    application.task_graph.nodes[sink]["start_time"] = _sink_start_time
    application.task_graph.nodes[sink]["finish_time"] = _sink_start_time
    
    application.is_accept = application.task_graph.nodes[sink]["finish_time"] - application.release_time <= application.deadline
           
    return True 

def adjust_task_start_finish_time(application:Application,edge_list,cloud:Cloud):
    global min_runtime
    node_list = [(_n,application.task_graph.nodes[_n]["start_time"]) for _n in application.task_graph.nodes()]
    node_list.sort(key = lambda x:0-x[1])
    node_list = [i[0] for i in node_list]
    # print(node_list)
    # print(application.is_accept)
    for _n in node_list:
        selected_node = application.task_graph.nodes[_n]["selected_node"]
        if _n == 0 or selected_node == len(edge_list):
            continue
        earliest_start_time = application.task_graph.nodes[_n]["start_time"]
        # if application.application_index == 191 and (_n == 6 or _n==16):
        #     print(application.task_graph.nodes[_n]["start_time"],application.task_graph.nodes[_n]["finish_time"])
        latest_finish_time = math.inf
        for _,v in application.task_graph.out_edges(_n):
            bandwidth = get_bandwidth(selected_node, application.task_graph.nodes[v]["selected_node"], edge_list, cloud)
            _finish_time = application.task_graph.nodes[v]["start_time"]-bandwidth*application.task_graph.edges[_n,v]["e"]
            latest_finish_time = min(latest_finish_time,_finish_time)
        latest_start_time = latest_finish_time - application.task_graph.nodes[_n]["w"]*edge_list[selected_node].process_data_rate
        start_sub_deadline = application.task_graph.nodes[_n]["start_sub_deadline"] + application.release_time
        #find the follow up task
        # print("latest_finish_time",latest_finish_time)
        if _n == application.task_graph.number_of_nodes()-1:
            actual_start_time = min(start_sub_deadline,latest_start_time)
        else:
            cpu = application.task_graph.nodes[_n]["cpu"] 
            is_interval_after = (edge_list[selected_node].planed_start_finish[cpu][:,0]-application.task_graph.nodes[_n]["finish_time"] < min_runtime) & (edge_list[selected_node].planed_start_finish[cpu][:,0]-application.task_graph.nodes[_n]["finish_time"] > 0 - min_runtime)
            interval_after_key = np.argmax(is_interval_after)
            if is_interval_after[interval_after_key]:
                follow_start_time_constraint = edge_list[selected_node].planed_start_finish[cpu][interval_after_key][1]-application.task_graph.nodes[_n]["w"]*edge_list[selected_node].process_data_rate
            else:
                follow_start_time_constraint = application.task_graph.nodes[_n]["start_time"]
            # print("start_sub_deadline,latest_start_time,follow_start_time_constraint",start_sub_deadline,latest_start_time,follow_start_time_constraint)
            actual_start_time = min(start_sub_deadline,latest_start_time,follow_start_time_constraint)
        # print(actual_start_time)
        actual_start_time = max(actual_start_time,earliest_start_time)
        # if application.application_index == 191 and (_n == 6 or _n==16):
        #     print(edge_list[selected_node].planed_start_finish[cpu])
        if actual_start_time > earliest_start_time:
            if _n == application.task_graph.number_of_nodes()-1:
                application.task_graph.nodes[_n]["start_time"] = actual_start_time
                application.task_graph.nodes[_n]["finish_time"] = actual_start_time + application.task_graph.nodes[_n]["w"]*edge_list[selected_node].process_data_rate
            else:
                edge_list[selected_node].delete_task(application.task_graph.nodes[_n]["start_time"],application.task_graph.nodes[_n]["finish_time"],application.task_graph.nodes[_n]["cpu"])
                # if application.application_index == 191 and (_n == 6 or _n==16):
                #     print(edge_list[selected_node].planed_start_finish[cpu][:,1] - np.maximum(actual_start_time,edge_list[selected_node].planed_start_finish[cpu][:,0]))
                #     print(application.task_graph.nodes[_n]["w"]*edge_list[selected_node].process_data_rate)
                
                is_in_interval= (edge_list[selected_node].planed_start_finish[cpu][:,0] <= actual_start_time) & (edge_list[selected_node].planed_start_finish[cpu][:,1] >= actual_start_time)
                selected_interval_key = np.argmax(is_in_interval)
                
                # if application.application_index == 191 and (_n == 6 or _n==16):
                #     print(actual_start_time)
                #     print(actual_start_time+application.task_graph.nodes[_n]["w"]*edge_list[selected_node].process_data_rate)
                #     print(edge_list[selected_node].planed_start_finish[cpu])
                #     print(edge_list[selected_node].planed_start_finish[cpu][selected_interval_key])
                edge_list[selected_node].set_cpu_state_by_planed(cpu, actual_start_time,
                                                            application.task_graph.nodes[_n]["w"]*edge_list[selected_node].process_data_rate,
                                                            application.application_index,
                                                            _n,
                                                            selected_interval_key)

                #last change
                application.task_graph.nodes[_n]["start_time"] = actual_start_time
                application.task_graph.nodes[_n]["finish_time"] = actual_start_time + application.task_graph.nodes[_n]["w"]*edge_list[selected_node].process_data_rate
        # if application.application_index == 191 and (_n == 6 or _n==16):
        #     print(edge_list[selected_node].planed_start_finish[cpu])
        # if application.application_index == 191 and (_n == 6 or _n==16):
        #     print(earliest_start_time,application.task_graph.nodes[_n]["start_time"],application.task_graph.nodes[_n]["start_sub_deadline"]+ application.release_time)
        #     # quit()
        #     print()

    return

def adjust(application:Application,edge_list,cloud):
    for _n in range(application.task_graph.number_of_nodes()):
        # only for edge task
        original_node = application.task_graph.nodes[_n]["selected_node"]
        if _n == 0 or _n == application.task_graph.number_of_nodes()-1:
            continue
        # if original_node == len(edge_list):
        #     continue

        if original_node < len(edge_list):
            min_cost_per_mip = edge_list[original_node].cost_per_mip
        else:
            min_cost_per_mip = cloud.cost_per_mip

        changed_node = -1
        changed_cpu = -1
        changed_selected_interval_key = -1
        changed_start_time = -1
        changed_finish_time = -1

        # find cheapest Substituted node
        for edge_index in range(len(edge_list)):
            if edge_index != original_node:
                _latest_start_time = 0
                for u,_ in application.task_graph.in_edges(_n):
                    bandwidth = get_bandwidth(application.task_graph.nodes[u]["selected_node"], edge_index, edge_list, cloud)
                    _latest_start_time = max(_latest_start_time,application.task_graph.nodes[u]["finish_time"]+bandwidth*application.task_graph.edges[u,_n]["e"])
                runtime = application.task_graph.nodes[_n]["w"]*edge_list[edge_index].process_data_rate
                _release_time = application.release_time
                actual_start_time, _cpu, selected_interval_key = edge_list[edge_index].find_actual_earliest_start_time_by_planed(_latest_start_time, runtime, _release_time)
                actual_finish_time = actual_start_time + runtime
                _can_be_assigned = True
                for _,v in application.task_graph.out_edges(_n):
                    bandwidth = get_bandwidth(edge_index, application.task_graph.nodes[v]["selected_node"], edge_list, cloud)
                    if actual_finish_time + bandwidth*application.task_graph.edges[_n,v]["e"] > application.task_graph.nodes[v]["start_time"]:
                        _can_be_assigned = False
                if _can_be_assigned:
                    if min_cost_per_mip > edge_list[edge_index].cost_per_mip:
                        min_cost_per_mip = edge_list[edge_index].cost_per_mip
                        changed_node = edge_index
                        changed_cpu = _cpu
                        changed_selected_interval_key = selected_interval_key
                        changed_start_time = actual_start_time
                        changed_finish_time = actual_finish_time
        
        if changed_node != -1:
            #first delete interval
            if original_node < len(edge_list):
                edge_list[original_node].delete_task(application.task_graph.nodes[_n]["start_time"],application.task_graph.nodes[_n]["finish_time"],application.task_graph.nodes[_n]["cpu"])

            # then add interval
            edge_list[changed_node].set_cpu_state_by_planed(changed_cpu, changed_start_time,
                                                       changed_finish_time - changed_start_time,
                                                       application.application_index,
                                                       _n,
                                                       changed_selected_interval_key)

            #last change
            application.task_graph.nodes[_n]["start_time"] = changed_start_time
            application.task_graph.nodes[_n]["finish_time"] = changed_finish_time
            application.task_graph.nodes[_n]["cpu"] = changed_cpu
            application.task_graph.nodes[_n]["selected_node"] = changed_node

    return

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