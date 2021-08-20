'''
Author: 娄炯
Date: 2021-08-04 11:11:52
LastEditors: loujiong
LastEditTime: 2021-08-16 12:38:16
Description: 
Email:  413012592@qq.com
'''

import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import read_in_task_graph

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
            
            # print("task list:{0}, task_num:{1}, edge_num:{2}".format(self.task_graph.nodes(),task_num,edge_num))
            # print("-"*100)

        source_node = 0
        sink_node = task_num + 1

        for i in range(1, task_num + 1):
            if self.task_graph.in_degree(i) == 0:
                self.task_graph.add_edge(source_node, i)
            if self.task_graph.out_degree(i) == 0:
                self.task_graph.add_edge(i, sink_node)

        # add weight
        for i in range(1, task_num + 1):
            self.task_graph.nodes[i]["w"] = 7+14*random.random() #rd(3, 20)  #rd(10, 15)
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
                self.task_graph.edges[u, v]["e"] = 1 + 6*random.random() #rd(2, 7)
            else:
                self.task_graph.edges[u, v]["e"] = 1 + 6*random.random() #rd(2, 7)

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