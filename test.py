'''
Author: 娄炯
Date: 2021-04-16 13:09:36
LastEditors: loujiong
LastEditTime: 2021-04-19 22:06:40
Description: a simple test of simulation
Email:  413012592@qq.com
'''
import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import utils_backup as utils
import draw
import random
import time
import math
import numpy as np
import pqdict


def test_scheduling(is_draw=False,
                      application_num=9,
                      application_average_interval=10,
                      edge_number=10,
                      scheduler = utils.get_node_with_least_start_time):
    # random setting
    st = time.time()
    random.seed(0)

    # environemnt

    # generate release time and application
    release_time_list = []
    release_time = 0
    for i in range(application_num):
        release_time_list.append(math.ceil(release_time))
        release_time += random.expovariate(1 / application_average_interval)
    application_list = [
        utils.Application(release_time=release_time_list[i],
                          task_num=rd(5, 15),
                          release_node=rd(0, edge_number - 1))
        for i in range(application_num)
    ]
    # draw.draw(application_list[0].task_graph)

    # scheduling baseline-random
    total_completion_time = 0

    # initiate edges and cloud
    cloud = utils.Cloud()
    edge_list = [
        utils.Edge(task_concurrent_capacity=rd(1, 3),
                   process_data_rate=rd(3, 10),
                   upload_data_rate=rd(5, 10)) for i in range(edge_number)
    ]

    # loops for scheduling applications
    for _application_index, _application in enumerate(application_list):
        # print(_application_index)
        # task is scheduled label
        scheduled = [1] * _application.task_graph.number_of_nodes()

        # calculate remain length for each task
        remian_length = utils.get_remain_length(_application.task_graph)

        while (sum(scheduled) > 0.5):
            # select the task with the longest remain path in terms of task workload
            selected_task_index = -1
            selected_remaining_length = -1
            for _index in _application.task_graph.nodes():
                if scheduled[_index] > 0.5:
                    if selected_remaining_length < remian_length[_index]:
                        selected_remaining_length = remian_length[_index]
                        selected_task_index = _index
            
            # randomly select edge node or cloud
            if selected_task_index == 0 or selected_task_index == _application.task_graph.number_of_nodes(
            ) - 1:
                # the source and sink
                selected_node = _application.release_node
            else:
                # greedy selecting node
                selected_node = scheduler(
                    selected_task_index, _application, edge_list, cloud)
            
            if selected_node == edge_number:
                # schedule to the cloud
                # earliest_state_time+ transimmission time + precedence job finish time
                precedence_task_finish_time = []
                for u, v in _application.task_graph.in_edges(
                        selected_task_index):
                    precedence_task_node = _application.task_graph.nodes[u][
                        "selected_node"]
                    if precedence_task_node == edge_number:
                        precedence_task_finish_time.append(
                            _application.task_graph.nodes[u]["finish_time"])
                    else:
                        precedence_task_finish_time.append(
                            _application.task_graph.edges[u, v]["e"] *
                            cloud.data_rate +
                            _application.task_graph.nodes[u]["finish_time"])
                # globally earliest start time is _application.release_time
                earliest_start_time = max(precedence_task_finish_time) if len(
                    precedence_task_finish_time
                ) > 0 else _application.release_time

                # set_latest_change_time
                _application.set_latest_change_time(selected_node,selected_task_index,edge_list,cloud,earliest_start_time)

                # set start time and node for each task
                _application.set_start_time(selected_task_index, selected_node,
                                            earliest_start_time, -1)
                _application.task_graph.nodes[selected_task_index]["finish_time"] = earliest_start_time + _application.task_graph.nodes[
                        selected_task_index]["w"] * cloud.process_data_rate
            else:
                # schedule to the edge
                # earliest_state_time+ transimmission time + precedence job finish time
                precedence_task_finish_time = []
                for u, v in _application.task_graph.in_edges(
                        selected_task_index):
                    precedence_task_node = _application.task_graph.nodes[u][
                        "selected_node"]
                    if precedence_task_node == edge_number:
                        # from the cloud
                        precedence_task_finish_time.append(
                            _application.task_graph.edges[u, v]["e"] *
                            cloud.data_rate +
                            _application.task_graph.nodes[u]["finish_time"])
                    elif precedence_task_node != selected_node:
                        # not same edge node
                        precedence_task_finish_time.append(
                            _application.task_graph.edges[u, v]["e"] *
                            edge_list[precedence_task_node].upload_data_rate +
                            _application.task_graph.nodes[u]["finish_time"])
                    else:
                        # same ege node
                        precedence_task_finish_time.append(
                            _application.task_graph.nodes[u]["finish_time"])

                # globally earliest start time is _application.release_time
                earliest_start_time = _application.release_time if len(
                    precedence_task_finish_time) == 0 else max(
                        precedence_task_finish_time)

                # run time
                estimated_runtime = _application.task_graph.nodes[
                    selected_task_index]["w"] * edge_list[
                        selected_node].process_data_rate

                # actual start time and _cpu
                actual_start_time, _cpu = edge_list[selected_node].find_actual_earliest_start_time(earliest_start_time, estimated_runtime)

                # set_latest_change_time
                _application.set_latest_change_time(selected_node,selected_task_index,edge_list,cloud,actual_start_time)
                
                # update cpu state
                edge_list[selected_node].set_cpu_state(_cpu, actual_start_time,
                                                       estimated_runtime,
                                                       _application_index,
                                                       selected_task_index,
                                                       )

                # set start time and node for each task
                _application.set_start_time(selected_task_index, selected_node,
                                            actual_start_time, _cpu)
                _application.task_graph.nodes[selected_task_index][
                    "finish_time"] = actual_start_time + estimated_runtime
            # finish the last node
            if selected_task_index == _application.task_graph.number_of_nodes(
            ) - 1:
                _application.finish_time = _application.task_graph.nodes[
                    selected_task_index]["finish_time"]
            scheduled[selected_task_index] = 0
        total_completion_time += _application.finish_time - _application.release_time
        # print( _application.release_time,_application.finish_time)
    print(total_completion_time)
    print(time.time() - st)
    # draw gantt
    if is_draw:
        draw.draw_gantt(application_list, edge_list, cloud)



if __name__ == '__main__':
    # setting
    is_draw=False
    application_num=500
    application_average_interval=10
    edge_number=10

    test_scheduling(is_draw=is_draw,
                    application_num=application_num,
                    application_average_interval=application_average_interval,
                    edge_number=edge_number,
                    scheduler = utils.get_node_by_random)
    test_scheduling(is_draw=is_draw,
                    application_num=application_num,
                    application_average_interval=application_average_interval,
                    edge_number=edge_number,
                    scheduler = utils.get_node_with_least_start_time)
    
