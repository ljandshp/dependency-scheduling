'''
Author: 娄炯
Date: 2021-04-18 11:21:32
LastEditors: loujiong
LastEditTime: 2021-06-02 15:17:38
Description: implement re-schedule
Email:  413012592@qq.com
'''

import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import utils_backup2 as tuils
import draw
import random
import time
import math
import numpy as np
import pqdict


def re_scheduling(is_draw=False,
                      application_num=9,
                      application_average_interval=10,
                      edge_number=10,
                      scheduler = utils.get_node_with_least_start_time):
    # random setting
    all_st = time.time()
    random.seed(0)

    finish_task_set = [pqdict.pqdict() for i in range(edge_number)]

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

    # pqdict.pqdict()可以全部替代
    unscheduled_tasks = pqdict.pqdict()
    remain_length_list = []
        
    # re-schedule
    for _release_index,_release_time in enumerate(release_time_list):
        each_application_time = 0
        print(_release_index)
        
        # pop task starting transmission
        while (len(unscheduled_tasks) > 0):
            _ap_index,_ta_index = unscheduled_tasks.top()
            _latest_transmission_time = unscheduled_tasks[(_ap_index,_ta_index)]
            if _latest_transmission_time < _release_time:
                application_list[_ap_index].task_graph.nodes[_ta_index]["is_scheduled"] = 1
                unscheduled_tasks.pop()
                if application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"] < edge_number:
                    finish_task_set[application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"]][(_ap_index,_ta_index)] = (application_list[_ap_index].task_graph.nodes()[_ta_index]["start_time"],application_list[_ap_index].task_graph.nodes()[_ta_index]["finish_time"])
            else:
                break

        # todo update self.planed_is_task_for_each_time, self.planed_task_for_each_time, update self.is_task_for_each_time, and self.task_for_each_time
        for _edge_index,_edge_node in enumerate(edge_list):
            _edge_node.update_plan_to_actural(_release_time,finish_task_set[_edge_index],application_list)
            _edge_node.generate_plan(_release_time)
            
        # current application and add unscheduled tasks
        _application = application_list[_release_index]
        for i in range(_application.task_graph.number_of_nodes()):
            if (_release_index,i) not in unscheduled_tasks:
                unscheduled_tasks[(_release_index,i)] = _release_time
        
        # add remain length -> customized list-based scheduling 
        remain_length_list.append(utils.get_remain_length(_application.task_graph))
        unscheduled_tasks_sorted_by_remain_length = pqdict.pqdict()
        for _ap_index,_ta_index in unscheduled_tasks.keys():
            if (_ap_index,_ta_index) not in unscheduled_tasks_sorted_by_remain_length:
                can_be_scheduled_label = 0
                for u, v in application_list[_ap_index].task_graph.in_edges(
                        _ta_index):
                    if application_list[_ap_index].task_graph.nodes[u]["is_scheduled"] == 0:
                        can_be_scheduled_label += 1 
                unscheduled_tasks_sorted_by_remain_length[(_ap_index,_ta_index)] = (can_be_scheduled_label,remain_length_list[_ap_index][_ta_index])
                
        print("current unscheduled task number: {0}".format(len(unscheduled_tasks)))
        while(len(unscheduled_tasks_sorted_by_remain_length) > 0):
            
            # select task with the least remain length
            _ap_index,selected_task_index = unscheduled_tasks_sorted_by_remain_length.pop()
            _current_application = application_list[_ap_index]

            # randomly select edge node or cloud
            if selected_task_index == 0 or selected_task_index == _current_application.task_graph.number_of_nodes(
            ) - 1:
                # the source and sink
                selected_node = _current_application.release_node
            else:
                # selecting node based on the scheduler
                selected_node = scheduler(
                    selected_task_index, _current_application, edge_list, cloud)
            
            if selected_node == edge_number:
                # schedule to the cloud
                # earliest_state_time+ transimmission time + precedence job finish time
                precedence_task_finish_time = []
                for u, v in _current_application.task_graph.in_edges(
                        selected_task_index):
                    precedence_task_node = _current_application.task_graph.nodes[u][
                        "selected_node"]
                    if precedence_task_node == edge_number:
                        precedence_task_finish_time.append(
                            _current_application.task_graph.nodes[u]["finish_time"])
                    else:
                        precedence_task_finish_time.append(
                            _current_application.task_graph.edges[u, v]["e"] *
                            cloud.data_rate +
                            _current_application.task_graph.nodes[u]["finish_time"])
                # globally earliest start time is _current_application.release_time
                earliest_start_time = max(precedence_task_finish_time) if len(
                    precedence_task_finish_time
                ) > 0 else _current_application.release_time

                # set_latest_change_time
                _current_application.set_latest_change_time(selected_node,selected_task_index,edge_list,cloud,earliest_start_time)

                # update scheduled tasks' latest schedule time
                unscheduled_tasks[(_ap_index,selected_task_index)] = _current_application.task_graph.nodes[selected_task_index]["latest_change_time"]
                
                # set start time and node for each task
                _current_application.set_start_time(selected_task_index, selected_node,
                                            earliest_start_time, -1)
                _current_application.task_graph.nodes[selected_task_index][
                    "finish_time"] = earliest_start_time + _current_application.task_graph.nodes[
                        selected_task_index]["w"] * cloud.process_data_rate
            else:
                
                # schedule to the edge
                # earliest_state_time+ transimmission time + precedence job finish time
                precedence_task_finish_time = []
                for u, v in _current_application.task_graph.in_edges(
                        selected_task_index):
                    precedence_task_node = _current_application.task_graph.nodes[u][
                        "selected_node"]
                    if precedence_task_node == edge_number:
                        # from the cloud
                        precedence_task_finish_time.append(
                            _current_application.task_graph.edges[u, v]["e"] *
                            cloud.data_rate +
                            _current_application.task_graph.nodes[u]["finish_time"])
                    elif precedence_task_node != selected_node:
                        # not same edge node
                        precedence_task_finish_time.append(
                            _current_application.task_graph.edges[u, v]["e"] *
                            edge_list[precedence_task_node].upload_data_rate +
                            _current_application.task_graph.nodes[u]["finish_time"])
                    else:
                        # same ege node
                        precedence_task_finish_time.append(
                            _current_application.task_graph.nodes[u]["finish_time"])

                # globally earliest start time is _current_application.release_time
                earliest_start_time = _current_application.release_time if len(
                    precedence_task_finish_time) == 0 else max(
                        precedence_task_finish_time)

                # run time
                estimated_runtime = _current_application.task_graph.nodes[
                    selected_task_index]["w"] * edge_list[
                        selected_node].process_data_rate

                st = time.time()
                # actual start time and _cpu
                actual_start_time, _cpu, selected_interval_key = edge_list[
                    selected_node].find_actual_earliest_start_time_by_planed(
                        earliest_start_time, estimated_runtime)
                each_application_time += time.time()-st

                # set_latest_change_time
                _current_application.set_latest_change_time(selected_node,selected_task_index,edge_list,cloud,actual_start_time)
                
                unscheduled_tasks[(_ap_index,selected_task_index)] = _current_application.task_graph.nodes[selected_task_index]["latest_change_time"]
                # update cpu state
                edge_list[selected_node].set_cpu_state_by_planed(_cpu, actual_start_time,
                                                       estimated_runtime,
                                                       _ap_index,
                                                       selected_task_index,
                                                       selected_interval_key)

                # set start time and node for each task
                _current_application.set_start_time(selected_task_index, selected_node,
                                            actual_start_time, _cpu)
                _current_application.task_graph.nodes[selected_task_index]["finish_time"] = actual_start_time + estimated_runtime

            
            # update can be scheduled label for other
            for u, v in application_list[_ap_index].task_graph.out_edges(selected_task_index):
                if (_ap_index,v) in unscheduled_tasks_sorted_by_remain_length:
                    _a,_b = unscheduled_tasks_sorted_by_remain_length[(_ap_index,v)]
                    unscheduled_tasks_sorted_by_remain_length[(_ap_index,v)] = (_a-1,_b)

        print(each_application_time)
        utils.check(application_list)
    for _application in application_list:        
        # finish the last node
        sink_task_index =  _application.task_graph.number_of_nodes() - 1
        _application.finish_time = _application.task_graph.nodes[sink_task_index]["finish_time"]
        total_completion_time += _application.finish_time - _application.release_time   

    print(total_completion_time)
    print(time.time() - all_st)

    if is_draw:
        draw.draw_gantt(application_list, edge_list, cloud)        
        

if __name__ == '__main__':
    is_draw=True
    application_num=9
    application_average_interval=10
    edge_number=10
    re_scheduling(is_draw=is_draw,
                application_num=application_num,
                application_average_interval=application_average_interval,
                edge_number=edge_number,
                scheduler = utils.get_node_by_random)