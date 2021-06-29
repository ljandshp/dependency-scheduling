'''
Author: 娄炯
Date: 2021-06-03 15:49:12
LastEditors: loujiong
LastEditTime: 2021-06-25 16:54:42
Description: no re_schedule
Email:  413012592@qq.com
'''


import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import utils
import draw
import random
import time
import math
import numpy as np
import pqdict
# import optimal_time_index

np.set_printoptions(suppress=True)

def re_scheduling(is_draw=False,
                      application_num=9,
                      application_average_interval=10,
                      edge_number=10,
                      scheduler = utils.get_node_with_least_start_time):
    deadline_alpha = 4/7
    # debug
    total_len_unscheduled_tasks_list = []
    
    # random setting
    all_st = time.time()
    random.seed(1.1)

    # 调度过程已经结束的所有task
    finish_task_set = [np.empty((0,4)) for i in range(edge_number)]

    # generate release time and application
    release_time_list = []
    release_time = 0
    for i in range(application_num):
        release_time_list.append(release_time)
        release_time += math.ceil(random.expovariate(1 / application_average_interval))
    application_list = [
        utils.Application(release_time=release_time_list[i],
                          task_num=rd(10, 20),
                          release_node=rd(0, edge_number - 1),application_index = i)
        for i in range(application_num)
    ]

    t_sum = 0
    for _application in application_list:
        t_sum += sum([_application.task_graph.nodes[_t]["w"] for _t in _application.task_graph.nodes()])
    print("average task weight sum:{0}".format(t_sum/application_num))
    
    # initial total completion time
    total_completion_time = 0

    # generate cost and process_data_rate list for edge node
    cost_per_mip_level = {i:15-i for i in range(5,11)}
    # cost_per_mip_level = {5:10,6:9.5,7:8.5,8:7,9:6,10:5}
    process_data_rate_list = [rd(5, 10) for _ in range(edge_number)]
    random.shuffle(process_data_rate_list)
    cost_per_mip_list = [cost_per_mip_level[i] for i in process_data_rate_list]


    # initiate edges and cloud
    cloud = utils.Cloud(cost_per_mip = 30)
    edge_list = [
        utils.Edge(task_concurrent_capacity=1,
                   process_data_rate=process_data_rate_list[i],
                   upload_data_rate=rd(2, 4),cost_per_mip = cost_per_mip_list[i]) for i in range(edge_number)
    ]

    for _app in application_list:
        utils.set_tmax(_app,edge_list,cloud)
        _app.deadline = deadline_alpha * _app.tmax
        # print("deadline:{0}".format(_app.deadline))

    edge_weight=(sum([_edge.upload_data_rate for _edge in edge_list])*(edge_number-1)+cloud.data_rate*2*edge_number)/((edge_number+1)*(edge_number+1))
    node_weight=(sum([_edge.process_data_rate for _edge in edge_list])+cloud.process_data_rate)/(edge_number+1)
    print("average node weight:{0}, average node number:{1}".format(sum([_edge.process_data_rate for _edge in edge_list])/edge_number,edge_number))
    print("average edge weight:{0}".format(edge_weight))
    # quit()
    
    unscheduled_tasks = pqdict.pqdict()
    remain_length_list = []

    each_application_time = 0
    
    # for _edge_index,_edge_node in enumerate(edge_list):
    #     print("edge {0}: {1}".format(_edge_index,_edge_node.interval_statistical()))
    #     print("cost_per_mip_for_{0}:{1}".format(_edge_index,_edge_node.cost_per_mip))
    #     print("process_data_rate_for_{0}:{1}".format(_edge_index,_edge_node.process_data_rate))
    #     print("data_rate_for_{0}:{1}".format(_edge_index,_edge_node.upload_data_rate))
    #     print()
    
    # print("cloud:")
    # print("cost_per_mip_for_cloud:{0}".format(cloud.cost_per_mip))
    # print("process_data_rate_for_cloud:{0}".format(cloud.process_data_rate))
    # print("data_rate_for_cloud:{0}".format(cloud.data_rate))
    
    # schedule each application
    for _release_index,_release_time in enumerate(release_time_list):
        # print(_release_index)
        
        # run completion_time_docplex2 
        # _TaskMatrix =[[-1]*application_list[_release_index].task_graph.number_of_nodes() for _ in range(application_list[_release_index].task_graph.number_of_nodes())]
        # for u,v in application_list[_release_index].task_graph.edges():
        #     _TaskMatrix[u][v] = application_list[_release_index].task_graph.edges[u, v]["e"]
        # _EdgeMatrix = [[edge_list[i].upload_data_rate]*edge_number for i in range(edge_number)]
        # _TaskEdgeRunTime = [[application_list[_release_index].task_graph.nodes[i]["w"]*edge_list[j].process_data_rate for j in range(edge_number)] for i in range(application_list[_release_index].task_graph.number_of_nodes())]
        # _ReleaseEdge = application_list[_release_index].release_node
        # _DeadLine = int(application_list[_release_index].deadline) 
        # _MaxTime = _DeadLine
        # _Cost = [edge_list[i].cost_per_mip/edge_list[i].process_data_rate for i in range(edge_number)]
        # print(_TaskMatrix,_EdgeMatrix,_TaskEdgeRunTime,_MaxTime,_DeadLine,_Cost,_ReleaseEdge)
        # draw.draw(application_list[_release_index].task_graph)
        # estimated_finish_time_list = optimal_time_index.completion_time_docplex2(_TaskMatrix,_EdgeMatrix,_TaskEdgeRunTime,_MaxTime,_DeadLine,_Cost,_ReleaseEdge)
        # print(estimated_finish_time_list)
        
        # pop task starting transmission
        while (len(unscheduled_tasks) > 0):
            _ap_index,_ta_index = unscheduled_tasks.top()
            _latest_transmission_time = unscheduled_tasks[(_ap_index,_ta_index)]
            if _latest_transmission_time <= _release_time:
                application_list[_ap_index].task_graph.nodes[_ta_index]["is_scheduled"] = 1
                unscheduled_tasks.pop()
                if application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"] < edge_number:
                    finish_task_set[application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"]] = np.vstack([finish_task_set[application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"]],np.array([_ap_index,_ta_index,application_list[_ap_index].task_graph.nodes()[_ta_index]["start_time"],application_list[_ap_index].task_graph.nodes()[_ta_index]["finish_time"]])])
            else:
                break
        
        print()
        print("unscheduled_tasks number:{0}".format(len(unscheduled_tasks)))
        print("_release_time:{0}, deadline:{1}".format(_release_time,application_list[_release_index].deadline))
        print("total task weight:{0}, total transfer:{1}".format(sum([application_list[_release_index].task_graph.nodes[_t]["w"] for _t in application_list[_release_index].task_graph.nodes()]),sum([application_list[_release_index].task_graph.edges[u, v]["e"] for u, v in application_list[_release_index].task_graph.edges()])))
        print("total task number:{0}, total edge number:{1}".format(len(application_list[_release_index].task_graph.nodes()),len(application_list[_release_index].task_graph.edges())))

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
        remain_length_list = utils.get_remain_length(_application.task_graph,edge_weight=edge_weight,node_weight=node_weight)
        for _t in _application.task_graph.nodes():
            _application.task_graph.nodes()[_t]["current_remain_length"] = remain_length_list[_t]
            if _application.dynamic_longest_remain_length < remain_length_list[_t]:
                _application.dynamic_longest_remain_length = remain_length_list[_t]

        # generate sub_deadline
        sub_deadline_list = utils.get_sub_deadline_list(_application.task_graph,remain_length_list,deadline = _application.deadline,edge_weight=edge_weight,node_weight=node_weight)
        print(sub_deadline_list)
        for _t in _application.task_graph.nodes():
            # _application.task_graph.nodes()[_t]["sub_deadline"] = estimated_finish_time_list[_t] 
            _application.task_graph.nodes()[_t]["sub_deadline"] = sub_deadline_list[_t]

        unscheduled_tasks_sorted_by_remain_length = pqdict.pqdict()
        for _ap_index,_ta_index in unscheduled_tasks.keys():
            if (_ap_index,_ta_index) not in unscheduled_tasks_sorted_by_remain_length:
                can_be_scheduled_label = 0
                for u, v in application_list[_ap_index].task_graph.in_edges(
                        _ta_index):
                    if application_list[_ap_index].task_graph.nodes[u]["is_scheduled"] == 0:
                        can_be_scheduled_label += 1 
                # unscheduled_tasks_sorted_by_remain_length[(_ap_index,_ta_index)] = (can_be_scheduled_label,application_list[_ap_index].dynamic_longest_remain_length,0-application_list[_ap_index].task_graph.nodes()[_ta_index]["current_remain_length"])
                unscheduled_tasks_sorted_by_remain_length[(_ap_index,_ta_index)] = (can_be_scheduled_label,application_list[_ap_index].deadline+application_list[_ap_index].release_time,0-application_list[_ap_index].task_graph.nodes()[_ta_index]["current_remain_length"])
                application_list[_ap_index].task_graph.nodes()[_ta_index]["is_scheduled_in_this_scheduling"] = 0
               
        # print("current unscheduled task number: {0}".format(len(unscheduled_tasks)))
        total_len_unscheduled_tasks_list.append(len(unscheduled_tasks))
        while(len(unscheduled_tasks_sorted_by_remain_length) > 0):
            # select task with the least remain length
            _ap_index, selected_task_index = unscheduled_tasks_sorted_by_remain_length.top()
            # print("application index:{0},task index:{1}".format(_ap_index, selected_task_index))
            _can_be_scheduled_label, application_level_length, task_level_length = unscheduled_tasks_sorted_by_remain_length[(_ap_index, selected_task_index)]
            unscheduled_tasks_sorted_by_remain_length.pop()
            _current_application = application_list[_ap_index]
            st = time.time()
            # randomly select edge node or cloud
            if selected_task_index == 0 or selected_task_index == _current_application.task_graph.number_of_nodes(
            ) - 1:
                # the source and sink
                selected_node = _current_application.release_node
            else:
                # selecting node based on the scheduler
                selected_node = scheduler(
                    selected_task_index, _current_application, edge_list, cloud, _release_time)
            _find_time = time.time()-st
            each_application_time += _find_time
            
            if selected_node == edge_number:
                # schedule to the cloud
                # earliest_state_time+ transimmission time + precedence job finish time
                precedence_task_finish_time = []
                for u, v in _current_application.task_graph.in_edges(selected_task_index):
                    precedence_task_node = _current_application.task_graph.nodes[u][
                        "selected_node"]
                    if precedence_task_node == edge_number:
                        precedence_task_finish_time.append(
                            max(_current_application.task_graph.nodes[u]["finish_time"],_release_time))
                    else:
                        # from edge to the cloud
                        precedence_task_finish_time.append(
                            _current_application.task_graph.edges[u, v]["e"] *
                            cloud.data_rate +
                            max(_current_application.task_graph.nodes[u]["finish_time"],_release_time))
                # globally earliest start time is _current_application.release_time
                earliest_start_time = max(precedence_task_finish_time) if len(
                    precedence_task_finish_time
                ) > 0 else _current_application.release_time

                # set_latest_change_time
                _current_application.set_latest_change_time(selected_node,selected_task_index,edge_list,cloud,earliest_start_time)

                # update scheduled tasks' latest schedule time
                # unscheduled_tasks[(_ap_index,selected_task_index)] = _current_application.task_graph.nodes[selected_task_index]["latest_change_time"]
                unscheduled_tasks[(_ap_index,selected_task_index)] = _current_application.release_time
                
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
                            cloud.data_rate + max(
                            _current_application.task_graph.nodes[u]["finish_time"],_release_time))
                    elif precedence_task_node != selected_node:
                        # not same edge node
                        precedence_task_finish_time.append(
                            _current_application.task_graph.edges[u, v]["e"] *
                            edge_list[precedence_task_node].upload_data_rate +
                            max(_current_application.task_graph.nodes[u]["finish_time"],_release_time))
                    else:
                        # same ege node
                        precedence_task_finish_time.append(
                            max(_current_application.task_graph.nodes[u]["finish_time"],_release_time))

                # globally earliest start time is _current_application.release_time
                earliest_start_time = _current_application.release_time if len(
                    precedence_task_finish_time) == 0 else max(
                        precedence_task_finish_time)

                if earliest_start_time< _release_time:
                    print(earliest_start_time,_release_time)
                    quit()
                    
                # run time
                estimated_runtime = _current_application.task_graph.nodes[
                    selected_task_index]["w"] * edge_list[
                        selected_node].process_data_rate

                
                # actual start time and _cpu
                actual_start_time, _cpu, selected_interval_key = edge_list[
                    selected_node].find_actual_earliest_start_time_by_planed(
                        earliest_start_time, estimated_runtime,_release_time)
                
                if actual_start_time< _release_time:
                    print(actual_start_time,_release_time)
                    quit()
                

                # set_latest_change_time
                _current_application.set_latest_change_time(selected_node,selected_task_index,edge_list,cloud,actual_start_time)
                
                # update scheduled tasks' latest schedule time
                # unscheduled_tasks[(_ap_index,selected_task_index)] = _current_application.task_graph.nodes[selected_task_index]["latest_change_time"]
                unscheduled_tasks[(_ap_index,selected_task_index)] = _current_application.release_time
                
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

            
            
            application_list[_ap_index].task_graph.nodes()[selected_task_index]["is_scheduled_in_this_scheduling"] = 1
            
            # unscheduled_tasks_sorted_by_remain_length can update for each time
            # if application_list[_ap_index].dynamic_longest_remain_length == application_list[_ap_index].task_graph.nodes()[selected_task_index]["current_remain_length"]:
            #     _longest_remain_length = 0
            #     for _t in application_list[_ap_index].task_graph.nodes():
            #         if application_list[_ap_index].task_graph.nodes()[_t]["is_scheduled_in_this_scheduling"] == 0  and application_list[_ap_index].task_graph.nodes()[_t]["current_remain_length"] > _longest_remain_length:
            #             _longest_remain_length = application_list[_ap_index].task_graph.nodes()[_t]["current_remain_length"]
            #     application_list[_ap_index].dynamic_longest_remain_length = _longest_remain_length
            #     for _t in application_list[_ap_index].task_graph.nodes():
            #         if (_ap_index, _t) in unscheduled_tasks_sorted_by_remain_length:
            #             _a, _b, _c = unscheduled_tasks_sorted_by_remain_length[(_ap_index, _t)]
            #             unscheduled_tasks_sorted_by_remain_length[(_ap_index,_t)] = (_a, _longest_remain_length, _c)

            # update can be scheduled label for other
            for u, v in application_list[_ap_index].task_graph.out_edges(selected_task_index):
                if (_ap_index,v) in unscheduled_tasks_sorted_by_remain_length:
                    _a, _b, _c = unscheduled_tasks_sorted_by_remain_length[(_ap_index,v)]
                    unscheduled_tasks_sorted_by_remain_length[(_ap_index,v)] = (_a-1, _b, _c)
        print("")
        print("current_application_index:{0}".format(_ap_index))
        print("remain_length_for_entry:{0}".format(application_list[_release_index].task_graph.nodes[0]["current_remain_length"]))
        print("(task_id,select_node_id,start_time,finish_time,sub_deadline)")
        print([(i,application_list[_release_index].task_graph.nodes[i]["selected_node"],application_list[_release_index].task_graph.nodes[i]["start_time"],application_list[_release_index].task_graph.nodes[i]["finish_time"],application_list[_release_index].task_graph.nodes[i]["sub_deadline"]+application_list[_release_index].release_time) for i in application_list[_release_index].task_graph.nodes()])
        # draw.draw(application_list[_release_index].task_graph)
        # utils.check(application_list)

    # 设置 finish_task_set
    # print(sum([len(i) for i in finish_task_set]))
    while (len(unscheduled_tasks) > 0):
        _ap_index,_ta_index = unscheduled_tasks.top()
        application_list[_ap_index].task_graph.nodes[_ta_index]["is_scheduled"] = 1
        unscheduled_tasks.pop()
        if application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"] < edge_number:
            finish_task_set[application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"]] = np.vstack([finish_task_set[application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"]],np.array([_ap_index,_ta_index,application_list[_ap_index].task_graph.nodes()[_ta_index]["start_time"],application_list[_ap_index].task_graph.nodes()[_ta_index]["finish_time"]])])
    
    # 生成最终的start_finish，也就是空闲的interval list
    for _edge_index,_edge_node in enumerate(edge_list):
        _edge_node.update_plan_to_actural(_release_time,finish_task_set[_edge_index],application_list)
    
    # output interval_statistical for each edge node
    print()
    print("edge resource utilization information")
    for _edge_index,_edge_node in enumerate(edge_list):
        print("edge {0}: {1}".format(_edge_index,_edge_node.interval_statistical()))
        finish_task_set[_edge_index] = finish_task_set[_edge_index][finish_task_set[_edge_index][:,3].argsort()]
        print(finish_task_set[_edge_index])
        print("cost_per_mip_for_{0}:{1}".format(_edge_index,_edge_node.cost_per_mip))
        print("process_data_rate_for_{0}:{1}".format(_edge_index,_edge_node.process_data_rate))
        print("data_rate_for_{0}:{1}".format(_edge_index,_edge_node.upload_data_rate))
        print()

    for _application in application_list:        
        # finish the last node
        sink_task_index =  _application.task_graph.number_of_nodes() - 1
        _application.finish_time = _application.task_graph.nodes[sink_task_index]["finish_time"]
        total_completion_time += _application.finish_time - _application.release_time   

    print("execution_time:{0}".format(each_application_time))
    # print(total_completion_time)
    # print(total_len_unscheduled_tasks_list)
    print([(_ap_index,_application.finish_time) for _ap_index,_application in enumerate(application_list)])
    print([(_ap_index,_application.release_time+_application.deadline) for _ap_index,_application in enumerate(application_list)])
    print([(_ap_index,_application.release_time) for _ap_index,_application in enumerate(application_list)])
    print([(_ap_index,_application.deadline) for _ap_index,_application in enumerate(application_list)])
    print([(_ap_index,_application.release_time+_application.deadline > _application.finish_time) for _ap_index,_application in enumerate(application_list)])
    print([(_ap_index,(_application.finish_time-_application.release_time)/_application.deadline) for _ap_index,_application in enumerate(application_list)])
    
    # output accept rate and total cost
    accept_rate = 0
    for _application in application_list:
        if _application.finish_time < _application.release_time+_application.deadline:
            accept_rate += 1
    print("accept_rate:{0}".format(accept_rate/len(application_list)))
    total_cost = 0
    cost_per_mip_list.append(cloud.cost_per_mip)
    for _application in application_list:
        _cost_application = sum([_application.task_graph.nodes()[_t]["w"]*cost_per_mip_list[_application.task_graph.nodes()[_t]["selected_node"]] for _t in  _application.task_graph.nodes()])
        total_cost += _cost_application
    print("total_cost:{0}".format(total_cost))
    
    print("application task number list:{0}".format([_application.task_graph.number_of_nodes() for _application in application_list]))
    
    if is_draw:
        draw.draw_gantt(application_list, edge_list, cloud) 



    for _edge_index,_edge_node in enumerate(edge_list):
        print("edge {0}: {1}".format(_edge_index,_edge_node.interval_statistical()))
        print("cost_per_mip_for_{0}:{1}".format(_edge_index,_edge_node.cost_per_mip))
        print("process_data_rate_for_{0}:{1}".format(_edge_index,_edge_node.process_data_rate))
        print("data_rate_for_{0}:{1}".format(_edge_index,_edge_node.upload_data_rate))
        print()
    
    print("cloud:")
    print("cost_per_mip_for_cloud:{0}".format(cloud.cost_per_mip))
    print("process_data_rate_for_cloud:{0}".format(cloud.process_data_rate))
    print("data_rate_for_cloud:{0}".format(cloud.data_rate))
        
if __name__ == '__main__':
    is_draw = True
    application_num=1
    application_average_interval=120
    edge_number=10
    # re_scheduling(is_draw=is_draw,
    #             application_num=application_num,
    #             application_average_interval=application_average_interval,
    #             edge_number=edge_number,
    #             scheduler = utils.get_node_with_least_start_time)
    
    re_scheduling(is_draw=is_draw,
                application_num=application_num,
                application_average_interval=application_average_interval,
                edge_number=edge_number,
                scheduler = utils.get_node_with_least_cost_constrained_by_subdeadline)