'''
Author: 娄炯
Date: 2021-07-06 21:30:50
LastEditors: loujiong
LastEditTime: 2021-07-07 19:34:33
Description: using partial critical path 
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

np.set_printoptions(suppress=True)


def re_scheduling(is_draw=False,
                  is_annotation=False,
                  application_num=9,
                  application_average_interval=10,
                  edge_number=10,
                  scheduler=utils.get_node_with_least_cost_constrained_by_subdeadline_for_pcp,
                  random_seed=1.11,
                  is_draw_task_graph=False):
    deadline_alpha = 1.5/7
    # debug
    total_len_unscheduled_tasks_list = []

    # random setting
    all_st = time.time()
    each_application_time = 0   #calculate the interested execution time
    interested_time = [0,0,0]
    random.seed(random_seed)

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

    # save task graph figures
    for i in range(application_num):
        print("application_{0} node number:{1} edge number:{2}".format(i, application_list[i].task_graph.number_of_nodes(), application_list[i].task_graph.number_of_edges()))
        if is_draw_task_graph:
            draw.draw(application_list[i].task_graph, is_save = True, _application_index = i)

    # calculate the total/average task weight
    t_sum = 0
    for _application in application_list:
        t_sum += sum([_application.task_graph.nodes[_t]["w"] for _t in _application.task_graph.nodes()])
    print("average task weight sum:{0}".format(t_sum/application_num))

    # generate cost and process_data_rate list for edge node
    cost_per_mip_level = {i:15-i for i in range(5,11)}
    # cost_per_mip_level = {5:10,6:9.5,7:8.5,8:7,9:6,10:5}
    process_data_rate_list = [rd(5, 10) for _ in range(edge_number)]
    random.shuffle(process_data_rate_list)
    cost_per_mip_list = [cost_per_mip_level[i] for i in process_data_rate_list]


    # initiate edges and cloud
    cloud = utils.Cloud(cost_per_mip = 30, data_rate = 15)
    edge_list = [
        utils.Edge(task_concurrent_capacity=1,
                   process_data_rate=process_data_rate_list[i],
                   upload_data_rate=rd(3, 6),cost_per_mip = cost_per_mip_list[i]) for i in range(edge_number)
    ]

    for _application in application_list:
        print()
        utils.set_tmax(_application,edge_list,cloud)
        _application.deadline = deadline_alpha * _application.tmax
        print("_application.tmax:{0}".format(_application.tmax))
        print("deadline:{0}".format(_application.deadline))
        _app_total_weight = sum([_application.task_graph.nodes[_t]["w"] for _t in _application.task_graph.nodes()])
        print("ideal local time:{0}".format(_app_total_weight/sum([1/_edge.process_data_rate for _edge in edge_list])))
        print("longgest path:{0}".format(min([_edge.process_data_rate for _edge in edge_list])*sum([_application.task_graph.nodes[_t]["w"] for _t in nx.dag_longest_path(_application.task_graph,weight="w")])))

    edge_weight=(sum([_edge.upload_data_rate for _edge in edge_list])*(edge_number-1)+cloud.data_rate*2*edge_number)/((edge_number+1)*(edge_number+1))
    node_weight=(sum([_edge.process_data_rate for _edge in edge_list])+cloud.process_data_rate)/(edge_number+1)
    # print("average node weight:{0}, average node number:{1}".format(sum([_edge.process_data_rate for _edge in edge_list])/edge_number,edge_number))
    # print("average edge weight:{0}".format(edge_weight))

    remain_length_list = []

    

    # schedule each application
    for _release_index,_release_time in enumerate(release_time_list):
        new_finish_task_set = [np.empty((0,4)) for i in range(edge_number)]
        # pop task starting transmission  -> remove task satisfying finish scheduling condition
        for _edge_index,_edge_node in enumerate(edge_list):
            _edge_node.generate_plan(_release_time)

                    # new_finish_task_set[_current_task_selected_node] = np.vstack([new_finish_task_set[_current_task_selected_node],np.array([_ap_index,_ta_index,application_list[_ap_index].task_graph.nodes()[_ta_index]["start_time"],application_list[_ap_index].task_graph.nodes()[_ta_index]["finish_time"]])])

        _application = application_list[_release_index]

        remain_length_list = utils.get_remain_length(_application.task_graph,edge_weight=edge_weight,node_weight=node_weight)
        for _t in _application.task_graph.nodes():
            _application.task_graph.nodes()[_t]["current_remain_length"] = remain_length_list[_t]
            if _application.dynamic_longest_remain_length < remain_length_list[_t]:
                _application.dynamic_longest_remain_length = remain_length_list[_t]

        # generate sub_deadline for each task
        sub_deadline_list = utils.get_sub_deadline_list(_application.task_graph,remain_length_list,deadline = _application.deadline,edge_weight=edge_weight,node_weight=node_weight)
        for _t in _application.task_graph.nodes():
            _application.task_graph.nodes()[_t]["sub_deadline"] = sub_deadline_list[_t]
    
        # start scheduling
        t_list =[]
        l_start_time = [math.inf] * _application.task_graph.number_of_nodes()
        # first labeling source and sink tasks
        _source, _sink = 0, _application.task_graph.number_of_nodes()-1
        _application.task_graph.nodes[_source]["is_scheduled"] = 1
        _application.task_graph.nodes[_sink]["is_scheduled"] = 1
        _application.set_start_time(_source,_application.release_node,_release_time,0)
        _application.set_start_time(_sink,_application.release_node,int(_release_time+_application.deadline),0)
        _application.task_graph.nodes[_source]["finish_time"] = _release_time
        _application.task_graph.nodes[_sink]["finish_time"] = int(_release_time+_application.deadline)
        l_start_time[_source] = _release_time
        l_start_time[_sink] = int(_release_time+_application.deadline)
        
        t_list.append(_sink)
        # update l_start_time
        l_update_list =[]
        l_update_list.append(_sink)
        while(len(l_update_list) > 0):
            _v = l_update_list.pop(0)
            for u,_ in _application.task_graph.in_edges(_v):
                if _application.task_graph.nodes[u]["is_scheduled"] == 0:
                    _e = _application.task_graph.edges[u, _v]["e"]
                    _w = _application.task_graph.nodes[u]["w"]
                    if l_start_time[_v]-_e*edge_weight-_w*node_weight < l_start_time[u]:
                        l_start_time[u] = l_start_time[_v]-_e*edge_weight-_w*node_weight
                        l_update_list.append(u)

        print("fuck")
        print(t_list)
        print("l_start_time",l_start_time)
        while(len(t_list) > 0):
            _t = t_list[-1]
            has_unscheduled_parent = sum([1-_application.task_graph.nodes[u]["is_scheduled"] for u,_ in _application.task_graph.in_edges(_t)])
            print("has_unscheduled_parent",has_unscheduled_parent)
            if has_unscheduled_parent > 0.5:
                _pcp = []
                #add current _t
                _pcp.append(_t)
                # find pcp
                while(has_unscheduled_parent > 0.5):
                    l_l_start_time = math.inf
                    critical_parent_t = -1
                    for u, _ in _application.task_graph.in_edges(_t):
                        print("is_scheduled",_application.task_graph.nodes[u]["is_scheduled"])
                        if _application.task_graph.nodes[u]["is_scheduled"] == 0:
                            if l_start_time[u] < l_l_start_time:
                                l_l_start_time = l_start_time[u]
                                critical_parent_t = u
                    _t = critical_parent_t
                    print(l_l_start_time)
                    _pcp.append(_t)
                    print(_t)
                    has_unscheduled_parent = sum([1-_application.task_graph.nodes[u]["is_scheduled"] for u,_ in _application.task_graph.in_edges(_t)])
                # assgin pcp
                selected_node = scheduler(
                    _pcp, _application, edge_list, cloud, _release_time)
                print(_pcp)
                print("selected_node",selected_node)
                #all in one node
                print("t_list",t_list)
                for i in _pcp[1:]:
                    t_list.append(i)
                print("t_list",t_list)
                break

                for i in _pcp:
                    _application.task_graph.nodes[i]["is_scheduled"] = 1
                # update l_start_time
                l_update_list =[]
                l_update_list.extend(_pcp)
                while(len(l_update_list) > 0):
                    _v = l_update_list.pop(0)
                    for u,_ in _application.task_graph.in_edges(_v):
                        if _application.task_graph.nodes[u]["is_scheduled"] == 0:
                            _e = _application.task_graph.edges[u, _v]["e"]
                            _w = _application.task_graph.nodes[u]["w"]
                            if l_start_time[_v]-_e*edge_weight-_w*node_weight < l_start_time[u]:
                                l_start_time[u] = l_start_time[_v]-_e*edge_weight-_w*node_weight
                                l_update_list.append(u)
                for i in _pcp[1:]:
                    t_list.append(i)
            else:
                t_list.pop(-1)
            
            
        


        for _edge_index,_edge_node in enumerate(edge_list):
            # print("------------------------{0}------------------------".format(_edge_index))
            _edge_node.update_plan_to_actural(_release_time,new_finish_task_set[_edge_index],application_list)

    quit()   
    # utils.check(application_list)

    # # 设置 finish_task_set
    # # print(sum([len(i) for i in finish_task_set]))
    # while (len(unscheduled_tasks) > 0):
    #     _ap_index,_ta_index = unscheduled_tasks.top()
    #     application_list[_ap_index].task_graph.nodes[_ta_index]["is_scheduled"] = 1
    #     unscheduled_tasks.pop()
    #     if application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"] < edge_number:
    #         finish_task_set[application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"]] = np.vstack([finish_task_set[application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"]],np.array([_ap_index,_ta_index,application_list[_ap_index].task_graph.nodes()[_ta_index]["start_time"],application_list[_ap_index].task_graph.nodes()[_ta_index]["finish_time"]])])

    # # 生成最终的start_finish，也就是空闲的interval list
    # for _edge_index,_edge_node in enumerate(edge_list):
    #     _edge_node.update_plan_to_actural(_release_time,finish_task_set[_edge_index],application_list)

    # # output interval_statistical for each edge node
    # print()
    # print("edge resource utilization information")
    # for _edge_index,_edge_node in enumerate(edge_list):
    #     print("edge {0}: {1}".format(_edge_index,_edge_node.interval_statistical()))
    #     finish_task_set[_edge_index] = finish_task_set[_edge_index][finish_task_set[_edge_index][:,3].argsort()]
    #     print(finish_task_set[_edge_index])
    #     print("cost_per_mip_for_{0}:{1}".format(_edge_index,_edge_node.cost_per_mip))
    #     print("process_data_rate_for_{0}:{1}".format(_edge_index,_edge_node.process_data_rate))
    #     print("data_rate_for_{0}:{1}".format(_edge_index,_edge_node.upload_data_rate))
    #     print()

    # for _application in application_list:
    #     # finish the last node
    #     sink_task_index =  _application.task_graph.number_of_nodes() - 1
    #     _application.finish_time = _application.task_graph.nodes[sink_task_index]["finish_time"]

    # print("execution_time:{0}".format(each_application_time))
    # print("interested_time:{0}".format(interested_time))

    # # print(total_len_unscheduled_tasks_list)
    # print([(_ap_index,_application.finish_time) for _ap_index,_application in enumerate(application_list)])
    # print([(_ap_index,_application.release_time+_application.deadline) for _ap_index,_application in enumerate(application_list)])
    # print([(_ap_index,_application.release_time) for _ap_index,_application in enumerate(application_list)])
    # print([(_ap_index,_application.deadline) for _ap_index,_application in enumerate(application_list)])
    # print([(_ap_index,_application.release_time+_application.deadline > _application.finish_time) for _ap_index,_application in enumerate(application_list)])
    # print([(_ap_index,(_application.finish_time-_application.release_time)/_application.deadline) for _ap_index,_application in enumerate(application_list)])

    # # output accept rate and total cost
    # accept_rate = 0
    # for _application in application_list:
    #     if _application.finish_time < _application.release_time+_application.deadline:
    #         accept_rate += 1
    # print("accept_rate:{0}".format(accept_rate/len(application_list)))
    # total_cost = 0
    # cost_per_mip_list.append(cloud.cost_per_mip)
    # for _application in application_list:
    #     _cost_application = sum([_application.task_graph.nodes()[_t]["w"]*cost_per_mip_list[_application.task_graph.nodes()[_t]["selected_node"]] for _t in  _application.task_graph.nodes()])
    #     total_cost += _cost_application
    # print("total_cost:{0}".format(total_cost))

    # print("application task number list:{0}".format([_application.task_graph.number_of_nodes() for _application in application_list]))

    # if is_draw:
    #     draw.draw_gantt(application_list, edge_list, cloud, is_annotation=is_annotation)

    # _cost_per_mip_list = []
    # _performance_list = []
    # _cost_per_mip_list.append(("cloud",cloud.cost_per_mip))
    # _performance_list.append(("cloud",cloud.process_data_rate))
    # for _edge_index,_edge_node in enumerate(edge_list):
    #     print("edge {0}: {1}".format(_edge_index,_edge_node.interval_statistical()))
    #     print("cost_per_mip_for_{0}:{1}".format(_edge_index,_edge_node.cost_per_mip))
    #     print("process_data_rate_for_{0}:{1}".format(_edge_index,_edge_node.process_data_rate))
    #     print("data_rate_for_{0}:{1}".format(_edge_index,_edge_node.upload_data_rate))
    #     print()
    #     _cost_per_mip_list.append((_edge_index,_edge_node.cost_per_mip))
    #     _performance_list.append((_edge_index,_edge_node.process_data_rate))

    # print("cloud:")
    # print("cost_per_mip_for_cloud:{0}".format(cloud.cost_per_mip))
    # print("process_data_rate_for_cloud:{0}".format(cloud.process_data_rate))
    # print("data_rate_for_cloud:{0}".format(cloud.data_rate))

    # print("")
    # print("cost:")
    # print(_cost_per_mip_list)
    # print("performance:")
    # print(_performance_list)
    # print(time.time()-all_st)

if __name__ == '__main__':
    is_draw = False
    is_annotation = True
    is_draw_task_graph = True
    application_num = 10
    application_average_interval = 120
    edge_number = 13
    random_seed = 1.2
    
    # re_scheduling(is_draw=is_draw,
    #             is_annotation = is_annotation
    #             application_num=application_num,
    #             application_average_interval=application_average_interval,
    #             edge_number=edge_number,
    #             scheduler = utils.get_node_with_earliest_finish_time,
    #             random_seed = random_seed,
    #               is_draw_task_graph = is_draw_task_graph)

    re_scheduling(
        is_draw=is_draw,
        is_annotation=is_annotation,
        application_num=application_num,
        application_average_interval=application_average_interval,
        edge_number=edge_number,
        scheduler=utils.get_node_with_least_cost_constrained_by_subdeadline_for_pcp,
        random_seed=random_seed,
        is_draw_task_graph=is_draw_task_graph)
