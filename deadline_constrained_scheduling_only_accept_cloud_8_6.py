'''
Author: 娄炯
Date: 2021-08-06 13:33:22
LastEditors: loujiong
LastEditTime: 2021-08-22 12:04:49
Description: 
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
import deadline_constrained_scheduling_only_accept_case_study_8_2
import sys, getopt
# import Edge
# import Application
# import Cloud

np.set_printoptions(suppress=True)

def resize_deadline(start_sub_deadline_list,sub_deadline_list,_application,edge_list):
    process_data_rate_list = [i.process_data_rate for i in edge_list]
    _reverse_average = sum([1/i for i in process_data_rate_list])/len(process_data_rate_list)
    process_data_rate_list.sort()
    l = []
    l.extend([[start_sub_deadline_list[i],0,i] for i in range(len(start_sub_deadline_list))])
    # calculate the latest children deadline
    for _n in _application.task_graph.nodes():
        _latest_children_t = start_sub_deadline_list[-1]
        for u,v in _application.task_graph.out_edges(_n):
            _latest_children_t = min(_latest_children_t,start_sub_deadline_list[v])
        l.append([_latest_children_t,1,_n])

    l.sort()
    current_w = 0
    last_time = 0
    total_w = 0
    for i in range(len(l)):
        _t = l[i][0]
        if current_w == 0:
            c_w = 0
            # 这个地方看看能不能修正
        elif current_w > len(process_data_rate_list):
            c_w = 1/(_reverse_average*current_w)
        else:
            c_w = 1/(sum([1/_p_d for _p_d in process_data_rate_list[:current_w]]))
        total_w = current_w*c_w*(_t-last_time)+total_w
        l[i].append(total_w)
        if l[i][1] == 0:
            current_w += 1
        else:
            current_w -= 1
        last_time = _t
    
    final_w = l[-1][-1]
    total_deadline = l[-1][0]
    resized_start_sub_deadline_list = [0]*_application.task_graph.number_of_nodes()
    resized_sub_deadline_list = [0]*_application.task_graph.number_of_nodes()
    for i in range(len(l)):
        _w = l[i][-1]
        _n = l[i][2]
        if l[i][1] == 0:
            resized_start_sub_deadline_list[_n] = _w/final_w*total_deadline
        else:
            resized_sub_deadline_list[_n] = _w/final_w*total_deadline
    
    return resized_start_sub_deadline_list
    
def re_scheduling(is_draw=False,
                  is_annotation=False,
                  application_num=9,
                  application_average_interval=10,
                  edge_number=10,
                  scheduler=utils.get_node_with_earliest_finish_time,
                  random_seed=1.11,
                  is_draw_task_graph=False,
                  is_multiple=True,
                  deadline_alpha=1.4/7,
                  base_deadline = []):

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
        release_time += random.expovariate(1 / application_average_interval)
    application_list = [
        utils.Application(release_time=release_time_list[i],
                          task_num=rd(10, 20),
                          release_node=rd(0, edge_number - 1),application_index = i)
        for i in range(application_num)
    ]

    # save task graph figures
    for i in range(application_num):
        if is_draw_task_graph:
            draw.draw(application_list[i].task_graph, is_save = True, _application_index = i)

    # calculate the total/average task weight
    t_sum = 0
    for _application in application_list:
        t_sum += sum([_application.task_graph.nodes[_t]["w"] for _t in _application.task_graph.nodes()])

    # generate cost and process_data_rate list for edge node
    cost_per_mip_level = {i:15-i for i in range(5,11)}
    process_data_rate_list = [rd(5, 10) for _ in range(edge_number)]
    random.shuffle(process_data_rate_list)
    cost_per_mip_list = [cost_per_mip_level[i] for i in process_data_rate_list]


    # initiate edges and cloud
    cloud = utils.Cloud(cost_per_mip = 30, data_rate = 15)
    edge_list = [
        utils.Edge(task_concurrent_capacity=1,
                   process_data_rate=process_data_rate_list[i],
                   upload_data_rate= 3+ 3*random.random(),cost_per_mip = cost_per_mip_list[i]) for i in range(edge_number)
    ]

    _cost_per_mip_list = []
    _performance_list = []
    _cost_per_mip_list.append(("cloud",cloud.cost_per_mip))
    _performance_list.append(("cloud",cloud.process_data_rate))
    for _edge_index,_edge_node in enumerate(edge_list):
        _cost_per_mip_list.append((_edge_index,_edge_node.cost_per_mip))
        _performance_list.append((_edge_index,_edge_node.process_data_rate))
    with open("edge_cloud_configuration.txt","w") as f:
        f.write("_cost_per_mip_list:\n{0} \n".format(_cost_per_mip_list))
        f.write("_performance_list:\n{0} \n".format(_performance_list))

    total_application_weight = 0
    total_deadline = 0
    for application_index,_application in enumerate(application_list):
        utils.set_tmax(_application,edge_list,cloud)
        if len(base_deadline) > 0:
            _application.deadline = math.ceil((1+deadline_alpha) * base_deadline[application_index])
        else:
            _application.deadline = math.ceil(deadline_alpha * _application.tmax)
        total_deadline += _application.deadline
        _app_total_weight = sum([_application.task_graph.nodes[_t]["w"] for _t in _application.task_graph.nodes()])
        total_application_weight += _app_total_weight

    edge_weight=(sum([_edge.upload_data_rate for _edge in edge_list])*(edge_number-1)+cloud.data_rate*2*edge_number)/((edge_number+1)*(edge_number+1))
    node_weight=(sum([_edge.process_data_rate for _edge in edge_list])+cloud.process_data_rate)/(edge_number+1)

    unscheduled_tasks = pqdict.pqdict()
    remain_length_list = []

    # schedule each application
    for _release_index,_release_time in enumerate(release_time_list):
        interested_time_st = time.time()

        #update self.planed_is_task_for_each_time, self.planed_task_for_each_time, update self.is_task_for_each_time, and self.task_for_each_time
        for _edge_index,_edge_node in enumerate(edge_list):
            _edge_node.generate_plan(_release_time)

        # current application and add unscheduled tasks
        _application = application_list[_release_index]
        for i in range(_application.task_graph.number_of_nodes()):
            if (_release_index,i) not in unscheduled_tasks:
                unscheduled_tasks[(_release_index,i)] = _release_time
        
        interested_time[0] += time.time() - interested_time_st
        interested_time_st = time.time()

        # add remain length -> customized list-based scheduling
        remain_length_list = utils.get_remain_length(_application.task_graph,edge_weight=edge_weight,node_weight=node_weight)
        for _t in _application.task_graph.nodes():
            _application.task_graph.nodes()[_t]["current_remain_length"] = remain_length_list[_t]
            if _application.dynamic_longest_remain_length < remain_length_list[_t]:
                _application.dynamic_longest_remain_length = remain_length_list[_t]

        # generate sub_deadline for each task
        sub_deadline_list = utils.get_sub_deadline_list(_application.task_graph,remain_length_list,deadline = _application.deadline,edge_weight=edge_weight,node_weight=node_weight)
        start_sub_deadline_list = utils.get_start_sub_deadline_list(_application.task_graph,remain_length_list,deadline = _application.deadline)
        
        # print(sub_deadline_list)

        resized_start_sub_deadline_list = resize_deadline(start_sub_deadline_list,sub_deadline_list,_application,edge_list)
        start_sub_deadline_list2 = utils.get_start_sub_deadline_list_pcp(_application.task_graph,remain_length_list,deadline = _application.deadline,edge_weight=edge_weight,node_weight=node_weight)
        
        # print(_release_index)
        # print("resized_start_sub_deadline_list:",[(i,j) for i,j in enumerate(resized_start_sub_deadline_list)])
        # print("start_sub_deadline_list:",[(i,j) for i,j in enumerate(start_sub_deadline_list)])
        # print("compare:",[resized_start_sub_deadline_list[i]>start_sub_deadline_list[i]  for i,j in enumerate(resized_start_sub_deadline_list)])

        for _t in _application.task_graph.nodes():
            _application.task_graph.nodes()[_t]["sub_deadline"] = sub_deadline_list[_t]
            _application.task_graph.nodes()[_t]["start_sub_deadline"] = min(start_sub_deadline_list2[_t],start_sub_deadline_list[_t],resized_start_sub_deadline_list[_t]) #min(start_sub_deadline_list[_t],resized_start_sub_deadline_list[_t])   #min(start_sub_deadline_list[_t],resized_start_sub_deadline_list[_t]) #start_sub_deadline_list[_t] #0   #resized_start_sub_deadline_list[_t] 

        interested_time[1] += time.time() - interested_time_st
        
        # schedule ready task > urgent application > remain length longer
        unscheduled_tasks_sorted_by_remain_length = pqdict.pqdict()
        for _ap_index,_ta_index in unscheduled_tasks.keys():
            if (_ap_index,_ta_index) not in unscheduled_tasks_sorted_by_remain_length:
                can_be_scheduled_label = 0
                for u, v in application_list[_ap_index].task_graph.in_edges(
                        _ta_index):
                    if application_list[_ap_index].task_graph.nodes[u]["is_scheduled"] == 0:
                        can_be_scheduled_label += 1
                unscheduled_tasks_sorted_by_remain_length[(_ap_index,_ta_index)] = (can_be_scheduled_label,application_list[_ap_index].deadline+application_list[_ap_index].release_time,10000000)
                # unscheduled_tasks_sorted_by_remain_length[(_ap_index,_ta_index)] = (can_be_scheduled_label,application_list[_ap_index].dynamic_longest_remain_length,0-application_list[_ap_index].task_graph.nodes()[_ta_index]["current_remain_length"])
                application_list[_ap_index].task_graph.nodes()[_ta_index]["is_scheduled_in_this_scheduling"] = 0

        #just for record
        total_len_unscheduled_tasks_list.append(len(unscheduled_tasks))

        # start scheduling
        while(len(unscheduled_tasks_sorted_by_remain_length) > 0):
            st = time.time()
            # select task with the least remain length
            _ap_index, selected_task_index = unscheduled_tasks_sorted_by_remain_length.top()

            _can_be_scheduled_label, application_level_length, task_level_length = unscheduled_tasks_sorted_by_remain_length[(_ap_index, selected_task_index)]
            unscheduled_tasks_sorted_by_remain_length.pop()

            _current_application = application_list[_ap_index]

            interested_time_st = time.time()
            if selected_task_index == 0 or selected_task_index == _current_application.task_graph.number_of_nodes(
            ) - 1:
                # the source and sink
                selected_node = _current_application.release_node
            else:
                # selecting node based on the scheduler
                is_in_deadline,selected_node = scheduler(
                    selected_task_index, _current_application, edge_list, cloud, _release_time)
            interested_time[2] += time.time() - interested_time_st

            if selected_node == edge_number:
                # schedule to the cloud
                precedence_task_finish_time = []
                for u, v in _current_application.task_graph.in_edges(selected_task_index):
                    precedence_task_node = _current_application.task_graph.nodes[u][
                        "selected_node"]
                    bandwidth = utils.get_bandwidth(precedence_task_node,selected_node,edge_list,cloud)
                    precedence_task_finish_time.append(max(_current_application.task_graph.edges[u, v]["e"] *bandwidth +_current_application.task_graph.nodes[u]["finish_time"],_release_time))
                
                # globally earliest start time is _current_application.release_time
                earliest_start_time = max(precedence_task_finish_time) if len(
                    precedence_task_finish_time
                ) > 0 else _release_time

                # set_latest_change_time
                _current_application.set_latest_change_time(selected_node,selected_task_index,edge_list,cloud,earliest_start_time)

                # only schedule once
                unscheduled_tasks[(_ap_index,selected_task_index)] = _release_time

                # set start time and node for each task
                _current_application.set_start_time(selected_task_index, selected_node,
                                            earliest_start_time, -1)
                _current_application.task_graph.nodes[selected_task_index][
                    "finish_time"] = earliest_start_time + _current_application.task_graph.nodes[
                        selected_task_index]["w"] * cloud.process_data_rate
            else:
                # schedule to the edge
                precedence_task_finish_time = []
                for u, v in _current_application.task_graph.in_edges(
                        selected_task_index):
                    precedence_task_node = _current_application.task_graph.nodes[u][
                        "selected_node"]
                    bandwidth = utils.get_bandwidth(precedence_task_node,selected_node,edge_list,cloud)
                    precedence_task_finish_time.append(_current_application.task_graph.edges[u, v]["e"] *bandwidth + max(_current_application.task_graph.nodes[u]["finish_time"],_release_time))

                # globally earliest start time is _current_application.release_time
                earliest_start_time = _release_time if len(
                    precedence_task_finish_time) == 0 else max(
                        precedence_task_finish_time)

                if earliest_start_time< _release_time:
                    print(earliest_start_time,_release_time,"error")
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
                    print(actual_start_time,_release_time,"error")
                    quit()

                # set_latest_change_time
                _current_application.set_latest_change_time(selected_node,selected_task_index,edge_list,cloud,actual_start_time)

                # update scheduled tasks' latest schedule time
                unscheduled_tasks[(_ap_index,selected_task_index)] = _release_time

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

            # update can be scheduled label for other
            for u, v in application_list[_ap_index].task_graph.out_edges(selected_task_index):
                if (_ap_index,v) in unscheduled_tasks_sorted_by_remain_length:
                    _a, _b, _c = unscheduled_tasks_sorted_by_remain_length[(_ap_index,v)]
                    _c = min(_c,0-(_current_application.task_graph.nodes[selected_task_index]["finish_time"]+ _current_application.task_graph.edges[u,v]["e"]*edge_list[selected_node].upload_data_rate-_current_application.release_time-_current_application.task_graph.nodes[v]["start_sub_deadline"]))

                    unscheduled_tasks_sorted_by_remain_length[(_ap_index,v)] = (_a-1, _b, _c)

            if selected_task_index == _current_application.task_graph.number_of_nodes() - 1:
                if _current_application.task_graph.nodes[selected_task_index]["finish_time"]- _current_application.release_time <= _current_application.deadline:
                    _current_application.is_accept = True

                # calculate the flexible time for each task
                if _current_application.is_accept:
                    for _n in _current_application.task_graph.nodes():
                        _latest_finish_time = math.inf
                        for _,v in _current_application.task_graph.out_edges(_n):
                            bandwidth = utils.get_bandwidth(_current_application.task_graph.nodes[_n]["selected_node"],_current_application.task_graph.nodes[v]["selected_node"],edge_list,cloud)
                            _lft = _current_application.task_graph.nodes[v]["start_time"]-_current_application.task_graph.edges[_n,v]["e"]*bandwidth
                            _latest_finish_time = min(_latest_finish_time,_lft)
                        if _latest_finish_time < math.inf:
                            _current_application.task_graph.nodes[_n]["flexible_time"] = _latest_finish_time- _current_application.task_graph.nodes[_n]["finish_time"]
                        else:
                            _current_application.task_graph.nodes[_n]["flexible_time"] = _current_application.deadline - _current_application.task_graph.nodes[_n]["finish_time"]

            _find_time = time.time()-st
            each_application_time += _find_time
            
        # if not application_list[_release_index].is_accept:
            #进行有关cloud的调度
            # re = utils.schedule_with_cloud(application_list[_release_index],edge_list,cloud)


        # utils.adjust_task_start_finish_time(application_list[_release_index],edge_list,cloud)

        # utils.adjust(application_list[_release_index],edge_list,cloud)

        new_finish_task_set = [np.empty((0,4)) for i in range(edge_number)]
        while (len(unscheduled_tasks) > 0):
            ((_ap_index,_ta_index),_latest_transmission_time) = unscheduled_tasks.topitem()
            application_list[_ap_index].task_graph.nodes[_ta_index]["is_scheduled"] = 1
            unscheduled_tasks.pop()
            _current_task_selected_node = application_list[_ap_index].task_graph.nodes()[_ta_index]["selected_node"]
            if  _current_task_selected_node < edge_number and application_list[_ap_index].is_accept:
                finish_task_set[_current_task_selected_node] = np.vstack([finish_task_set[_current_task_selected_node],np.array([_ap_index,_ta_index,application_list[_ap_index].task_graph.nodes()[_ta_index]["start_time"],application_list[_ap_index].task_graph.nodes()[_ta_index]["finish_time"]])])
                new_finish_task_set[_current_task_selected_node] = np.vstack([new_finish_task_set[_current_task_selected_node],np.array([_ap_index,_ta_index,application_list[_ap_index].task_graph.nodes()[_ta_index]["start_time"],application_list[_ap_index].task_graph.nodes()[_ta_index]["finish_time"]])])
        
        for _edge_index,_edge_node in enumerate(edge_list):
            if is_multiple and new_finish_task_set[_edge_index].shape[0]>0:
                _edge_node.update_plan_to_actural(_release_time,new_finish_task_set[_edge_index],application_list)
        
        # print("application:{0},is_accept:{1}".format(_release_index,application_list[_release_index].is_accept))

    utils.check(is_multiple, application_list, edge_list, cloud)
    
    # output interval_statistical for each edge node
    for _edge_index,_edge_node in enumerate(edge_list):
        finish_task_set[_edge_index] = finish_task_set[_edge_index][finish_task_set[_edge_index][:,3].argsort()]

    for _application in application_list:
        # finish the last node
        sink_task_index =  _application.task_graph.number_of_nodes() - 1
        _application.finish_time = _application.task_graph.nodes[sink_task_index]["finish_time"]

    # output accept rate and total cost
    accept_rate = sum([_application.is_accept for _application in application_list])/len(application_list)

    total_cost = 0
    cost_per_mip_list.append(cloud.cost_per_mip)
    average_deadline_utilization = 0
    cost_list = []
    for _application in application_list:
        _cost_application = sum([_application.task_graph.nodes()[_t]["w"]*cost_per_mip_list[_application.task_graph.nodes()[_t]["selected_node"]] for _t in  _application.task_graph.nodes()])
        cost_list.append(_cost_application)
        if _application.is_accept:
            total_cost += _cost_application
            average_deadline_utilization += (_application.finish_time - _application.release_time)/_application.deadline


    if is_draw:
        draw.draw_gantt(application_list, edge_list, cloud, is_annotation=is_annotation, is_only_accept = True, gantt_name = scheduler.__name__)

    _cost_per_mip_list = []
    _performance_list = []
    _cost_per_mip_list.append(("cloud",cloud.cost_per_mip))
    _performance_list.append(("cloud",cloud.process_data_rate))
    for _edge_index,_edge_node in enumerate(edge_list):
        _cost_per_mip_list.append((_edge_index,_edge_node.cost_per_mip))
        _performance_list.append((_edge_index,_edge_node.process_data_rate))

    # print(time.time()-all_st)
    # print(interested_time)
    print("total_application_weight:{0}".format(total_application_weight))
    print("edge_weight:{0}, node_weight:{1}".format(edge_weight,node_weight))
    print("average_deadline_utilization:{0}".format(average_deadline_utilization/(accept_rate*len(application_list))))
    # print("cost_list")
    # print(cost_list)
    # print("accept_list")
    # print([_application.is_accept for _application in application_list])
    return(accept_rate,total_cost,application_list)

if __name__ == '__main__':
    resultfile = ''
    try:
      opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
    for opt, arg in opts:
        if opt in ("-o", "--ofile"):
            resultfile = arg
         
    is_draw = False
    is_annotation = True
    is_draw_task_graph = False
    application_num = 200
    application_average_interval = 120
    edge_number = 20
    random_seed = 1.2
    is_multiple = False
    deadline_alpha = 0.2

    for application_average_interval in range(150,550,400):
        for deadline_alpha in range(10):
            deadline_alpha = 0 + 0.05*deadline_alpha
            exp_num = 1
            a_list = [0,0]
            c_list = [0,0]
            for _exp_num in range(exp_num):
                random_seed = 1+0.1*_exp_num
                _a, _c, application_list = deadline_constrained_scheduling_only_accept_case_study_8_2.re_scheduling(
                    is_draw=is_draw,
                    is_annotation=is_annotation,
                    application_num=application_num,
                    application_average_interval=application_average_interval,
                    edge_number=edge_number,
                    scheduler=utils.get_node_with_earliest_finish_time_without_cloud,
                    random_seed=random_seed,
                    is_draw_task_graph=is_draw_task_graph,
                    is_multiple=False,
                    deadline_alpha=deadline_alpha,
                    base_deadline = [])
                a_list[0]+=_a
                c_list[0]+=_c

                base_deadline = []
                for a in application_list:
                    _sink = a.task_graph.number_of_nodes()-1
                    base_deadline.append(a.task_graph.nodes[_sink]["finish_time"]-a.release_time)
                    
                _a, _c, _ = re_scheduling(
                        is_draw=is_draw,
                        is_annotation=is_annotation,
                        application_num=application_num,
                        application_average_interval=application_average_interval,
                        edge_number=edge_number,
                        scheduler=utils.
                        get_node_with_least_cost_constrained_by_start_subdeadline_without_cloud,
                        random_seed=random_seed,
                        is_draw_task_graph=is_draw_task_graph,
                        is_multiple=is_multiple,
                        deadline_alpha=deadline_alpha,
                        base_deadline = base_deadline)
                a_list[1]+=_a
                c_list[1]+=_c
            print("deadline_alpha",deadline_alpha)
            print("application_average_interval",application_average_interval)
            print([i/exp_num for i in a_list])
            print([i/exp_num for i in c_list])
            print([c_list[i]/(a_list[i]+0.00000000001) for i in range(len(a_list))])
            # quit()
            print()
            if resultfile != '':
                with open(resultfile,mode="a") as f:
                    f.write("deadline_alpha:{0}\n".format(deadline_alpha))
                    f.write("application_average_interval:{0}\n".format(application_average_interval))
                    f.write(str([i/exp_num for i in a_list])+"\n")
                    f.write(str([i/exp_num for i in c_list])+"\n")
                    f.write(str([c_list[i]/(a_list[i]+0.00000000001) for i in range(len(a_list))])+"\n")

