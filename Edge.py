'''
Author: 娄炯
Date: 2021-08-04 11:07:37
LastEditors: loujiong
LastEditTime: 2021-08-04 11:07:37
Description: 
Email:  413012592@qq.com
'''
import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import math
import numpy as np

class Edge():
    def __init__(self, task_concurrent_capacity, process_data_rate,
                 upload_data_rate, cost_per_mip = 1):
        # set 10000 time slots first
        self.task_concurrent_capacity = task_concurrent_capacity
        self.planed_start_finish = [np.array([[0,10000000000000,-1,-1,-1,-1]]) for i in range(self.task_concurrent_capacity)]
        self.start_finish = [np.array([[0,10000000000000,-1,-1,-1,-1]]) for i in range(self.task_concurrent_capacity)]
        self.process_data_rate = process_data_rate
        self.upload_data_rate = upload_data_rate
        self.cost_per_mip = cost_per_mip
        self.assigned_task_list = [np.empty((0,5)) for i in range(self.task_concurrent_capacity)]
        self.assigned_task_since_release_time = [0 for i in range(self.task_concurrent_capacity)]
        self.last_release_time = -1

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
            _st = max(start_time,self.planed_start_finish[_cpu][interval_key][0])
            if min_start_time > _st:
                min_start_time = _st
                selected_interval_key = interval_key
                selected_cpu = _cpu
        return (min_start_time, selected_cpu, selected_interval_key)

    def find_actual_earliest_start_time_by_planed_modify(self, start_time, runtime, deadline, _release_time):
        selected_cpu = -1
        min_start_time = math.inf
        for _cpu in range(self.task_concurrent_capacity):
            if runtime == 0:
                return (start_time,0, 0) 
            for _t_item in self.assigned_task_since_release_time[_cpu]:
                pass

    def set_cpu_state_by_planed(self, _cpu, actual_start_time,
                                estimated_runtime, _application_index,
                                selected_index, selected_interval_key):
        # do not set for virtual task
        if estimated_runtime != 0:   
            _start = self.planed_start_finish[_cpu][selected_interval_key][0]
            _end = self.planed_start_finish[_cpu][selected_interval_key][1]
            _before_ap_index = self.planed_start_finish[_cpu][selected_interval_key][2]
            _before_ta_index = self.planed_start_finish[_cpu][selected_interval_key][3]
            _after_ap_index = self.planed_start_finish[_cpu][selected_interval_key][4]
            _after_ta_index = self.planed_start_finish[_cpu][selected_interval_key][5]
            min_end_time = actual_start_time + estimated_runtime - 1
            self.planed_start_finish[_cpu] = np.delete(self.planed_start_finish[_cpu], selected_interval_key, 0)
            if _end == 10000000000000:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([min_end_time + 1, _end,_application_index,selected_index,-1,-1])])
            elif _end > min_end_time:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([min_end_time + 1, _end,_application_index,selected_index,_after_ap_index,_after_ta_index])])
            if _start < actual_start_time:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([_start, actual_start_time - 1,_before_ap_index,_before_ta_index,_application_index,selected_index])])

    '''  
    def update_plan_to_actural(self, _release_time, new_finish_task_set,
                            application_list):
        for item in new_finish_task_set:
            _ap_index, _ta_index, start_time, finish_time = [int(i) for i in item]
            cpu = application_list[_ap_index].task_graph.nodes()[_ta_index]["cpu"]
            self.assigned_task_list[cpu] = np.vstack([self.assigned_task_list[cpu],np.array([_ap_index, _ta_index, start_time, finish_time, application_list[_ap_index].task_graph.nodes[_ta_index]["flexible_time"]])])
            
            if start_time == finish_time:
                continue
            
            # first find the item in self.start_finish[cpu]
            # print(self.start_finish[cpu])
            is_in_interval = (self.start_finish[cpu][:,0] <= start_time) * 1 + (self.start_finish[cpu][:,1] >= finish_time-1) * 1
            interval_key = np.argmax(is_in_interval)
            # print(interval_key)

            # then try to split the time interval 
            _start = self.start_finish[cpu][interval_key][0]
            _end = self.start_finish[cpu][interval_key][1]
            _before_ap_index = self.start_finish[cpu][interval_key][2]
            _before_ta_index = self.start_finish[cpu][interval_key][3]
            _after_ap_index = self.start_finish[cpu][interval_key][4]
            _after_ta_index = self.start_finish[cpu][interval_key][5]
            # delete time interval
            self.start_finish[cpu] = np.delete(self.start_finish[cpu], interval_key, 0)
            if _end == 10000000000000:
                self.start_finish[cpu] = np.vstack([self.start_finish[cpu], np.array([finish_time, _end,_ap_index,_ta_index,-1,-1])])
            elif _end > finish_time - 1:
                self.start_finish[cpu] = np.vstack([self.start_finish[cpu], np.array([finish_time, _end,_ap_index,_ta_index,_after_ap_index,_after_ta_index])])
            if _start < start_time:
                self.start_finish[cpu] = np.vstack([self.start_finish[cpu], np.array([_start, start_time - 1,_before_ap_index,_before_ta_index,_ap_index,_ta_index])])
        
            # print("self.start_finish",self.start_finish)
    '''

    def update_plan_to_actural(self, _release_time, new_finish_task_set,
                               application_list):
        for item in new_finish_task_set:
            _ap_index, _ta_index, start_time, finish_time = [int(i) for i in item]
            cpu = application_list[_ap_index].task_graph.nodes()[_ta_index]["cpu"]
            self.assigned_task_list[cpu] = np.vstack([self.assigned_task_list[cpu],np.array([_ap_index, _ta_index, start_time, finish_time, application_list[_ap_index].task_graph.nodes[_ta_index]["flexible_time"]])])
            
            if start_time == finish_time:
                continue
        # print("self.last_release_time",self.last_release_time)
        # first delete
        for _cpu in range(self.task_concurrent_capacity):
            self.start_finish[_cpu] = self.start_finish[_cpu][self.start_finish[_cpu][:,1]<self.last_release_time,:]

        # then copy from the planed
        for _cpu in range(self.task_concurrent_capacity):
            # print(self.start_finish[_cpu])
            # print(self.planed_start_finish[_cpu])
            self.start_finish[_cpu] =  np.vstack([self.start_finish[_cpu],self.planed_start_finish[_cpu]]) 
            # print(self.start_finish[_cpu])
            # print()
             
    def generate_plan(self, _release_time):
        self.last_release_time = _release_time
        self.planed_start_finish = [0 for i in range(self.task_concurrent_capacity)]
        self.assigned_task_since_release_time = [0 for i in range(self.task_concurrent_capacity)]
        for _cpu in range(self.task_concurrent_capacity):
            self.planed_start_finish[_cpu] = self.start_finish[_cpu].copy()
            self.planed_start_finish[_cpu] = self.planed_start_finish[_cpu][self.planed_start_finish[_cpu][:,1]>=_release_time,:] 
            self.assigned_task_since_release_time[_cpu] = self.assigned_task_list[_cpu]
            self.assigned_task_since_release_time[_cpu] = self.assigned_task_since_release_time[_cpu][self.assigned_task_since_release_time[_cpu][:,2]>= _release_time,:]


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