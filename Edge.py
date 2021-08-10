'''
Author: 娄炯
Date: 2021-08-04 11:07:37
LastEditors: loujiong
LastEditTime: 2021-08-09 23:48:39
Description: 
Email:  413012592@qq.com
'''
import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import math
import numpy as np
np.set_printoptions(suppress=True)

class Edge():
    def __init__(self, task_concurrent_capacity, process_data_rate,
                 upload_data_rate, cost_per_mip = 1, edge_index = 1):
        # set 10000 time slots first
        self.task_concurrent_capacity = task_concurrent_capacity
        self.planed_start_finish = [np.array([[0.0,10000000000000,-1,-1,-1,-1]]) for i in range(self.task_concurrent_capacity)]
        self.start_finish = [np.array([[0.0,10000000000000,-1,-1,-1,-1]]) for i in range(self.task_concurrent_capacity)]
        self.process_data_rate = process_data_rate
        self.upload_data_rate = upload_data_rate
        self.cost_per_mip = cost_per_mip
        self.assigned_task_list = [np.empty((0,5)) for i in range(self.task_concurrent_capacity)]
        self.assigned_task_since_release_time = [0 for i in range(self.task_concurrent_capacity)]
        self.planed_update_start_finish_list = {}
        self.edge_index = edge_index

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

    def find_actual_earliest_start_time_by_planed_modify(self, start_time, runtime, _release_time, application_list):
        '''
        @Description: 

        @Parameters: 
        start_time, runtime, deadline, _release_time, application_list
        @Returns: 
        selected_cpu,selected_a,selected_t,min_start_time
        '''        
        selected_cpu = -1
        min_start_time = math.inf
        selected_a = -1
        selected_t = -1
        # print(self.planed_start_finish[0])
        # print(self.assigned_task_since_release_time[0])
        for _cpu in range(self.task_concurrent_capacity):
            # for sink and source tasks or empty tasks
            if runtime == 0:
                return (0, -1, -1,start_time) 

            selected_a_cpu = -1
            selected_t_cpu = -1
            min_start_time_cpu = math.inf

            # print(self.assigned_task_since_release_time[_cpu])
            for _t_item in self.assigned_task_since_release_time[_cpu]:
                _a,_t = int(_t_item[0]),int(_t_item[1])
                # find the interval before
                _index = (self.planed_start_finish[_cpu][:,4]==_a) & (self.planed_start_finish[_cpu][:,5]==_t)
                interval_before = self.planed_start_finish[_cpu][_index,:]
                
                # find the interval after
                _index = (self.planed_start_finish[_cpu][:,2]==_a) & (self.planed_start_finish[_cpu][:,3]==_t)
                interval_after =  self.planed_start_finish[_cpu][_index,:]
                
                if interval_before.shape[0] > 1 or interval_after.shape[0] > 1:
                    print("interval after before error")
                    quit()

                if interval_before.shape[0] == 1:
                    _start =  interval_before[0][0]
                else:
                    _start = _t_item[2]
                
                if interval_after.shape[0] == 1:
                    _finish = interval_after[0][1]
                else:
                    _finish = _t_item[3]

                _f = min(_finish,application_list[_a].task_graph.nodes[_t]["flexible_time"]) 
                _s = max(start_time,_start)

                if _f -_s >= application_list[_a].task_graph.nodes[_t]["finish_time"] - application_list[_a].task_graph.nodes[_t]["start_time"] + runtime:
                    if min_start_time_cpu > _s:
                        min_start_time_cpu = _s
                        selected_a_cpu = _a
                        selected_t_cpu = _t
        
            # no task modified
            is_in_interval = self.planed_start_finish[_cpu][:,1] - np.maximum(start_time,self.planed_start_finish[_cpu][:,0]) >= runtime
            st_time = (1 - is_in_interval)* 1000000000000000 + np.maximum(start_time,self.planed_start_finish[_cpu][:,0])
            interval_key = np.argmin(st_time)

            
            _min_start_time_cpu = max(start_time,self.planed_start_finish[_cpu][interval_key][0])
            # print(_min_start_time_cpu)
            if  min_start_time_cpu > _min_start_time_cpu:
                min_start_time_cpu = _min_start_time_cpu
                selected_a_cpu = -1
                selected_t_cpu = -1
            
            if min_start_time > min_start_time_cpu:
                selected_a = selected_a_cpu
                selected_t = selected_t_cpu
                selected_cpu = _cpu
                min_start_time = min_start_time_cpu

        return(selected_cpu,selected_a,selected_t,min_start_time)

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

    def set_cpu_state_by_planed_modified(self, _cpu, actual_start_time,
                                estimated_runtime, _application_index,
                                selected_index, modified_a_t):
        modified_a,modified_t = modified_a_t

        # do not set for virtual task
        if estimated_runtime != 0 and modified_a != -1:   
            # find the interval before
            _index = (self.planed_start_finish[_cpu][:,4]==modified_a) & (self.planed_start_finish[_cpu][:,5]==modified_t)
            interval_before = self.planed_start_finish[_cpu][_index,:]
            # find the interval after
            _index = (self.planed_start_finish[_cpu][:,2]==modified_a) & (self.planed_start_finish[_cpu][:,3]==modified_t)
            interval_after = self.planed_start_finish[_cpu][_index,:]
            if interval_before.shape[0] > 1 or interval_after.shape[0] > 1:
                print("interval after before error")
                quit()
            
            # delete interval_before and  interval_after
            _index = (self.planed_start_finish[_cpu][:,2]!=modified_a) | (self.planed_start_finish[_cpu][:,3]!=modified_t)
            self.planed_start_finish[_cpu] = self.planed_start_finish[_cpu][_index,:]

            _index = (self.planed_start_finish[_cpu][:,4]!=modified_a) | (self.planed_start_finish[_cpu][:,5]!=modified_t)
            self.planed_start_finish[_cpu] = self.planed_start_finish[_cpu][_index,:]

            # add modified interval
            # modify before interval if exists
            if interval_before.shape[0] == 1:
                if actual_start_time > interval_before[0][0]:
                    self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([interval_before[0][0], actual_start_time ,interval_before[0][2],interval_before[0][3],_application_index,selected_index])])
            
            _index = (self.assigned_task_since_release_time[_cpu][:,0]==modified_a) & (self.assigned_task_since_release_time[_cpu][:,1]==modified_t)
            modified_item = self.assigned_task_since_release_time[_cpu][_index,:][0]
            modified_s = modified_item[2]
            modified_f = modified_item[3]
            if actual_start_time + estimated_runtime < modified_s:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([actual_start_time + estimated_runtime, modified_s , _application_index, selected_index, modified_a, modified_t])])
            elif actual_start_time + estimated_runtime > modified_s:
                modified_f += actual_start_time + estimated_runtime - modified_s
                modified_s = actual_start_time + estimated_runtime
                self.planed_update_start_finish_list[(modified_a,modified_t)] = (modified_s,modified_f)
                _index = (self.assigned_task_since_release_time[_cpu][:,0]==modified_a) & (self.assigned_task_since_release_time[_cpu][:,1]==modified_t)
                self.assigned_task_since_release_time[_cpu][_index,2] = modified_s
                self.assigned_task_since_release_time[_cpu][_index,3] = modified_f
            
            # modify before interval if exists
            if interval_after.shape[0] == 1:
                if modified_f < interval_after[0][1]:
                    self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], np.array([modified_f, interval_after[0][1] , modified_a, modified_t, interval_after[0][4], interval_after[0][5]])])
        # print(_cpu, actual_start_time,
                                # estimated_runtime, _application_index,
                                # selected_index, modified_a_t)
        if estimated_runtime != 0 and modified_a == -1:
            index = np.where((self.planed_start_finish[_cpu][:,0] <= actual_start_time)&(self.planed_start_finish[_cpu][:,1] >= actual_start_time+estimated_runtime)) 
            index = index[0][0]
            # print(index)

            _s = self.planed_start_finish[_cpu][index][0]
            add_item = np.array([_s, actual_start_time , self.planed_start_finish[_cpu][index][2], self.planed_start_finish[_cpu][index][3], _application_index, selected_index])

            if actual_start_time+estimated_runtime <= self.planed_start_finish[_cpu][index][1]:
                # print("before",self.planed_start_finish[_cpu][index][0])
                self.planed_start_finish[_cpu][index][0]= actual_start_time+estimated_runtime
                self.planed_start_finish[_cpu][index][2]= _application_index
                self.planed_start_finish[_cpu][index][3]= selected_index
                # print(actual_start_time+estimated_runtime,self.planed_start_finish[_cpu][index][0])
            else:
                self.planed_start_finish[_cpu] = np.delete(self.planed_start_finish[_cpu], index, 0)
            # print("ffff",self.planed_start_finish[_cpu][index])
            # print(_s,actual_start_time)
            if _s < actual_start_time:
                self.planed_start_finish[_cpu] = np.vstack([self.planed_start_finish[_cpu], add_item])
            
    def update_plan_to_actural(self, new_finish_task_set,
                               application_list,_release_time):
        for item in new_finish_task_set:

            _ap_index, _ta_index, start_time, finish_time = [i for i in item]
            _ap_index = int(_ap_index)
            _ta_index = int(_ta_index)
            cpu = application_list[_ap_index].task_graph.nodes()[_ta_index]["cpu"]

            if start_time == finish_time:
                continue
            cpu = application_list[_ap_index].task_graph.nodes()[_ta_index]["cpu"]
            self.assigned_task_list[cpu] = np.vstack([self.assigned_task_list[cpu],np.array([_ap_index, _ta_index, start_time, finish_time, application_list[_ap_index].task_graph.nodes[_ta_index]["flexible_time"]])])
            
        # first delete
        for _cpu in range(self.task_concurrent_capacity):
            self.start_finish[_cpu] = self.start_finish[_cpu][self.start_finish[_cpu][:,1]<_release_time,:]

        # then copy from the planed
        for _cpu in range(self.task_concurrent_capacity):
            self.start_finish[_cpu] =  np.vstack([self.start_finish[_cpu],self.planed_start_finish[_cpu]]) 
        
        for modified_a_t in self.planed_update_start_finish_list:
            modified_a,modified_t = modified_a_t
            modified_s,modified_f = self.planed_update_start_finish_list[modified_a_t]
            application_list[modified_a].task_graph.nodes[modified_t]["start_time"] = modified_s
            application_list[modified_a].task_graph.nodes[modified_t]["finish_time"] = modified_f
            _cpu = application_list[modified_a].task_graph.nodes[modified_t]["cpu"]

            _index = (self.assigned_task_list[_cpu][:,0]==modified_a) & (self.assigned_task_list[_cpu][:,1]==modified_t)
            self.assigned_task_list[_cpu][_index,2] = modified_s
            self.assigned_task_list[_cpu][_index,3] = modified_f
             
    def generate_plan(self, _release_time):
        self.planed_start_finish = [0 for i in range(self.task_concurrent_capacity)]
        self.assigned_task_since_release_time = [0 for i in range(self.task_concurrent_capacity)]
        for _cpu in range(self.task_concurrent_capacity):
            self.planed_start_finish[_cpu] = self.start_finish[_cpu].copy()
            self.planed_start_finish[_cpu] = self.planed_start_finish[_cpu][self.planed_start_finish[_cpu][:,1]>=_release_time,:] 
            self.assigned_task_since_release_time[_cpu] = self.assigned_task_list[_cpu].copy()
            self.assigned_task_since_release_time[_cpu] = self.assigned_task_since_release_time[_cpu][self.assigned_task_since_release_time[_cpu][:,2]>= _release_time,:]
        self.planed_update_start_finish_list = {}

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