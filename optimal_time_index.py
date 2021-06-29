'''
Author: 娄炯
Date: 2021-06-21 15:48:09
LastEditors: loujiong
LastEditTime: 2021-06-22 20:55:55
Description: 
Email:  413012592@qq.com
'''

import pseudoflow
import networkx as nx
import queue
import docplex.mp.model as cpx
import cplex
import numpy as np
from cplex.exceptions import CplexError
 
def completion_time_docplex1(TaskMatrix,EdgeMatrix,TaskEdgeRunTime,MaxTime,DeadLine,Cost,ReleaseEdge):
    '''
    @description: 利用cplex计算deadline constrained情况下 cost minimization的方案。
    @param {TaskMatrix,EdgeMatrix,TaskEdgeRunTime,MaxTime,DeadLine,Cost,ReleaseEdge}
    @return {sol}
    '''    
    opt_model = cpx.Model(name="MIP Model")
    opt_model.parameters.lpmethod =  2
    numOfTask= len(TaskMatrix[0])  #task数量
    numOfEdge = len(EdgeMatrix[0])  #edge node数量      
    # xNameList = [[[] for j in range(numOfEdge)] for i in range(numOfTask)] # shape: numOfEdge * numOfTask
    # nVariable = [[[] for j in range(numOfEdge)] for i in range(numOfTask)] # shape: numOfEdge * numOfTask
    # tNameList = [[[[0 for j2 in range(numOfEdge)] for i2 in range(numOfTask)] for j1 in range(numOfEdge)] for i1 in range(numOfTask)]   
    # tVariable = [[[[] for i2 in range(numOfTask)] for j1 in range(numOfEdge)] for i1 in range(numOfTask)]
    # aNameList = [[] for i in range(numOfTask)] # shape: numOfTask
    # aVariable = [[] for i in range(numOfTask)] # shape: numOfTask    
    # ct_assembly = []   #constraint list  
    FNameList = []
    FVariable = []
    aNameList = [[] for i in range(numOfTask)] # shape: numOfTask
    aVariable = [[] for i in range(numOfTask)] # shape: numOfTask
    xNameList = [[] for i in range(numOfTask)]
    xVariable = [[] for i in range(numOfTask)]
    #---------------------------generate variable----------------------------
    var_cnt = 0
    for i in range(numOfTask):
        FName = "F_{0}".format(i)
        FNameList.append(FName)  
        FVariable.append(opt_model.continuous_var(lb = 0,name = FName)) 
        var_cnt += 1
        
    for i in range(numOfTask):
        for j in range(numOfEdge):
            aName="a_{0}_{1}".format(i,j)
            aNameList[i].append(aName)
            aVariable[i].append(opt_model.continuous_var(0,1,aName))
            var_cnt += 1

    for i1 in range(numOfTask):
        for i2 in range(numOfTask):
            xName="x_{0}_{1}".format(i1,i2)
            xNameList[i1].append(xName)
            xVariable[i1].append(opt_model.continuous_var(lb=0, ub=1, name = xName))
            var_cnt += 1
    #---------------------------generate variable----------------------------
    # k 时刻task i在edge j上是否结束
    # for i in range(numOfTask):
    #     for j in range(numOfEdge):
    #         for k in range(MaxTime): #MaxTime time index 最大的数量
    #             xName="x_{0}_{1}_{2}".format(i,j,k)       
    #             xNameList[i][j].append(xName)
                
    # # task i是否在edge j上运行
    # for i in range(numOfTask):
    #     for j in range(numOfEdge):
    #         aName="a_{0}_{1}".format(i,j)
    #         aNameList[i].append(aName)
    
    # for i1 in range(numOfTask):
    #     for j1 in range(numOfEdge):
    #         for i2 in range(numOfTask):
    #             for j2 in range(numOfEdge):
    #                 tName = "t_{0}_{1}_{2}_{3}".format(i1,j1,i2,j2)
    #                 tNameList[i1][j1][i2][j2] = tName 

    # var_cnt = 0
    # # 0-1 变量 x
    # for i in range(numOfTask):
    #     for j in range(numOfEdge):
    #         for k in range(MaxTime):
    #             nVariable[i][j].append(opt_model.continuous_var(0,1,xNameList[i][j][k]))    #结束时间variable binary_var
    #             var_cnt += 1

    # # 0-1 变量 alpha
    # for i in range(numOfTask):
    #     for j in range(numOfEdge):
    #        aVariable[i].append(opt_model.continuous_var(0,1,aNameList[i][j]))     
               

    # for i1 in range(numOfTask):
    #     for j1 in range(numOfEdge):
    #         for i2 in range(numOfTask):
    #             for j2 in range(numOfEdge):
    #                 tVariable[i1][j1][i2].append(opt_model.continuous_var(0,1,tNameList[i1][j1][i2][j2]))   
    #                 var_cnt += 1     

    print("var_cnt:{0}".format(var_cnt))
    #---------------------------generate constraints----------------------------
    # 限制条件(1) task的数量等于任意时刻结束的task的数量：
    # for i in range(numOfTask):
    #     for j in range(numOfEdge):
    #         ct_assembly.append(opt_model.add_constraint(opt_model.sum(nVariable[i][j][k] for k in range(MaxTime)) == aVariable[i][j]))

    # (2) Application要在结束时间前完成：
    # ct_assembly.append(opt_model.add_constraint(opt_model.sum(nVariable[numOfTask-1][j][k]*k for k in range(MaxTime) for j in range(numOfEdge)) <= DeadLine))
    opt_model.add_constraint(FVariable[numOfTask-1] <= DeadLine)


    # (3) 每个task只能运行一次：
    # for i in range(numOfTask):
    #     ct_assembly.append(opt_model.add_constraint(opt_model.sum(aVariable[i][j] for j in range(numOfEdge)) == 1))
    opt_model.add_constraints(opt_model.sum(aVariable[i][j] for j in range(numOfEdge)) == 1 for i in range(numOfTask))


    opt_model.add_constraints(xVariable[i1][i2]+xVariable[i2][i1] <= 1 for i1 in range(numOfTask) for i2 in range(numOfTask) if i1 != i2)

    # (4) 在任意时间片(s-1,s]，每个edge上最多只能有一个task：
    opt_model.add_constraints(FVariable[i1]-FVariable[i2])
    # for j in range(numOfEdge):
    #     for k in range(MaxTime):
    #         ct_assembly.append(opt_model.add_constraint(opt_model.sum(nVariable[i][j][s] for i in range(1,numOfTask-1) for s in range(k,min(MaxTime,k+TaskEdgeRunTime[i][j]))) <= 1)) #TaskEdgeRunTime 为task i在edge j上运行的时间
    #         print(j,k)
            # for i in range(1,numOfTask-1):
            #     for s in range(k,min(MaxTime,k+TaskEdgeRunTime[i][j])):
            #         # print(range(k,min(MaxTime,k+TaskEdgeRunTime[i][j])))
            #         # print(k+TaskEdgeRunTime[i][j]-1)
            #         print(i,j,s)

    print("fuckfuckfuckfuckfuckfuckfuckfuckfuckfuck")
    # (5) Task之间的precedence限制
    for i1 in range(numOfTask):
        for i2 in range(numOfTask):
            if(TaskMatrix[i1][i2] > 0):
                ct_assembly.append(opt_model.add_constraint(opt_model.sum(nVariable[i1][j][k]*k for k in range(MaxTime) for j in range(numOfEdge))  #i1的结束时间
                + opt_model.sum(TaskEdgeRunTime[i2][j]*aVariable[i2][j] for j in range(numOfEdge))    #i2的运行时间
                + opt_model.sum(TaskMatrix[i1][i2]*EdgeMatrix[j1][j2]*tVariable[i1][j1][i2][j2] for j1 in range(numOfEdge) for j2 in range(numOfEdge))  #i1和i2的数据传输时间
                <= opt_model.sum(nVariable[i2][j][k]*k for k in range(MaxTime) for j in range(numOfEdge))))       #小于i2的结束时间
    
    for i1 in range(numOfTask):
        for i2 in range(numOfTask):
            if(TaskMatrix[i1][i2] > 0):
                for j1 in range(numOfEdge):
                    for j2 in range(numOfEdge):
                        ct_assembly.append(opt_model.add_constraint(tVariable[i1][j1][i2][j2] >= aVariable[i1][j1]+aVariable[i2][j2]-1))
                        
    # (6) 头尾task需要在同一个release edge上执行
    ct_assembly.append(opt_model.add_constraint(aVariable[0][ReleaseEdge] == 1))
    ct_assembly.append(opt_model.add_constraint(aVariable[numOfTask-1][ReleaseEdge] == 1))
    
    # 目标
    print("fuckfuck")
    objective = opt_model.sum(TaskEdgeRunTime[i][j]*Cost[j]*aVariable[i][j] for i in range(numOfTask) for j in range(numOfEdge))
    opt_model.minimize(objective)    
    opt_model.print_information()
    sol = opt_model.solve(log_output=True)
    print("fuckfuck1111111111111111111111111111")
    # res = sol.get_all_values()
    print(sol)
    # for i in range(len(ct_assembly)):
    #     print(ct_assembly[i])
    #     print()
    return sol




def completion_time_docplex2(TaskMatrix,EdgeMatrix,TaskEdgeRunTime,MaxTime,DeadLine,Cost,ReleaseEdge):
    '''
    @description: 利用cplex计算deadline constrained情况下 cost minimization的方案。
    @param {TaskMatrix,EdgeMatrix,TaskEdgeRunTime,MaxTime,DeadLine,Cost,ReleaseEdge}
    @return {sol}
    '''    
    opt_model = cpx.Model(name="MIP Model")
    opt_model.parameters.lpmethod =  2
    numOfTask= len(TaskMatrix[0])  #task数量
    numOfEdge = len(EdgeMatrix[0])  #edge node数量      
    xNameList = [[[] for j in range(numOfEdge)] for i in range(numOfTask)] # shape: numOfEdge * numOfTask
    nVariable = [[[] for j in range(numOfEdge)] for i in range(numOfTask)] # shape: numOfEdge * numOfTask
    tNameList = [[[[0 for j2 in range(numOfEdge)] for i2 in range(numOfTask)] for j1 in range(numOfEdge)] for i1 in range(numOfTask)]   
    tVariable = [[[[] for i2 in range(numOfTask)] for j1 in range(numOfEdge)] for i1 in range(numOfTask)]
    aNameList = [[] for i in range(numOfTask)] # shape: numOfTask
    aVariable = [[] for i in range(numOfTask)] # shape: numOfTask    
    ct_assembly = []   #constraint list  

    #---------------------------generate variable----------------------------
    # k 时刻task i在edge j上是否结束
    for i in range(numOfTask):
        for j in range(numOfEdge):
            for k in range(MaxTime): #MaxTime time index 最大的数量
                xName="x_{0}_{1}_{2}".format(i,j,k)       
                xNameList[i][j].append(xName)
                
    # task i是否在edge j上运行
    for i in range(numOfTask):
        for j in range(numOfEdge):
            aName="a_{0}_{1}".format(i,j)
            aNameList[i].append(aName)
    
    for i1 in range(numOfTask):
        for j1 in range(numOfEdge):
            for i2 in range(numOfTask):
                for j2 in range(numOfEdge):
                    tName = "t_{0}_{1}_{2}_{3}".format(i1,j1,i2,j2)
                    tNameList[i1][j1][i2][j2] = tName 

    var_cnt = 0
    # 0-1 变量 x
    for i in range(numOfTask):
        for j in range(numOfEdge):
            for k in range(MaxTime):
                nVariable[i][j].append(opt_model.continuous_var(0,1,xNameList[i][j][k]))    #结束时间variable binary_var
                var_cnt += 1

    # 0-1 变量 alpha
    for i in range(numOfTask):
        for j in range(numOfEdge):
           aVariable[i].append(opt_model.continuous_var(0,1,aNameList[i][j]))     
               

    for i1 in range(numOfTask):
        for j1 in range(numOfEdge):
            for i2 in range(numOfTask):
                for j2 in range(numOfEdge):
                    tVariable[i1][j1][i2].append(opt_model.continuous_var(0,1,tNameList[i1][j1][i2][j2]))   
                    var_cnt += 1     

    print("var_cnt:{0}".format(var_cnt))
    print(aNameList)
    #---------------------------generate constraints----------------------------
    # 限制条件(1) task的数量等于任意时刻结束的task的数量：
    # for i in range(numOfTask):
    #     for j in range(numOfEdge):
    #         ct_assembly.append(opt_model.add_constraint(opt_model.sum(nVariable[i][j][k] for k in range(MaxTime)) == aVariable[i][j]))
    opt_model.add_constraints(opt_model.sum(nVariable[i][j][k] for k in range(MaxTime)) == aVariable[i][j] for i in range(numOfTask) for j in range(numOfEdge))

    
    # (2) Application要在结束时间前完成：
    ct_assembly.append(opt_model.add_constraint(opt_model.sum(nVariable[numOfTask-1][j][k]*k for k in range(MaxTime) for j in range(numOfEdge)) <= DeadLine))

    # (3) 每个task只能运行一次：
    for i in range(numOfTask):
        ct_assembly.append(opt_model.add_constraint(opt_model.sum(aVariable[i][j] for j in range(numOfEdge)) == 1))
    # print(MaxTime)
    
    # (4) 在任意时间片(s-1,s]，每个edge上最多只能有一个task：
    # for j in range(numOfEdge):
    #     for k in range(MaxTime):
    #         ct_assembly.append(opt_model.add_constraint(opt_model.sum(nVariable[i][j][s] for i in range(1,numOfTask-1) for s in range(k,min(MaxTime,k+TaskEdgeRunTime[i][j]))) <= 1)) #TaskEdgeRunTime 为task i在edge j上运行的时间
    #         print(j,k)
    # quit()
            # for i in range(1,numOfTask-1):
            #     for s in range(k,min(MaxTime,k+TaskEdgeRunTime[i][j])):
            #         # print(range(k,min(MaxTime,k+TaskEdgeRunTime[i][j])))
            #         # print(k+TaskEdgeRunTime[i][j]-1)
            #         print(i,j,s)

    # (5) Task之间的precedence限制
    # for i1 in range(numOfTask):
    #     for i2 in range(numOfTask):
    #         if(TaskMatrix[i1][i2] > 0):
    #             ct_assembly.append(opt_model.add_constraint(opt_model.sum(nVariable[i1][j][k]*k for k in range(MaxTime) for j in range(numOfEdge))  #i1的结束时间
    #             + opt_model.sum(TaskEdgeRunTime[i2][j]*aVariable[i2][j] for j in range(numOfEdge))    #i2的运行时间
    #             + opt_model.sum(TaskMatrix[i1][i2]*EdgeMatrix[j1][j2]*tVariable[i1][j1][i2][j2] for j1 in range(numOfEdge) for j2 in range(numOfEdge))  #i1和i2的数据传输时间
    #             <= opt_model.sum(nVariable[i2][j][k]*k for k in range(MaxTime) for j in range(numOfEdge))))       #小于i2的结束时间
    

    opt_model.add_constraints(opt_model.sum(nVariable[i][j][s] for i in range(1,numOfTask-1) for s in range(k,min(MaxTime,k+TaskEdgeRunTime[i][j]))) <= 1 for j in range(numOfEdge) for k in range(MaxTime))

    opt_model.add_constraints(opt_model.sum(nVariable[i1][j][k]*k for k in range(MaxTime) for j in range(numOfEdge))  #i1的结束时间
                + opt_model.sum(TaskEdgeRunTime[i2][j]*aVariable[i2][j] for j in range(numOfEdge))    #i2的运行时间
                + opt_model.sum(TaskMatrix[i1][i2]*EdgeMatrix[j1][j2]*tVariable[i1][j1][i2][j2] for j1 in range(numOfEdge) for j2 in range(numOfEdge))  #i1和i2的数据传输时间
                <= opt_model.sum(nVariable[i2][j][k]*k for k in range(MaxTime) for j in range(numOfEdge)) for i1 in range(numOfTask) for i2 in range(numOfTask) if TaskMatrix[i1][i2] > 0)
    # for i1 in range(numOfTask):
    #     for i2 in range(numOfTask):
    #         if(TaskMatrix[i1][i2] > 0):
    #             for j1 in range(numOfEdge):
    #                 for j2 in range(numOfEdge):
    #                     ct_assembly.append(opt_model.add_constraint(tVariable[i1][j1][i2][j2] >= aVariable[i1][j1]+aVariable[i2][j2]-1))
    opt_model.add_constraints(tVariable[i1][j1][i2][j2] >= aVariable[i1][j1]+aVariable[i2][j2]-1 for i1 in range(numOfTask) for i2 in range(numOfTask) for j1 in range(numOfEdge) for j2 in range(numOfEdge) if TaskMatrix[i1][i2] > 0)
                        
    # (6) 头尾task需要在同一个release edge上执行
    ct_assembly.append(opt_model.add_constraint(aVariable[0][ReleaseEdge] == 1))
    ct_assembly.append(opt_model.add_constraint(aVariable[numOfTask-1][ReleaseEdge] == 1))
    # ct_assembly.append(opt_model.add_constraint(nVariable[numOfTask-1][ReleaseEdge][MaxTime-1] == 1))
    
    # 目标
    print("fuckfuck")
    objective = opt_model.sum(TaskEdgeRunTime[i][j]*Cost[j]*aVariable[i][j] for i in range(numOfTask) for j in range(numOfEdge))
    opt_model.minimize(objective)    
    opt_model.print_information()
    sol = opt_model.solve(log_output=True)
    # res = sol.get_all_values()
    # print(sol)
    estimated_finish_time_list = []
    for i in range(numOfTask):
        # sol.get_value(
        estimated_finish_time = sum([sol.get_value(xNameList[i][j][k])*k for k in range(MaxTime) for j in range(numOfEdge)])
        estimated_finish_time_list.append(estimated_finish_time) 
        print("task {0}, estimated finish time:{1}".format(i,estimated_finish_time))

    # rescale
    estimated_finish_time_list = [i*MaxTime/estimated_finish_time_list[-1] for i in estimated_finish_time_list]
    # for i in range(len(ct_assembly)):
    #     print(ct_assembly[i])
    #     print()
    return estimated_finish_time_list

# TaskMatrix = [[0,1],[0,0]]
# EdgeMatrix = [[0,1],[1,0]]
# TaskEdgeRunTime = [[1,2],[1,1]]
# MaxTime = 5
# DeadLine = 3
# Cost = [3,2]
# res = completion_time_docplex2(TaskMatrix,EdgeMatrix,TaskEdgeRunTime,MaxTime,DeadLine,Cost)
# print(len(res))