'''
Author: 娄炯
Date: 2021-09-07 18:34:35
LastEditors: loujiong
LastEditTime: 2021-09-09 11:12:53
Description: draw simulation results
Email:  413012592@qq.com
'''

import matplotlib.pyplot as plt
import numpy as np
import json


def draw_bar():
    # TODO open file
    # TODO read lines
    # TODO find interested data
    # TODO edit legend 
    x=np.arange(5)
    y1=[1,3,5,4,2]
    y2=[2,5,3,1,6]

    bar_width=0.3
    tick_label=['1','2','3','4','5']

    plt.bar(x,y1,bar_width,color='salmon',label='type A')
    plt.bar(x+bar_width,y2,bar_width,color='orchid',label='type B')

    plt.legend()
    plt.xticks(x+bar_width/2,tick_label)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('the length')
    plt.ylabel("f1") 
    plt.title("A simple plot")
    plt.show()
    return

def draw_line():
    # TODO open file
    # TODO read lines
    # TODO find interested data
    # TODO edit legend 
    names = [str(x) for x in list(range(8,21))]
    
    x = range(len(names))
    y_train = [0.840,0.839,0.834,0.832,0.824,0.831,0.823,0.817,0.814,0.812,0.812,0.807,0.805]
    y_test  = [0.838,0.840,0.840,0.834,0.828,0.814,0.812,0.822,0.818,0.815,0.807,0.801,0.796]
    
    
    plt.plot(x, y_train, marker='o', mec='r', mfc='w',label='uniprot90_train')
    plt.plot(x, y_test, marker='*', ms=10,label='uniprot90_test')
    plt.legend() 
    plt.xticks(x, names, rotation=1)
    
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('the length')
    plt.ylabel("f1") 
    plt.yticks([0.750,0.800,0.850])
    plt.title("A simple plot")
    plt.show()

def first_graph():
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    with open("dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "deadline_alpha" in i:
                deadline_alpha = float(i[15:-1])
                x.append(1+round(deadline_alpha,2))
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                dasa.append(success_number[1])

    with open("baseline_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_heft.append(success_number[1])
    
    with open("baseline_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_pcp.append(success_number[1])
    
    with open("baseline_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_prolis.append(success_number[1])
    
    with open("baseline_BDAS.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_BDAS.append(success_number[1])
    
    plt.plot(x, dasa, marker='o', ms=10,label='DASA')
    plt.plot(x, baseline_heft, marker='*', ms=10,label='HEFT')
    plt.plot(x, baseline_pcp, marker='s', ms=10,label='PCP')
    plt.plot(x, baseline_prolis, marker='.', ms=10,label='ProLis')
    plt.plot(x, baseline_BDAS, marker='p', ms=10,label='BDAS')
    plt.legend() 
    plt.xticks(x, x, rotation=1)
    
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel(r'$\beta$')
    plt.ylabel("SR") 
    plt.ylim(0,1.05)
    # plt.title("A simple plot")
    plt.show()

def second_graph():
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    with open("dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "deadline_alpha" in i:
                deadline_alpha = float(i[15:-1])
                x.append(1+round(deadline_alpha,2))
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                dasa.append(success_number[1])

    with open("baseline_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_heft.append(success_number[1])
    
    with open("baseline_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_pcp.append(success_number[1])
    
    with open("baseline_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_prolis.append(success_number[1])
    
    with open("baseline_BDAS.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_BDAS.append(success_number[1])
    
    plt.plot(x, dasa, marker='o', ms=10,label='DASA')
    plt.plot(x, baseline_heft, marker='*', ms=10,label='HEFT')
    plt.plot(x, baseline_pcp, marker='s', ms=10,label='PCP')
    plt.plot(x, baseline_prolis, marker='.', ms=10,label='ProLis')
    plt.plot(x, baseline_BDAS, marker='p', ms=10,label='BDAS')
    plt.legend() 
    plt.xticks(x, x, rotation=1)
    
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel(r'$\beta$')
    plt.ylabel("NC") 
    plt.ylim(5,15)
    # plt.title("A simple plot")
    plt.show()

def third_graph():
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    with open("offline_only_edge_ccr04_dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "deadline_alpha" in i:
                deadline_alpha = float(i[15:-1])
                x.append(1+round(deadline_alpha,2))
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                dasa.append(success_number[1])

    with open("offline_only_edge_ccr04_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_heft.append(success_number[1])
    
    with open("offline_only_edge_ccr04_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_pcp.append(success_number[1])
    
    with open("offline_only_edge_ccr04_ProLis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_prolis.append(success_number[1])
    
    with open("offline_only_edge_ccr04_BDAS.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_BDAS.append(success_number[1])
    
    plt.plot(x, dasa, marker='o', ms=10,label='DASA')
    plt.plot(x, baseline_heft, marker='*', ms=10,label='HEFT')
    plt.plot(x, baseline_pcp, marker='s', ms=10,label='PCP')
    plt.plot(x, baseline_prolis, marker='.', ms=10,label='ProLis')
    plt.plot(x, baseline_BDAS, marker='p', ms=10,label='BDAS')
    plt.legend() 
    plt.xticks(x, x, rotation=1)
    
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel(r'$\beta$')
    plt.ylabel("SR") 
    plt.ylim(0,1.05)
    # plt.title("A simple plot")
    plt.show()


def four_graph():
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    with open("offline_only_edge_ccr04_dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "deadline_alpha" in i:
                deadline_alpha = float(i[15:-1])
                x.append(1+round(deadline_alpha,2))
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                dasa.append(success_number[1])

    with open("offline_only_edge_ccr04_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_heft.append(success_number[1])
    
    with open("offline_only_edge_ccr04_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_pcp.append(success_number[1])
    
    with open("offline_only_edge_ccr04_ProLis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_prolis.append(success_number[1])
    
    with open("offline_only_edge_ccr04_BDAS.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_BDAS.append(success_number[1])
    
    plt.plot(x, dasa, marker='o', ms=10,label='DASA')
    plt.plot(x, baseline_heft, marker='*', ms=10,label='HEFT')
    plt.plot(x, baseline_pcp, marker='s', ms=10,label='PCP')
    plt.plot(x, baseline_prolis, marker='.', ms=10,label='ProLis')
    plt.plot(x, baseline_BDAS, marker='p', ms=10,label='BDAS')
    plt.legend() 
    plt.xticks(x, x, rotation=1)
    
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel(r'$\beta$')
    plt.ylabel("NC") 
    plt.ylim(5,15)
    # plt.title("A simple plot")
    plt.show()

def draw_sr():
    ccr_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    with open("offline_only_edge_ccrall_dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "deadline_alpha" in i:
                deadline_alpha = float(i[15:-1])
                x.append(1+round(deadline_alpha,2))
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                dasa.append(success_number[1])
            if "ccr:" in i:
                ccr = float(i[4:-1])
                ccr_list.append(ccr)

    with open("offline_only_edge_ccrall_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_heft.append(success_number[1])
    
    with open("offline_only_edge_ccrall_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_pcp.append(success_number[1])
    
    with open("offline_only_edge_ccrall_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_prolis.append(success_number[1])
    
    with open("offline_only_edge_ccrall_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_BDAS.append(success_number[1])
    
    for ccr in [0.1,0.3,0.5,0.7,0.9,1.1]:
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DASA')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        plt.legend() 
        plt.xticks(_x, _x, rotation=1)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$')
        plt.ylabel("SR") 
        plt.ylim(0,1.05)
        # plt.show()
        plt.savefig('offline_only_edge_ccr{0}_sr.pdf'.format(int(ccr*10)))
        plt.clf()

def draw_nc():
    ccr_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    with open("offline_only_edge_ccrall_dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "deadline_alpha" in i:
                deadline_alpha = float(i[15:-1])
                x.append(1+round(deadline_alpha,2))
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                dasa.append(success_number[1])
            if "ccr:" in i:
                ccr = float(i[4:-1])
                ccr_list.append(ccr)

    with open("offline_only_edge_ccrall_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_heft.append(success_number[1])
    
    with open("offline_only_edge_ccrall_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_pcp.append(success_number[1])
    
    with open("offline_only_edge_ccrall_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_prolis.append(success_number[1])
    
    with open("offline_only_edge_ccrall_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_BDAS.append(success_number[1])
    
    for ccr in [0.1,0.3,0.5,0.7,0.9,1.1]:
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DASA')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        plt.legend() 
        plt.xticks(_x, _x, rotation=1)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$')
        plt.ylabel("NC") 
        plt.ylim(5,15)
        # plt.show()
        plt.savefig('offline_only_edge_ccr{0}_nc.pdf'.format(int(ccr*10)))
        plt.clf()

if __name__ == '__main__':
    # draw_bar()
    draw_nc()
