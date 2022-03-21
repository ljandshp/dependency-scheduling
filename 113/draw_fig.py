'''
Author: 娄炯
Date: 2021-09-07 18:34:35
LastEditors: loujiong
LastEditTime: 2021-11-05 09:53:02
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
    
    plt.plot(x, dasa, marker='o', ms=10,label='DCDS')
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
    
    plt.plot(x, dasa, marker='o', ms=10,label='DCDS')
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
    
    plt.plot(x, dasa, marker='o', ms=10,label='DCDS')
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
    
    plt.plot(x, dasa, marker='o', ms=10,label='DCDS')
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
    plt.ylim(7,15)
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
    
    for ccr in [0.1,1]:
        print(ccr)
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("SR",fontsize=17) 
        plt.ylim(0,1.05)
        # plt.show()
        plt.tight_layout()
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
    
    for ccr in [0.1,1]:
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("NC",fontsize=17) 
        ymin = min(min(_dasa),min(_baseline_heft),min(_baseline_pcp),min(_baseline_prolis),min(_baseline_BDAS))
        ymax = max(max(_dasa),max(_baseline_heft),max(_baseline_pcp),max(_baseline_prolis),max(_baseline_BDAS))
        plt.ylim(ymin*0.95,ymax*1.05)
        # plt.show()
        plt.tight_layout()
        plt.savefig('offline_only_edge_ccr{0}_nc.pdf'.format(int(ccr*10)))
        plt.clf()

def draw_sr_cloud():
    ccr_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    with open("offline_cloud_ccrall_dasa.out","r") as f:
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

    with open("offline_cloud_ccrall_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_heft.append(success_number[1])
    
    with open("offline_cloud_ccrall_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_pcp.append(success_number[1])
    
    with open("offline_cloud_ccrall_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_prolis.append(success_number[1])
    
    with open("offline_cloud_ccrall_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_BDAS.append(success_number[1])
    
    for ccr in [0.1,1]:
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("SR",fontsize=17) 
        plt.ylim(0,1.05)
        # plt.show()
        plt.tight_layout()
        plt.savefig('offline_cloud_ccr{0}_sr.pdf'.format(int(ccr*10)))
        plt.clf()

def draw_nc_cloud():
    ccr_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    with open("offline_cloud_ccrall_dasa.out","r") as f:
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

    with open("offline_cloud_ccrall_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_heft.append(success_number[1])
    
    with open("offline_cloud_ccrall_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_pcp.append(success_number[1])
    
    with open("offline_cloud_ccrall_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_prolis.append(success_number[1])
    
    with open("offline_cloud_ccrall_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_BDAS.append(success_number[1])
    
    for ccr in [0.1,1]:
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        ymin = min(min(_dasa),min(_baseline_heft),min(_baseline_pcp),min(_baseline_prolis),min(_baseline_BDAS))
        ymax = max(max(_dasa),max(_baseline_heft),max(_baseline_pcp),max(_baseline_prolis),max(_baseline_BDAS))
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("NC",fontsize=17)
        plt.ylim(ymin*0.95,ymax*1.05)
        # print(ymin,ymax)
        # maxy = max([max(_dasa),max(_baseline_heft),max(_baseline_pcp),max(_baseline_prolis),max(_baseline_BDAS),]) 
        # plt.ylim(0,maxy)
        # plt.show()
        plt.tight_layout()
        plt.savefig('offline_cloud_ccr{0}_nc.pdf'.format(int(ccr*10)))
        plt.clf()

'''
def draw_sr_cloud_bandwidth():
    ccr_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    dasa_w_refine = []
    with open("offline_cloud_edge_ccr5_bandwidth8_dasa.out","r") as f:
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

    with open("offline_cloud_edge_ccr5_bandwidth8_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_heft.append(success_number[1])
    
    with open("offline_cloud_edge_ccr5_bandwidth8_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_pcp.append(success_number[1])
    
    with open("offline_cloud_edge_ccr5_bandwidth8_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_prolis.append(success_number[1])
    
    with open("offline_cloud_edge_ccr5_bandwidth8_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_BDAS.append(success_number[1])
    
    # with open("offline_cloud_edge_ccr5_bandwidth8_dasa_w_refine.out","r") as f:
    #     t = f.readlines()
    #     for i in t:
    #         if "success_number:" in i:
    #             success_number = json.loads(i[15:-1])
    #             dasa_w_refine.append(success_number[1])
    
    for ccr in [0.5]:
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]
        # _dasa_w_refine = [dasa_w_refine[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        # plt.plot(_x, _dasa_w_refine, marker='h', ms=10,label='dasa without refine')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("SR",fontsize=17) 
        plt.ylim(0,1.05)
        # plt.show()
        plt.savefig('offline_cloud_edge_ccr{0}_bandwidth8_sr.pdf'.format(int(ccr*10)))
        plt.clf()
'''


def draw_runtime_bar():
    plt.figure(figsize=(8,4))
    # TODO open file
    # TODO read lines
    # TODO find interested data
    # TODO edit legend 
    x=np.arange(3)
    heft=np.array([7.298434257507324,15.060241460800171,31.8154239654541])/500
    pcp = np.array([7.396198034286499,15.522457361221313,32.53854274749756])/500
    prolis = np.array([7.36017107963562,15.915173530578613,32.133084774017334])/500
    bdas = np.array([7.462753772735596,15.605095148086548,32.37430262565613])/500
    dasaw = np.array([10.96363091468811,23.914133548736572,51.30663514137268])/500
    dasa = np.array([20.989537477493286,44.801427125930786,95.70630598068237])/500


    bar_width=0.15
    tick_label=['Montage_25','Montage_50','Montage_100']

    plt.bar(x-2*bar_width,dasaw,bar_width,label='DCDS-EdgeSchduling',edgecolor='black',linewidth=2)
    plt.bar(x-2*bar_width,dasa-dasaw,bar_width,label='DCDS-CloudOffloading',bottom=dasaw,edgecolor='black',linewidth=2)
    plt.bar(x-1*bar_width,heft,bar_width,label='HEFT',edgecolor='black',linewidth=2)
    plt.bar(x-0*bar_width,pcp,bar_width,label='PCP',edgecolor='black',linewidth=2)
    plt.bar(x+1*bar_width,prolis,bar_width,label='ProLis',edgecolor='black',linewidth=2)
    plt.bar(x+2*bar_width,bdas,bar_width,label='BDAS',edgecolor='black',linewidth=2)

    plt.legend(fontsize=15)
    plt.xticks(x,tick_label,fontsize=15)
    plt.yticks(fontsize=15)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    # plt.xlabel('Workflow')
    plt.ylabel("Runtime (s)",fontsize=15) 
    plt.ylim(0,0.2)
    plt.xlim(-0.5,2.5)
    plt.tight_layout()
    plt.savefig('montage_runtime.pdf')
    return

def draw_sr_cloud_bandwidth(bandwidth_factor = 2):
    ccr_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    dasa_w_refine = []
    with open("offline_cloud_edge_ccr5_bandwidth{0}_dasa.out".format(bandwidth_factor),"r") as f:
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

    with open("offline_cloud_edge_ccr5_bandwidth{0}_heft.out".format(bandwidth_factor),"r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_heft.append(success_number[1])
    
    with open("offline_cloud_edge_ccr5_bandwidth{0}_pcp.out".format(bandwidth_factor),"r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_pcp.append(success_number[1])
    
    with open("offline_cloud_edge_ccr5_bandwidth{0}_prolis.out".format(bandwidth_factor),"r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_prolis.append(success_number[1])
    
    with open("offline_cloud_edge_ccr5_bandwidth{0}_bdas.out".format(bandwidth_factor),"r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_BDAS.append(success_number[1])
    
    # with open("offline_cloud_edge_ccr5_bandwidth8_dasa_w_refine.out","r") as f:
    #     t = f.readlines()
    #     for i in t:
    #         if "success_number:" in i:
    #             success_number = json.loads(i[15:-1])
    #             dasa_w_refine.append(success_number[1])
    
    for ccr in [0.5]:
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]
        # _dasa_w_refine = [dasa_w_refine[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        # plt.plot(_x, _dasa_w_refine, marker='h', ms=10,label='dasa without refine')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("SR",fontsize=17) 
        plt.ylim(0,1.05)
        # plt.show()
        plt.tight_layout()
        plt.savefig('offline_cloud_edge_ccr{0}_bandwidth{1}_sr.pdf'.format(int(ccr*10),bandwidth_factor))
        plt.clf()

def draw_nc_cloud_bandwidth(bandwidth_factor = 2):
    ccr_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    dasa_w_refine = []
    with open("offline_cloud_edge_ccr5_bandwidth{0}_dasa.out".format(bandwidth_factor),"r") as f:
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

    with open("offline_cloud_edge_ccr5_bandwidth{0}_heft.out".format(bandwidth_factor),"r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_heft.append(success_number[1])
    
    with open("offline_cloud_edge_ccr5_bandwidth{0}_pcp.out".format(bandwidth_factor),"r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_pcp.append(success_number[1])
    
    with open("offline_cloud_edge_ccr5_bandwidth{0}_prolis.out".format(bandwidth_factor),"r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_prolis.append(success_number[1])
    
    with open("offline_cloud_edge_ccr5_bandwidth{0}_bdas.out".format(bandwidth_factor),"r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_BDAS.append(success_number[1])
    
    # with open("offline_cloud_edge_ccr5_bandwidth8_dasa_w_refine.out","r") as f:
    #     t = f.readlines()
    #     for i in t:
    #         if "normalized_cost:" in i:
    #             success_number = json.loads(i[16:-1])
    #             dasa_w_refine.append(success_number[1])
    
    for ccr in [0.5]:
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        # print(baseline_prolis)
        # print(baseline_pcp)
        # print(baseline_heft)
        # print(ccr_list)
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]
        # _dasa_w_refine = [dasa_w_refine[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        # plt.plot(_x, _dasa_w_refine, marker='h', ms=10,label='dasa without refine')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("NC",fontsize=17)
        ymin = min(min(_dasa),min(_baseline_heft),min(_baseline_pcp),min(_baseline_prolis),min(_baseline_BDAS))
        ymax = max(max(_dasa),max(_baseline_heft),max(_baseline_pcp),max(_baseline_prolis),max(_baseline_BDAS))
        plt.ylim(ymin*0.95,ymax*1.05)
        # plt.show()
        plt.tight_layout()
        plt.savefig('offline_cloud_edge_ccr{0}_bandwidth{1}_nc.pdf'.format(int(ccr*10),bandwidth_factor))
        plt.clf()


def draw_sr_cloud_edge_number():
    ccr_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    dasa_w_refine = []
    with open("edge_number_dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "edge_number" in i:
                edge_number = float(i[12:-1])
                x.append(edge_number)
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                dasa.append(success_number[1])
            if "ccr:" in i:
                ccr = float(i[4:-1])
                ccr_list.append(ccr)

    with open("edge_number_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_heft.append(success_number[1])
    
    with open("edge_number_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_pcp.append(success_number[1])
    
    with open("edge_number_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_prolis.append(success_number[1])
    
    with open("edge_number_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_BDAS.append(success_number[1])
    
    # with open("offline_cloud_edge_ccr5_bandwidth8_dasa_w_refine.out","r") as f:
    #     t = f.readlines()
    #     for i in t:
    #         if "success_number:" in i:
    #             success_number = json.loads(i[15:-1])
    #             dasa_w_refine.append(success_number[1])
    
    for ccr in [0.5]:
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]
        # _dasa_w_refine = [dasa_w_refine[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        # plt.plot(_x, _dasa_w_refine, marker='h', ms=10,label='dasa without refine')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel("Number of Edge Servers",fontsize=17)
        plt.ylabel("SR",fontsize=17) 
        plt.ylim(0,1.05)
        # plt.show()
        plt.tight_layout()
        plt.savefig('edge_number_cloud_sr.pdf')
        plt.clf()

def draw_nc_cloud_edge_number():
    ccr_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    dasa_w_refine = []
    with open("edge_number_dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "edge_number" in i:
                edge_number = float(i[12:-1])
                x.append(edge_number)
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                dasa.append(success_number[1])
            if "ccr:" in i:
                ccr = float(i[4:-1])
                ccr_list.append(ccr)

    with open("edge_number_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_heft.append(success_number[1])
    
    with open("edge_number_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_pcp.append(success_number[1])
    
    with open("edge_number_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_prolis.append(success_number[1])
    
    with open("edge_number_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_BDAS.append(success_number[1])
    
    # with open("offline_cloud_edge_ccr5_bandwidth8_dasa_w_refine.out","r") as f:
    #     t = f.readlines()
    #     for i in t:
    #         if "normalized_cost:" in i:
    #             success_number = json.loads(i[16:-1])
    #             dasa_w_refine.append(success_number[1])
    
    for ccr in [0.5]:
        _x = [x[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _dasa = [dasa[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if ccr_list[i] == ccr]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if ccr_list[i] == ccr]
        # _dasa_w_refine = [dasa_w_refine[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        # plt.plot(_x, _dasa_w_refine, marker='h', ms=10,label='dasa without refine')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel("Number of Edge Servers",fontsize=17)
        plt.ylabel("NC",fontsize=17)
        ymin = min(min(_dasa),min(_baseline_heft),min(_baseline_pcp),min(_baseline_prolis),min(_baseline_BDAS))
        ymax = max(max(_dasa),max(_baseline_heft),max(_baseline_pcp),max(_baseline_prolis),max(_baseline_BDAS))
        plt.ylim(ymin*0.95,ymax*1.05)
        # plt.show()
        plt.tight_layout()
        plt.savefig("edge_number_cloud_nc.pdf")
        plt.clf()



def draw_runtime_line():
    plt.figure(figsize=(8,4))
    # TODO open file
    # TODO read lines
    # TODO find interested data
    # TODO edit legend 
    x=np.arange(10,60,10)
    heft=np.array([0.75645,1.538787,2.158245325,2.81918168,3.47499632])/100
    pcp = np.array([0.7745509,1.456953,2.1187353,2.7837326,3.447077035])/100
    prolis = np.array([0.7623217,1.4467082,2.14018845,2.800899267,3.445009946])/100
    bdas = np.array([0.766621,1.43118643,2.15509653,2.7783434391,3.49207234])/100
    dasa = np.array([2.159525,4.117906,5.9288601,7.7615580,9.70042133])/100


    bar_width=0.15
    tick_label=['Montage_25','Montage_50','Montage_100']

    # plt.bar(x-2*bar_width,dasaw,bar_width,label='DCDS-LSTS')

    plt.plot(x, dasa, marker='o', ms=10,label='DCDS')
    plt.plot(x, heft, marker='*', ms=10,label='HEFT')
    plt.plot(x, pcp, marker='s', ms=10,label='PCP')
    plt.plot(x, prolis, marker='.', ms=10,label='ProLis')
    plt.plot(x, bdas, marker='p', ms=10,label='BDAS')

    plt.legend(fontsize =15)
    plt.xticks(x,x,fontsize =15)
    plt.yticks(fontsize =15)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('Number of edge servers',fontsize =15)
    plt.ylabel("Runtime (s)",fontsize =15) 
    plt.ylim(0,0.1)
    plt.xlim(5,55)
    plt.tight_layout()
    plt.savefig('edge_number_runtime.pdf')
    return

def draw_sr_cloud_lambda():
    application_average_interval_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    # dasa_w_refine = []
    with open("only_cloud_edge_lambdaall_dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "deadline_alpha" in i:
                deadline_alpha = float(i[15:-1])
                x.append(1+round(deadline_alpha,2))
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                dasa.append(success_number[1])
            if "application_average_interval:" in i:
                application_average_interval = float(i[29:-1])
                application_average_interval_list.append(application_average_interval)

    with open("only_cloud_edge_lambdaall_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_heft.append(success_number[1])
    
    with open("only_cloud_edge_lambdaall_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_pcp.append(success_number[1])
    
    with open("only_cloud_edge_lambdaall_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_prolis.append(success_number[1])
    
    with open("only_cloud_edge_lambdaall_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_BDAS.append(success_number[1])
    
    # with open("offline_cloud_edge_ccr5_bandwidth8_dasa_w_refine.out","r") as f:
    #     t = f.readlines()
    #     for i in t:
    #         if "success_number:" in i:
    #             success_number = json.loads(i[15:-1])
    #             dasa_w_refine.append(success_number[1])
    
    for application_average_interval in [50,250]:
        _x = [x[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _dasa = [dasa[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        # _dasa_w_refine = [dasa_w_refine[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        # plt.plot(_x, _dasa_w_refine, marker='h', ms=10,label='dasa without refine')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("SR",fontsize=17) 
        plt.ylim(0,1.05)
        # plt.show()
        plt.tight_layout()
        plt.savefig('online_cloud_edge_lambda{0}_sr.pdf'.format(application_average_interval))
        plt.clf()

def draw_nc_cloud_lambda():
    application_average_interval_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    # dasa_w_refine = []
    with open("only_cloud_edge_lambdaall_dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "deadline_alpha" in i:
                deadline_alpha = float(i[15:-1])
                x.append(1+round(deadline_alpha,2))
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                dasa.append(success_number[1])
            if "application_average_interval:" in i:
                application_average_interval = float(i[29:-1])
                application_average_interval_list.append(application_average_interval)

    with open("only_cloud_edge_lambdaall_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_heft.append(success_number[1])
    
    with open("only_cloud_edge_lambdaall_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_pcp.append(success_number[1])
    
    with open("only_cloud_edge_lambdaall_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_prolis.append(success_number[1])
    
    with open("only_cloud_edge_lambdaall_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_BDAS.append(success_number[1])
    
    # with open("offline_cloud_edge_ccr5_bandwidth8_dasa_w_refine.out","r") as f:
    #     t = f.readlines()
    #     for i in t:
    #         if "normalized_cost:" in i:
    #             success_number = json.loads(i[16:-1])
    #             dasa_w_refine.append(success_number[1])
    
    for application_average_interval in [50,250]:
        _x = [x[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _dasa = [dasa[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if application_average_interval_list[i] ==  application_average_interval]
        # _dasa_w_refine = [dasa_w_refine[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        # plt.plot(_x, _dasa_w_refine, marker='h', ms=10,label='dasa without refine')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("NC",fontsize=17)
        ymin = min(min(_dasa),min(_baseline_heft),min(_baseline_pcp),min(_baseline_prolis),min(_baseline_BDAS))
        ymax = max(max(_dasa),max(_baseline_heft),max(_baseline_pcp),max(_baseline_prolis),max(_baseline_BDAS))
        plt.ylim(ymin*0.95,ymax*1.05)
        plt.tight_layout()
        plt.savefig('online_cloud_edge_lambda{0}_nc.pdf'.format(application_average_interval))
        plt.clf()

def draw_sr_edge_lambda():
    application_average_interval_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    # dasa_w_refine = []
    with open("online_only_edge_lambdaall_dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "deadline_alpha" in i:
                deadline_alpha = float(i[15:-1])
                x.append(1+round(deadline_alpha,2))
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                dasa.append(success_number[1])
            if "application_average_interval:" in i:
                application_average_interval = float(i[29:-1])
                application_average_interval_list.append(application_average_interval)

    with open("online_only_edge_lambdaall_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_heft.append(success_number[1])
    
    with open("online_only_edge_lambdaall_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_pcp.append(success_number[1])
    
    with open("online_only_edge_lambdaall_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_prolis.append(success_number[1])
    
    with open("online_only_edge_lambdaall_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "success_number:" in i:
                success_number = json.loads(i[15:-1])
                baseline_BDAS.append(success_number[1])
    
    for application_average_interval in [50,250]:
        _x = [x[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _dasa = [dasa[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        # _dasa_w_refine = [dasa_w_refine[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        # plt.plot(_x, _dasa_w_refine, marker='h', ms=10,label='dasa without refine')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("SR",fontsize=17) 
        plt.ylim(0,1.05)
        # plt.show()
        plt.tight_layout()
        plt.savefig('online_only_edge_lambda{0}_sr.pdf'.format(application_average_interval))
        plt.clf()

def draw_nc_edge_lambda():
    application_average_interval_list = []
    x = []
    dasa= []
    baseline_heft = []
    baseline_pcp = []
    baseline_prolis = []
    baseline_BDAS = []
    # dasa_w_refine = []
    with open("online_only_edge_lambdaall_dasa.out","r") as f:
        t = f.readlines()
        for i in t:
            if "deadline_alpha" in i:
                deadline_alpha = float(i[15:-1])
                x.append(1+round(deadline_alpha,2))
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                dasa.append(success_number[1])
            if "application_average_interval:" in i:
                application_average_interval = float(i[29:-1])
                application_average_interval_list.append(application_average_interval)

    with open("online_only_edge_lambdaall_heft.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_heft.append(success_number[1])
    
    with open("online_only_edge_lambdaall_pcp.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_pcp.append(success_number[1])
    
    with open("online_only_edge_lambdaall_prolis.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_prolis.append(success_number[1])
    
    with open("online_only_edge_lambdaall_bdas.out","r") as f:
        t = f.readlines()
        for i in t:
            if "normalized_cost:" in i:
                success_number = json.loads(i[16:-1])
                baseline_BDAS.append(success_number[1])
    
    # with open("offline_cloud_edge_ccr5_bandwidth8_dasa_w_refine.out","r") as f:
    #     t = f.readlines()
    #     for i in t:
    #         if "normalized_cost:" in i:
    #             success_number = json.loads(i[16:-1])
    #             dasa_w_refine.append(success_number[1])
    
    for application_average_interval in [50,250]:
        _x = [x[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _dasa = [dasa[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_heft = [baseline_heft[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_pcp = [baseline_pcp[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_prolis = [baseline_prolis[i] for i in range(len(x)) if application_average_interval_list[i] == application_average_interval]
        _baseline_BDAS = [baseline_BDAS[i] for i in range(len(x)) if application_average_interval_list[i] ==  application_average_interval]
        # _dasa_w_refine = [dasa_w_refine[i] for i in range(len(x)) if ccr_list[i] == ccr]


        plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
        plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
        plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
        plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
        plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
        # plt.plot(_x, _dasa_w_refine, marker='h', ms=10,label='dasa without refine')
        plt.legend(fontsize=17) 
        plt.xticks(_x, _x, rotation=1,fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(r'$\beta$',fontsize=17)
        plt.ylabel("NC",fontsize=17)
        ymin = min(min(_dasa),min(_baseline_heft),min(_baseline_pcp),min(_baseline_prolis),min(_baseline_BDAS))
        ymax = max(max(_dasa),max(_baseline_heft),max(_baseline_pcp),max(_baseline_prolis),max(_baseline_BDAS))
        plt.ylim(ymin*0.95,ymax*1.05)
        plt.tight_layout()
        plt.savefig('online_only_edge_lambda{0}_nc.pdf'.format(application_average_interval))
        # plt.show()
        plt.clf()



def draw_sr_deadline_cloud():
    _x = [5,10,15,20,25,30]
    _dasa = [0.998,0.998,1.0,1.0,1.0,1.0]
    _baseline_heft = [1,1,1,1,1,1]
    _baseline_pcp = [0.94,0.844,0.718,0.758,0.696,0.768]
    _baseline_prolis = [0.944,0.786,0.668,0.728,0.638,0.748]
    _baseline_BDAS = [0.644,0.354,0.382,0.474,0.442,0.502]


    plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
    plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
    plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
    plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
    plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
    plt.legend(fontsize=17) 
    plt.xticks(_x, _x, rotation=1,fontsize=17)
    plt.yticks(fontsize=17)
    
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel("Edge Server Number",fontsize=17)
    plt.ylabel("SR",fontsize=17) 
    plt.ylim(0,1.05)
    # plt.show()
    plt.tight_layout()
    plt.savefig('deadline_cloud_sr.pdf')
    plt.clf()

def draw_nc_deadline_cloud():
    _x = [5,10,15,20,25,30]
    _dasa = [12.028,10.561,9.910,9.815,9.831,9.743]
    _baseline_heft = [15.982,14.113,13.570,14.463,14.378,14.567]
    _baseline_pcp = [13.167,11.696,11.132,11.029,11.150,10.860]
    _baseline_prolis = [12.152,11.040,10.626,10.487,10.668,10.374]
    _baseline_BDAS = [11.713,10.765,10.278,9.908,10.157,9.868]


    plt.plot(_x, _dasa, marker='o', ms=10,label='DCDS')
    plt.plot(_x, _baseline_heft, marker='*', ms=10,label='HEFT')
    plt.plot(_x, _baseline_pcp, marker='s', ms=10,label='PCP')
    plt.plot(_x, _baseline_prolis, marker='.', ms=10,label='ProLis')
    plt.plot(_x, _baseline_BDAS, marker='p', ms=10,label='BDAS')
    plt.legend(fontsize=17) 
    plt.xticks(_x, _x, rotation=1,fontsize=17)
    plt.yticks(fontsize=17)
    
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel("Edge Server Number",fontsize=17)
    plt.ylabel("NC",fontsize=17) 
    plt.ylim(0,16.5)
    # plt.show()
    plt.tight_layout()
    plt.savefig('deadline_cloud_nc.pdf')
    plt.clf()


if __name__ == '__main__':
    # draw_runtime_line()
    # draw_nc_cloud_bandwidth()
    # draw_sr_cloud_bandwidth()
    # draw_sr_cloud_lambda()
    # draw_nc_cloud_lambda()
    # draw_nc_edge_lambda()
    # draw_sr_edge_lambda()
    # draw_runtime_bar()

    # draw_sr_cloud_lambda()
    # draw_nc_cloud_lambda()
    # draw_nc_edge_lambda()
    # draw_sr_edge_lambda()

    # draw_nc_cloud_bandwidth()
    # draw_sr_cloud_bandwidth()

    # draw_sr()
    # draw_nc()
    # draw_sr_cloud()
    # draw_nc_cloud()
    # draw_sr_cloud_bandwidth()
    # draw_nc_cloud_bandwidth()
    # draw_sr_edge_lambda()
    # draw_nc_edge_lambda()

    # draw_sr_cloud_lambda()
    # draw_nc_cloud_lambda()
    # draw_runtime_line()
    # draw_runtime_bar()

    # draw_runtime_bar()

    # draw_sr_deadline_cloud()
    # draw_nc_deadline_cloud()



    draw_sr()
    draw_nc()
    draw_sr_cloud()
    draw_nc_cloud()
    draw_nc_cloud_bandwidth(2)
    draw_sr_cloud_bandwidth(2)
    draw_nc_cloud_bandwidth(8)
    draw_sr_cloud_bandwidth(8)
    draw_nc_cloud_edge_number()
    draw_sr_cloud_edge_number()
    draw_nc_edge_lambda()
    draw_sr_edge_lambda()
    draw_sr_cloud_lambda()
    draw_nc_cloud_lambda()

