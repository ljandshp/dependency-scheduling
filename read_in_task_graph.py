'''
Author: 娄炯
Date: 2021-08-15 22:37:43
LastEditors: loujiong
LastEditTime: 2021-08-17 22:39:21
Description: read in the workflow task graph
Email:  413012592@qq.com
'''

import networkx
import  xml.dom.minidom
import random

def get_cybershake_30():
    dom = xml.dom.minidom.parse('CyberShake_30.xml')
    root = dom.documentElement
    G = networkx.DiGraph()
    for child in root.childNodes:
        if child.nodeName == "child":
            child_node = child.getAttribute("ref")
            for parent in child.childNodes:
                if parent.nodeName == "parent":
                    parent_node = parent.getAttribute("ref") 
                    # print("parent_{0} -> child_{1}".format(parent_node,child_node))
                    G.add_edge(parent_node,child_node)
    
    node_list = list(networkx.topological_sort(G))
    _map = {item:index+1 for index,item in enumerate(node_list)}
    # print(_map)
    G = networkx.relabel_nodes(G,_map)
    # print(list(G.nodes()))
    return G

# def get_cybershake_30():
#     dom = xml.dom.minidom.parse('CyberShake_30.xml')
#     root = dom.documentElement
#     G = networkx.DiGraph()
#     for child in root.childNodes:
#         if child.nodeName == "child":
#             child_node = child.getAttribute("ref")
#             for parent in child.childNodes:
#                 if parent.nodeName == "parent":
#                     parent_node = parent.getAttribute("ref") 
#                     # print("parent_{0} -> child_{1}".format(parent_node,child_node))
#                     G.add_edge(parent_node,child_node)
    
#     node_list = list(networkx.topological_sort(G))
#     _map = {item:index+1 for index,item in enumerate(node_list)}
#     # print(_map)
#     G = networkx.relabel_nodes(G,_map)
#     # print(list(G.nodes()))
#     return G

def get_workflow():
    rd = random.randint(0,4)
    workflow_list = ['CyberShake_50.xml','Epigenomics_46.xml','Inspiral_50.xml','Montage_50.xml','Sipht_60.xml']
    dom = xml.dom.minidom.parse(workflow_list[rd])
    root = dom.documentElement
    G = networkx.DiGraph()
    for child in root.childNodes:
        if child.nodeName == "child":
            child_node = child.getAttribute("ref")
            for parent in child.childNodes:
                if parent.nodeName == "parent":
                    parent_node = parent.getAttribute("ref") 
                    # print("parent_{0} -> child_{1}".format(parent_node,child_node))
                    G.add_edge(parent_node,child_node)
    
    node_list = list(networkx.topological_sort(G))
    _map = {item:index+1 for index,item in enumerate(node_list)}
    # print(_map)
    G = networkx.relabel_nodes(G,_map)
    # print(list(G.nodes()))
    return G


if __name__ == '__main__':
    get_cybershake_30()