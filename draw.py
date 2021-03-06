'''
Author: 娄炯
Date: 2021-04-16 16:18:15
LastEditors: loujiong
LastEditTime: 2021-08-08 21:33:46
Description: draw task graph
Email:  413012592@qq.com
'''
import networkx as nx
from random import randint as rd
import matplotlib.pyplot as plt
import plotly as py
import plotly.figure_factory as ff
from datetime import datetime
import random

def get_layered_pos(G):
    edge_rad = dict()
    edge_pos = dict()
    longest_path = nx.dag_longest_path(G)
    positive_pos = [1]*len(longest_path)
    negative_pos = [1]*len(longest_path)
    pos = {}
    for index,item in enumerate(longest_path):
        pos[item] = [0,0-index]
    for i in G.nodes():
        if i not in longest_path:
            _path = [j for j in nx.all_simple_paths(G, 0, i)]
            length = max([len(j) for j in _path]) -1
            if positive_pos[length] <= negative_pos[length]:
                pos[i] = [positive_pos[length],0-length]
                positive_pos[length] += 1
            else:
                pos[i] = [0-negative_pos[length],0-length]
                negative_pos[length] += 1

    edge_info = dict()
    proccessed_edge = set()
    for index in range(len(longest_path) - 1):
        u = longest_path[index]
        v = longest_path[index+1]
        edge_rad[(u,v)] = 0
        slope = (pos[u][0]-pos[v][0])/(pos[u][1]-pos[v][1])
        intercept = pos[u][0] - pos[u][1]*slope
        if (slope,intercept) not in edge_info:
            edge_info[(slope,intercept)] = {}
        edge_info[(slope,intercept)][(u,v)] = (pos[u][1],pos[v][1])
        proccessed_edge.add((u,v))

    for u,v in G.edges():
        m1 = [pos[u][0]*1/2+pos[v][0]*1/2,pos[u][1]*1/2+pos[v][1]*1/2]
        edge_pos[(u,v)] = m1
        if (u,v) in proccessed_edge:
            continue
        flag = False
        slope = (pos[u][0]-pos[v][0])/(pos[u][1]-pos[v][1])
        intercept = pos[u][0] - pos[u][1]*slope
        if (slope,intercept) not in edge_info:
            edge_info[(slope,intercept)] = {}
            edge_info[(slope,intercept)][(u,v)] = (pos[u][1],pos[v][1])
        else:
            for edge1 in edge_info[(slope,intercept)]:
                y1,y2 = edge_info[(slope,intercept)][edge1]
                if (y1>=pos[u][1] and pos[u][1]>y2) or (pos[u][1] >=y1 and y1>pos[v][1]) :
                    flag = True
                    break
            edge_info[(slope,intercept)][(u,v)] = (pos[u][1],pos[v][1])
        if flag:
            edge_rad[(u,v)] = 0.3
        else:
            edge_rad[(u,v)] = 0
        c1 = [m1[0]+1/2*edge_rad[(u,v)]*(pos[v][1]-pos[u][1]),m1[1]-1/2*edge_rad[(u,v)]*(pos[v][0]-pos[u][0])]
        edge_pos[(u,v)] = c1
    return pos,edge_rad,edge_pos


def draw(G, is_save=True, _application_index=0):
    pos, edge_rad, edge_pos = get_layered_pos(G)
    nx.draw_networkx_nodes(G, pos)
    labels = {i: (i, G.nodes[i]["w"]) for i in G.nodes()}
    edge_labels = {(u, v): (G.edges[u, v]["e"]) for u, v in G.edges()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    # print(edge_labels)
    for e in edge_pos:
        nx.draw_networkx_labels(G,
                            {e:edge_pos[e]},
                            {e:edge_labels[e]},
                            font_size=8,
                            font_color="red")
    # nx.draw_networkx_labels(G,
    #                         edge_pos,
    #                         edge_labels,
    #                         font_size=8,
    #                         font_color="red")
    
    ax = plt.gca()
    for e in G.edges:
        nx.draw_networkx_edges(
            G, pos, edgelist = [e],
            connectionstyle="arc3,rad=rrr".replace(
                    'rrr', str(edge_rad[(e[0], e[1])])))
        # nx.draw_networkx_edge_labels(G, pos, edge_labels = {e:edge_labels[e]}, rotate= False,label_pos = 0.5)
        # ax.annotate(
        #     "",
        #     xy=pos[e[0]],
        #     xycoords='data',
        #     xytext=pos[e[1]],
        #     textcoords='data',
        #     arrowprops=dict(
        #         arrowstyle="<-",
        #         shrinkA=5,
        #         shrinkB=5,
        #         patchA=None,
        #         patchB=None,
        #         connectionstyle="arc3,rad=rrr".replace(
        #             'rrr', str(edge_rad[(e[0], e[1])])),
        #     ),
        # )

    plt.axis('equal')
    if is_save:
        plt.savefig('task_graph/task_graph_{0}.png'.format(_application_index))
        plt.clf()
    else:
        plt.show()

def draw2(A, is_save=True, _application_index=0, current_task_index = 0):
    G = A.task_graph
    pos, edge_rad, edge_pos = get_layered_pos(G)

    finished_task_list = [i for i in G.nodes() if G.nodes()[i]["is_scheduled_in_this_scheduling"] == 1]
    nx.draw_networkx_nodes(G, pos,nodelist=finished_task_list)
    unfinished_task_list = [i for i in G.nodes() if G.nodes()[i]["is_scheduled_in_this_scheduling"] == 0]
    nx.draw_networkx_nodes(G, pos,nodelist=unfinished_task_list,node_color="grey")
    current_task_list = [current_task_index]
    nx.draw_networkx_nodes(G, pos,nodelist=current_task_list,node_color="red")

    
    labels = {i: (i, G.nodes[i]["w"]) for i in G.nodes()}
    edge_labels = {(u, v): (G.edges[u, v]["e"]) for u, v in G.edges()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    # print(edge_labels)
    for e in edge_pos:
        nx.draw_networkx_labels(G,
                            {e:edge_pos[e]},
                            {e:edge_labels[e]},
                            font_size=8,
                            font_color="red")
    # nx.draw_networkx_labels(G,
    #                         edge_pos,
    #                         edge_labels,
    #                         font_size=8,
    #                         font_color="red")
    
    ax = plt.gca()
    for e in G.edges:
        nx.draw_networkx_edges(
            G, pos, edgelist = [e],
            connectionstyle="arc3,rad=rrr".replace(
                    'rrr', str(edge_rad[(e[0], e[1])])))
        # nx.draw_networkx_edge_labels(G, pos, edge_labels = {e:edge_labels[e]}, rotate= False,label_pos = 0.5)
        # ax.annotate(
        #     "",
        #     xy=pos[e[0]],
        #     xycoords='data',
        #     xytext=pos[e[1]],
        #     textcoords='data',
        #     arrowprops=dict(
        #         arrowstyle="<-",
        #         shrinkA=5,
        #         shrinkB=5,
        #         patchA=None,
        #         patchB=None,
        #         connectionstyle="arc3,rad=rrr".replace(
        #             'rrr', str(edge_rad[(e[0], e[1])])),
        #     ),
        # )

    plt.axis('equal')
    if is_save:
        plt.savefig('task_graph/task_graph_{0}.png'.format(_application_index))
        plt.clf()
    else:
        plt.show()

def draw_gantt(_application_list,edge_list,cloud,is_annotation = False, is_only_accept = False,  gantt_name = "result"):
    pyplt = py.offline.plot
    df = []
    # add data
    yaxis = []
    yaxis.append("cloud")
    # 增加dummy task以确保所有的node cpu都在
    df.append(dict(Task="cloud", Start=-10, Finish=-10,Resource = "dummy"))
    for _edge_index,_edge in enumerate(edge_list):
        for _cpu in range(_edge.task_concurrent_capacity):
            yaxis.append("{0}_{1}".format(_edge_index,_cpu))
            # 增加dummy task以确保所有的node cpu都在
            df.append(dict(Task="{0}_{1}".format(_edge_index,_cpu), Start=-10, Finish=-10,Resource = "dummy"))
    for _application_index,_application in enumerate(_application_list):
        if is_only_accept and not _application.is_accept:
            continue
        for i in _application.task_graph.nodes():
            if _application.task_graph.nodes[i]["selected_node"] == len(edge_list):
                machine = "cloud"
            else:
                machine = "{0}_{1}".format(_application.task_graph.nodes[i]["selected_node"],_application.task_graph.nodes[i]["cpu"])
            df.append(dict(Task=machine, Start=_application.task_graph.nodes[i]["start_time"], Finish=_application.task_graph.nodes[i]["finish_time"],Resource = "A_"+str(_application_index)))
    df.sort(key=lambda x: yaxis.index(x["Task"]), reverse=True)
    # 设置task graph的颜色
    all_the_colors = list("rgb({0},{1},{2})".format(x,y,z) for x in range(256) for y in range(256) for z in range(256))
    colors_list = random.sample(all_the_colors,len(_application_list)+1)
    colors_list.sort()
    colors = {"A_{0}".format(i):colors_list[i] for i in range(len(_application_list))}
    colors["dummy"] = colors_list[-1]
    
    # 获取gantt图
    fig = ff.create_gantt(df, colors=colors, show_colorbar=True,group_tasks=True,index_col='Resource',showgrid_x=True,showgrid_y=True)
    # 修改x轴
    fig['layout']['xaxis'].update({'type': None})
    # fig.layout['xaxis'].update(range=[-10, 1000])

    # draw annotations
    if is_annotation:
        for _application_index,_application in enumerate(_application_list):
            if is_only_accept and not _application.is_accept:
                continue
            for i in _application.task_graph.nodes():
                if  _application.task_graph.nodes[i]["selected_node"] != len(edge_list): #i != 0 and i != _application.task_graph.number_of_nodes() and
                    text = "{0}-{1}".format(_application_index,i)
                    if _application.task_graph.nodes[i]["selected_node"] == len(edge_list):
                        _machine = "cloud"
                    else:
                        _machine = "{0}_{1}".format(_application.task_graph.nodes[i]["selected_node"],_application.task_graph.nodes[i]["cpu"])
                    y_pos = yaxis.index(_machine)
                    x_pos = (_application.task_graph.nodes[i]["start_time"]+_application.task_graph.nodes[i]["finish_time"])/2
                    text_font = dict(size=12, color='black')
                    fig['layout']['annotations'] += tuple([dict(x=x_pos, y=y_pos, text=text, textangle=-30, showarrow=False, font=text_font)])
    for edge_index,item in enumerate(edge_list):
        fig['layout']['annotations'] += tuple([dict(x=-150, y=edge_index+1, text="({0},{1},{2})".format(item.cost_per_mip,item.process_data_rate,item.upload_data_rate), textangle=0, showarrow=False, font=text_font)])
    fig['layout']['annotations'] += tuple([dict(x=-150, y=0, text="({0},{1},{2})".format(cloud.cost_per_mip,cloud.process_data_rate,cloud.data_rate), textangle=0, showarrow=False, font=text_font)])
    # 画边框
    fig.update_traces(mode='lines',
                      line_color='black',
                      selector=dict(fill='toself'))
    pyplt(fig, filename='tmp/{0}.html'.format(gantt_name))
    return

if __name__ == '__main__':
    pass