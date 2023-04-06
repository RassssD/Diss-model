

import networkx as nx
import numpy as np


import plotly.graph_objects as go



class plotly_sim_drawing(object):
    '''HELPER FUNCTIONS'''
    '''MODIFY THESE TO TAKE A SPECIFIC TIME INSTEAD'''
    # gets a tuple of arrays of edge positions
    def go_get_edge_positions(self, graph, graph_layout):

        edge_x = []
        edge_y = []

        for edge in graph.edges():
            x0, y0 = graph_layout[edge[0]]
            x1, y1 = graph_layout[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)


        return (edge_x, edge_y)

    # gets a edge trace based on a wanted layout and graph
    #to add color list later: https://stackoverflow.com/questions/62601052/option-to-add-edge-colouring-in-networkx-trace-using-plotly

    def go_get_edge_trace(self, graph, graph_layout):#, edge_color_list):
        edge_x, edge_y = self.go_get_edge_positions(graph, graph_layout)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            hoverinfo=None,
            mode='lines',

            line=dict(
                width=0.5, 
                color='rgba(100, 100, 100, 0.35)')
        )

        return edge_trace


    # returns a tuple with an array for each of x pos and y pos
    def go_get_node_positions(self, graph, graph_layout):

        node_x_list = []
        node_y_list = []

        for node in graph.nodes():
            node_x, node_y = graph_layout[node]
            node_x_list.append(node_x)
            node_y_list.append(node_y)


        return (node_x_list, node_y_list)

    # gets a node trace based on a wanted layout and graph
    def go_get_node_trace(self, graph, graph_layout, node_color_list, node_text):
        node_x, node_y = self.go_get_node_positions(graph, graph_layout)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            hoverinfo='text',
            mode='markers',
            text=node_text,
            #IDEA: add text for number of friends in each group

            marker=dict(
                size=5,
                color=node_color_list),
        )

        return node_trace


    def get_legend_text(self, simulation, graph):
        # share of H friends
        H_shares = simulation.helper_functions().average_neighbour_type_per_SES(graph)
        text_H_shares = f"Mean share of High-SES friends:<br> • High SES: {round(H_shares[0], 2)}<br> • Low SES: {round(H_shares[1], 2)}<br>"

        # attribute assortativity
        attr_assortativity = nx.assortativity.attribute_assortativity_coefficient(graph, 'SES_High')
        text_attr_assortativity = f"High SES assortativity: {round(attr_assortativity, 2)}"

        # edge types
        edge_type_counts = simulation.helper_functions().SES_edge_classifier_all(graph, return_counts = True)[2]
        
        within_count, cross_count = int(edge_type_counts[0][1]), int(edge_type_counts[1][1])
        total_count = within_count + cross_count
        text_edge_type_counts = f"<br>Edge types<br> • Within-SES: {round(within_count / total_count, 2)}<br> • Across-SES: {round(cross_count / total_count, 2)}<br> • Total: {total_count}"


        # largest connected component / unconnected nodes
        n_in_largest_cc = len(max(nx.connected_components(graph), key=len))
        text_largest_cc = f"<br> Nodes in largest CC: {n_in_largest_cc} / {simulation.T + simulation.initial_n}"

        # expected number of connections per node without bias vs realised average degree and variance
        degree_hist = nx.degree_histogram(graph)
        avg_degree = round(np.average(degree_hist), 2)
        var_degree = round(np.var(degree_hist), 2)

        no_bias_expected_degree = simulation.m*simulation.pm_o + simulation.n*simulation.pn_o
        realised_degree_dist = (avg_degree, var_degree)
        text_encpn = f"<br>Expected average degree <br>without bias: {no_bias_expected_degree}<br>Realised degree <br>mean, var: {realised_degree_dist}"


        # parameters?
        text_parameters = f"<br><br><br>Parameters<br> • Initial nodes: {simulation.initial_n}<br> • T: {simulation.T}<br> • m: {simulation.m}<br> • pm_o: {simulation.pm_o}<br> • n: {simulation.n}<br> • pn_o: {simulation.pn_o}<br> • p_SES_high: {simulation.p_SES_high}<br> • rho: {simulation.rho}<br> • pm_x: {simulation.pm_o / simulation.rho}<br> • pn_x: {simulation.pn_o / simulation.rho}"

        # combine everything
        all_texts = [text_H_shares, text_attr_assortativity, text_edge_type_counts, text_encpn, text_largest_cc, text_parameters]
        
        final_text = '<br>'.join(all_texts)
        return final_text




    '''ACTUAL DRAWINGS'''

    # basic plotly draw
    def plotly_draw(self, simulation, t=-1, layout='spring', draw_largest_CC=True, legend=False, title = None):
        graph = simulation.graph_history[t]

        # only draw the largest connected component (since there are always a few nodes with no connections which makes the graph ugly)
        if draw_largest_CC == True:
            largest_CC = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_CC).copy()

        SES_list = list(simulation.helper_functions().get_select_node_attributes(graph, 'SES_High', graph.nodes()).values())
        SES_color_list = ['blue' if SES == 1 else 'red' for SES in SES_list]


        # node texts
        '''also add share of H/L neighbour and degree'''
        node_text = [str(i) + '_' + str(j) for i, j in zip(list(graph.nodes()), SES_list)]

        graph_layout = nx.spring_layout(graph, seed=42, scale=10) if layout =='spring' else nx.random_layout(graph)

        edge_trace = self.go_get_edge_trace(graph, graph_layout)
        node_trace = self.go_get_node_trace(graph, graph_layout, SES_color_list, node_text)


        # Get annotations if needed
        if legend == True:
            # legend: parameters, realised values of a bunch of things
            legend_text = self.get_legend_text(simulation, graph)
            annotations = [dict(
                            text=legend_text,
                            align='left',
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=1.35, y=1)]
            l_margin = 205
            width, height = 800, 600
        
        else:
            annotations = []
            l_margin = 5
            width, height = 600, 600


        # make custom title
        if title == None:
            title_dict = None
            t_margin = 5

        else:
            title = f'' if title == '' else title
            title_dict = {'text': title, 'x': 0.5, 'y': 0.98}
            t_margin = 30

        '''actual figure'''
        fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title_dict,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=5,l=5,r=l_margin,t=t_margin),
                        annotations=annotations,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        width=width, height=height)                      
                        )


        return fig

