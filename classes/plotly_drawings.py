
import networkx as nx
import numpy as np
import plotly.express as px

import plotly.graph_objects as go


class plotly_lines(object):
    
    basic_dict = {
        'title': None, 
        'font':dict(size = 18),
        'width': 500, 'height': 500,
        'legend': dict(x=0.75,y=1),
        'legend_title_text': '',
        'margin': dict(b=5,l=5,r=5,t=5),
        'xaxis': dict(showgrid = False, showline = True, linecolor = 'rgb(0,0,0)', linewidth = 2), 
        'yaxis': dict(showgrid = False, showline = True, linecolor = 'rgb(0,0,0)', linewidth = 2),
        'plot_bgcolor': 'rgba(0,0,0,0)'
    }

    # simple line graph
    def line_graph(self, x_vals, y_vals, param_dict, colors = None, hline_y = None, hline_text = None, vline_x = None, vline_text = None):
        simple_graph = px.line(x=x_vals, y=y_vals)


        layout_dict = param_dict | self.basic_dict

        full_graph = simple_graph.update_layout(layout_dict)

        # add lines if needed
        if (hline_y != None) & (hline_text != None):
            full_graph.add_hline(y=hline_y, line_width=1, annotation_text=hline_text)
        if (vline_x != None) & (vline_text != None):
            full_graph.add_vline(x=vline_x, line_width=1, annotation_text=vline_text)

        # add colours
        if colors != None:
            for i in range(len(colors)):
                full_graph.data[i].line.color = colors[i]

        return full_graph
    
    # scatter
    def scatter_graph(self, x_vals, y_vals, param_dict, colors = None, hline_y = None, hline_text = None, vline_x = None, vline_text = None):
        simple_graph = px.scatter(x=x_vals, y=y_vals)


        layout_dict = param_dict | self.basic_dict

        full_graph = simple_graph.update_layout(layout_dict)

        # add lines if needed
        if (hline_y != None) & (hline_text != None):
            full_graph.add_hline(y=hline_y, line_width=1, annotation_text=hline_text)
        if (vline_x != None) & (vline_text != None):
            full_graph.add_vline(x=vline_x, line_width=1, annotation_text=vline_text)

        # add colours
        if colors != None:
            for i in range(len(colors)):
                full_graph.data[i].line.color = colors[i]

        return full_graph


    # line with error bars
    def line_with_error(self, data, x_var, y_var, color, legend_text):

        x = data.groupby(x_var, as_index=False).mean()[x_var]
        y = data.groupby(x_var, as_index=False).mean()[y_var]
        y_upper = y + 1.96*data.groupby(x_var, as_index=False).std()[[y_var]][y_var] / len(data.groupby(x_var, as_index=False))
        y_lower = y - 1.96*data.groupby(x_var, as_index=False).std()[[y_var]][y_var] / len(data.groupby(x_var, as_index=False))

        #color = 'rgba(0,100,80,0)'
        color_std = color[:-3] + ',0.25)'

        fig = go.Figure([
            go.Scatter(
                x=x, 
                y=y,
                line=dict(color=color, width=2),
                mode='lines',
                showlegend=True,
                name=legend_text,
            ),
            go.Scatter(
                x=x, # x, then x reversed
                y=y_upper, # upper, then lower reversed
                #fill='toself',
                #fillcolor='rgba(0,100,80,0.2)',
                line=dict(color=color_std, width=2),
                mode='lines',
                showlegend=False
            ),
                go.Scatter(
                x=x, # x, then x reversed
                y=y_lower, # upper, then lower reversed
                #fill='toself',
                #fillcolor='rgba(0,100,80,0.2)',
                fillcolor='rgba(68, 68, 68, 0.2)',
                fill='tonexty',
                line=dict(color=color_std, width=2),
                mode='lines',
                showlegend=False
            )
        ])

        return fig
    
    # plot MTO results: assumes that we have "predicted" versus something else
    def plot_MTO_sim_results(self, data, x_var, y_var, x_range=[0,1], y_range=[0,1], x_title='', y_title='', legend_text = '', with_predicted=False, colors=['rgba(0,0,255,1)', 'rgba(255,0,0,1)']):

        if with_predicted:
            fig_realised = self.line_with_error(data, x_var, y_var, color=colors[0], legend_text='Realised')
            fig_predicted = self.line_with_error(data, x_var, f'Predicted_{y_var}', color=colors[1], legend_text='Predicted')
            fig = go.Figure(data = fig_realised.data + fig_predicted.data)
        
        else:
            fig_realised = self.line_with_error(data, x_var, y_var, color=colors[0], legend_text=legend_text)
            fig = go.Figure(data = fig_realised.data)


        fig.update_layout(title = None, 
            font=dict(size = 18),
            xaxis_title = x_title, 
            yaxis_title = y_title,
            width = 400, height = 400,
            showlegend = True, 
            legend=dict(x=0.6,y=1),
            legend_title_text = '',
            xaxis_range = x_range,
            yaxis_range = y_range,
            margin=dict(b=5,l=5,r=5,t=5),
            xaxis = dict(showgrid = False, showline = True, linecolor = 'rgb(0,0,0)', linewidth = 2),
            yaxis = dict(showgrid = False, showline = True, linecolor = 'rgb(0,0,0)', linewidth = 2),
            plot_bgcolor='rgba(0,0,0,0)')


        return fig




