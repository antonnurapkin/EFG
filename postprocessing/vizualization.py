from matplotlib import pyplot as plt
from plotly import graph_objects as go
import numpy as np
from scipy.interpolate import griddata

from postprocessing.crack import create_upper_crack_bound, create_lower_crack_bound
from postprocessing.helpers import calculate_coeff, get_max


def show_displacement(nodes, nodes_coords):

    u = np.array([node.u_real for node in nodes])
    v = np.array([node.v_real for node in nodes])

    x = nodes_coords[0]
    y = nodes_coords[1]

    xr = np.linspace(0, 1, 100)
    yr = np.linspace(0, 1, 100)
    xr, yr = np.meshgrid(xr, yr)

    U = griddata((x, y), u, (xr, yr), method='cubic')
    V = griddata((x, y), v, (xr, yr), method='cubic')

    max_data = get_max(disp=u, x=x, y=y)
    create_contourplot(x=xr[0], y=yr[:, 0], z=U, axis="X", value="Перемещения", max=max_data)

    max_data = get_max(disp=v, x=x, y=y)
    create_contourplot(x=xr[0], y=yr[:, 0], z=V, axis="Y", value="Перемещения", max=max_data)


def create_contourplot(x, y, z, axis, value, max=None):
    fig = go.Figure(go.Contour(x=x, y=y, z=z,
                               colorscale='jet',
                               ncontours=12,
                               contours=dict(start=np.min(z),
                                             end=np.max(z),
                                             size=(np.max(z) - np.min(z)) / 8),
                               colorbar=dict(
                                   exponentformat='power', showexponent="last"
                               )))

    y_up, x_up = create_upper_crack_bound()
    y_low, x_low = create_lower_crack_bound()

    fig.add_trace(go.Scatter(x=x_low, y=y_low, mode="lines", name='', line=dict(width=0.1, color='rgb(255, 255, 255)')))
    fig.add_trace(go.Scatter(x=x_up, y=y_up, mode="lines", fill='tonexty', fillcolor="white", name='',
                             line=dict(width=0.1, color='rgb(255, 255, 255)')))


    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    fig.update_layout(title_text=f'{value} вдоль оси {axis}',
                      title_x=0.5,
                      width=800, height=800)

    fig.show()


def create_scatterplot(x1, y1, x2, y2, indexes):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x1, y=y1, text=indexes, mode="markers", name="Недеформированное состояние", opacity=0.5))
    fig.add_trace(go.Scatter(x=x2, y=y2, text=indexes, mode="markers", name="Деформированное состояние"))

    fig.update_layout(title_text=f'Деформированное и недеформированное состояние',
                      title_x=0.5,
                      width=1000, height=800)

    fig.show()


def show_deformed_shape(nodes, nodes_coords, a, b):
    u = np.array([node.u_real for node in nodes])
    v = np.array([node.v_real for node in nodes])

    coeff = calculate_coeff(a=a, b=b, u=u, v=v)

    x1 = nodes_coords[0]
    y1 = nodes_coords[1]

    x2 = np.array([node.x + node.u_real * coeff for node in nodes])
    y2 = np.array([node.y + node.v_real * coeff for node in nodes])

    global_indexes = np.array([node.global_index for node in nodes])

    create_scatterplot(x1=x1, y1=y1, x2=x2, y2=y2, indexes=global_indexes)


def show_geometry(nodes_coords, integration_points):
    x = [point.x for point in integration_points]
    y = [point.y for point in integration_points]

    fig = go.Figure(go.Scatter(x=nodes_coords[0], y=nodes_coords[1], mode="markers", name="Узлы"))
    fig.add_trace(go.Scatter(x=x, y=y, opacity=0.5, mode="markers", name="Точки Гаусса"))
    fig.update_layout(width=800, height=800)
    fig.show()


def show_stress(nodes_coords, stress, integration_points):

    x, y = [], []

    for point in integration_points:
        x.append(point.x)
        y.append(point.y)

    fig = go.Figure(go.Contour(x=x, y=y, z=stress[0],

                               colorscale='jet',
                               ncontours=12,
                               # contours=dict(start=np.min(stress[0]),
                               #               end=np.max(stress[0]),
                               #               size=(np.max(stress[0]) - np.min(stress[0])) / 8),
                               line=dict(width=0),
                               colorbar=dict(exponentformat='power', showexponent="last")))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    y_up, x_up = create_upper_crack_bound()
    y_low, x_low = create_lower_crack_bound()

    fig.add_trace(go.Scatter(x=x_low, y=y_low, mode="lines", name='', line=dict(width=0.1, color='rgb(255, 255, 255)')))
    fig.add_trace(go.Scatter(x=x_up, y=y_up, mode="lines", fill='tonexty', fillcolor="white", name='',
                             line=dict(width=0.1, color='rgb(255, 255, 255)')))

    fig.update_layout(title_text=f'Напряжения вдоль оси X',
                      title_x=0.5,
                      width=800, height=800)

    fig.show()

    fig = go.Figure(go.Contour(x=x, y=y, z=stress[1],
                               colorscale='jet',
                               ncontours=12,
                               # contours=dict(start=np.min(stress[1]),
                               #               end=np.max(stress[1]),
                               #               size=(np.max(stress[1]) - np.min(stress[1])) / 8),
                               line=dict(width=0),
                               colorbar=dict(exponentformat='power', showexponent="last")))

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    y_up, x_up = create_upper_crack_bound()
    y_low, x_low = create_lower_crack_bound()

    fig.add_trace(go.Scatter(x=x_low, y=y_low, mode="lines", name='', line=dict(width=0.1, color='rgb(255, 255, 255)')))
    fig.add_trace(go.Scatter(x=x_up, y=y_up, mode="lines", fill='tonexty', fillcolor="white", name='',
                             line=dict(width=0.1, color='rgb(255, 255, 255)')))

    fig.update_layout(title_text=f'Напряжения вдоль оси Y',
                      title_x=0.5,
                      width=800, height=800)

    fig.show()






