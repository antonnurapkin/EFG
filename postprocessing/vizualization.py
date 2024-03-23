from plotly import graph_objects as go
import numpy as np
from scipy.interpolate import griddata
from params import R0


def show_displacement(nodes, nodes_coords):
    x = nodes_coords[0]
    y = nodes_coords[1]

    xr = np.linspace(0, 1, 100)
    yr = np.linspace(0, 1, 100)
    xr, yr = np.meshgrid(xr, yr)

    u = np.array([node.u_real for node in nodes])
    v = np.array([node.v_real for node in nodes])

    # evaluate the z-values at the regular grid through cubic interpolation
    U = griddata((x, y), u, (xr, yr), method='cubic')
    V = griddata((x, y), v, (xr, yr), method='cubic')

    create_contourplot(x=xr[0], y=yr[:, 0], z=U, axis="X", value="Перемещения")
    create_contourplot(x=xr[0], y=yr[:, 0], z=V, axis="Y", value="Перемещения")


def create_contourplot(x, y, z, axis, value):
    fig = go.Figure(go.Contour(x=x, y=y, z=z,
                               colorscale='jet',
                               contours=dict(start=np.nanmin(z),
                                             end=np.nanmax(z))))
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  fillcolor="White",
                  x0=-R0, y0=-R0, x1=R0, y1=R0,
                  line_width=0,
                  )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(title_text=f'{value} вдоль оси {axis}',
                      title_x=0.5,
                      width=800, height=800)

    fig.show()


def calculate_coeff(a, b, u, v):
    related_coeff = 0.05  # Хочу, чтобы перемещения были примерно 1/10 от среднего размеры
    average_size = (a + b) / 2

    average_max_displ = (np.max(u) + np.max(v)) / 2

    return related_coeff * average_size / average_max_displ


def create_scatterplot(x1, y1, x2, y2, indexes):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x1, y=y1, text=indexes, mode="markers", name="Недеформированное состояние",
                             fillcolor='rgba(100,100,255,0.5)'))
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


def show_nodes(nodes_coords):
    fig = go.Figure(go.Scatter(x=nodes_coords[0], y=nodes_coords[1], mode="markers"))
    fig.update_layout(width=800, height=800)
    fig.show()


def show_stress(stress, integration_points):
    x = [point.x for point in integration_points]
    y = [point.x for point in integration_points]

    stress_xx = np.array([stress[0][i] for i in range(len(stress[0]))])
    stress_yy = np.array([stress[1][i] for i in range(len(stress[0]))])

    create_contourplot(x=x, y=y, z=stress_xx, axis="X", value="Напряжения")
    create_contourplot(x=x, y=y, z=stress_yy, axis="Y", value="Напряжения")
