from plotly import graph_objects as go
import numpy as np
from helpers import get_x_coord, get_y_coord
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def show_displacement(nodes, r0, nodes_number):
    x = np.array([node.x for node in nodes])
    y = np.array([node.y for node in nodes])

    xr = np.linspace(0, 1, 100)
    yr = np.linspace(0, 1, 100)
    xr, yr = np.meshgrid(xr, yr)

    u = np.array([node.u_real for node in nodes])
    v = np.array([node.v_real for node in nodes])

    # evaluate the z-values at the regular grid through cubic interpolation
    U = griddata((x, y), u, (xr, yr), method='cubic')
    V = griddata((x, y), v, (xr, yr), method='cubic')

    # f, ax = plt.subplots()
    # ax.tricontourf(xr, yr, U, 20)  # choose 20 contour levels, just to show how good its interpolation is
    # plt.show()

    create_figure(x=xr, y=yr, z=U, axis="X", r=r0)
    create_figure(x=xr, y=yr, z=V, axis="Y", r=r0)


def create_figure(x, y, z, axis, r):
    fig = go.Figure(go.Contour(x=x[0], y=y[:, 0], z=z,
                               colorscale='jet',
                               contours=dict(start=np.nanmin(z),
                                             end=np.nanmax(z))))
    # fig.add_shape(type="circle",
    #               xref="x", yref="y",
    #               fillcolor="White",
    #               x0=-r, y0=-r, x1=r, y1=r,
    #               line_width=0,
    #               )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(title_text=f'Перемещения вдоль оси {axis}',
                      title_x=0.5,
                      width=800, height=800)

    fig.show()


