from plotly import graph_objects as go
import numpy as np
from scipy.interpolate import griddata


def show_displacement(nodes, r0):
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

    create_contourplot(x=xr, y=yr, z=U, axis="X", r=r0)
    create_contourplot(x=xr, y=yr, z=V, axis="Y", r=r0)


def create_contourplot(x, y, z, axis, r):
    fig = go.Figure(go.Contour(x=x[0], y=y[:, 0], z=z,
                               colorscale='jet',
                               contours=dict(start=np.nanmin(z),
                                             end=np.nanmax(z))))
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  fillcolor="White",
                  x0=-r, y0=-r, x1=r, y1=r,
                  line_width=0,
                  )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(title_text=f'Перемещения вдоль оси {axis}',
                      title_x=0.5,
                      width=800, height=800)

    fig.show()


def calculate_coeff(a, b, u, v):
    related_coeff = 0.05 # Хочу, чтобы перемещения были примерно 1/10 от среднего размеры
    average_size = (a + b) / 2

    average_max_displ = (np.max(u) + np.max(v)) / 2

    return related_coeff * average_size / average_max_displ


def create_scatterplot(x1, y1, x2, y2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x1, y=y1, mode="markers", name="Недеформированное состояние", fillcolor='rgba(100,100,255,0.5)'))
    fig.add_trace(go.Scatter(x=x2, y=y2, mode="markers", name="Деформированное состояние"))

    fig.update_layout(title_text=f'Деформированное и недеформированное состояние',
                      title_x=0.5,
                      width=1000, height=800)

    fig.show()

def show_deformed_shape(nodes, a, b):
    u = np.array([node.u_real for node in nodes])
    v = np.array([node.v_real for node in nodes])

    coeff =calculate_coeff(a=a, b=b, u=u, v=v)

    x1 = np.array([node.x for node in nodes])
    y1 = np.array([node.y for node in nodes])

    x2 = np.array([node.x + node.u_real * coeff for node in nodes])
    y2 = np.array([node.y + node.v_real * coeff for node in nodes])

    create_scatterplot(x1=x1, y1=y1, x2=x2, y2=y2)
















