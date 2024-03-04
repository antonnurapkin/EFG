import numpy as np
from plotly import graph_objects as go
import pandas as pd

from nodes import create_nodes, get_nodes_coords
from integration_points import create_integration_points, create_integration_points_bound
from matrices.K_matrix import K_global
from matrices.f_vector import f_global
from matrices.G_matrix import G_global


a = 1
b = 1
r0 = a / 4

nodes_number = 4
elems = nodes_number - 1
t = np.array([[0], [100]])

fi_delta = np.pi / (2 * (elems * 2))
nodes = create_nodes(nodes_number=nodes_number, a=a, b=b, fi_delta=fi_delta, r0=r0)
nodes_coords = get_nodes_coords(nodes=nodes)
integration_points = create_integration_points(nodes_coords=nodes_coords, nodes_number=nodes_number)

K = K_global(integration_points=integration_points, nodes=nodes, nodes_coords=nodes_coords)
f = f_global(nodes=nodes, nodes_coords=nodes_coords, t=t, x_bound=True, x_value=1)
G = G_global(
    nodes=nodes,
    nodes_coords=nodes_coords,
    nodes_number=nodes_number,
    x_bound=True,
    x_value=0.0,
    y_bound=True,
    y_value=0.0
)

K_extended = np.block([[K, G], [np.transpose(G), np.zeros((G.shape[1], G.shape[1]))]])
f_extended = np.vstack((f, np.zeros((G.shape[1], 1))))

df = pd.DataFrame(G)
df.to_excel("G.xlsx")
# u = np.dot(np.linalg.inv(K_extended), f_extended)


#
#
# points = []
# for point in integration_points:
#     points.append([point.x, point.y])
#
# points = np.array(points)
#
# indexes = [str(node.global_index) for node in nodes]
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers'))
# fig.add_trace(go.Scatter(x=nodes_coords[0], y=nodes_coords[1], mode='markers', text=indexes))
# fig.show()