import numpy as np
from plotly import graph_objects as go

from nodes import create_nodes, get_nodes_coords
from integration_points import create_integration_points


a = 1
b = 1
r0 = a / 4

nodes_number = 4
elems = nodes_number - 1

fi_delta = np.pi / (2 * (elems * 2))
nodes = create_nodes(nodes_number=nodes_number, a=a, b=b, fi_delta=fi_delta, r0=r0)
nodes_coords = get_nodes_coords(nodes=nodes)
integration_points = create_integration_points(nodes_coords=nodes_coords, nodes_number=nodes_number)


points = []
for point in integration_points:
    points.append([point.x, point.y])

points = np.array(points)

indexes = [str(node.global_index) for node in nodes]

fig = go.Figure()
fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers'))
fig.add_trace(go.Scatter(x=nodes_coords[0], y=nodes_coords[1], mode='markers', text=indexes))
fig.show()