import numpy as np

from nodes import create_nodes, get_nodes_coords
from integration_points import create_integration_points
from matrices.K_matrix import K_global
from matrices.f_vector import f_global
from matrices.G_matrix import G_global
from matrices.q_vector import q_global
from displacements import get_real_displacements
from params import a, nodes_number, elems, t, b, r0
from vizualization import show_displacement, show_deformed_shape


fi_delta = np.pi / (2 * (elems * 2))
nodes = create_nodes(nodes_number=nodes_number, a=a, b=b, fi_delta=fi_delta, r0=r0)
nodes_coords = get_nodes_coords(nodes=nodes)
integration_points = create_integration_points(nodes_coords=nodes_coords, nodes_number=nodes_number)

K = K_global(integration_points=integration_points, nodes=nodes, nodes_coords=nodes_coords)

f = f_global(nodes=nodes, nodes_coords=nodes_coords, t=t, y_bound=True, y_value=1)
G = G_global(
    nodes=nodes,
    nodes_coords=nodes_coords,
    nodes_number=nodes_number,
    x_bound=True,
    x_value=0.0,
    y_bound=True,
    y_value=0.0
)
q = q_global(
    nodes=nodes,
    nodes_coords=nodes_coords,
    rows=G.shape[1],
    x_bound=True,
    x_value=0.0,
    y_bound=True,
    y_value=0.0
)

K_extended = np.block([[K, G], [np.transpose(G), np.zeros((G.shape[1], G.shape[1]))]])
f_extended = np.vstack((f, np.zeros((G.shape[1], 1))))

u_extended = np.dot(np.linalg.inv(K_extended), f_extended)
print("Уравнение решено...")

u = u_extended[:G.shape[0]]
nodes = get_real_displacements(nodes=nodes, u=u, coords=nodes_coords)

show_displacement(nodes=nodes, r0=r0)
show_deformed_shape(nodes=nodes, a=a, b=b)
