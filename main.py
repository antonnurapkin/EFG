import numpy as np
from plotly import graph_objects as go
import pandas as pd

from G_matrix import G_global
from K_matrix import K_global
from f_vector import f_global
from K_penalty import K_penalty_global
from meshing import create_cells
from params import n_x, n_y, l_x, l_y, elem_x, elem_y
from displacements import get_displaments_array

from numpy.linalg import eigvals


cells = create_cells(n_x=n_x, n_y=n_y, l_x=l_x, l_y=l_y)

K = K_global(cells=cells, n_x=n_x, n_y=n_y)
f = f_global(cells=cells, n_x=n_x, n_y=n_y)

# Lagrange multiplyers
# G = G_global(cells=cells, n_x=n_x, n_y=n_y)
# G_T = np.transpose(G)
# E = np.zeros((G.shape[1], G.shape[1]))
# q = np.zeros((G.shape[1], 1))
# K_extend = np.block([[K, G], [G_T, E]])
# f_extend = np.block([[f], [q]])

# Penalty method
K_penalty = K_penalty_global(cells=cells, n_x=n_x, n_y=n_y)
K_extend = K + K_penalty
f_extend = f
print("ГУ учтены")


u = np.linalg.solve(K_extend, f_extend)
U = get_displaments_array(cells, u[:f.shape[0]], n_x, n_y)

# u = np.transpose(np.reshape(u[:f.shape[0]], (n_y, n_x)))
df = pd.DataFrame(U)
df.to_excel("U.xlsx")


fig = go.Figure()
fig.add_trace(
    go.Surface(z=U, x=np.arange(0, l_x, 1 / elem_x), y=np.arange(0, l_y, 1 / elem_y))
)

fig.update_layout(yaxis_range=[-0.001, 0.001])

fig.show()