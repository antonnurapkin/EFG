import numpy as np

from nodes import create_nodes, get_nodes_coords
from integration_points import create_integration_points
from matrices.K_matrix import K_global
from matrices.f_vector import f_global
from matrices.G_matrix import G_global
from matrices.q_vector import q_global
from postprocessing.displacements import get_real_displacements
from params import A, B
from postprocessing.stress import calculate_stress
from postprocessing.vizualization import show_displacement, show_deformed_shape, show_geometry, show_stress
from plotly import graph_objects as go
from exact_solution import u_radial
import pandas as pd


def main():
    # Создание узлов
    nodes = create_nodes()

    # Создание матрицы с координатами для удобства
    nodes_coords = get_nodes_coords(nodes=nodes)

    # Точки Гаусса
    integration_points = create_integration_points()

    show_geometry(nodes_coords=nodes_coords, integration_points=integration_points)

    # # Матрица жесткости
    K = K_global(nodes=nodes, integration_points=integration_points, nodes_coords=nodes_coords)

    # Вектор сил(алгоритм аналогичен матрице K)
    f = f_global(nodes=nodes, nodes_coords=nodes_coords, y_bound=True, y_value=1)

    # Матрица для соблюдения ГУ
    G = G_global(
        nodes=nodes,
        nodes_coords=nodes_coords,
        x_bound=True,
        x_value=0.0,
        y_bound=True,
        y_value=0.0
    )
    q = q_global(
        nodes_coords=nodes_coords,
        rows=G.shape[1],
        x_bound=True,
        x_value=0.0,
        y_bound=True,
        y_value=0.0
    )

    K_extended = np.block([[K, G], [np.transpose(G), np.zeros((G.shape[1], G.shape[1]))]])
    f_extended = np.vstack((f, q))

    u_extended = np.dot(np.linalg.inv(K_extended), f_extended)
    print("Уравнение решено...")

    u = u_extended[:G.shape[0]]
    # Так как функции формы не соответсвуют символу Кронекера, то полученное решение не является действительным перемещением
    # Вычисление реальных перемещений на основе полученных узловых параметров
    nodes = get_real_displacements(nodes=nodes, u=u, coords=nodes_coords)

    # Перемещения v при y=0
    x = nodes_coords[0]
    ids = np.where(x == 0)[0]
    y_bound = nodes_coords[1][ids]

    u_bound = np.array([node.v_real for node in nodes[ids]])
    v_bound = np.array([node.v_real for node in nodes[ids]])

    fig = go.Figure(go.Scatter(x=y_bound, y=v_bound, name="Приближенное решение"))
    fig.add_trace(go.Scatter(x=y_bound, y=u_radial(r=y_bound, tetta=0), name="Точное решение"))
    fig.update_layout(title_text="Перемещение v на стороне x=0",
                      title_x=0.5,
                      width=1000,
                      height=800)
    fig.show()

    # По диагонали
    coords = np.array([])
    ids = np.array(([]))
    for i in range(len(nodes_coords[0])):
        if np.round(nodes_coords[0][i], 5) == np.round(nodes_coords[1][i], 5):
            coords = np.append(coords, nodes_coords[1][i])
            ids = np.append(ids, i)

    r_arr = np.sqrt(np.square(coords) + np.square(coords))
    v_bound = np.array([node.v_real for node in nodes[ids.astype(int)]])

    fig = go.Figure(go.Scatter(x=r_arr, y=v_bound, name="Приближенное решение"))
    fig.add_trace(go.Scatter(x=r_arr, y=u_radial(r=r_arr, tetta=np.pi / 4), name="Точное решение"))

    fig.update_layout(title_text="Перемещение v при tetta=Pi/2",
                      title_x=0.5,
                      width=1000,
                      height=800)
    fig.show()

    show_displacement(nodes=nodes, nodes_coords=nodes_coords)
    show_deformed_shape(nodes=nodes, a=A, b=B, nodes_coords=nodes_coords)
    show_stress(nodes=nodes, nodes_coords=nodes_coords, u=u)


if __name__ == "__main__":
    main()
