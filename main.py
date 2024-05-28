import numpy as np

from matrices.K_penalty import K_penalty
from matrices.f_penalty import f_penalty
from nodes import create_nodes, get_nodes_coords
from integration_points import create_integration_points
from matrices.K_matrix import K_global
from matrices.f_vector import f_global
from matrices.G_matrix import G_global
from matrices.q_vector import q_global
from postprocessing.displacements import get_real_displacements
from postprocessing.stress import calculate_stress
from postprocessing.vizualization import show_displacement, show_deformed_shape, show_geometry, show_stress
from plotly import graph_objects as go
from params import METHOD, B, A


def main():
    # Создание узлов
    nodes, crack_top_ind = create_nodes()

    # Создание матрицы с координатами для удобства
    nodes_coords = get_nodes_coords(nodes=nodes)

    # Точки Гаусса
    integration_points = create_integration_points()

    show_geometry(nodes_coords=nodes_coords, integration_points=integration_points)

    # # Матрица жесткости
    K = K_global(
        nodes=nodes,
        integration_points=integration_points,
        nodes_coords=nodes_coords,
        crack_top_ind=crack_top_ind
    )

    # Вектор сил(алгоритм аналогичен матрице K)
    f = f_global(nodes=nodes, nodes_coords=nodes_coords, y_bound=True, y_value=1)

    if METHOD == "Lagrange":
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

        u = u_extended[:G.shape[0]]

    elif METHOD == "Penalty":
        K_pen = K_penalty(
            nodes=nodes,
            nodes_coords=nodes_coords,
            x_bound=True,
            x_value=0.0,
            y_bound=True,
            y_value=0.0
        )

        f_pen = f_penalty(
            nodes=nodes,
            nodes_coords=nodes_coords,
            x_bound=True,
            x_value=0.0,
            y_bound=True,
            y_value=0.0
        )

        K_extended = K + K_pen
        f_extended = f + f_pen

        u = np.dot(np.linalg.inv(K_extended), f_extended)
    print("Уравнение решено...")


    # Так как функции формы не соответсвуют символу Кронекера, то полученное решение не является действительным перемещением
    # Вычисление реальных перемещений на основе полученных узловых параметров
    nodes = get_real_displacements(nodes=nodes, u=u, coords=nodes_coords)

    show_displacement(nodes=nodes, nodes_coords=nodes_coords)
    show_deformed_shape(nodes=nodes, a=A, b=B, nodes_coords=nodes_coords)

    stress = calculate_stress(
        nodes_coords=nodes_coords,
        nodes=nodes,
        crack_top_ind=crack_top_ind,
        integration_points=integration_points
    )
    show_stress(nodes_coords=nodes_coords, stress=stress, integration_points=integration_points)


if __name__ == "__main__":
    main()
