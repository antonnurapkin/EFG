import numpy as np

from nodes import create_nodes, get_nodes_coords
from integration_points import create_integration_points
from matrices.K_matrix import K_global
from matrices.f_vector import f_global
from matrices.G_matrix import G_global
from matrices.q_vector import q_global
from postprocessing.displacements import get_real_displacements
from params import A, NODES_NUMBER_TETTA, NODES_NUMBER_RADIAL, B, R0, FI_DELTA
from postprocessing.vizualization import show_displacement, show_deformed_shape


def main():
    # Создание узлов
    nodes = create_nodes(
        nodes_number_tetta=NODES_NUMBER_TETTA,
        nodes_number_radial=NODES_NUMBER_RADIAL,
        a=A,
        b=B,
        fi_delta=FI_DELTA,
        r0=R0
    )

    # Создание матрицы с координатами для удобства
    nodes_coords = get_nodes_coords(nodes=nodes)

    # Точки Гаусса
    integration_points = create_integration_points(nodes_coords=nodes_coords, nodes_number=NODES_NUMBER_RADIAL)

    # Матрица жесткости
    K = K_global(integration_points=integration_points, nodes=nodes, nodes_coords=nodes_coords)

    # Вектор сил(алгоритм аналогичен матрице K)
    f = f_global(nodes=nodes, nodes_coords=nodes_coords, y_bound=True, y_value=1)

    # Матрица для соблюдения ГУ
    G = G_global(
        nodes=nodes,
        nodes_coords=nodes_coords,
        nodes_number=NODES_NUMBER_RADIAL,
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

    show_displacement(nodes=nodes, r0=R0)
    show_deformed_shape(nodes=nodes, a=A, b=B)

if __name__ == "__main__":
    main()
