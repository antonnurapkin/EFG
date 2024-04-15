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
from exact_solution import u_radial, stress_yy
from params import METHOD


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
    stress, points = calculate_stress(nodes_coords=nodes_coords, nodes=nodes)

    # # Перемещения v при y=0
    # x = nodes_coords[0]
    # ids = np.where(x == 0)[0]
    # y_bound = nodes_coords[1][ids]
    #
    # v_bound = np.array([node.v_real for node in nodes[ids]])
    #
    # fig = go.Figure(go.Scatter(x=y_bound, y=v_bound, name="Приближенное решение"))
    # fig.add_trace(go.Scatter(x=y_bound, y=u_radial(r=y_bound, tetta=0), name="Точное решение"))
    # fig.update_layout(title_text="Перемещение v на стороне x=0",
    #                   title_x=0.5,
    #                   width=1000,
    #                   height=800)
    #
    # fig.update_xaxes(
    #     title_text="y",
    #     title_font={"size": 25},
    #     title_standoff=25)
    #
    # fig.update_yaxes(
    #     exponentformat='power',
    #     showexponent="last",
    #     title_text="u",
    #     title_font={"size": 25},
    #     title_standoff=25
    # )
    #
    # fig.show()
    #
    # # Перемещения u при x=0
    # y = nodes_coords[1]
    # ids = np.where(y == 0)[0]
    # bottom = nodes_coords[0][ids]
    #
    # u_bound = np.array([node.u_real for node in nodes[ids]])
    #
    # fig = go.Figure(go.Scatter(x=bottom, y=u_bound, name="Приближенное решение"))
    # fig.add_trace(go.Scatter(x=bottom, y=u_radial(r=bottom, tetta=np.pi / 2), name="Точное решение"))
    # fig.update_layout(title_text="Перемещение u стороне y=0",
    #                   title_x=0.5,
    #                   width=1000,
    #                   height=800)
    #
    # fig.update_xaxes(
    #     title_text="x",
    #     title_font={"size": 25},
    #     title_standoff=25)
    #
    # fig.update_yaxes(
    #     exponentformat='power',
    #     showexponent="last",
    #     title_text="v",
    #     title_font={"size": 25},
    #     title_standoff=25
    # )
    # fig.show()
    #
    # show_displacement(nodes=nodes, nodes_coords=nodes_coords)
    # show_deformed_shape(nodes=nodes, a=A, b=B, nodes_coords=nodes_coords)

    coords = np.array([point.x for point in points])
    fig = go.Figure(go.Scatter(x=coords, y=stress[1], mode="lines", name="Приближенное решение"))
    fig.add_trace(go.Scatter(x=coords, y=stress_yy(r=coords, tetta=np.pi / 2), name="Точное решение"))

    fig.update_layout(title_text="Напряжения syy стороне y=0",
                                        title_x=0.5,
                                        width=1000,
                                        height=800)

    fig.update_xaxes(
        title_text="x",
        title_font={"size": 25},
        title_standoff=25)

    fig.update_yaxes(
        exponentformat='power',
        showexponent="last",
        title_text="Syy",
        title_font={"size": 25},
        title_standoff=25
    )

    fig.show()


if __name__ == "__main__":
    main()
