import numpy as np


def calculate_coeff(a, b, u, v):
    related_coeff = 0.05  # Хочу, чтобы перемещения были примерно 1/10 от среднего размеры
    average_size = (a + b) / 2

    average_max_displ = (np.max(u) + np.max(v)) / 2

    return related_coeff * average_size / average_max_displ


def get_max(x, y, disp):
    disp = np.abs(disp)
    ind_max = np.where(disp == disp.max())[0]
    max_data = [disp[ind_max][0], x[ind_max][0], y[ind_max][0]]
    print(disp[ind_max])

    return max_data