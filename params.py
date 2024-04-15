import numpy as np


'''ПАРАМЕТРЫ МАТЕРИАЛА'''
mu = 0.3
E = 2e11
G = E / (2 * (1 + mu))

D_init_array = [[1, mu, 0],
                [mu, 1, 0],
                [0, 0, (1 - mu) / 2]]

D_init_const = E / (1 - mu ** 2)

D = D_init_const * np.array(D_init_array)


'''ПАРАМЕТРЫ ГЕОМЕТРИИ'''
A = 1  # Размер пластины в направлении оси Х
B = 2  # Размер пластины в направлении оси Y
R0 = A / 5 # Радиус отверстия


'''ПАРАМЕТРЫ АППРОКСИМАЦИИ'''
METHOD = "Lagrange" # Cubic

NODES_NUMBER_ON_BOUND_X = 7
NODES_NUMBER_ON_BOUND_Y = 14
CELLS_NUMBER = 10

PENALTY = E * 100


ALPHA_X = 4  # Размер области поддержки
DC_X = np.min([A / NODES_NUMBER_ON_BOUND_X, B / NODES_NUMBER_ON_BOUND_Y])

DS = ALPHA_X * DC_X

WEIGHT_FUNCTION_TYPE = "cubic"  # quadratic


'''НАГРУЗКА'''
P = 10000

