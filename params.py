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
B = 1  # Размер пластины в направлении оси Y
R0 = A / 10  # Радиус отверстия


'''ПАРАМЕТРЫ АППРОКСИМАЦИИ'''
NODES_NUMBER_TETTA = 6
NODES_NUMBER_RADIAL = 12

FI_DELTA = np.pi / (2 * ((NODES_NUMBER_TETTA - 1) * 2))  # Шаг угла для разбиения

# Размер области поддержки
# x
ALPHA_X = 2.5
DC_X = ((1 / 4 * ALPHA_X) ** 2) / (np.sqrt(12) - 1)  # характеристическая длина ( расстояние между двумя узлами )

DS = ALPHA_X * DC_X

# Вид весовой функции
WEIGHT_FUNCTION_TYPE = "cubic"  # quadratic

'''НАГРУЗКА'''
t = np.array([[0], [1000]])  # [[tx], [ty]]


