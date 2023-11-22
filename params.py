import numpy as np

'''ПАРАМЕТРЫ МАТЕРИАЛА и ГЕОМЕТРИИ'''
mu = 0.3
E = 2e11

# Толщина пластина
THICKNESS = 0.05

D_init_array = [[1, mu, 0],
                [mu, 1, 0],
                [0, 0, (1 - mu) / 2]]

D_init_const = (E * THICKNESS ** 3) / (12 * (1 - mu ** 2))

D = D_init_const * np.array(D_init_array)

# Длина сторон пластины
l_x = 2
l_y = 2

'''ПАРАМЕТРЫ СЕТКИ'''
# Количество узлов
n_x = 15
n_y = 15
n = n_x * n_y

# Количество элементов
elem_x = n_x - 1
elem_y = n_y - 1

step_x = l_x / (n_x - 1)
step_y = l_y / (n_y - 1)

'''ПАРАМЕТРЫ АППРОКСИМАЦИИ'''
# Размер области поддержки
# x
dc_x = step_x  # характеристическая длина ( расстояние между двумя узлами )
alpha_x = 4

ds_x = alpha_x * dc_x

# y
dc_y = step_y
alpha_y = alpha_x

ds_y = alpha_y * dc_y

# Коэффициент штрафа
penalty_alpha = 10e6 * E

# Вид весовой функции
WEIGHT_FUNCTION_TYPE = "quadratic"

'''ПАРАМЕТРЫ НАГРУЖЕНИЯ'''
b = np.array([[0], [0], [-5e6]])
