import numpy as np


#TODO: Посмотреть как подсчитыватся коэффициенты функций форм для нерегелурных сеток

'''ПАРАМЕТРЫ МАТЕРИАЛА и ГЕОМЕТРИИ'''
mu = 0.3
E = 2e11
G = E / (2 * (1 + mu))

D_init_array = [[1, mu, 0],
                [mu, 1, 0],
                [0, 0, (1 - mu) / 2]]

D_init_const = E / (1 - mu ** 2)

D = D_init_const * np.array(D_init_array)


'''ПАРАМЕТРЫ АППРОКСИМАЦИИ'''
# Размер области поддержки
# x
alpha_x = 3.5
dc_x = ((1 / 4 * alpha_x) ** 2) / (np.sqrt(12) - 1)# характеристическая длина ( расстояние между двумя узлами )


ds = alpha_x * dc_x
#
# # y
# dc_x = 1 / 6
# alpha = 2.5

# ds = alpha * dc_x

# Вид весовой функции
WEIGHT_FUNCTION_TYPE = "cubic"


a = 1
nodes_number = 9
elems = nodes_number - 1
t = np.array([[0], [100]])
b = 1
r0 = a / 10
