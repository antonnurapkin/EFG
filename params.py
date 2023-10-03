import numpy as np


# Степень аппроксимирующего полинома
m = 1

""" ДАННЫЕ О СЕТКЕ """
n_x = 5
n_y = 5
n = n_x * n_y

# Длина сторон пластины
l_x = 1
l_y = 1

# Количество элементов
elem_x = n_x - 1
elem_y = n_y - 1

step_x = l_x / (n_x - 1)
step_y = l_y / (n_y - 1)

# Size of support domain
# x
dc_x = step_x  # charasteristic length ( length beetwen two nodes)
alpha_x = 2

ds_x = alpha_x * dc_x

# y
dc_y = step_y
alpha_y = 2

ds_y = alpha_y * dc_y


""" ПАРАМЕТРЫ МАТЕРИАЛА"""

mu = 0.3
E = 2e5

D = (E / 1 - mu ** 2) * np.array(
                                [
                                    [1, mu, 0],
                                    [mu, 1, 0],
                                    [0, 0, (1 - mu) / 2]
                                ]
                            )

"""НАГРУЖЕНИЕ"""
b = np.array([[0],[0],[-100]])



