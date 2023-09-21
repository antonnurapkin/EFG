# Степень аппроксимирующего полинома
m = 1

# Данные о сетке
n_x = 5
n_y = 5
n = n_x * n_y

# Number of elems
elem_x = n_x - 1
elem_y = n_y - 1

l_x = 1
l_y = 1

step_x = l_x / (n_x - 1)
step_y = l_y / (n_y - 1)

# Size of support domain
dc_x = step_x #charasteristic length ( length beetwen two nodes)
alpha_x = 2

ds_x = alpha_x * dc_x 


dc_y = step_y
alpha_y = 2

ds_y = alpha_y * dc_y
