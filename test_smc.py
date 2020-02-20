from math import sqrt
from smc_function import *
import time

# 仅限二维数组
def matrix_sum(matrix):
    sum = 0
    matrix = numpy.array(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
           sum += matrix[i][j]
    return sum

data_x_width = 100
data_x_height = 100
data_y_width = 1
data_y_height = 1
entity_smc = smc_functions(10, 1e-10)
x_1 = numpy.random.uniform(-1, 1,(data_x_width, data_x_height))
x_2 = numpy.random.uniform(-1, 1,(data_x_width, data_x_height))
y_1 = numpy.random.uniform(0, 0.5,(data_y_width, data_y_height))
y_2 = numpy.random.uniform(0, 0.5,(data_y_width, data_y_height))

x = x_1 + x_2
y = y_1 + y_2

# time_start = time.time()
# for i in range(30):
#     spm_1, spm_2 = entity_smc.SecPointMul(x_1, x_2, y_1, y_2)
# time_end = time.time()
# print("Time is : ", time_end - time_start)
#
# ssigmoid_1, ssigmoid_2 = entity_smc.SSigmoid(y_1, y_2)
# stanh_1, stanh_2 = entity_smc.STanh(y_1, y_2)

# print("Point Multiplication")
# # print(x * y)
# # print(spm_1 + spm_2)
# print((x * y - (spm_1 + spm_2)).max())
#
# print("Dot Multiplication")
# # print(numpy.dot(x, y))
# # print(sdm_1 + sdm_2)
# print((numpy.dot(x, y) - (sdm_1 + sdm_2)).max())
#
#
se_1, se_2 = entity_smc.SecExp(y_1, y_2)
print("Exponentiation")
# print(numpy.exp(y))
# print(se_1 + se_2)
print((numpy.exp(y) - (se_1 + se_2)).max())
#print((numpy.exp(-y) - (se_3 + se_4)).max())
#
# print("Reciprocal")
# # print(1 / y)
# # print(si_1 + si_2)
# print((1 / y - (si_1 + si_2)).max())

# print("Sigmoid")
# # print(1.0 / (1.0 + numpy.exp(-y)))
# # print(ssigmoid_1 + ssigmoid_2)
# # print((1.0 / (1.0 + numpy.exp(-y)) - (ssigmoid_1 + ssigmoid_2)).max())
# print((1.0 / (1.0 + numpy.exp(-y)) - (ssigmoid_1 + ssigmoid_2)).sum() / (data_y_width * data_y_height))
#
# print("Tanh")
# # # print(2.0 / (1.0 + numpy.exp(-2 * y)) - 1.0)
# # # print(stanh_1 + stanh_2)
# #print(((2.0 / (1.0 + numpy.exp(-2 * y)) - 1.0) - (stanh_1 + stanh_2)).max())
# sum = matrix_sum((2.0 / (1.0 + numpy.exp(-2 * y)) - 1.0) - (stanh_1 + stanh_2))
# #print(((2.0 / (1.0 + numpy.exp(-2 * y)) - 1.0) - (stanh_1 + stanh_2)))
# print(sum / (data_y_width * data_y_height))

# print("Ln x")
# ln_1, ln_2 = entity_smc.SecLog(y_1, y_2)
# print(numpy.log(y))
# print(entity_smc.SecLog(y_1, y_2))
# print((numpy.log(y) - (ln_1 + ln_2)).max())