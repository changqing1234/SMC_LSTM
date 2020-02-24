import numpy as numpy
import random as random

def initParameters(p):
    p = numpy.array(p)
    p_1 = numpy.random.uniform(p.min(), p.max(), p.shape)
    p_2 = p - p_1
    return p_1, p_2

def random_generator(shape):
    r = numpy.random.uniform(0, 1, shape)  # [0-1)里面随机产数，样本大小为shape，输出元组
    r_1 = r - numpy.random.uniform(0, r.min(), shape)  # 这步意思是为了保证下面的r2为正值
    r_2 = r - r_1
    return r_1, r_2


def secPointMul(x_1, x_2, y_1, y_2):
    a_1, a_2 = random_generator(numpy.array(x_1).shape)
    b_1, b_2 = random_generator(numpy.array(y_1).shape)


if __name__ == '__main__':
    x = [[12, 32, 4],
         [23, 4, 22],
         [34, 5, 33]]

    y =[[23, 5, 54],
        [44, 66, 3],
        [1, 3, 5] ]

    #数据拆分
    x_1, x_2 = initParameters(x)
    y_1, y_2 = initParameters(y)

    #构建bevear三元组，生成两个随机数a, b,和c=a*b, 拆分a, b, c
    a_1, a_2 = random_generator(numpy.array(x_1).shape)
    b_1, b_2 = random_generator(numpy.array(y_1).shape)

    c = (a_1+a_2) * (b_1+b_2)
    c_1 = numpy.random.uniform(0, c.min(), c.shape)
    c_2 = c - c_1

    #安全乘法的计算过程
    alpha = (x_1 - a_1) + (x_2 - a_2)
    beta = (y_1 - b_1) + (y_2 - b_2)






