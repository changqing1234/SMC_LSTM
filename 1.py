import numpy as numpy
import random as random

#拆分矩阵数据p为p_1, p_2两个子矩阵
def initParameters(p):
    p = numpy.array(p)
    p_1 = numpy.random.uniform(p.min(), p.max(), p.shape)
    p_2 = p - p_1
    return p_1, p_2

#生成指定形状的随机数矩阵， 矩阵的值都在[0, 1)
def random_generator(shape):
    r = numpy.random.uniform(0, 1, shape)  # [0-1)里面随机产数，样本大小为shape，输出元组
    r_1 = r - numpy.random.uniform(0, r.min(), shape)  # 这步意思是为了保证下面的r2为正值
    r_2 = r - r_1
    return r_1, r_2

#矩阵点乘，对应位置的数据相乘。（必须是同形矩阵才能乘）
def secPointMul(x_1, x_2, y_1, y_2):
    a_1, a_2 = random_generator(numpy.array(x_1).shape)
    b_1, b_2 = random_generator(numpy.array(y_1).shape)

    #构建bevear三元组，生成两个随机数a, b,和c=a*b, 拆分a, b, c
    a_1, a_2 = random_generator(numpy.array(x_1).shape)
    b_1, b_2 = random_generator(numpy.array(y_1).shape)

    c = (a_1+a_2) * (b_1+b_2)
    c_1 = numpy.random.uniform(0, c.min(), c.shape)
    c_2 = c - c_1

    #安全乘法的计算过程

    #用随机数a, b加密输入数据x, y,分别发给对方，得到混淆后的alpha, beta
    alpha = (x_1 - a_1) + (x_2 - a_2)
    beta = (y_1 - b_1) + (y_2 - b_2)

    f_1 = c_1 + b_1 * alpha + a_1 * beta
    numpy.seterr(divide='ignore', invalid='ignore')
    f_2 = c_2 + b_2 * alpha + a_2 * beta + alpha * beta

    return f_1, f_2

#矩阵相乘，第一个矩阵的列等于第二个矩阵的行数




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

    f_1, f_2 = secPointMul(x_1, x_2, y_1, y_2)

    print(f_1,'\n\n', f_2)


    print(numpy.array(x) * numpy.array(y), '\n\n')
    print(f_1 + f_2)