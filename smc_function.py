import numpy as numpy
import random as random
numpy.seterr(divide='ignore', invalid='ignore')

class smc_functions(object):
    def __init__(self, iteration_round, precision):
        # iteration round. Determine the precision of approximation of SecExp, SecInv, SecSqrt and SecLog
        self.iteration_round = iteration_round
        self.precision = precision

    # split the input parameters p into two shares p = p_1 + p_2
    def initParameters(self, p):
        p = numpy.array(p)
        p_1 = numpy.random.uniform(p.min(), p.max(), p.shape)
        p_2 = p - p_1
        return p_1, p_2

    # generate uniform random r and split it into two shares where r_1 + r_2 = r
    def random_generator(self, shape):
        r = numpy.random.uniform(0, 1, shape)  # [0-1)里面随机产数，样本大小为shape，输出元组
        r_1 = r - numpy.random.uniform(0, r.min(), shape) # 这步意思是为了保证下面的r2为正值
        r_2 = r - r_1
        return r_1, r_2

    #f(x, y) = x * y
    def SecPointMul(self, x_1, x_2, y_1, y_2):
        a_1, a_2 = self.random_generator(numpy.array(x_1).shape)
        b_1, b_2 = self.random_generator(numpy.array(y_1).shape)

        c = (a_1 + a_2) * (b_1 + b_2)
        c_1 = c - numpy.random.uniform(0, c.min(), c.shape)
        c_2 = c - c_1

        alpha = (x_1 - a_1) + (x_2 - a_2)
        beta = (y_1 - b_1) + (y_2 - b_2)

        f_1 = c_1 + b_1 * alpha + a_1 * beta
        numpy.seterr(divide='ignore', invalid='ignore')
        f_2 = c_2 + b_2 * alpha + a_2 * beta + alpha * beta
        return f_1, f_2

    # #f(x, y) = x dot y
    def SecDotMul(self, x_1, x_2, y_1, y_2):
        a_1, a_2 = self.random_generator(numpy.array(x_1).shape)
        b_1, b_2 = self.random_generator(numpy.array(y_1).shape)
        c = numpy.dot((a_1 + a_2), (b_1 + b_2))
        c_1 = c - numpy.random.uniform(0, c.min(), c.shape)
        c_2 = c - c_1

        alpha = x_1 - a_1 + x_2 - a_2
        beta = y_1 - b_1 + y_2 - b_2

        f_1 = c_1 + numpy.dot(alpha, b_1) + numpy.dot(a_1, beta)
        f_2 = c_2 + numpy.dot(alpha, b_2) + numpy.dot(a_2, beta) + numpy.dot(alpha, beta)

        return f_1, f_2

    # f(x) = e^x
    def SecExp(self, u_1, u_2):
        g_1, g_2 = self.SecPointMul(u_1, u_2, 0.5 * u_1, 0.5 * u_2)
        f_1 = 1 + u_1 + g_1
        f_2 = u_2 + g_2
        index = 1

        #while numpy.fabs((g_1 + g_2)).max() > self.precision:
        while index < self.iteration_round:
            constant = 1 / (index + 2)
            g_1, g_2 = self.SecPointMul(g_1, g_2, constant * u_1, constant * u_2)
            f_1 += g_1
            f_2 += g_2
            index += 1
        #print(index)

        return f_1, f_2

    # f(x) = 1/x
    def SecInv(self, u_1, u_2):
        r_1, r_2 = self.random_generator(numpy.array(u_1).shape)
        alpha_1 = r_1 + u_1
        alpha_2 = r_2 + u_2

        f_1 = 0.5 / (u_1 + alpha_2)
        f_2 = 0.5 / (u_2 + alpha_1)
        index = 0

        while index < self.iteration_round:
            g_1, g_2 = self.SecPointMul(f_1, f_2, u_1, u_2)
            g_1 = 2 - g_1
            g_2 = - g_2
            f_1, f_2 = self.SecPointMul(f_1, f_2, g_1, g_2)
            index += 1

        return f_1, f_2

    # f(x) = sqrt(x)
    def SecSqrt(self, u_1, u_2):
        f_1 = u_1 / 2
        f_2 = u_2 / 2
        index = 0

        while index < self.iteration_round:
            w_1, w_2 = self.SecInv(f_1, f_2)
            g_1, g_2 = self.SecPointMul(w_1, w_2, u_1, u_2)
            f_1 = 0.5 * (g_1 + f_1)
            f_2 = 0.5 * (g_2 + f_2)
            index += 1

        return f_1, f_2

    # f(x) = ln x
    def SecLog(self, u_1, u_2):
        d_1, d_2 = self.SecInv(u_1 + 1, u_2)
        v_1, v_2 = self.SecPointMul(u_1 - 1, u_2, d_1, d_2)
        s_1, s_2 = self.SecPointMul(v_1, v_2, v_1, v_2)

        g_1 = v_1
        g_2 = v_2
        f_1 = g_1
        f_2 = g_2
        index = 1

        while numpy.fabs((g_1 + g_2).max()) > self.precision:
            g_1, g_2 = self.SecPointMul(g_1, g_2, s_1, s_2)
            f_1 += (1.0 /(2.0 * index + 1)) * g_1
            f_2 += (1.0 /(2.0 * index + 1)) * g_2
            index += 1

        return 2 * f_1, 2 * f_2

    # f(x) = sigmoid x
    def SSigmoid(self, u_1, u_2):
        f_1, f_2 = self.SecExp(-u_1, -u_2)
        f_1 += 1
        f_1, f_2 = self.SecInv(f_1, f_2)
        return f_1, f_2

    # f(x) = tanh x
    def STanh(self, u_1, u_2):
        f_1, f_2 = self.SSigmoid(2 * u_1, 2 * u_2)
        f_1 = 2 * f_1 - 1
        f_2 = 2 * f_2
        return f_1, f_2

    # f(x) = derivative of sigmoid x
    def Deri_SSigmoid(self, u_1, u_2):
        f_1, f_2 = self.SSigmoid(u_1, u_2)
        f_1, f_2 = self.SecPointMul(f_1, f_2, 1 - f_1, -f_2)
        return f_1, f_2

    # f(x) = derivative of tanh x
    def Deri_STanh(self, u_1, u_2):
        f_1, f_2 = self.STanh(u_1, u_2)
        f_1, f_2 = self.SecPointMul(f_1, f_2, f_1, f_2)
        f_1 = 1 - f_1
        f_2 = - f_2
        return f_1, f_2