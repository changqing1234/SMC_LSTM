import numpy as numpy
from smc_function import *
import time

max_iteration_round = 10            # secure function's max iteration rounds
approximation_precision = 1e-10     # secure function's iterative precision

class SigmoidActivator(object):
    def __init__(self):
        self.smc = smc_functions(max_iteration_round, approximation_precision)

    def forward(self, u_1, u_2):
        return self.smc.SSigmoid(u_1, u_2)

    def backward(self, u_1, u_2):
        return self.smc.SecPointMul(u_1, u_2, 1 - u_1, -u_2)


class TanhActivator(object):
    def __init__(self):
        self.smc = smc_functions(max_iteration_round, approximation_precision)

    def forward(self, u_1, u_2):
        return self.smc.STanh(u_1, u_2)

    def backward(self, u_1, u_2):
        f_1, f_2 = self.smc.SecPointMul(u_1, u_2, u_1, u_2)
        f_1 = 1- f_1
        f_2 = -f_2
        return f_1, f_2

class orginalSigmoidActivator(object):
    def forward(self, weighted_inumpyut):
        return 1.0 / (1.0 + numpy.exp(-weighted_inumpyut))

    def backward(self, output):
        return output * (1 - output)


class originalTanhActivator(object):
    def forward(self, weighted_inumpyut):
        return 2.0 / (1.0 + numpy.exp(-2 * weighted_inumpyut)) - 1.0

    def backward(self, output):
        return 1 - output * output

class SMC_LstmLayer(object):
    #################################################################
                            #矩阵初始化过程#
        # N = input_width;   D = feature_num   H = state_width #
    #################################################################
    def __init__(self, input_width, state_width, feature_num, learning_rate):
        self.smc = smc_functions(max_iteration_round, approximation_precision)
        self.input_width = input_width
        self.state_width = state_width
        self.feature_num = feature_num
        self.learning_rate = learning_rate
        # 门的激活函数
        self.gate_activator = SigmoidActivator()
        self.original_gate_activator = orginalSigmoidActivator()
        # 输出的激活函数
        self.output_activator = TanhActivator()
        self.orginal_output_activator = originalTanhActivator()
        # 当前时刻初始化为t0
        self.times = 0
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_vec()
        self.c_list_1 = self.init_state_vec()
        self.c_list_2 = self.init_state_vec()
        # 各个时刻的输出向量h
        self.h_list = self.init_state_vec()
        self.h_list_1 = self.init_state_vec()
        self.h_list_2 = self.init_state_vec()
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_vec()
        self.f_list_1 = self.init_state_vec()
        self.f_list_2 = self.init_state_vec()
        # 各个时刻的输入门i
        self.i_list = self.init_state_vec()
        self.i_list_1 = self.init_state_vec()
        self.i_list_2 = self.init_state_vec()
        # 各个时刻的输出门o
        self.o_list = self.init_state_vec()
        self.o_list_1 = self.init_state_vec()
        self.o_list_2 = self.init_state_vec()
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_vec()
        self.ct_list_1 = self.init_state_vec()
        self.ct_list_2 = self.init_state_vec()
        # 遗忘门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wfh, self.Wfx, self.bf = (
            self.init_weight_mat())
        self.Wfh_1, self.Wfh_2 = self.smc.initParameters(self.Wfh)
        self.Wfx_1, self.Wfx_2 = self.smc.initParameters(self.Wfx)
        self.bf_1, self.bf_2 = self.smc.initParameters(self.bf)
        # 输入门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wih, self.Wix, self.bi = (
            self.init_weight_mat())
        self.Wih_1, self.Wih_2 = self.smc.initParameters(self.Wih)
        self.Wix_1, self.Wix_2 = self.smc.initParameters(self.Wix)
        self.bi_1, self.bi_2 = self.smc.initParameters(self.bi)
        # 输出门权重矩阵Wfh, Wfx, 偏置项bf
        self.Woh, self.Wox, self.bo = (
            self.init_weight_mat())
        self.Woh_1, self.Woh_2 = self.smc.initParameters(self.Woh)
        self.Wox_1, self.Wox_2 = self.smc.initParameters(self.Wox)
        self.bo_1, self.bo_2 = self.smc.initParameters(self.bo)
        # 单元状态权重矩阵Wfh, Wfx, 偏置项bf
        self.Wch, self.Wcx, self.bc = (
            self.init_weight_mat())
        self.Wch_1, self.Wch_2 = self.smc.initParameters(self.Wch)
        self.Wcx_1, self.Wcx_2 = self.smc.initParameters(self.Wcx)
        self.bc_1, self.bc_2 = self.smc.initParameters(self.bc)

    def init_state_vec(self):
        '''
        初始化保存状态的向量  # N = input_width;   D = feature_num   H = state_width #
        '''
        state_vec_list = []
        state_vec_list.append(numpy.zeros(
            (self.input_width, self.state_width)))
        return state_vec_list

    def init_weight_mat(self):
        '''
        初始化权重矩阵  # N = input_width;   D = feature_num   H = state_width #
        '''
        Wh = numpy.random.uniform(-1e-2, 1e-2,
            (self.state_width, self.state_width))
        Wx = numpy.random.uniform(-1e-2, 1e-2,
            (self.feature_num, self.state_width))
        b = numpy.zeros((1, self.state_width))
        return Wh, Wx, b

    def reset_state(self):
        # 当前时刻初始化为t0
        self.times = 0
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_vec()
        self.c_list_1 = self.init_state_vec()
        self.c_list_2 = self.init_state_vec()
        # 各个时刻的输出向量h
        self.h_list = self.init_state_vec()
        self.h_list_1 = self.init_state_vec()
        self.h_list_2 = self.init_state_vec()
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_vec()
        self.f_list_1 = self.init_state_vec()
        self.f_list_2 = self.init_state_vec()
        # 各个时刻的输入门i
        self.i_list = self.init_state_vec()
        self.i_list_1 = self.init_state_vec()
        self.i_list_2 = self.init_state_vec()
        # 各个时刻的输出门o
        self.o_list = self.init_state_vec()
        self.o_list_1 = self.init_state_vec()
        self.o_list_2 = self.init_state_vec()
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_vec()
        self.ct_list_1 = self.init_state_vec()
        self.ct_list_2 = self.init_state_vec()





    #################################################################
        # 前向计算过程
    #################################################################
    def forward(self, x, x_1, x_2):
        '''
        根据式1-式6进行前向计算
        '''
        self.times += 1
        forget_time_start = time.time()
        # 遗忘门
        fg, fg_1, fg_2 = self.calc_gate(x, x_1, x_2, self.Wfx, self.Wfx_1, self.Wfx_2, self.Wfh, self.Wfh_1, self.Wfh_2,
                            self.bf, self.bf_1, self.bf_2, self.gate_activator, self.original_gate_activator)
        self.f_list.append(fg)
        self.f_list_1.append(fg_1)
        self.f_list_2.append(fg_2)
        forget_time_end = time.time()
        #print("forget Computation time is : ", forget_time_end - forget_time_start)
        #print("\n")

        # 输入门
        input_time_start_1 = time.time()
        ig, ig_1, ig_2 = self.calc_gate(x, x_1, x_2, self.Wix, self.Wix_1, self.Wix_2, self.Wih, self.Wih_1, self.Wih_2,
                            self.bi, self.bi_1, self.bi_2, self.gate_activator, self.original_gate_activator)
        self.i_list.append(ig)
        self.i_list_1.append(ig_1)
        self.i_list_2.append(ig_2)
        input_time_end_1 = time.time()

        # 输出门
        output_time_start_1 = time.time()
        og, og_1, og_2 = self.calc_gate(x, x_1, x_2, self.Wox, self.Wox_1, self.Wox_2, self.Woh, self.Woh_1, self.Woh_2,
                            self.bo, self.bo_1, self.bo_2, self.gate_activator, self.original_gate_activator)
        self.o_list.append(og)
        self.o_list_1.append(og_1)
        self.o_list_2.append(og_2)
        output_time_end_1 = time.time()
        
        # 即时状态
        input_time_start_2 = time.time()
        ct, ct_1, ct_2 = self.calc_gate(x, x_1, x_2, self.Wcx, self.Wcx_1, self.Wcx_2, self.Wch, self.Wch_1, self.Wch_2,
                            self.bc, self.bc_1, self.bc_2, self.output_activator, self.orginal_output_activator)
        self.ct_list.append(ct)
        self.ct_list_1.append(ct_1)
        self.ct_list_2.append(ct_2)

        # 单元状态
        c = fg * self.c_list[self.times - 1] + ig * ct
        c_11, c_12 = self.smc.SecPointMul(fg_1, fg_2, self.c_list_1[self.times - 1], self.c_list_2[self.times - 1])
        c_21, c_22 = self.smc.SecPointMul(ig_1, ig_2, ct_1, ct_2)
        c_1 = c_11 + c_21
        c_2 = c_12 + c_22
        input_time_end_2 = time.time()
        #print("Input Computation time is : ",
        #      (input_time_end_1 - input_time_start_1) + (input_time_end_2 - input_time_start_2))
        #print("\n")

        self.c_list.append(c)
        self.c_list_1.append(c_1)
        self.c_list_2.append(c_2)

        # 输出
        output_time_start_2 = time.time()
        h = og * self.orginal_output_activator.forward(c)
        h_11, h_12 = self.output_activator.forward(c_1, c_2)
        h_1, h_2 = self.smc.SecPointMul(og_1, og_2, h_11, h_12)
        output_time_end_2 = time.time()
        #print("Output Computation time is : ",
        #      (output_time_end_1 - output_time_start_1) + (output_time_end_2 - output_time_start_2))
        #print("\n")

        self.h_list.append(h)
        self.h_list_1.append(h_1)
        self.h_list_2.append(h_2)

    def calc_gate(self, x, x_1, x_2, Wx, Wx_1, Wx_2, Wh, Wh_1, Wh_2, b, b_1, b_2, activator, original_activator):
        '''
        计算门
        '''
        h = self.h_list[self.times - 1] # 上次的LSTM输出
        h_1 = self.h_list_1[self.times - 1]  # 上次的LSTM输出
        h_2 = self.h_list_2[self.times - 1]  # 上次的LSTM输出

        net = numpy.dot(h, Wh) + numpy.dot(x, Wx) + b
        dot_11, dot_12 = self.smc.SecDotMul(h_1, h_2, Wh_1, Wh_2)
        dot_21, dot_22 = self.smc.SecDotMul(x_1, x_2, Wx_1, Wx_2)
        net_1 = dot_11 + dot_21 + b_1
        net_2 = dot_12 + dot_22 + b_2
        print(net)
        # gate = activator.forward(net)
        gate = original_activator.forward(net)
        gate_1, gate_2 = activator.forward(net_1, net_2)
        '''print(gate)
        print("------------------------------------------------------")
        print(gate_1+gate_2)'''
        debug_gate = gate_1 + gate_2
        return gate, gate_1, gate_2


    #################################################################
        # 反向传播
    #################################################################
    def backward(self, x, x_1, x_2, delta_h, delta_h_1, delta_h_2):
        '''
        实现LSTM训练算法
        '''
        self.calc_delta(delta_h, delta_h_1, delta_h_2)
        self.calc_gradient(x, x_1, x_2)

    def calc_delta(self, delta_h, delta_h_1, delta_h_2):
        # 初始化各个时刻的误差项
        self.delta_h_list = self.init_delta()   # 输出误差项
        self.delta_h_list_1, self.delta_h_list_2 = self.smc.initParameters(self.delta_h_list)
        self.delta_o_list = self.init_delta()   # 输出门误差项
        self.delta_o_list_1, self.delta_o_list_2 = self.smc.initParameters(self.delta_o_list)
        self.delta_i_list = self.init_delta()   # 输入门误差项
        self.delta_i_list_1, self.delta_i_list_2 = self.smc.initParameters(self.delta_i_list)
        self.delta_f_list = self.init_delta()   # 遗忘门误差项
        self.delta_f_list_1, self.delta_f_list_2 = self.smc.initParameters(self.delta_f_list)
        self.delta_ct_list = self.init_delta()  # 即时输出误差项
        self.delta_ct_list_1, self.delta_ct_list_2 = self.smc.initParameters(self.delta_ct_list)

        # 保存从上一层传递下来的当前时刻的误差项
        self.delta_h_list[-1] = delta_h
        self.delta_h_list_1[-1] = delta_h_1
        self.delta_h_list_2[-1] = delta_h_2

        # 迭代计算每个时刻的误差项
        for k in range(self.times, 0, -1):
            self.calc_delta_k(k)

    def init_delta(self):
        '''
        初始化误差项    # N = input_width;   D = feature_num   H = state_width #
        '''
        delta_list = []
        for i in range(self.times + 1):
            delta_list.append(numpy.zeros(
                (self.input_width, self.state_width)))
        return delta_list

    def calc_delta_k(self, k):
        '''
        根据k时刻的delta_h，计算k时刻的delta_f、
        delta_i、delta_o、delta_ct，以及k-1时刻的delta_h
        '''
        # 获得k时刻前向计算的值
        ig = self.i_list[k]
        ig_1 = self.i_list_1[k]
        ig_2 = self.i_list_2[k]

        og = self.o_list[k]
        og_1 = self.o_list_1[k]
        og_2 = self.o_list_2[k]

        fg = self.f_list[k]
        fg_1 = self.f_list_1[k]
        fg_2 = self.f_list_2[k]

        ct = self.ct_list[k]
        ct_1 = self.ct_list_1[k]
        ct_2 = self.ct_list_2[k]

        c = self.c_list[k]
        c_1 = self.c_list_1[k]
        c_2 = self.c_list_2[k]

        c_prev = self.c_list[k - 1]
        c_prev_1 = self.c_list_1[k - 1]
        c_prev_2 = self.c_list_2[k - 1]

        pre_start_time = time.time()
        tanh_c = self.orginal_output_activator.forward(c)
        tanh_c_1, tanh_c_2 = self.output_activator.forward(c_1, c_2) # tanh, 输出为c
        d_tanh_c_1, d_tanh_c_2 = self.smc.Deri_STanh(c_1, c_2) # tanh 的导数，输入为c

        delta_k = self.delta_h_list[k]
        delta_k_1 = self.delta_h_list_1[k]
        delta_k_2 = self.delta_h_list_2[k]

        # 计算 delta_k * og * (1 - tanh_c * tanh_c)
        delta_temp_1, delta_temp_2 = self.smc.SecPointMul(delta_k_1, delta_k_2, og_1, og_2)
        delta_temp_1, delta_temp_2 = self.smc.SecPointMul(delta_temp_1, delta_temp_2, d_tanh_c_1, d_tanh_c_2)
        pre_end_time = time.time()
        #print("Preparation time is : ", pre_end_time - pre_start_time)

        o_start_time = time.time()
        delta_o = (delta_k * tanh_c * self.original_gate_activator.backward(og))
        delta_o_11, delta_o_12 = self.smc.SecPointMul(delta_k_1, delta_k_2, tanh_c_1, tanh_c_2)
        delta_o_21, delta_o_22 = self.gate_activator.backward(og_1, og_2)
        delta_o_1, delta_o_2 = self.smc.SecPointMul(delta_o_11, delta_o_12, delta_o_21, delta_o_22)
        o_end_time = time.time()
        #print("O time is : ", o_end_time - o_start_time)

        f_start_time = time.time()
        delta_f = (delta_k * og * (1 - tanh_c * tanh_c) * c_prev * self.original_gate_activator.backward(fg))
        delta_f_11, delta_f_12 = self.smc.SecPointMul(delta_temp_1, delta_temp_2, c_prev_1, c_prev_2)
        delta_f_21, delta_f_22 = self.gate_activator.backward(fg_1, fg_2)
        delta_f_1, delta_f_2 = self.smc.SecPointMul(delta_f_11, delta_f_12, delta_f_21, delta_f_22)
        f_end_time = time.time()
        #print("f time is : ", f_end_time - f_start_time)

        i_start_time = time.time()
        delta_i = (delta_k * og * (1 - tanh_c * tanh_c) * ct * self.original_gate_activator.backward(ig))
        delta_i_11, delta_i_12 = self.smc.SecPointMul(delta_temp_1, delta_temp_2, ct_1, ct_2)
        delta_i_21, delta_i_22 = self.gate_activator.backward(ig_1, ig_2)
        delta_i_1, delta_i_2 = self.smc.SecPointMul(delta_i_11, delta_i_12, delta_i_21, delta_i_22)
        i_end_time = time.time()
        #print("i time is : ", i_end_time - i_start_time)

        ct_start_time = time.time()
        delta_ct = (delta_k * og * (1 - tanh_c * tanh_c) * ig * self.orginal_output_activator.backward(ct))
        delta_ct_11, delta_ct_12 = self.smc.SecPointMul(delta_temp_1, delta_temp_2, ig_1, ig_2)
        delta_ct_21, delta_ct_22 = self.output_activator.backward(ct_1, ct_2)
        delta_ct_1, delta_ct_2 = self.smc.SecPointMul(delta_ct_11, delta_ct_12, delta_ct_21, delta_ct_22)
        ct_end_time = time.time()
        #print("ct time is : ", ct_end_time - ct_start_time)

        delta_h_prev = (
            numpy.dot(delta_o, self.Woh) +
            numpy.dot(delta_i, self.Wih) +
            numpy.dot(delta_f, self.Wfh) +
            numpy.dot(delta_ct, self.Wch)
        )
        dot_o_1, dot_o_2 = self.smc.SecDotMul(delta_o_1, delta_o_2, self.Woh_1, self.Woh_2)
        dot_i_1, dot_i_2 = self.smc.SecDotMul(delta_i_1, delta_i_2, self.Wih_1, self.Wih_2)
        dot_f_1, dot_f_2 = self.smc.SecDotMul(delta_f_1, delta_f_2, self.Wfh_1, self.Wfh_2)
        dot_ct_1, dot_ct_2 = self.smc.SecDotMul(delta_ct_1, delta_ct_2, self.Wch_1, self.Wch_2)
        delta_h_prev_1 = (dot_o_1 + dot_i_1 + dot_f_1 + dot_ct_1)
        delta_h_prev_2 = (dot_o_2 + dot_i_2 + dot_f_2 + dot_ct_2)

        # 保存全部delta值
        self.delta_h_list[k - 1] = delta_h_prev
        self.delta_h_list_1[k - 1] = delta_h_prev_1
        self.delta_h_list_2[k - 1] = delta_h_prev_2

        self.delta_f_list[k] = delta_f
        self.delta_f_list_1[k] = delta_f_1
        self.delta_f_list_2[k] = delta_f_2

        self.delta_i_list[k] = delta_i
        self.delta_i_list_1[k] = delta_i_1
        self.delta_i_list_2[k] = delta_i_2

        self.delta_o_list[k] = delta_o
        self.delta_o_list_1[k] = delta_o_1
        self.delta_o_list_2[k] = delta_o_2

        self.delta_ct_list[k] = delta_ct
        self.delta_ct_list_1[k] = delta_ct_1
        self.delta_ct_list_2[k] = delta_ct_2


    def calc_gradient(self, x, x_1, x_2):
        # 初始化遗忘门权重梯度矩阵和偏置项
        self.Wfh_grad, self.Wfx_grad, self.bf_grad, \
        self.Wfh_grad_1, self.Wfh_grad_2, self.Wfx_grad_1, self.Wfx_grad_2, self.bf_grad_1, self.bf_grad_2 = (
            self.init_weight_gradient_mat())
        # 初始化输入门权重梯度矩阵和偏置项
        self.Wih_grad, self.Wix_grad, self.bi_grad,\
        self.Wih_grad_1, self.Wih_grad_2, self.Wix_grad_1, self.Wix_grad_2, self.bi_grad_1, self.bi_grad_2 = (
            self.init_weight_gradient_mat())
        # 初始化输出门权重梯度矩阵和偏置项
        self.Woh_grad, self.Wox_grad, self.bo_grad,\
        self.Woh_grad_1, self.Woh_grad_2, self.Wox_grad_1, self.Wox_grad_2, self.bo_grad_1, self.bo_grad_2 = (
            self.init_weight_gradient_mat())
        # 初始化单元状态权重梯度矩阵和偏置项
        self.Wch_grad, self.Wcx_grad, self.bc_grad,\
        self.Wch_grad_1, self.Wch_grad_2, self.Wcx_grad_1, self.Wcx_grad_2, self.bc_grad_1, self.bc_grad_2 = (
            self.init_weight_gradient_mat())

        # 计算对上一次输出h的权重梯度
        update_start_time = time.time()
        for t in range(self.times, 0, -1):
            # 计算各个时刻的梯度
            (Wfh_grad, bf_grad, Wfh_grad_1, Wfh_grad_2, bf_grad_1, bf_grad_2,
             Wih_grad, bi_grad, Wih_grad_1, Wih_grad_2, bi_grad_1, bi_grad_2,
             Woh_grad, bo_grad, Woh_grad_1, Woh_grad_2, bo_grad_1, bo_grad_2,
             Wch_grad, bc_grad, Wch_grad_1, Wch_grad_2, bc_grad_1, bc_grad_2) = (
                self.calc_gradient_t(t))

            # 实际梯度是各时刻梯度之和
            self.Wfh_grad += Wfh_grad
            self.Wfh_grad_1 += Wfh_grad_1
            self.Wfh_grad_2 += Wfh_grad_2
            self.bf_grad = numpy.sum(bf_grad, axis=0)
            self.bf_grad_1 = numpy.sum(bf_grad_1, axis=0)
            self.bf_grad_2 = numpy.sum(bf_grad_2, axis=0)

            self.Wih_grad += Wih_grad
            self.Wih_grad_1 += Wih_grad_1
            self.Wih_grad_2 += Wih_grad_2
            self.bi_grad = numpy.sum(bi_grad, axis=0)
            self.bi_grad_1 = numpy.sum(bi_grad_1, axis=0)
            self.bi_grad_2 = numpy.sum(bi_grad_2, axis=0)

            self.Woh_grad += Woh_grad
            self.Woh_grad_1 += Woh_grad_1
            self.Woh_grad_2 += Woh_grad_2
            self.bo_grad = numpy.sum(bo_grad, axis=0)
            self.bo_grad_1 = numpy.sum(bo_grad_1, axis=0)
            self.bo_grad_2 = numpy.sum(bo_grad_2, axis=0)

            self.Wch_grad += Wch_grad
            self.Wch_grad_1 += Wch_grad_1
            self.Wch_grad_2 += Wch_grad_2
            self.bc_grad = numpy.sum(bc_grad, axis=0)
            self.bc_grad_1 = numpy.sum(bc_grad_1, axis=0)
            self.bc_grad_2 = numpy.sum(bc_grad_2, axis=0)

        # 计算对本次输入x的权重梯度
        xt = x.transpose()
        xt_1 = x_1.transpose()
        xt_2 = x_2.transpose()
        self.Wfx_grad = numpy.dot(xt, self.delta_f_list[-1])
        self.Wfx_grad_1, self.Wfx_grad_2 = \
            self.smc.SecDotMul(xt_1, xt_2, self.delta_f_list_1[-1], self.delta_f_list_2[-1])
        self.Wix_grad = numpy.dot(xt, self.delta_i_list[-1])
        self.Wix_grad_1, self.Wix_grad_2 = \
            self.smc.SecDotMul(xt_1, xt_2, self.delta_i_list_1[-1], self.delta_i_list_2[-1])
        self.Wox_grad = numpy.dot(xt, self.delta_o_list[-1])
        self.Wox_grad_1, self.Wox_grad_2 = \
            self.smc.SecDotMul(xt_1, xt_2, self.delta_o_list_1[-1], self.delta_o_list_2[-1])
        self.Wcx_grad = numpy.dot(xt, self.delta_ct_list[-1])
        self.Wcx_grad_1, self.Wcx_grad_2 = \
            self.smc.SecDotMul(xt_1, xt_2, self.delta_ct_list_1[-1], self.delta_ct_list_2[-1])

        update_end_time = time.time()
        #print("Update time is : ", update_end_time - update_start_time)

    def init_weight_gradient_mat(self):
        '''
        初始化权重矩阵      # N = input_width;   D = feature_num   H = state_width #
        '''
        Wh_grad = numpy.zeros((self.state_width, self.state_width))
        Wh_grad_1, Wh_grad_2 = self.smc.initParameters(Wh_grad)
        Wx_grad = numpy.zeros((self.input_width, self.state_width))
        Wx_grad_1, Wx_grad_2 = self.smc.initParameters(Wx_grad)
        b_grad = numpy.zeros((1, self.state_width))
        b_grad_1, b_grad_2 = self.smc.initParameters(b_grad)
        return Wh_grad, Wx_grad, b_grad, Wh_grad_1, Wh_grad_2, Wx_grad_1, Wx_grad_2, b_grad_1, b_grad_2

    def calc_gradient_t(self, t):
        '''
        计算每个时刻t权重的梯度
        '''
        h_prev = self.h_list[t - 1].transpose()
        h_prev_1 = self.h_list_1[t - 1].transpose()
        h_prev_2 = self.h_list_2[t - 1].transpose()

        Wfh_grad = numpy.dot(h_prev, self.delta_f_list[t])
        bf_grad = self.delta_f_list[t]
        Wfh_grad_1, Wfh_grad_2 = self.smc.SecDotMul(h_prev_1, h_prev_2, self.delta_f_list_1[t], self.delta_f_list_2[t])
        bf_grad_1 = self.delta_f_list_1[t]
        bf_grad_2 = self.delta_f_list_2[t]

        Wih_grad = numpy.dot(h_prev, self.delta_i_list[t])
        bi_grad = self.delta_f_list[t]
        Wih_grad_1, Wih_grad_2 = self.smc.SecDotMul(h_prev_1, h_prev_2, self.delta_i_list_1[t], self.delta_i_list_2[t])
        bi_grad_1 = self.delta_f_list_1[t]
        bi_grad_2 = self.delta_f_list_2[t]

        Woh_grad = numpy.dot(h_prev, self.delta_o_list[t])
        bo_grad = self.delta_f_list[t]
        Woh_grad_1, Woh_grad_2 = self.smc.SecDotMul( h_prev_1, h_prev_2, self.delta_o_list_1[t], self.delta_o_list_2[t])
        bo_grad_1 = self.delta_f_list_1[t]
        bo_grad_2 = self.delta_f_list_2[t]

        Wch_grad = numpy.dot(h_prev, self.delta_ct_list[t])
        bc_grad = self.delta_ct_list[t]
        Wch_grad_1, Wch_grad_2 = self.smc.SecDotMul(h_prev_1, h_prev_2, self.delta_ct_list_1[t], self.delta_ct_list_2[t])
        bc_grad_1 = self.delta_ct_list_1[t]
        bc_grad_2 = self.delta_ct_list_2[t]

        return Wfh_grad, bf_grad, Wfh_grad_1, Wfh_grad_2, bf_grad_1, bf_grad_2, \
                Wih_grad, bi_grad, Wih_grad_1, Wih_grad_2, bi_grad_1, bi_grad_2, \
                Woh_grad, bo_grad, Woh_grad_1, Woh_grad_2, bo_grad_1, bo_grad_2, \
                Wch_grad, bc_grad, Wch_grad_1, Wch_grad_2, bc_grad_1, bc_grad_2

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        self.Wfh -= self.learning_rate * self.Wfh_grad
        self.Wfh_1 -= self.learning_rate * self.Wfh_grad_1
        self.Wfh_2 -= self.learning_rate * self.Wfh_grad_2
        self.Wfx -= self.learning_rate * self.Wfx_grad
        self.Wfx_1 -= self.learning_rate * self.Wfx_grad_1
        self.Wfx_2 -= self.learning_rate * self.Wfx_grad_2
        self.bf -= self.learning_rate * self.bf_grad
        self.bf_1 -= self.learning_rate * self.bf_grad_1
        self.bf_2 -= self.learning_rate * self.bf_grad_2
        self.Wih -= self.learning_rate * self.Wih_grad
        self.Wih_1 -= self.learning_rate * self.Wih_grad_1
        self.Wih_2 -= self.learning_rate * self.Wih_grad_2
        self.Wix -= self.learning_rate * self.Wix_grad
        self.Wix_1 -= self.learning_rate * self.Wox_grad_1
        self.Wix_2 -= self.learning_rate * self.Wox_grad_2
        self.bi -= self.learning_rate * self.bi_grad
        self.bi_1 -= self.learning_rate * self.bi_grad_1
        self.bi_2 -= self.learning_rate * self.bi_grad_2
        self.Woh -= self.learning_rate * self.Woh_grad
        self.Woh_1 -= self.learning_rate * self.Woh_grad_1
        self.Woh_2 -= self.learning_rate * self.Woh_grad_2
        self.Wox -= self.learning_rate * self.Wox_grad
        self.Wox_1 -= self.learning_rate * self.Wox_grad_1
        self.Wox_2 -= self.learning_rate * self.Wox_grad_2
        self.bo -= self.learning_rate * self.bo_grad
        self.bo_1 -= self.learning_rate * self.bo_grad_1
        self.bo_2 -= self.learning_rate * self.bo_grad_2
        self.Wch -= self.learning_rate * self.Wch_grad
        self.Wch_1 -= self.learning_rate * self.Wch_grad_1
        self.Wch_2 -= self.learning_rate * self.Wch_grad_2
        self.Wcx -= self.learning_rate * self.Wcx_grad
        self.Wcx_1 -= self.learning_rate * self.Wcx_grad_1
        self.Wcx_2 -= self.learning_rate * self.Wcx_grad_2
        self.bc -= self.learning_rate * self.bc_grad
        self.bc_1 -= self.learning_rate * self.bc_grad_1
        self.bc_2 -= self.learning_rate * self.bc_grad_2