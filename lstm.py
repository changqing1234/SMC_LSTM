from lstm_layer import *
from smc_lstm_layer import *
from smc_function import *
import read_data

def data_set():
    x = [numpy.array([[1], [2], [3]]),
         numpy.array([[2], [3], [4]])]
    #d = numpy.array([[1], [2]])
    return x

lr = 0.001                  # learning rate
hidden_unit = 16
batch_size = 1
n_inputs = 24               # MNIST data input(image shape:28*28)
n_step = 1                  # time steps
n_classes = 4               # MNIST classes (0-9 digits)
iteration = 10            # iteration round

# 申请基于SMC的安全运算符
instance_smc = smc_functions(max_iteration_round, approximation_precision)

# 设计一个误差函数，取所有节点输出项之和
error_function = lambda o: o.sum()
smc_error_function = lambda o_1, o_2: o_1.sum() + o_2.sum()

# lstm = LstmLayer(3, 2, lr)
smc_lstm = SMC_LstmLayer(n_step, hidden_unit, n_inputs, lr)

# x = data_set()
# lstm.forward(x[0])
# lstm.forward(x[1])
# # 求取sensitivity map
# sensitivity_array = numpy.ones(lstm.h_list[-1].shape,
#                             dtype=numpy.float64)
# # 计算梯度
# lstm.backward(x[1], sensitivity_array)
# print(lstm.delta_h_list[0])
# print('@@@@@@@@@@@@@@@')

# 读取数据
x, d = read_data.read_data(n_step)
x = x[0: batch_size]
# print(numpy.array(x[0]).shape)
x_1, x_2 = instance_smc.initParameters(x)

def lstm_training():
    # 计算forward值
    #d_1, d_2 = instance_smc.initParameters(d)
    #lstm.forward(x[0])
    #lstm.forward(x[1])

    index = 0
    fc_ave_per_error = 0
    for i in range(batch_size):
        fc_time_start = time.time()
        for j in range(n_step):
            smc_lstm.forward(x[i][j], x_1[i][j], x_2[i][j])
        fc_time_end = time.time()
        #print("FC Computation time is : ", fc_time_end - fc_time_start)
        #print("\n")

        fc_ave_error = numpy.fabs((smc_lstm.h_list[-1]) - (smc_lstm.h_list_1[-1] + smc_lstm.h_list_2[-1])).sum() / (n_inputs * n_step)
        fc_original_value = numpy.array(smc_lstm.h_list[-1]).sum() / (n_inputs * n_step)
        #print((fc_ave_error / fc_original_value).sum()/ (n_inputs * n_step))
        index += 1
        fc_ave_per_error += numpy.fabs((fc_ave_error / fc_original_value))

        fc_ave_per_error = fc_ave_per_error / batch_size
        #print(fc_ave_per_error)

        # 求取sensitivity map
        sensitivity_array = numpy.ones(smc_lstm.h_list[-1].shape,
                                dtype=numpy.float64)
        sensitivity_array_1 = numpy.random.uniform(0, 1.0, sensitivity_array.shape)
        sensitivity_array_2 = sensitivity_array - sensitivity_array_1

        # 计算梯度
        bp_time_start = time.time()
        #lstm.backward(x[1], sensitivity_array)
        smc_lstm.backward(x[index - 1], x_1[index - 1], x_2[index - 1], sensitivity_array, sensitivity_array_1, sensitivity_array_2)
        bp_time_end = time.time()
        #print("BP Computation time is : ", bp_time_end - bp_time_start)
        #print("\n")

        #print(numpy.array(smc_lstm.delta_h_list[0]).max())
        #(numpy.array(smc_lstm.delta_h_list_1[0] + smc_lstm.delta_h_list_2[0]).max())
        bp_ave_error = numpy.fabs((smc_lstm.delta_h_list[0]) -
                        (smc_lstm.delta_h_list_1[0] + smc_lstm.delta_h_list_2[0])).sum() / (n_inputs * n_step)
        bp_original_value = numpy.array(smc_lstm.delta_h_list[-1])
        bp_ave_per_error = (bp_ave_error / bp_original_value).sum()/ (n_inputs * n_step)
        fc_time_end = time.time()

        #print((bp_ave_error / bp_original_value).sum()/ (n_inputs * n_step))
        #print("\n")

    return fc_ave_error, bp_ave_error
#a,b = lstm_training()
#print(a)
#print("----------------------")
#print(b)
def gradient_check():
    '''
    梯度检查
    '''
    fc_train_error = 0
    bp_train_error = 0

    for i in range(iteration):
        fc_error, bp_error = lstm_training()
        fc_train_error += fc_error
        bp_train_error += bp_error

        #print(i)

        if i % 9 == 0 and i != 0:
            print("Current Iteration Round is : ", i)
            print("FC computation error")
            print(fc_train_error / i)
            print("BP computation error")
            print(bp_train_error / i)
            print("\n")

        # print("Output of LSTM:")
        # print(smc_lstm.h_list)
        # print("Gradient of LSTM:")
        # print(smc_lstm.delta_h_list)
        # print("SMC Output of LSTM:")
        # print(numpy.add(smc_lstm.h_list_1, smc_lstm.h_list_2))
        # print("SMC Gradient of LSTM:")
        # print(numpy.add(smc_lstm.delta_h_list_1, smc_lstm.delta_h_list_2))

        #检查梯度
        # epsilon = 10e-4
        # for i in range(smc_lstm.Wfh.shape[0]):
        #     for j in range(smc_lstm.Wfh.shape[1]):
        #         smc_lstm.Wfh[i, j] += epsilon
        #         # SMC 常量加法，只需要加其中一个sharing
        #         smc_lstm.Wfh_1[i, j] += epsilon
        #
        #         smc_lstm.reset_state()
        #
        #         #lstm.forward(x[0])
        #         #lstm.forward(x[1])
        #         # smc_lstm.forward(x[0], x_1[0], x_2[0])
        #         # smc_lstm.forward(x[1], x_1[1], x_2[1])
        #         for j in range(batch_size):
        #             smc_lstm.forward(x[j], x_1[j], x_2[j])
        #
        #         err1 = error_function(smc_lstm.h_list[-1])
        #         smc_err_1 = smc_error_function(smc_lstm.h_list_1[-1], smc_lstm.h_list_2[-1])
        #
        #         smc_lstm.Wfh[i,j] -= 2 * epsilon
        #         # SMC 常量加法，只需要加其中一个sharing
        #         smc_lstm.Wfh_1[i, j] -= 2 * epsilon
        #
        #         #lstm.reset_state()
        #         smc_lstm.reset_state()
        #
        #         #lstm.forward(x[0])
        #         #lstm.forward(x[1])
        #         # smc_lstm.forward(x[0], x_1[0], x_2[0])
        #         # smc_lstm.forward(x[1], x_1[1], x_2[1])
        #         for j in range(batch_size):
        #             smc_lstm.forward(x[j], x_1[j], x_2[j])
        #
        #         err2 = error_function(smc_lstm.h_list[-1])
        #         smc_err_2 = smc_error_function(smc_lstm.h_list_1[-1], smc_lstm.h_list_2[-1])
        #
        #         expect_grad = (err1 - err2) / (2 * epsilon)
        #         smc_expect_grad = (smc_err_1 - smc_err_2) / (2 * epsilon)
        #
        #         smc_lstm.Wfh[i, j] += epsilon
        #         smc_lstm.Wfh_1[i, j] += epsilon
        #
        #         print('weights(%d,%d): expected - actural %.4e - %.4e' % (
        #             i, j, expect_grad, smc_lstm.Wfh_grad[i,j]))
        #
        #         print('smc weights(%d,%d): expected - actural %.4e - %.4e' % (
        #             i, j, smc_expect_grad, smc_lstm.Wfh_grad_1[i, j] + smc_lstm.Wfh_grad_2[i, j]))
    # return lstm
gradient_check()