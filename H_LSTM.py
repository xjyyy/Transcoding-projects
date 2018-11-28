# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

VECTOR_LENGTH_LIST = [18, 38, 54]
LSTM_READ_LENGTH = 1  # online
LSTM_DEPTH = 2
n_steps=30
input_size_64=VECTOR_LENGTH_LIST[2]
input_size_32=VECTOR_LENGTH_LIST[1]
input_size_16=VECTOR_LENGTH_LIST[0]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)#0.01
    return tf.Variable(initial)

def conv2d(x0, w0,stride):
    return tf.nn.conv2d(x0, w0, strides=[1, stride, stride, 1], padding='VALID')

def Lstm64(x1,x2,x3,x4,x5,x6, state_in_tensor, input_size,name):
        #卷积层
        x1= tf.transpose(x1, [1, 0,2])#(batchsize,1,64)->(1,batchsize,64)
        x1=tf.reshape(x1,[1,-1,8,8,1])#->(1,batchsize,8,8,1)
        x2 = tf.transpose(x2, [1, 0, 2])
        x2 = tf.reshape(x2, [1, -1, 8, 8, 1])
        x3 = tf.transpose(x3, [1, 0, 2])
        x3 = tf.reshape(x3, [1, -1, 4, 4, 1])
        x4 = tf.transpose(x4, [1, 0, 2])
        x4 = tf.reshape(x4, [1, -1, 4, 4, 1])
        x5 = tf.transpose(x5, [1, 0, 2])
        x5 = tf.reshape(x5, [1, -1, 8, 8, 1])
        x6 = tf.transpose(x6, [1, 0, 2])
        x6 = tf.reshape(x6, [1, -1, 8, 8, 1])
        with tf.variable_scope(name):
            with tf.variable_scope("conv1"):
                w_conv_n_steps1 = tf.get_variable('Variable',[4, 4, 1, 1])
                b_conv_n_steps1 = tf.get_variable('Variable_1',[1])
                w_conv_n_steps2 = tf.get_variable('Variable_2',[4, 4, 1, 1])
                b_conv_n_steps2 = tf.get_variable('Variable_3',[1])
                w_conv_n_steps3 = tf.get_variable('Variable_4',[2, 2, 1, 1])
                b_conv_n_steps3 = tf.get_variable('Variable_5',[1])
                w_conv_n_steps4 = tf.get_variable('Variable_6',[2, 2, 1, 1])
                b_conv_n_steps4 = tf.get_variable('Variable_7',[1])
                w_conv_n_steps5 = tf.get_variable('Variable_8',[4, 4, 1, 1])
                b_conv_n_steps5 = tf.get_variable('Variable_9',[1])
                w_conv_n_steps6 = tf.get_variable('Variable_10',[4, 4, 1, 1])
                b_conv_n_steps6 = tf.get_variable('Variable_11',[1])

            with tf.variable_scope("lstm"):
                W = tf.get_variable('Variable',[input_size, 1])
                b = tf.get_variable('Variable_1',[1])

            y1_conv=tf.nn.relu(conv2d(x1[0], w_conv_n_steps1,stride=2)) + b_conv_n_steps1
            y1_bn=tf.layers.batch_normalization(y1_conv, training=False, trainable=True)
            y1 = tf.reshape(y1_bn, [LSTM_READ_LENGTH, -1, 3 * 3])
            # y1 = tf.reshape(y1_conv, [LSTM_READ_LENGTH, -1, 3 * 3])
            y2_conv=tf.nn.relu(conv2d(x2[0], w_conv_n_steps2,stride=2)) + b_conv_n_steps2
            y2_bn=tf.layers.batch_normalization(y2_conv, training=False, trainable=True)
            y2 = tf.reshape(y2_bn, [LSTM_READ_LENGTH, -1, 3 * 3])

            y3_conv=tf.nn.relu(conv2d(x3[0], w_conv_n_steps3,stride=1)) + b_conv_n_steps3
            y3_bn=tf.layers.batch_normalization(y3_conv, training=False, trainable=True)
            y3 = tf.reshape(y3_bn, [LSTM_READ_LENGTH, -1, 3 * 3])

            y4_conv=tf.nn.relu(conv2d(x4[0], w_conv_n_steps4,stride=1)) + b_conv_n_steps4
            y4_bn=tf.layers.batch_normalization(y4_conv, training=False, trainable=True)
            y4 = tf.reshape(y4_bn, [LSTM_READ_LENGTH, -1, 3 * 3])

            y5_conv=tf.nn.relu(conv2d(x5[0], w_conv_n_steps5,stride=2)) + b_conv_n_steps5
            y5_bn=tf.layers.batch_normalization(y5_conv, training=False, trainable=True)
            y5 = tf.reshape(y5_bn, [LSTM_READ_LENGTH,-1, 3 * 3])

            y6_conv=tf.nn.relu(conv2d(x6[0], w_conv_n_steps6,stride=2)) + b_conv_n_steps6
            y6_bn=tf.layers.batch_normalization(y6_conv, training=False, trainable=True)
            y6 = tf.reshape(y6_bn, [LSTM_READ_LENGTH, -1, 3 * 3])

            _X = tf.concat([y1, y2, y3, y4, y5, y6], 2)
            #连接lstm网络
            _X = tf.reshape(_X, [-1, input_size])  # (n_steps*batch_size, (n_input_3+n_input_4+n_input_2))
            with tf.variable_scope("rnn"):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(input_size, forget_bias=0.9, state_is_tuple=True)##????
                lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
                cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * LSTM_DEPTH)
                state_in = []
                for i_depth in range(LSTM_DEPTH):
                    state_in_tuple_temp = tf.contrib.rnn.LSTMStateTuple(
                        tf.reshape(state_in_tensor[:, i_depth, 0, :], [-1, input_size]),
                        tf.reshape(state_in_tensor[:, i_depth, 1, :], [-1, input_size]))
                    state_in.append(state_in_tuple_temp)
                XN = tf.split(_X, 1, 0)  # n_steps * (batch_size, n_hidden)
                cell_outputs = []
                state = state_in
                (cell_output_one_step, state) = cell(XN[0], state)
                cell_outputs.append(cell_output_one_step)
                state_out = state
                Y_ = tf.reshape(cell_outputs[0], [-1, input_size])
                y_pred_list = tf.matmul(Y_, W) + b
                y_pred_list = tf.sigmoid(y_pred_list)
                state_out_tensor_list = []
                for i_depth in range(LSTM_DEPTH):
                    state_out_tensor_c = tf.reshape(state_out[i_depth][0], [-1, 1, 1, input_size])
                    state_out_tensor_h = tf.reshape(state_out[i_depth][1], [-1, 1, 1, input_size])
                    state_out_tensor_curr_depth = tf.concat([state_out_tensor_c, state_out_tensor_h], axis=2)
                    state_out_tensor_list.append(state_out_tensor_curr_depth)
                state_out_tensor = tf.concat(state_out_tensor_list, axis=1)

        return y_pred_list, state_out_tensor
def Lstm32(x1,x2,x3,x4,x5,x6, state_in_tensor, input_size,name):
    x1 = tf.transpose(x1, [1, 0, 2])  # (batchsize,1,64)->(1,batchsize,64)
    x1 = tf.reshape(x1, [1, -1, 4, 4, 1])  # ->(1,batchsize,8,8,1)
    x2 = tf.transpose(x2, [1, 0, 2])
    x2 = tf.reshape(x2, [1, -1, 4, 4, 1])
    x3 = tf.transpose(x3, [1, 0, 2])
    x3 = tf.reshape(x3, [1, -1, 2, 2, 1])
    x4 = tf.transpose(x4, [1, 0, 2])
    x4 = tf.reshape(x4, [1, -1, 2, 2, 1])
    x5 = tf.transpose(x5, [1, 0, 2])
    x5 = tf.reshape(x5, [1, -1, 4, 4, 1])
    x6 = tf.transpose(x6, [1, 0, 2])
    x6 = tf.reshape(x6, [1, -1, 4, 4, 1])

    with tf.variable_scope(name):
        with tf.variable_scope("conv1"):
            w_conv_n_steps1 = tf.get_variable('Variable', [2, 2, 1, 1])
            b_conv_n_steps1 = tf.get_variable('Variable_1', [1])
            w_conv_n_steps2 = tf.get_variable('Variable_2', [2, 2, 1, 1])
            b_conv_n_steps2 = tf.get_variable('Variable_3', [1])
            w_conv_n_steps3 = tf.get_variable('Variable_4', [2, 2, 1, 1])
            b_conv_n_steps3 = tf.get_variable('Variable_5', [1])
            w_conv_n_steps4 = tf.get_variable('Variable_6', [2, 2, 1, 1])
            b_conv_n_steps4 = tf.get_variable('Variable_7', [1])
            w_conv_n_steps5 = tf.get_variable('Variable_8', [2, 2, 1, 1])
            b_conv_n_steps5 = tf.get_variable('Variable_9', [1])
            w_conv_n_steps6 = tf.get_variable('Variable_10', [2, 2, 1, 1])
            b_conv_n_steps6 = tf.get_variable('Variable_11', [1])
        with tf.variable_scope("lstm"):
            W = tf.get_variable('Variable', [input_size, 1])
            b = tf.get_variable('Variable_1', [1])

       # [1*batch_size,3,3,1]
        y1_conv = tf.nn.relu(conv2d(x1[0], w_conv_n_steps1, stride=1)) + b_conv_n_steps1
        y1_bn = tf.layers.batch_normalization(y1_conv, training=False, trainable=True)
        y1 = tf.reshape(y1_bn, [LSTM_READ_LENGTH, -1, 3 * 3])

        y2_conv = tf.nn.relu(conv2d(x2[0], w_conv_n_steps2, stride=1)) + b_conv_n_steps2
        y2_bn = tf.layers.batch_normalization(y2_conv, training=False, trainable=True)
        y2 = tf.reshape(y2_bn, [LSTM_READ_LENGTH, -1, 3 * 3])

        y3_conv = tf.nn.relu(conv2d(x3[0], w_conv_n_steps3, stride=1)) + b_conv_n_steps3
        y3_bn = tf.layers.batch_normalization(y3_conv, training=False, trainable=True)
        y3 = tf.reshape(y3_bn, [LSTM_READ_LENGTH, -1, 1])

        y4_conv = tf.nn.relu(conv2d(x4[0], w_conv_n_steps4, stride=1)) + b_conv_n_steps4
        y4_bn = tf.layers.batch_normalization(y4_conv, training=False, trainable=True)
        y4 = tf.reshape(y4_bn, [LSTM_READ_LENGTH, -1, 1])

        y5_conv = tf.nn.relu(conv2d(x5[0], w_conv_n_steps5, stride=1)) + b_conv_n_steps5
        y5_bn = tf.layers.batch_normalization(y5_conv, training=False, trainable=True)
        y5 = tf.reshape(y5_bn, [LSTM_READ_LENGTH, -1, 3 * 3])

        y6_conv = tf.nn.relu(conv2d(x6[0], w_conv_n_steps6, stride=1)) + b_conv_n_steps6
        y6_bn = tf.layers.batch_normalization(y6_conv, training=False, trainable=True)
        y6 = tf.reshape(y6_bn, [LSTM_READ_LENGTH, -1, 3 * 3])

        _X = tf.concat([y1, y2, y3, y4, y5, y6], 2)
        # 连接lstm网络
        _X = tf.reshape(_X, [-1, input_size])  # (1*batch_size,n_input)
        with tf.variable_scope("rnn"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(input_size, forget_bias=0.9, state_is_tuple=True)  ##????
            lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * LSTM_DEPTH)
            state_in = []
            for i_depth in range(LSTM_DEPTH):
                state_in_tuple_temp = tf.contrib.rnn.LSTMStateTuple(
                    tf.reshape(state_in_tensor[:, i_depth, 0, :], [-1, input_size]),
                    tf.reshape(state_in_tensor[:, i_depth, 1, :], [-1, input_size]))
                state_in.append(state_in_tuple_temp)
            XN = tf.split(_X, 1, 0)  # n_steps * (batch_size, n_hidden)
            cell_outputs = []
            state = state_in
            (cell_output_one_step, state) = cell(XN[0], state)
            cell_outputs.append(cell_output_one_step)
            state_out = state
            Y_ = tf.reshape(cell_outputs[0], [-1, input_size])
            y_pred_list = tf.matmul(Y_, W) + b
            y_pred_list = tf.sigmoid(y_pred_list)
            # y_pred_list = tf.round(y_pred_list)
            state_out_tensor_list = []
            for i_depth in range(LSTM_DEPTH):
                state_out_tensor_c = tf.reshape(state_out[i_depth][0], [-1, 1, 1, input_size])
                state_out_tensor_h = tf.reshape(state_out[i_depth][1], [-1, 1, 1, input_size])
                state_out_tensor_curr_depth = tf.concat([state_out_tensor_c, state_out_tensor_h], axis=2)
                state_out_tensor_list.append(state_out_tensor_curr_depth)
            state_out_tensor = tf.concat(state_out_tensor_list, axis=1)
    return y_pred_list, state_out_tensor

def Lstm16(_X, state_in_tensor, input_size,name):
    with tf.variable_scope(name):
        with tf.variable_scope("lstm"):
            W = tf.get_variable('Variable', [input_size, 1])
            b = tf.get_variable('Variable_1', [1])
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, input_size])  # (n_steps*batch_size, (n_input_3+n_input_4+n_input_2))
    with tf.variable_scope(name):
        with tf.variable_scope("rnn"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(input_size, forget_bias=0.9, state_is_tuple=True)  ##????
            lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * LSTM_DEPTH)
            state_in = []
            for i_depth in range(LSTM_DEPTH):
                state_in_tuple_temp = tf.contrib.rnn.LSTMStateTuple(
                    tf.reshape(state_in_tensor[:, i_depth, 0, :], [-1, input_size]),
                    tf.reshape(state_in_tensor[:, i_depth, 1, :], [-1, input_size]))
                state_in.append(state_in_tuple_temp)
            XN = tf.split(_X, 1, 0)  # n_steps * (batch_size, n_hidden)
            cell_outputs = []
            state = state_in
            (cell_output_one_step, state) = cell(XN[0], state)
            cell_outputs.append(cell_output_one_step)
            state_out = state
            Y_ = tf.reshape(cell_outputs[0], [-1, input_size])
            y_pred_list = tf.matmul(Y_, W) + b
            y_pred_list = tf.sigmoid(y_pred_list)
            # y_pred_list = tf.round(y_pred_list)
            state_out_tensor_list = []
            for i_depth in range(LSTM_DEPTH):
                state_out_tensor_c = tf.reshape(state_out[i_depth][0], [-1, 1, 1, input_size])
                state_out_tensor_h = tf.reshape(state_out[i_depth][1], [-1, 1, 1, input_size])
                state_out_tensor_curr_depth = tf.concat([state_out_tensor_c, state_out_tensor_h], axis=2)
                state_out_tensor_list.append(state_out_tensor_curr_depth)
            state_out_tensor = tf.concat(state_out_tensor_list, axis=1)
    return y_pred_list, state_out_tensor

def net(x_64_1, x_64_2,x_64_3,x_64_4,x_64_5,x_64_6,x_32_1,x_32_2,x_32_3,x_32_4,x_32_5,x_32_6, x_16, state_in_tensor_64, state_in_tensor_32, state_in_tensor_16):

    y_pred_flat_64_list, state_out_tensor_64 = Lstm64(x_64_1, x_64_2,x_64_3,x_64_4,x_64_5,x_64_6, state_in_tensor_64, input_size_64,"64x64")
    # opt_vars_all_LSTM64 = [v for v in tf.trainable_variables()]
    opt_vars_all_LSTM64 = [v for v in tf.global_variables()]
    y_pred_flat_32_list, state_out_tensor_32 = Lstm32(x_32_1,x_32_2,x_32_3,x_32_4,x_32_5,x_32_6, state_in_tensor_32, input_size_32,"32x32")
    opt_vars_all_LSTM32 = [v for v in tf.global_variables() if v not in opt_vars_all_LSTM64]
    # opt_vars_all_LSTM32 = [v for v in tf.trainable_variables()if v not in opt_vars_all_LSTM64]
    opt_vars_all_LSTM6432 = [v for v in tf.global_variables()]

    y_pred_flat_16_list, state_out_tensor_16 = Lstm16(x_16, state_in_tensor_16, input_size_16,'16x16')
    opt_vars_all_LSTM16 = [v for v in tf.global_variables() if v not in opt_vars_all_LSTM6432]
    # opt_vars_all_LSTM16 = [v for v in tf.trainable_variables() if v not in opt_vars_all_LSTM6432]
    # state_out_total = tf.concat([state_out_tensor_64, state_out_tensor_32, state_out_tensor_16], axis=3)

    y_pred_flat_64 = tf.reshape(tf.concat(y_pred_flat_64_list, axis=1), [-1, 1])
    y_pred_flat_32 = tf.reshape(tf.concat(y_pred_flat_32_list, axis=1), [-1, 4])
    y_pred_flat_16 = tf.reshape(tf.concat(y_pred_flat_16_list, axis=1), [-1, 16])

    # variables in LSTM are all trainable variables except those in CNN
    opt_vars_all_LSTM = [v for v in tf.trainable_variables()]
    # vs = tf.global_variables()
    # print('There are %d global_variables in the Graph: ' % len(vs))
    # for v in vs:
    #     print(v)

    return opt_vars_all_LSTM64,opt_vars_all_LSTM32,opt_vars_all_LSTM16, y_pred_flat_64, y_pred_flat_32, y_pred_flat_16, state_out_tensor_64, state_out_tensor_32, state_out_tensor_16
