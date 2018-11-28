# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import numpy
import H_LSTM as nt
import math
import tensorflow as tf
import os
import time

VECTOR_LENGTH_LIST = [18, 38, 54]
LSTM_DEPTH = 2

#第一层分类器的输入
x_64_1 = tf.placeholder(tf.float32, [None, 1,64])
x_64_2 = tf.placeholder(tf.float32, [None, 1,64])
x_64_3 = tf.placeholder(tf.float32, [None,1, 16])
x_64_4 = tf.placeholder(tf.float32, [None,1, 16])
x_64_5 = tf.placeholder(tf.float32, [None,1, 64])
x_64_6 = tf.placeholder(tf.float32, [None,1, 64])
#第二层分类器的输入
x_32_1 = tf.placeholder(tf.float32, [None, 1,16])
x_32_2 = tf.placeholder(tf.float32, [None, 1,16])
x_32_3 = tf.placeholder(tf.float32, [None,1, 4])
x_32_4 = tf.placeholder(tf.float32, [None,1, 4])
x_32_5 = tf.placeholder(tf.float32, [None,1, 16])
x_32_6 = tf.placeholder(tf.float32, [None,1, 16])

x_16 = tf.placeholder(tf.float32, [None, 1, 18])

state_in_tensor_64 = tf.placeholder(tf.float32, [None, LSTM_DEPTH, 2, VECTOR_LENGTH_LIST[2]])
state_in_tensor_32 = tf.placeholder(tf.float32, [None, LSTM_DEPTH, 2, VECTOR_LENGTH_LIST[1]])
state_in_tensor_16 = tf.placeholder(tf.float32, [None, LSTM_DEPTH, 2, VECTOR_LENGTH_LIST[0]])

def send_init_signal(init_file):
    f = open(init_file, 'w+')
    f.write('1')
    f.close()
    print('Python: Tensorflow initialized.')

def send_complete_signal(complete_file):
    f = open(complete_file, 'w+')
    f.write('1')
    f.close()
    print('Python: Operation completed.')

def get_command(command_file):
    f = open(command_file, 'r+')
    line = f.readline()
    str_arr = line.split(' ')
    if len(str_arr) == 5 and str_arr[4] == '[end]':
        i_frame = int(str_arr[0])
        frame_width = int(str_arr[1])
        frame_height = int(str_arr[2])
        qp_seq = int(str_arr[3])
    else:
        i_frame = -1
        frame_width = -1
        frame_height = -1
        qp_seq = -1
    f.close()
    return i_frame, frame_width, frame_height, qp_seq

def get_images_from_one_file(CU_file, PU_file, TU_file, frame_width, frame_height, i_frame, CUwidth):
    valid_height = math.ceil(frame_height / CUwidth)
    valid_width = math.ceil(frame_width / CUwidth)
    num_banch = valid_height * valid_width
    a=video
    f1_64 =open(Ress_begin+a+ '_Ress_64x64_22.dat','rb')
    f2_64 = open(Pattern_begin + a+'_Pattern_64x64_22.dat', 'rb')
    f3_64 = open(MBbits_begin + a+'_MBbits_64x64_22.dat','rb')
    f4 = open(CU_file, 'rb')
    f5 = open(PU_file, 'rb')
    f6 = open(TU_file, 'rb')
    lens1_64 = 64;lens2_64 = 64;lens3_64 = 16;lens4_64=16;lens5_64=64;lens6_64=64;
    lens1_32 = 16;lens2_32 = 16;lens3_32 = 4;lens1_16 = 4;lens2_16 = 4;lens3_16 = 1;

    f1_64.seek(num_banch * i_frame * 4 * lens1_64, 0)
    image1_64 = f1_64.read(num_banch * lens1_64 * 4)
    img1_64 = numpy.frombuffer(image1_64, dtype=numpy.int32)
    f2_64.seek(num_banch * i_frame * 4 * lens2_64, 0)
    image2_64 = f2_64.read(num_banch * 4 * lens2_64)
    img2_64 = numpy.frombuffer(image2_64, dtype=numpy.int32)
    f3_64.seek(num_banch * i_frame * 4 * lens3_64, 0)
    image3_64 = f3_64.read(num_banch * 4 * lens3_64)
    img3_64 = numpy.frombuffer(image3_64, dtype=numpy.int32)
    image4 = f4.read()
    image5 = f5.read()
    image6 = f6.read()
    img4 = numpy.frombuffer(image4, dtype=numpy.int8)
    img5 = numpy.frombuffer(image5, dtype=numpy.int8)
    img6 = numpy.frombuffer(image6, dtype=numpy.int8)
    f1_64.close()
    f2_64.close()
    f3_64.close()
    f4.close()
    f5.close()
    f6.close()

    data_img1_64 = numpy.reshape(img1_64, [num_banch, lens1_64])
    data_img1_64 = numpy.reshape(data_img1_64, [num_banch,1, lens1_64])
    data_img2_64 = numpy.reshape(img2_64, [num_banch, lens2_64])
    data_img2_64 = numpy.reshape(data_img2_64, [num_banch,1, lens2_64])
    data_img3_64 = numpy.reshape(img3_64, [num_banch, lens3_64])
    data_img3_64 = numpy.reshape(data_img3_64, [num_banch,1, lens3_64])
    data_img4_64_ori = numpy.reshape(img4, [num_banch, lens4_64])
    data_img4_64_ori = numpy.reshape(data_img4_64_ori, [num_banch,1, lens4_64])

    data_img5_64_ori = numpy.reshape(img5, [num_banch, lens5_64])
    data_img5_64_ori = numpy.reshape(data_img5_64_ori, [num_banch, 1, lens5_64])
    data_img6_64_ori = numpy.reshape(img6, [num_banch, lens6_64])
    data_img6_64_ori = numpy.reshape(data_img6_64_ori, [num_banch, 1, lens6_64])#特征不合起来


    data_img4_64=[[-88] * 16 for row in range(num_banch)]; data_img5_64=[[-88] * 64 for row in range(num_banch)]; data_img6_64=[[-88] * 64 for row in range(num_banch)];

    data_img1_32=[[-88] * 16 for row in range(num_banch*4)];data_img2_32=[[-88] * 16 for row in range(num_banch*4)];
    data_img3_32 = [[-88] * 4 for row in range(num_banch * 4)];data_img4_32=[[-88] * 4 for row in range(num_banch * 4)];
    data_img5_32=[[-88] * 16 for row in range(num_banch*4)]; data_img6_32=[[-88] * 16 for row in range(num_banch*4)];
    data_img1_16=[[-88] * 4 for row in range(num_banch*16)];data_img2_16=[[-88] * 4 for row in range(num_banch*16)];
    data_img3_16=[[-88] * 1 for row in range(num_banch*16)];data_img4_16=[[-88] * 1 for row in range(num_banch*16)];
    data_img5_16=[[-88] * 4 for row in range(num_banch*16)];data_img6_16=[[-88] * 4 for row in range(num_banch*16)];


    for ord_i in range(num_banch):
        ff=data_img1_64[ord_i, :]
        data_img4_64[ord_i] = numpy.reshape(data_img4_64_ori[ord_i, :], [4, 4], order='F')
        data_img5_64[ord_i]=numpy.reshape(data_img5_64_ori[ord_i, :],[8,8],order='F')
        data_img6_64[ord_i] = numpy.reshape(data_img6_64_ori[ord_i, :], [8, 8], order='F')

        F164 = numpy.reshape(data_img1_64[ord_i, :], [8, 8],order='F')
        F264 = numpy.reshape(data_img2_64[ord_i, :], [8, 8],order='F')
        F364 = numpy.reshape(data_img3_64[ord_i, :], [4, 4],order='F')

        F464 = numpy.reshape(data_img4_64_ori[ord_i, :], [4, 4],order='C')
        F564 = numpy.reshape(data_img5_64_ori[ord_i, :], [8, 8],order='C')
        F664 = numpy.reshape(data_img6_64_ori[ord_i, :], [8, 8],order='C')
        for n_i32 in range(2):
            for n_j32 in range(2):
                data_img1_32[ord_i * 4 + (n_i32) * 2 + n_j32] = numpy.reshape(
                    F164[(4 * n_i32):(n_i32 + 1) * 4, 4 * n_j32:(n_j32 + 1) * 4], [1, 16],order='F')
                data_img2_32[ord_i * 4 + (n_i32) * 2 + n_j32] = numpy.reshape(
                    F264[(4 * n_i32):(n_i32 + 1) * 4 , 4 * n_j32:(n_j32 + 1) * 4 ], [1, 16],order='F')
                data_img3_32[ord_i * 4 + (n_i32) * 2 + n_j32] = numpy.reshape(
                    F364[(2 * n_i32):(n_i32 + 1) * 2 , 2 * n_j32:(n_j32 + 1) * 2 ], [1, 4],order='F')
                data_img4_32[ord_i * 4 + (n_i32) * 2 + n_j32] = numpy.reshape(
                    F464[(2 * n_i32):(n_i32 + 1) * 2 , 2 * n_j32:(n_j32 + 1) * 2 ], [1, 4],order='F')
                data_img5_32[ord_i * 4 + (n_i32) * 2 + n_j32] = numpy.reshape(
                    F564[(4 * n_i32):(n_i32 + 1) * 4 , 4 * n_j32:(n_j32 + 1) * 4 ], [1, 16],order='F')
                data_img6_32[ord_i * 4 + (n_i32) * 2 + n_j32] = numpy.reshape(
                    F664[(4 * n_i32):(n_i32 + 1) * 4 , 4 * n_j32:(n_j32 + 1) * 4 ], [1, 16],order='F')
        for n_i16 in range(4):
            for n_j16 in range(4):
                data_img1_16[ord_i * 16 + (n_i16) * 4 + n_j16] = numpy.reshape(
                    F164[(2 * n_i16):(n_i16 + 1) * 2 , 2 * n_j16:(n_j16 + 1) * 2 ], [1, 4],order='F')
                data_img2_16[ord_i * 16 + (n_i16) * 4 + n_j16] = numpy.reshape(
                    F264[(2 * n_i16):(n_i16 + 1) * 2 , 2 * n_j16:(n_j16 + 1) * 2 ], [1, 4],order='F')
                data_img3_16[ord_i * 16 + (n_i16) * 4 + n_j16] =  numpy.reshape(F364[n_i16, n_j16],[1,1],order='F')
                data_img4_16[ord_i * 16 + (n_i16) * 4 + n_j16] = numpy.reshape(F464[n_i16, n_j16],[1,1],order='F')
                data_img5_16[ord_i * 16 + (n_i16) * 4 + n_j16] = numpy.reshape(
                    F564[(2 * n_i16):(n_i16 + 1) * 2 , 2 * n_j16:(n_j16 + 1) * 2 ], [1, 4],order='F')
                data_img6_16[ord_i * 16 + (n_i16) * 4 + n_j16] = numpy.reshape(
                    F664[(2 * n_i16):(n_i16 + 1) * 2 , 2 * n_j16:(n_j16 + 1) * 2 ], [1, 4],order='F')

    img4_64=numpy.array(data_img4_64,dtype=numpy.float32)
    data_img4_64_after=numpy.reshape(img4_64,[num_banch,1, lens4_64])
    img5_64 = numpy.array(data_img5_64, dtype=numpy.float32)
    data_img5_64_after = numpy.reshape(img5_64, [num_banch, 1, lens5_64])
    img6_64 = numpy.array(data_img6_64, dtype=numpy.float32)
    data_img6_64_after = numpy.reshape(img6_64, [num_banch, 1, lens6_64])

    img1_32 = numpy.array(data_img1_32,dtype=numpy.float32)
    img2_32 = numpy.array(data_img2_32,dtype=numpy.float32)
    img3_32 = numpy.array(data_img3_32,dtype=numpy.float32)
    img4_32 = numpy.array(data_img4_32,dtype=numpy.float32)
    img5_32 = numpy.array(data_img5_32,dtype=numpy.float32)
    img6_32 = numpy.array(data_img6_32,dtype=numpy.float32)
    data_img1_16 = numpy.array(data_img1_16, dtype=numpy.float32)
    data_img2_16 = numpy.array(data_img2_16, dtype=numpy.float32)
    data_img3_16 = numpy.array(data_img3_16, dtype=numpy.float32)
    data_img4_16 = numpy.array(data_img4_16, dtype=numpy.float32)
    data_img5_16 = numpy.array(data_img5_16, dtype=numpy.float32)
    data_img6_16 = numpy.array(data_img6_16, dtype=numpy.float32)

    img_16 = numpy.concatenate([data_img1_16, data_img2_16, data_img3_16, data_img4_16, data_img5_16, data_img6_16],
                            axis=2)

    img_64_1 = data_img1_64.astype(numpy.float32)
    img_64_2 = data_img2_64.astype(numpy.float32)
    img_64_3 = data_img3_64.astype(numpy.float32)
    img_64_4 = data_img4_64_after.astype(numpy.float32)
    img_64_5 = data_img5_64_after.astype(numpy.float32)
    img_64_6 = data_img6_64_after.astype(numpy.float32)

    return img_64_1,img_64_2,img_64_3,img_64_4,img_64_5,img_64_6, img1_32,img2_32 ,img3_32,img4_32,img5_32,img6_32,img_16, num_banch

def get_state_in_from_one_file(state_file64, state_file32, state_file16, num_vectors, i_frame):
    if (i_frame-1)%30==0:
        state_in64 = numpy.zeros((num_vectors, LSTM_DEPTH, 2, VECTOR_LENGTH_LIST[2]))
        state_in32 = numpy.zeros((num_vectors * 4, LSTM_DEPTH, 2, VECTOR_LENGTH_LIST[1]))
        state_in16 = numpy.zeros((num_vectors * 16, LSTM_DEPTH, 2, VECTOR_LENGTH_LIST[0]))
    else :
        f_in64 = open(state_file64, 'rb')
        state_in_buf64 = f_in64.read(num_vectors * LSTM_DEPTH * 2 * VECTOR_LENGTH_LIST[2] * 4)  # *2?
        state_in64 = numpy.frombuffer(state_in_buf64, dtype=numpy.float32)
        state_in64 = numpy.reshape(state_in64, [num_vectors, LSTM_DEPTH, 2, VECTOR_LENGTH_LIST[2]])
        f_in64.close()
        f_in32 = open(state_file32, 'rb')
        state_in_buf32 = f_in32.read(num_vectors * LSTM_DEPTH * 2 * 4 * VECTOR_LENGTH_LIST[1] * 4)
        state_in32 = numpy.frombuffer(state_in_buf32, dtype=numpy.float32)
        state_in32 = numpy.reshape(state_in32, [num_vectors * 4, LSTM_DEPTH, 2, VECTOR_LENGTH_LIST[1]])
        f_in32.close()
        f_in16 = open(state_file16, 'rb')
        state_in_buf16 = f_in16.read(num_vectors * LSTM_DEPTH * 2 * 16 * VECTOR_LENGTH_LIST[0] * 4)
        state_in16 = numpy.frombuffer(state_in_buf16, dtype=numpy.float32)
        state_in16 = numpy.reshape(state_in16, [num_vectors * 16, LSTM_DEPTH, 2, VECTOR_LENGTH_LIST[0]])
        f_in16.close()

    return state_in64, state_in32, state_in16


def predict_cu_depth(images64_1, images64_2,images64_3,images64_4,images64_5,images64_6,images32_1,images32_2, images32_3,images32_4,images32_5,images32_6,images16, state_in64, state_in32, state_in16, qp_seq, i_frame):

    index_start = 0
    y_pred_flat_64_temp, y_pred_flat_32_temp, y_pred_flat_16_temp, state_out_tensor_64_temp, state_out_tensor_32_temp, state_out_tensor_16_temp = sess.run(
        [y_pred_flat_64, y_pred_flat_32, y_pred_flat_16, state_out_64, state_out_32, state_out_16],
        feed_dict={x_64_1: images64_1,x_64_2: images64_2,x_64_3: images64_3,x_64_4: images64_4,x_64_5: images64_5,x_64_6: images64_6,
                   x_32_1: images32_1,x_32_2: images32_2,x_32_3: images32_3,x_32_4: images32_4, x_32_5: images32_5,x_32_6: images32_6,
                   x_16: images16,
                   state_in_tensor_64: state_in64,state_in_tensor_32: state_in32, state_in_tensor_16: state_in16})
    depth_out = numpy.concatenate([y_pred_flat_64_temp, y_pred_flat_32_temp, y_pred_flat_16_temp], axis=1)#shape:[batch_size,1+4+16]?
    return depth_out, state_out_tensor_64_temp, state_out_tensor_32_temp, state_out_tensor_16_temp


def save_cu_depth_and_state(depth_out, state_out64, state_out32, state_out16, save_file, state_file64, state_file32,
                            state_file16, end_file, num_vectors):
    depth_out_line = numpy.reshape(depth_out.astype(numpy.float32), [1, num_vectors * (1 + 4 + 16)])
    state_out_line_64 = numpy.reshape(state_out64.astype(numpy.float32), [1, num_vectors * LSTM_DEPTH * 2 * VECTOR_LENGTH_LIST[2]])
    state_out_line_32 = numpy.reshape(state_out32.astype(numpy.float32), [1, num_vectors * LSTM_DEPTH * 2 * 4*VECTOR_LENGTH_LIST[1]])#??
    state_out_line_16 = numpy.reshape(state_out16.astype(numpy.float32), [1, num_vectors * LSTM_DEPTH * 2 * 16*VECTOR_LENGTH_LIST[0]])
    # print(save_file)
    f_out = open(save_file, 'wb')
    f_out.write(depth_out_line)
    # print(depth_out_line)
    f_out.close()
    f_out64 = open(state_file64, 'wb')
    f_out64.write(state_out_line_64)
    f_out64.close()
    f_out32 = open(state_file32, 'wb')
    f_out32.write(state_out_line_32)
    f_out32.close()
    f_out16 = open(state_file16, 'wb')
    f_out16.write(state_out_line_16)
    f_out16.close()

    f_out = open(end_file, 'wb')  # generate ending signal
    f_out.close()
#
if __name__ == "__main__":
    with tf.device("/cpu:0"):

        Ress_begin='G:/单步预测特征输入_149帧/22/'
        Pattern_begin = 'G:/单步预测特征输入_149帧/22/'
        MBbits_begin = 'G:/单步预测特征输入_149帧/22/'
        #初始化视频名和帧数
        video=' '
        frameNum=0


        opt_vars_all_LSTM64,opt_vars_all_LSTM32,opt_vars_all_LSTM16, y_pred_flat_64, y_pred_flat_32, y_pred_flat_16, state_out_64, state_out_32, state_out_16 = nt.net(x_64_1,x_64_2,x_64_3,x_64_4,x_64_5,x_64_6, x_32_1,x_32_2,x_32_3,x_32_4,x_32_5,x_32_6, x_16, state_in_tensor_64, state_in_tensor_32, state_in_tensor_16)

        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        # sess = tf.Session(config=config)
        sess = tf.Session()
        saver_LSTM64 = tf.train.Saver(opt_vars_all_LSTM64)#创建Saver对象
        saver_LSTM32 = tf.train.Saver(opt_vars_all_LSTM32)
        saver_LSTM16 = tf.train.Saver(opt_vars_all_LSTM16)
        n_frame_total = 0
        qp_seq = 0
        #前一帧的
        CU_file = 'CU.dat'
        PU_file = 'PU.dat'
        TU_file = 'TU.dat'
        #用来存状态
        state_file64 = 'state64.dat'
        state_file32 = 'state32.dat'
        state_file16 = 'state16.dat'
        save_file = 'cu_depth.dat'
        command_file = 'command.dat'

        start_file = 'pred_start.sig'
        end_file = 'pred_end.sig'
        order=1
        while True:
            time.sleep(0.001)
            if os.path.isfile(start_file):
                # order += 1
                i_frame, frame_width, frame_height, qp_seq_temp = get_command(command_file)
                if i_frame==1:
                    with open("encoder_yuv_source_auto.cfg","r")  as f_cfg:
                        for i in f_cfg:
                            matchObj = re.match(r'InputFile                     : (.*).yuv', i)
                            matchObj2 = re.match(r'FramesToBeEncoded             : (.*)', i)
                            # print(i)
                            if matchObj:
                                video=matchObj.group(1).split('_')[0]
                                print('videoName: ' + video)
                                # break
                            if matchObj2:
                                frameNumStr=matchObj2.group(1).split('_')[0]
                                print('frameNum: ' + frameNumStr)
                                frameNum=int(frameNumStr)
                                break
                    print('第0帧已编码')
                print('预测第' + str(i_frame) + '帧， 分辨率：' + str(frame_width) + 'x' + str(frame_height))
                # print(str(i_frame)+' '+str(frame_width)+' '+str(frame_height)+' '+str(qp_seq_temp))

                if i_frame >= 1:
                    qp_seq_last = qp_seq
                    qp_seq = qp_seq_temp
                    os.remove(start_file)
                    model_LSTM_file64= 'F:/model/64x64/22/model.ckpt-900000'
                    model_LSTM_file32 = 'F:/model/32x32/22/model.ckpt-3750000'
                    model_LSTM_file16 = 'F:/model/16x16/22/model.ckpt-56700000'


                    saver_LSTM64.restore(sess, model_LSTM_file64)
                    saver_LSTM32.restore(sess, model_LSTM_file32)
                    saver_LSTM16.restore(sess, model_LSTM_file16)

                    data_img_64_1,data_img_64_2,data_img_64_3,data_img_64_4,data_img_64_5,data_img_64_6, data_img_32_1,data_img_32_2,data_img_32_3,data_img_32_4,data_img_32_5,data_img_32_6, data_img_16, num_vectors = get_images_from_one_file(CU_file, PU_file, TU_file,
                                                                                                  frame_width, frame_height,
                                                                                                  i_frame, 64)

                    state_in64, state_in32, state_in16 = get_state_in_from_one_file(state_file64, state_file32,
                                                                                    state_file16, num_vectors, i_frame)
                    depth_out, state_out64, state_out32, state_out16 = predict_cu_depth(data_img_64_1,data_img_64_2,data_img_64_3,data_img_64_4,data_img_64_5,data_img_64_6,data_img_32_1,data_img_32_2,data_img_32_3,data_img_32_4,data_img_32_5,data_img_32_6,
                                                                                        data_img_16, state_in64, state_in32,
                                                                                        state_in16, qp_seq, i_frame)
                    save_cu_depth_and_state(depth_out, state_out64, state_out32, state_out16, save_file, state_file64,
                                            state_file32, state_file16, end_file, num_vectors)
                    n_frame_total += 1
                    print('%d frames predicted.' % n_frame_total)
                    del data_img_64_1,data_img_64_2,data_img_64_3,data_img_64_6, data_img_32_1,data_img_32_2,data_img_32_3,data_img_32_6, data_img_16

                    if n_frame_total==(frameNum-1):
                        print("-----------" + video + "--finished!----------")
                        n_frame_total=0



