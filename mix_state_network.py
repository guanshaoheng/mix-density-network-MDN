#!/usr/bin/env python
# encoding: utf-8
"""

@file: mix_state_network.py
@time: 2020/7/21 10:52
@author: Luke
@email: guanshaoheng@qq.com
@application：
             当神经网络撞上薛定谔：混合密度网络入门
             mixture density network
@reffernce: [1] https://zhuanlan.zhihu.com/p/37992239
            [2] https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%80%BC%E5%87%BD%E6%95%B0
            [3] https://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler


class Network:
    def __init__(self, x_data, y_data):
        self.x_data, self.y_data = x_data, y_data
        self.x = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='x')
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y')
        self.train_op, self.loss, self.p, self.mu, self.sigma = self.net_model()
        self.sess = self.train()
        self.test(self.sess)

    def net_model(self):
        num_gaussion = 5
        hidden_1 = tf.layers.dense(inputs=self.x, units=64, activation=tf.nn.relu, name='hidden_1')
        hidden_2 = tf.layers.dense(inputs=hidden_1, units=128, activation=tf.nn.tanh, name='hidden_2')
        output = tf.layers.dense(hidden_2, units=num_gaussion*3, activation=None, name='output')
        p, mu, sigma = tf.split(output, 3, axis=1)
        p = tf.nn.softmax(p)
        sigma = tf.exp(sigma)
        factor = 1./np.sqrt(2.*np.pi)
        temp = -tf.square((self.y-mu)/sigma)/2.
        y_posibility = tf.exp(temp)*factor/sigma
        loss = tf.reduce_sum(tf.multiply(y_posibility, p), axis=1, keep_dims=True)
        loss = -tf.log(loss)
        loss = tf.reduce_mean(loss)
        train_step = tf.train.AdamOptimizer().minimize(loss)
        return train_step, loss, p, mu, sigma

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        n_epoch = 10000
        loss_vals = np.zeros(n_epoch)
        for i in range(n_epoch):
            _, loss_val = sess.run([self.train_op, self.loss], feed_dict={self.x: self.x_data, self.y: self.y_data})
            loss_vals[i] = loss_val
            if i % 500 == 0:
                print('{}/{} loss: {}'.format(i, n_epoch, loss_val))
        # plot the  training process
        plt.figure(figsize=(8, 8))
        plt.plot(loss_vals)
        plt.show()
        plt.savefig('traing_process.svg', dpi=600, format='svg')
        return sess

    def test(self, sess):
        x_test = np.linspace(-15., 15., 100).reshape(-1, 1)
        p, mu, sigma = sess.run([self.p, self.mu, self.sigma], feed_dict={self.x: x_test})
        # plot the distribution of the p, mu and sigma
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 12))
        ax1.plot(x_test, p)
        ax1.set_title('P(mixture). $\Pi$ (Y-axis must sum to one.)')
        ax2.plot(x_test, sigma)
        ax2.set_title('$\sigma$ of each mixture.')
        ax3.plot(x_test, mu)
        ax3.set_title('$\mu$ (mean value for each mixture.)')
        plt.xlim([-15, 15])
        plt.show()
        plt.savefig('distribution of p mu sigma.svg', dpi=600, format='svg')

        # plot the mixed gaussian function
        plt.figure(figsize=(8, 8))
        for mu_k, sigma_k in zip(mu.T, sigma.T):
            plt.plot(x_test, mu_k)
            plt.fill_between(x_test.flatten(), mu_k - sigma_k, mu_k + sigma_k, alpha=0.1)
        plt.scatter(self.x_data, self.y_data, marker='.', lw=0, alpha=0.2, c='black')
        plt.title('Gaussian Function Mixing Process')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.show()
        plt.savefig('gaussian function mixing process.svg', dpi=600, format='svg')

        data_len = len(p)
        repeat_num = 5
        random_uniform = np.random.uniform(0, 1, [data_len, repeat_num])
        random_normal = np.random.normal(size=[data_len, repeat_num])
        y_prediction = np.zeros(shape=[data_len, repeat_num])
        for i in range(data_len):
            for j in range(repeat_num):
                # 获取随机的第几个样本
                accumulate = 0.
                for k in range(p.shape[1]):
                    accumulate += p[i, k]
                    if accumulate >= random_uniform[i, j]:
                        break
                y_prediction[i, j] = mu[i, k] + random_normal[i, j] * sigma[i, k]

        # 作图
        plt.figure()
        plt.plot(self.x_data, self.y_data, 'ro', x_test, y_prediction, 'bo', alpha=0.2)
        plt.show()
        plt.savefig('prediction.svg', dpi=600, format='svg')


sample_num = 1000
x_train = np.random.uniform(-10.5, 10.5, size=(sample_num, 1))
y_train = np.sin(0.75 * x_train) * 7.0 + 0.5 * x_train + np.random.normal(size=(sample_num, 1))
mix_net = Network(x_data=y_train, y_data=x_train)
