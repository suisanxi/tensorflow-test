# --*-- coding: utf-8 --*--
# @Author: zhang
# @Email: your email
# @Time: 2019/9/6 17:10
# @File: test.py
# @Software: PyCharm
# desc: 测试用

import tensorflow as tf

class NN(tf.keras.Model):
    def __int__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            100, #units 神经元个数
            activation= tf.nn.relu
        )
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self,inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output

model = NN()