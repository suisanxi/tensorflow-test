# --*-- coding: utf-8 --*--
# @Author: zhang
# @Email: your email
# @Time: 2019/9/7 16:26
# @File: model_visualize302.py
# @Software: PyCharm
# desc:数据可视化例子
import tensorflow as tf
from mnist201 import MLP
from mnist201 import MNISTLoader

num_epochs = 0.1
num_batches = 200
batch_size = 50
learning_rate = 0.001
model = MLP()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
summary_writer = tf.summary.create_file_writer('./tensorboard')     # 实例化记录器
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
        with summary_writer.as_default():  # 指定记录器
            tf.summary.scalar("loss", loss, step=batch_index)  # 将当前损失函数的值写入记录器
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for test_batch_index in range(num_batches):
        start_index, end_index = test_batch_index * batch_size, (test_batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    with summary_writer.as_default():  # 指定记录器
        tf.summary.scalar("pred", sparse_categorical_accuracy.result(), step=batch_index)  # 将当前损失函数的值写入记录器
    print("test accuracy: %f" % sparse_categorical_accuracy.result())
