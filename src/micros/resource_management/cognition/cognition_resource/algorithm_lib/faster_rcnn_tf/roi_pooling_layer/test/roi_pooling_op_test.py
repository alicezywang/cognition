import tensorflow as tf
import numpy as np
import roi_pooling_op
import roi_pooling_op_grad

number_stored = 1

def get_number():
    global number_stored
    number_stored = (number_stored * 101 + 59) % 32768
    return number_stored

array = np.zeros((32, 100, 100, 3))

for index1 in xrange(32):
    for index2 in xrange(100):
        for index3 in xrange(100):
            array[index1][index2][index3][0] = get_number()/32768.0
            array[index1][index2][index3][1] = get_number()/32768.0
            array[index1][index2][index3][2] = get_number()/32768.0

data = tf.convert_to_tensor(array, dtype=tf.float32)
rois = tf.convert_to_tensor([[0, 10, 10, 20, 20], [31, 30, 30, 40, 40]], dtype=tf.float32)

W = tf.Variable(np.zeros((3, 3, 3, 1)), dtype=tf.float32)
h = tf.nn.conv2d(data, W, strides=[1, 1, 1, 1], padding='SAME')

[y, argmax] = roi_pooling_op.roi_pool(h, rois, 6, 6, 1.0/3)

y_data = tf.convert_to_tensor(np.ones((2, 6, 6, 1)), dtype=tf.float32)
print y_data, y, argmax

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

file_out = open("b.txt", "w")

for step in xrange(10):
    sess.run(train)
    file_out.write(str(step+1))
    file_out.write("\n\n");
    temp = sess.run(W).reshape(-1)
    file_out.write("\n".join("{:.6f}".format(i) for i in temp))
    file_out.write("\n\n");
    temp = sess.run(y).reshape(-1)
    file_out.write("\n".join("{:.6f}".format(i) for i in temp))
    file_out.write("\n\n");

file_out.close()

# with tf.device('/gpu:0'):
#   result = module.roi_pool(data, rois, 1, 1, 1.0/1)
#   print result.eval()
# with tf.device('/cpu:0'):
#   run(init)
