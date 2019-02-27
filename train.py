import tensorflow as tf
import random
import matplotlib.pyplot as plt

rand_op = (lambda x: 0.5 * float(x) + 2.0 + (random.randrange(0, 10)-5) / 20)
op = (lambda x: 0.5 * float(x) + 2.0)

x_data = list(float(x) for x in range(-5, 5))
ground_y_data = [rand_op(x) for x in x_data]
valid_x = list(float(x)/10 for x in range(-50, 50))
predict_y_data = [op(x) for x in valid_x]

print(x_data)
print(ground_y_data)
#x_data = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
#ground_y_data = [-0.65, -0.2, 0.6, 0.75, 1.35, 1.95, 2.5, 2.8, 3.65, 3.75]

W1 = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
W2 = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
W3 = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
B = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

feedX = tf.placeholder(tf.float32)
feedY = tf.placeholder(tf.float32)

predic = W1 * feedX * feedX * feedX + W2 * feedX * feedX + W3 * feedX + B

cost = tf.reduce_mean(tf.square(feedY - predic))
opt = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(2800) :
		_ = sess.run(opt, feed_dict={feedX:x_data, feedY:ground_y_data})
	print(sess.run(W1))
	print(sess.run(W2))
	print(sess.run(W3))
	print(sess.run(B))
	result = sess.run(predic, feed_dict={feedX:valid_x})


plt.scatter(x_data, ground_y_data)
plt.plot(valid_x, predict_y_data, c= 'red')
plt.plot(valid_x, result, c= 'black')
plt.show()

