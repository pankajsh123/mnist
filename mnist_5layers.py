import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
pkeep = tf.placeholder(tf.float32)

L = 200
M = 100
N = 60
O = 30

W1 = tf.Variable(tf.contrib.layers.xavier_initializer(uniform = True)(shape = [784,L]))
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.contrib.layers.xavier_initializer(uniform = True)(shape = [L, M]))
B2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.contrib.layers.xavier_initializer(uniform = True)(shape = [M, N]))
B3 = tf.Variable(tf.zeros([N]))
W4 = tf.Variable(tf.contrib.layers.xavier_initializer(uniform = True)(shape = [N, O]))
B4 = tf.Variable(tf.zeros([O]))
W5 = tf.Variable(tf.contrib.layers.xavier_initializer(uniform = True)(shape = [O, 10]))
B5 = tf.Variable(tf.zeros([10]))

XX = tf.reshape(X, [-1, 784])
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y1_ = tf.contrib.layers.batch_norm(Y1,center = True,scale = True,is_training = True)
Y2 = tf.nn.sigmoid(tf.matmul(Y1_, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)
Y2_ = tf.contrib.layers.batch_norm(Y2,center = True,scale = True,is_training = True)
Y3 = tf.nn.sigmoid(tf.matmul(Y2_, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)
Y3_ = tf.contrib.layers.batch_norm(Y3,center = True,scale = True,is_training = True)
Y4 = tf.nn.sigmoid(tf.matmul(Y3_, W4) + B4)
Y4_ = tf.contrib.layers.batch_norm(Y4,center = True,scale = True,is_training = True)
Ylogits = tf.matmul(Y4_, W5) + B5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

learning_rate = tf.constant(0.001)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	
	for i in range(10000):
		batch_X, batch_Y = mnist.train.next_batch(100)
		train_data = {X: batch_X , Y_: batch_Y}

		sess.run(train_step, feed_dict=train_data)

		a,c = sess.run([accuracy, cross_entropy], feed_dict = train_data)

	test_data =  {X:mnist.test.images, Y_: mnist.test.labels}
	a,c =  sess.run([accuracy, cross_entropy], feed_dict = test_data)
	print(a,c)

