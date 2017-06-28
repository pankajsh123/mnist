import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets("data",one_hot = True ,reshape = False, validation_size = 0)

lr = tf.constant(0.001)		#learning rate

X = tf.placeholder(tf.float32,[None,28,28,1])
Y_ = tf.placeholder(tf.float32,[None, 10])

W = tf.Variable(tf.contrib.layers.xavier_initializer(uniform = True)(shape = [784,10]))
b = tf.Variable(tf.zeros([10]))

logits_ = tf.add(tf.matmul(tf.reshape(X,[-1,784]),W), b)	#defining model
Y = tf.nn.softmax(logits_)
Y__ = tf.nn.softmax_cross_entropy_with_logits(logits = logits_,labels = Y_)
cross_entropy = 100*tf.reduce_mean(Y__)

is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)		#minimising cost function

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess1:
	sess1.run(init)
	
	for i in range(10000):
		batch_X, batch_Y = mnist.train.next_batch(100)
		train_data = {X: batch_X , Y_: batch_Y}
		sess1.run(train_step, feed_dict=train_data)
		a,c = sess1.run([accuracy, cross_entropy], feed_dict = train_data)
	saver.save(sess1, 'my-model')

	'''test_data =  {X:mnist.test.images, Y_: mnist.test.labels}
	a,c =  sess.run([accuracy, cross_entropy], feed_dict = test_data)
	print(a," ",c)'''

