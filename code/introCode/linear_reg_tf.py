import numpy as np
import tensorflow as tf

# Model linear regression y_hat = a + bx

# Placeholder variables for the input data
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# Model parameters are Variables in Tensorflow
a = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1, 1]))

# Calculating the prediction of the linear model
y_hat = a + tf.matmul(x,b)

# Loss function =  sum((y_hat-y)^2)
cost = tf.reduce_mean(tf.square(y_hat - y))

# Training using Gradient Descent using momentum to minimize cost
train_step = tf.train.MomentumOptimizer(learning_rate = 0.0005, momentum = 0.1).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
steps = 20000
for i in range(steps):
    # Input data for the model
    xs = np.array([[10], [20], [30], [40], [50]])
    ys = np.array([[592], [1090], [1604], [2122], [2620]])

    # Defining a feed dictionary; this is the set of 
    # (placeholder : value) pairs for calculations
    feed = { x : xs, y : ys }

    # Printing out some info every 1000 iterations
    if i % 1000 == 0:
        print("Iteration = " + str(i) + ", loss = " + str(sess.run(cost, feed_dict = feed)))
    # Running a single training step

    sess.run(train_step, feed_dict = feed)

# Printing the results    
print("a: %f" % sess.run(b))
print("b: %f" % sess.run(a))

sess.close()
