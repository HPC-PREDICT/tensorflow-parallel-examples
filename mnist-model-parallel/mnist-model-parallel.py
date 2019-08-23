from  __future__  import division, print_function, absolute_import

import math
import time
import tensorflow as tf
import os
import subprocess

def get_data():
    # Get data
    from tensorflow.examples.tutorials.mnist import input_data
    dataDir = os.path.join(os.environ['SCRATCH'], 'tensorflow', 'data')
    print("Downloading data to {}".format(dataDir))
    return input_data.read_data_sets(dataDir, one_hot=True)

tf.test.gpu_device_name()

# Some numbers
batch_size = 128
display_step = 10
num_input = 784
num_classes = 10

def conv_layer(inputs, channels_in, channels_out, strides=1):

        # Create variables
        w=tf.Variable(tf.random_normal([3, 3, channels_in, channels_out]))
        b=tf.Variable(tf.random_normal([channels_out]))

        # We can double check the device that this variable was placed on
        print(w.device)
        print(b.device)

        # Define Ops
        x = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)

        # Non-linear activation
        return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

clausesApplied = 0
def calculateDeviceIdx():
    numClauses = 6
    global clausesApplied
    numDevices = len(nodeList)
    numClausesPerDevice = numClauses/numDevices
    idx = math.floor(clausesApplied / numClausesPerDevice)
    clausesApplied += 1
    return devices[idx]

# Create model
def CNN(x, devices):

    with tf.device(calculateDeviceIdx()): # <----------- Put first half of network on device 0

        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1=conv_layer(x, 1, 32, strides=1)
        pool1=maxpool2d(conv1)

    with tf.device(calculateDeviceIdx()):
        # Convolution Layer
        conv2=conv_layer(pool1, 32, 64, strides=1)
        pool2=maxpool2d(conv2)

    with tf.device(calculateDeviceIdx()):  # <----------- Put second half of network on device 1
        # Fully connected layer
        fc1 = tf.reshape(pool2, [-1, 7*7*64])
        w1=tf.Variable(tf.random_normal([7*7*64, 1024]))
        b1=tf.Variable(tf.random_normal([1024]))
        fc1 = tf.add(tf.matmul(fc1,w1),b1)
        fc1=tf.nn.relu(fc1)

    with tf.device(calculateDeviceIdx()):
        # Output layer
        w2=tf.Variable(tf.random_normal([1024, num_classes]))
        b2=tf.Variable(tf.random_normal([num_classes]))
        out = tf.add(tf.matmul(fc1,w2),b2)

        # Check devices for good measure
        print(w1.device)
        print(b1.device)
        print(w2.device)
        print(b2.device)

    return out

nodeListProcess = subprocess.run(['scontrol', 'show', 'hostnames', os.environ['SLURM_STEP_NODELIST']], stdout=subprocess.PIPE, encoding='ascii')
nodeList = nodeListProcess.stdout.strip().split('\n')

nodeList = ["{}:{}".format(node, 2222+port) for node,port in zip(nodeList, range(len(nodeList)))]

# Define devices that we wish to split our graph over
devices=["/job:worker/task:{}".format(id) for id in range(len(nodeList))] # (device0, device1)

# This line should match the same cluster definition in the Helper_Server.ipynb
cluster_spec = tf.train.ClusterSpec({'worker' : nodeList})

task_idx=int(os.environ.get('SLURM_PROCID', 0))
server = tf.train.Server(cluster_spec, job_name='worker', task_index=task_idx)

# Check the server definition
print(server.server_def)

if task_idx == 0:
    mnist = get_data()


    tf.reset_default_graph() # Reset graph

    # Construct model
    with tf.device(calculateDeviceIdx()):
        X = tf.placeholder(tf.float32, [None, num_input]) # Input images feedable
        Y = tf.placeholder(tf.float32, [None, num_classes]) # Ground truth feedable

    logits = CNN(X, devices) # Unscaled probabilities

    merged = None
    with tf.device(calculateDeviceIdx()):

        prediction = tf.nn.softmax(logits) # Class-wise probabilities

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()


    # Start training
    with tf.Session(server.target,config=tf.ConfigProto(log_device_placement=True)) as sess:  # <----- IMPORTANT: Pass the server target to the session definition
        train_writer = tf.summary.FileWriter(os.path.join(os.environ['SCRATCH'], 'tensorflow', 'logs', 'train', sess.graph)

        # Run the initializer
        sess.run(init)

        start = time.time()
        for step in range(10000):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # Run optimization op (backprop)
            summary, _ = sess.run([merged, train_op], feed_dict={X: batch_x, Y: batch_y}, options=run_options, run_metadata=run_metadata)

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                train_writer.add_run_metadata(run_metadata, 'step%d' % step)
                train_writer.add_summary(summary, step)

                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y : batch_y})
                print("Step {}, Minibatch Loss={:.4f}, Training Accuracy={:.3f}".format(step, loss, acc))

        print("Time running: {}".format(time.time()-start))

        train_writer.flush()
        train_writer.close()
        # Get test set accuracy
        print("Testing Accuracy:",sess.run(accuracy, feed_dict={X: mnist.test.images[:256],Y: mnist.test.labels[:256]}))
else:
    server.join()
