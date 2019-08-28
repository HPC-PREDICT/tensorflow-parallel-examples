import unittest
import os
import tensorflow as tf
import numpy as np
import time
import datetime


# Linear regression test for Horovod with disjoint unit vector data set
# (should converge completely within one GradientDescentOptimizer iteration)


class TestHorovodLinearRegression(unittest.TestCase):

    def setUp(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'


    def test_horovod_linear_regression(self):

        import horovod.tensorflow as hvd
        # Horovod: initialize Horovod.
        hvd.init()

        logdir = args.logdir + '/horovod_test/'
        if hvd.rank() == 0:
            if not os.path.exists(logdir):
                os.makedirs(logdir)

        assert args.training_size % hvd.size() == 0
        assert (args.training_size // hvd.size()) % args.batch_size == 0

        training_data_filename = logdir + 'training_data.npy'
        if hvd.rank() == 0:
            with open(training_data_filename, 'w') as f:
                full_training_data = np.random.random(size=(args.training_size,))
                full_training_data.tofile(f)
            print("Full training data:")
            print(full_training_data)
        hvd.allgather(tf.constant([0]))


        with open(training_data_filename, 'r') as f:
            training_data = np.fromfile(f)
            training_data_size = training_data.shape[0]
            local_training_data_size = (training_data.shape[0]+hvd.size()-1)//hvd.size()
            local_training_data_begin = hvd.rank() * local_training_data_size
            if hvd.rank() == hvd.size()-1 and training_data.shape[0] % local_training_data_size != 0:
                local_training_data_size = training_data.shape[0] % local_training_data_size
            local_training_data = training_data[local_training_data_begin:local_training_data_begin + local_training_data_size]
            print("Local training data:")
            print(local_training_data)

        # Define Tensorflow graph
        graph = tf.Graph()

        with graph.as_default():
            x_ph = tf.placeholder(tf.float32, shape=[None,training_data_size], name='x')
            y_ph = tf.placeholder(tf.float32, shape=[None], name='y')
            w = tf.Variable(np.zeros((training_data_size,)), dtype=tf.float32, name='w')
            loss_func = tf.constant(0.5)*tf.reduce_sum(tf.square(y_ph-tf.tensordot(x_ph,w, axes=1)))

            opt = tf.train.GradientDescentOptimizer(learning_rate=hvd.size()*1.0)
            # Horovod: wrap local optimizer in distributed Horovod optimizer.
            opt = hvd.DistributedOptimizer(opt)

            #train_step = opt.minimize(loss_func)
            #grads_and_vars = opt.compute_gradients(loss_func)
            train_step = opt.minimize(loss_func) # apply_gradients(grads_and_vars)

            config = tf.ConfigProto()
            config.intra_op_parallelism_threads = 22
            config.inter_op_parallelism_threads = 8
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True

            # Horovod
            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            bcast = hvd.broadcast_global_variables(0)

        print("Local training data:")
        print(local_training_data)
        print('Opening tf.Session...')
        with tf.Session(graph=graph, config=config) as sess:
            # We must initialize all variables before we use them.
            init.run()
            bcast.run()
            print('Initialized all Horovod ranks.')
            print('Begin training - batch size = {}.'.format(args.batch_size), flush=True)

            if hvd.rank() == 0:
                training_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

            for i in range((local_training_data_size+args.batch_size-1)//args.batch_size):
                batch_begin = i*args.batch_size
                batch_size = args.batch_size \
                    if i != (local_training_data_size+args.batch_size-1)//args.batch_size else \
                    local_training_data_size % args.batch_size
                x = np.zeros(shape=(batch_size, training_data_size))
                x[np.arange(x.shape[0]), local_training_data_begin + batch_begin + np.arange(x.shape[0])] = 1.0
                y = local_training_data[batch_begin:batch_begin+batch_size]

                feed_train = {
                    x_ph: x,
                    y_ph: y
                }

                # Only compute shuffle indices, do not compute shuffled data set to avoid memory error
                time_before = time.time()
                #grad_op = grads_and_vars[0][0]
                loss, _ = sess.run([loss_func, train_step], feed_dict=feed_train)
                time_after = time.time()
                print('Step {0:5d}  -  loss: {1:6.2f}  -  latency: {2:6.2f} ms.'.format(i, loss,
                                                                                        1000 * (time_after - time_before)),
                      flush=True)

            print('Finished training - local residual w - y_training is:')
            local_residual = sess.run([w[local_training_data_begin:
                                         local_training_data_begin+local_training_data_size]-y_ph],
                                       feed_dict={y_ph: local_training_data})
            print(local_residual)
            print('Locally trained variable components')
            print(sess.run([w[local_training_data_begin:
                              local_training_data_begin+local_training_data_size]]))
            print('Local training data')
            print(local_training_data) 
            self.assertTrue(np.allclose(local_residual,
                                        np.zeros(len(local_residual),),rtol=1e-7))

        if hvd.rank() == 0:
            os.remove(training_data_filename)
        hvd.allgather(tf.constant([0]))
        hvd.shutdown()


if __name__ == '__main__':

    # Parse arguments
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description='Training for Jonas\' MRI reconstruction method')
        parser.add_argument('--training-size', type=int,
                            help='Training data size')
        parser.add_argument('--batch-size', type=int,
                            help='Size of the local batch')
        parser.add_argument('--logdir', type=str,  # "path_mat_out
                            help='Where to write tf.summary event files to be visualized in Tensorboard.')
        parser.add_argument('unittest_args', nargs='*')
        return parser.parse_args()

    args = parse_args()

    # unittest does its own parsing (will not understand the above options)
    import sys
    sys.argv[1:] = args.unittest_args
    unittest.main()
