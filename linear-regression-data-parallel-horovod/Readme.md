### Test for data-parallel distributed SGD Training with Horovod 

Linear regression model (without offset)

  y = <w,x>

is used with

  loss = 0.5 * (<w,x_i> - y_i)^2

with x_i = e_i the i-th unit vector of the standard basis. This converges within a single step of the GradientDescentOptimizer (Horovod's averaging of node-wise gradients during allreduce in SGD training step must be compensated). The test verifies that this is done correctly.

An example can be run using the following command after loading the daint-gpu, Tensorflow and Horovod modules:
```
module load daint-gpu; 
module load TensorFlow/1.12.0-CrayGNU-18.08-cuda-9.1-python3; 
module load Horovod/0.16.0-CrayGNU-18.08-tf-1.12.0

srun -n 16 -C gpu -o logs/out_%t.log -e  logs/err_%t.log python linearRegressionHorovod.py --training-size 64 --batch-size 2  --logdir logs
```

