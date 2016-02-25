import mxnet as mx
import logging

## define lenet
# input
data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                          kernel=(2,2), stride=(2,2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                          kernel=(2,2), stride=(2,2))
# first fullc
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
batch_size=1000
num_epoch = 20
num_gpus = 1
logging.basicConfig(level=logging.DEBUG)
gpus = [mx.gpu(i) for i in range(num_gpus)]
model = mx.model.FeedForward(ctx=gpus, symbol=lenet, num_epoch=num_epoch,
                                 learning_rate=0.05, momentum=0.9, wd=0.0001,
                                 initializer=mx.init.Uniform(0.07))
train_dataiter = mx.io.ImageRecordIter(
        path_imgrec="chars/train.rec",
        mean_img="chars/mean.bin",
        rand_crop=True,
        rand_mirror=True,
        data_shape=(1,20,20),
        batch_size=batch_size,
        preprocess_threads=1)
test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="chars/train.rec",
        mean_img="chars/mean.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(1,20,20),
        batch_size=batch_size,
        preprocess_threads=1)
model.fit(X=train_dataiter, eval_data=test_dataiter,
          batch_end_callback=mx.callback.Speedometer(100))
model.save('chars/lenetweights',num_epoch)
