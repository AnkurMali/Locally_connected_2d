class locallyconnected_tf(object):
  def __init__(self, h,w,in_channels, out_channels, output_size, kernel_size, stride, padding='VALID', bias = False):
    super(locallyconnected_tf, self).__init__()
    initializer = tf.compat.v1.keras.initializers.Orthogonal()
    total_input_size = (kernel_size**2) * in_channels
    output_size = (w - kernel_size)/stride + 1
    output_size = int(output_size)
    self.weight = tf.Variable(initializer(shape=(1, output_size, output_size, total_input_size, out_channels)))
    #self.bias = tf.Variable(initializer(shape=(1, output_size, output_size, out_channels)))
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

  def forward(self,x):
    _,h,w,c = x.shape # Assuming it is coming from numpy or tensor
    x_windows = tf.compat.v1.extract_image_patches(x, ksizes =[1,self.kernel_size,self.kernel_size,1], strides = [1,self.stride,self.stride,1], rates = [1,1,1,1], padding=self.padding)
    x_windows = tf.expand_dims(x_windows, axis=4)
    output = tf.math.multiply(x_windows, self.weight)
    final_output = tf.math.reduce_sum(output,axis=3)
    return final_output
  
### Script to test the function

# # Create input
batch_size = 5
in_channels = 3
h, w = 24, 24
x = tf.random.normal(shape=(batch_size, h, w, in_channels))
# x = torch.randn(batch_size, in_channels, h, w)

# # Create layer and test if backpropagation works
out_channels = 2
output_size = 22 # This will be calculated on the fly, once h, w and in_channels are provided using (W - F + 2P)/S)+1
kernel_size = 3
stride = 1
## Important provide input_height (h) and output_height (h)
## Call the operator
conv = locallyconnected_tf(h,w,in_channels, out_channels, output_size, kernel_size, stride, bias=False)
## This will give you the output
output = conv.forward(x)
print(output)
## This will give you the weights
print(conv.weight.shape)


### Using multiple locallyconnected

## First get shape
_, h,w,in_channels = output.shape
## As number of output_channels will become input_channel for layer below
conv2 = locallyconnected_tf(h,w,in_channels, out_channels, output_size, kernel_size, stride, padding = 'VALID', bias=False)
output2 = conv2.forward(output)
print(conv2.weight.shape)
## Now repeat this get shape, change variables as needed

_, h,w,in_channels = output2.shape
## As number of output_channels will become input_channel for layer below
conv3 = locallyconnected_tf(h,w,in_channels, out_channels, output_size, kernel_size, stride, padding = 'VALID', bias=False)
output3 = conv3.forward(output2)
print(conv3.weight.shape)



#### Test-2 Adding locallyconnected with max_pooling layer (Will this code work, let's find out)

# # Create input
batch_size = 5
in_channels = 64
h, w = 32, 32
x = tf.random.normal(shape=(batch_size, h, w, in_channels))
# x = torch.randn(batch_size, in_channels, h, w)

# # Create layer and test if backpropagation works
out_channels = 128
output_size = 30
kernel_size = 3
stride = 1
conv = locallyconnected_tf(h,w,in_channels, out_channels, output_size, kernel_size, stride, padding = 'VALID', bias=False)

output = conv.forward(x)
#print(conv.weight.shape)
#print(output.shape)
max_operation = tf.nn.max_pool2d(output, ksize=[1,3,3,1], strides=[1,1,1,1], padding = 'VALID', data_format='NHWC', name=None)
### Using multiple locallyconnected

## First get shape
_, h,w,in_channels = max_operation.shape
print(max_operation.shape)
## As number of output_channels will become input_channel for layer below
conv2 = locallyconnected_tf(h,w,in_channels, out_channels, output_size, kernel_size, stride, padding = 'VALID', bias=False)
output2 = conv2.forward(max_operation)
print(conv2.weight.shape)
## Now repeat this get shape, change variables as needed
max_operation2 = tf.nn.max_pool2d(output2, ksize=kernel_size, strides=stride, padding = 'VALID', data_format='NHWC', name=None)
_, h,w,in_channels = max_operation2.shape
## As number of output_channels will become input_channel for layer below
conv3 = locallyconnected_tf(h,w,in_channels, out_channels, output_size, kernel_size, stride, padding = 'VALID', bias=False)
output3 = conv3.forward(max_operation2)
print(conv3.weight.shape)

### It's working
