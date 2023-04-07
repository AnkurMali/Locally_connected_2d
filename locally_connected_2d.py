class locallyconnected_tf(object):
  def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, padding='VALID', bias = False):
    super(locallyconnected_tf, self).__init__()
    initializer = tf.compat.v1.keras.initializers.Orthogonal()
    total_input_size = (kernel_size**2) * in_channels
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
output_size = 22
kernel_size = 3
stride = 1

## Call the operator
conv = locallyconnected_tf(in_channels, out_channels, output_size, kernel_size, stride, bias=False)
## This will give you the output
output = conv.forward(x)
print(output)
## This will give you the weights
print(conv.weight.shape)
