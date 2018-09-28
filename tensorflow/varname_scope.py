import tensorflow as tf
# 1>
print('\n')
tf.reset_default_graph()
with tf.name_scope('varscope'):
    a = tf.constant(1.0, name='oneconst')
    b = tf.Variable(1.0, name='VariableName')
    c = tf.get_variable(name='getvariable', initializer=1.0)

print(a)
print(b)
print(c)

# Tensor("varscope/oneconst:0", shape=(), dtype=float32)
# <tf.Variable 'varscope/VariableName:0' shape=() dtype=float32_ref>
# <tf.Variable 'getvariable:0' shape=() dtype=float32_ref>

# 2>
print('\n')
tf.reset_default_graph()
with tf.variable_scope('varscope'):
    a = tf.constant(1.0, name='oneconst')
    b = tf.Variable(1.0, name='VariableName')
    c = tf.get_variable(name='getvariable', initializer=1.0)

print(a)
print(b)
print(c)

# Tensor("varscope/oneconst:0", shape=(), dtype=float32)
# <tf.Variable 'varscope/VariableName:0' shape=() dtype=float32_ref>
# <tf.Variable 'varscope/getvariable:0' shape=() dtype=float32_ref>


# 3>
print('\n')
# the `get_variable()` function creates a new variable or
# returns one created earlier by `get_variable()`.
# It won't return a variable created using tf.Variable()

tf.reset_default_graph()
var1 = tf.Variable(1.0, name='Variable')
# `tf.Variable` always create new variable
var2 = tf.Variable(1.0, name='Variable')
var3 = tf.get_variable(name='Variable', initializer=1.0)
# `tf.get_variable` Gets an existing variable with these parameters or create a new one
with tf.variable_scope("", reuse=True): # root variable scope
    var4 = tf.get_variable(name='Variable')

print(var1)
print(var2)
print(var3)
print(var4)

# <tf.Variable 'Variable:0' shape=() dtype=float32_ref>
# <tf.Variable 'Variable_1:0' shape=() dtype=float32_ref>
# <tf.Variable 'Variable_2:0' shape=() dtype=float32_ref>


# 4>
print('\n')
tf.reset_default_graph()
# https://www.tensorflow.org/guide/variables#sharing_variables

# TensorFlow supports two ways of sharing variables:
# - Explicitly passing tf.Variable objects around.
# - Implicitly wrapping tf.Variable objects within tf.variable_scope objects.

# While code which explicitly passes variables around is very clear, it is sometimes
# convenient to write TensorFlow functions that implicitly use variables in their implementations.
# Most of the functional layers from tf.layers use this approach, as well as all tf.metrics, and a few other library utilities.

input1 = tf.random_normal([1, 10, 10, 32])
input2 = tf.random_normal([1, 20, 20, 32])

def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

#  Calling conv_relu in different scopes, however, clarifies that we want to create new variables:
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])


# If you do want the variables to be shared, you have two options.
# First, you can create a scope with the same name using `reuse=True`:
with tf.variable_scope("model"):
    output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
    output2 = my_image_filter(input2)
