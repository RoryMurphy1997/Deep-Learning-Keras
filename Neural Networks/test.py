import theano
from theano import tensor
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from keras import backend

a = tensor.dscalar()
b = tensor.dscalar()

c = a + b

f = theano.function([a, b], c)
# Bind 1.5 to a, 2.5 to b and evaluate c
result = f(1.5, 2.5)
print(result)


# Test tensorflow

# Declare two symbolic floating-point scalars
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
#create a simple symbolic expression using "add"
add = tf.add(a,b)
#bind 1.5 to a, 2.5 to b and evaluate c
sess = tf.Session()
binding = {a: 1.5, b:2.5}
c = sess.run(add, feed_dict=binding)

print(c)

#Test Keres
print(backend._BACKEND)

