import tensorflow as tf

constVec = tf.constant([[1.,2.,3.],[4.,5.,6.]])
# print(constVec)
# print(constVec.shape)
# print(constVec.dtype)

# print(constVec[:, 1:])
# print(constVec[..., 1, tf.newaxis])


constVec2 = tf.constant([[1.,2.,3.],[4.,5.,6.]])

print(constVec2 @ tf.transpose(constVec))

constScal = tf.constant(42)
# print(constScal)

print(constVec2[:, 1:])

print(constVec2+3)

print(tf.add(constVec2,4))

print(tf.multiply(constVec2 ,6))

print(tf.square(constVec2))

print(tf.reduce_sum(constVec2))

constVec3 = tf.constant([True ,False])
print(tf.reduce_all(constVec3)) #logical And operation

print(tf.exp(constVec2))

print(tf.math.log(constVec2))

from tensorflow import keras
K = keras.backend

print(K.square(K.transpose(constVec2))+10)

print(tf.cast(constVec2, tf.int16))


varVec = tf.Variable([10,30,40])

print(varVec[0].assign(20))
print(varVec.assign(varVec*10))

print(varVec[:2].assign([100,90]))

def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = tf.Variable(5.), tf.Variable(3.)

with tf.GradientTape(persistent=True) as hessian_tape:
    with tf.GradientTape() as jacobian_tape:
        z = f(w1, w2)
        jacobians = jacobian_tape.gradient(z, [w1, w2])
    hessians = [hessian_tape.gradient(jacobian, [w1, w2])
    for jacobian in jacobians]
    print(hessians)
del hessian_tape