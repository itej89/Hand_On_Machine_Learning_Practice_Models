import tensorflow as tf

X = tf.range(10)

dataset = tf.data.Dataset.from_tensor_slices(X)

for item in dataset:
    print(item)
print("\n\n")

dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)
dataset = dataset.unbatch()
print("\n\n")

dataset = dataset.map(lambda x: x*2)
for item in dataset:
    print(item)
print("\n\n")

# dataset = dataset.filter(lambda x  : x<8)
# for item in dataset:
#     print(item)
# print("\n\n")

# for item in dataset.take(3):
#     print(item)

dataset  = dataset.shuffle(buffer_size = 5, seed = 42).batch(7)
for item in dataset:
    print(item)
print("\n\n")