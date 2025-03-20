import tensorflow as tf
import numpy as np

# TensorFlow Basics - Machine Learning Library
print("TensorFlow Basics Examples:")

# 1. Creating Tensors
print("\n1. Creating Tensors:")
# Create a constant tensor
tensor1 = tf.constant([[1, 2], [3, 4]])
print("Constant tensor:")
print(tensor1)

# Create a tensor filled with zeros
zeros = tf.zeros([3, 3])
print("\nZeros tensor:")
print(zeros)

# Create a tensor filled with ones
ones = tf.ones([2, 2])
print("\nOnes tensor:")
print(ones)

# 2. Basic Operations
print("\n2. Basic Operations:")
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

print("Addition:")
print(tf.add(a, b))

print("\nMultiplication:")
print(tf.multiply(a, b))

print("\nMatrix multiplication:")
print(tf.matmul(a, b))

# 3. Simple Neural Network
print("\n3. Simple Neural Network Example:")
# Create a simple sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Display model summary
model.summary()

# 4. Working with Variables
print("\n4. Working with Variables:")
# Create a TensorFlow variable
var = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
print("Variable:")
print(var)

# Update variable
var.assign(tf.multiply(var, 2.0))
print("\nUpdated variable:")
print(var)

# 5. Simple Training Example
print("\n5. Simple Training Example:")
# Generate some random data
X = np.random.randn(100, 2)
y = np.random.randint(0, 2, 100)

# Convert to TensorFlow datasets
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(32)

# Compile the model
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Train for just 2 epochs as example
print("\nTraining the model:")
model.fit(dataset, epochs=2)

print("\nNote: This is a basic example. Real-world applications would require more data, tuning, and validation.")
