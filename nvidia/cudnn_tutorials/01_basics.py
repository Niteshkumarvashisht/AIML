import tensorflow as tf

print("cuDNN Basics Tutorial")
print("=====================")

# 1. cuDNN Acceleration in TensorFlow
print("\n1. cuDNN Acceleration:")

# Check if GPU is available
print("GPU Available:", tf.test.is_gpu_available())

# Check if cuDNN is enabled
print("cuDNN Enabled:", tf.test.is_built_with_cuda())

# Simple CNN to demonstrate cuDNN usage
print("\n2. Simple CNN:")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

print("\nNote: This tutorial demonstrates basic cuDNN usage with TensorFlow. Ensure TensorFlow is installed with GPU support.")
