import tensorflow as tf
import numpy as np

print("TensorFlow Basics Tutorial")
print("=========================")

# 1. Tensors
print("\n1. Basic Tensor Operations:")
# Create tensors
tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[5, 6], [7, 8]])

print("Tensor 1:")
print(tensor1)
print("\nTensor 2:")
print(tensor2)

# Basic operations
print("\nAddition:")
print(tf.add(tensor1, tensor2))

print("\nMultiplication:")
print(tf.multiply(tensor1, tensor2))

print("\nMatrix multiplication:")
print(tf.matmul(tensor1, tensor2))

# 2. Variables
print("\n2. Variables:")
# Create a variable
var = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
print("Variable:")
print(var)

# Update variable
var.assign(tf.multiply(var, 2.0))
print("\nUpdated variable:")
print(var)

# 3. Basic Neural Network
print("\n3. Simple Neural Network:")
# Create a simple sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Display model summary
print("\nModel Summary:")
model.summary()

# 4. Data Processing
print("\n4. Data Processing:")
# Create sample data
X = np.random.randn(100, 2)
y = np.random.randint(0, 2, 100)

# Convert to TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(32)

print("\nDataset structure:")
for batch_x, batch_y in dataset.take(1):
    print("X shape:", batch_x.shape)
    print("y shape:", batch_y.shape)

# 5. Model Training
print("\n5. Basic Model Training:")
# Compile the model
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Train for just 2 epochs as example
print("\nTraining:")
history = model.fit(dataset, epochs=2)

# 6. Making Predictions
print("\n6. Making Predictions:")
# Generate test data
test_data = np.random.randn(5, 2)
predictions = model.predict(test_data)
print("\nPredictions for 5 samples:")
print(predictions)

# 7. Save and Load Models
print("\n7. Save and Load Models:")
# Save model
model.save('basic_model')
print("Model saved to 'basic_model'")

# Load model
loaded_model = tf.keras.models.load_model('basic_model')
print("Model loaded successfully")

# Verify predictions
print("\nPredictions from loaded model:")
new_predictions = loaded_model.predict(test_data)
print(new_predictions)

# 8. Custom Training Loop
print("\n8. Custom Training Loop:")
# Create a simple model
inputs = tf.keras.Input(shape=(2,))
outputs = tf.keras.layers.Dense(1)(inputs)
simple_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Define loss function
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# Custom training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = simple_model(x)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, simple_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, simple_model.trainable_variables))
    return loss

# Training loop
print("\nCustom training loop:")
X_train = np.random.randn(32, 2)
y_train = np.random.randn(32, 1)

for epoch in range(2):
    loss = train_step(X_train, y_train)
    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

print("\nNote: This is a basic introduction to TensorFlow. Real-world applications would require more data, proper model architecture, and validation procedures.")
