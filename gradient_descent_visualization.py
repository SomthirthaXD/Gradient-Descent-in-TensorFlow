# This is the section of dependency imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# This is a randomly generated data
X=np.random.rand(100)
y=4+3*X+np.random.rand(100) # y=4+3X+c

# This is a Sequential Model

# Training loop
epochs=100
model=tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(units=1, activation='linear')])

# Model is compiled here
optimizer=tf.keras.optimizers.experimental.SGD() # Parameter decay=lr/epochs can give a learning rate scheduled decay per epoch
model.compile(optimizer=optimizer, loss='mse')

plt.scatter(X, y, label='True')

loss=[]
print("Training Epoch:", end=" ")
for epoch in range(epochs):
    print((epoch+1), end=", ")
    if(epoch==0):
      plt.plot(X, model.predict(X, verbose=0), label='Pred 1')

    # Model is trained for one epoch here
    with tf.GradientTape() as tape:
      logits=model(X) # Logit generation
      losses=tf.reduce_mean(tf.keras.losses.MeanSquaredError()(y, logits)) # Loss calculation
    gradient=tape.gradient(losses, model.trainable_variables) # Gradient generation
    optimizer.apply_gradients(zip(gradient, model.trainable_variables)) # Applying gradient descent on the model's trainable parameters

    if (epoch==epochs-1):
      plt.plot(X, model.predict(X, verbose=0), label=f'Pred {epochs}')
    loss.append(model.evaluate(X, y, verbose=0))

plt.title(f"Target vs Predictions in epochs 0 and {epochs}")
plt.legend()
plt.show()
plt.close()

# Loss Graph is here
plt.title("Visualization of Loss Minimization in the Fully Connected Layer")
for epoch in range(epochs):
  plt.scatter(epoch, loss[epoch])
plt.show()
plt.close()
print("Initial loss:", loss[0])
print("Final loss:", loss[epochs-1])
