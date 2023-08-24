import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

import time
import json

# example of loading the mnist dataset
# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))

# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# convert from integers to floats
trainX = trainX.astype('float32')
testX = testX.astype('float32')
# normalize to range 0-1
trainX = trainX / 255.0
testX = testX / 255.0

#defining the model
model = tf.keras.models.Sequential([
                            tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3), padding='same', kernel_initializer='glorot_uniform', activation='selu', input_shape=(28, 28, 1)),
                            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(64, kernel_initializer='glorot_uniform', activation='selu'),
                            tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])

epoch_data = {
    "epoch": [],
    "epoch_memory_usage": [],
    "epoch_time": [],
    "batch_processing_time": [],
    "throughput": [],
    "accuracy": [],
    "loss": [],
}

# Training configuration
batch_size = 64
num_epochs = 10

# Initialize variables for monitoring
total_training_time = 0
total_memory_usage = 0

throughputs = []
accs = []
losss = []

# Start training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Initialize variables for epoch-level monitoring
    total_batch_processing_time = 0
    epoch_memory_use = 0



    epoch_start_time = time.time()

    for batch in range(0, len(trainX), batch_size):
        batch_start_time = time.time()

        # Extract a batch of data
        batch_x = trainX[batch:batch + batch_size]
        batch_y = trainY[batch:batch + batch_size]

        # Perform training step on CPU (no need to specify device)
        batch_history = model.train_on_batch(batch_x, batch_y)

        batch_processing_time = time.time() - batch_start_time
        total_batch_processing_time += batch_processing_time



    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    total_training_time += epoch_time

    num_samples = len(trainX)
    steps_per_epoch = num_samples // batch_size
    throughput = num_samples / epoch_time

    throughputs.append(throughput)

    print(f" - Epoch Time: {epoch_time:.2f} seconds")
    print(f" - Batch Processing Time: {total_batch_processing_time:.2f} seconds")
    print(f" - Throughput: {throughput:.2f} samples/second")

    # Evaluate accuracy and convergence
    eval_results = model.evaluate(testX, testY, verbose=0)
    accuracy = eval_results[1]
    loss = eval_results[0]

    accs.append(accuracy)
    losss.append(loss)

    print(f" - Accuracy: {accuracy:.4f}")
    print(f" - Loss: {loss:.4f}")

    epoch_data["epoch"].append(epoch + 1)
    epoch_data["epoch_memory_usage"].append(0)
    epoch_data["epoch_time"].append(epoch_time)
    epoch_data["batch_processing_time"].append(total_batch_processing_time)
    epoch_data["throughput"].append(throughput)
    epoch_data["accuracy"].append(accuracy)
    epoch_data["loss"].append(loss)



with open('./json/CPU_PYTHON_epoch_data.json', 'w') as json_file:
    json.dump(epoch_data, json_file, indent=4)

print(f"Total Training Time: {total_training_time:.2f} seconds")
print(f"Average Throughput: {sum(throughputs) / num_epochs:.2f} samples/second")
print(f"Average Accuracy: {sum(accs) / num_epochs:.2f}")
print(f"Average Loss: {sum(losss) / num_epochs:.2f}")
