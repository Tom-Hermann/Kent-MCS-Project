using Flux
using Flux.Data: DataLoader
using MLDatasets
using Statistics
using JSON
using CUDA


# Load MNIST dataset
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

# Reshape dataset
train_X = reshape(train_X, (28, 28, 1, :))
test_X = reshape(test_X, (28, 28, 1, :))

# One-hot encode target values
train_y = Flux.onehotbatch(train_y, 0:9)
test_y = Flux.onehotbatch(test_y, 0:9)

# Normalize data
train_X /= 255.0
test_X /= 255.0

println("Shape of train_X: $(size(train_X))")
println("Shape of test_X: $(size(test_X))")

# Define model
calc_device = gpu

model = Chain(
    Conv((3, 3), 1=>32, pad=(1, 1), selu),
    MaxPool((2, 2)),
    Conv((3, 3), 32=>64, pad=(1, 1), selu),
    MaxPool((2, 2)),
    x -> reshape(x, :, size(x, 4)),  # Flatten layer
    Dense(3136, 64),
    selu,
    Dense(64, 10),  # No activation here
    softmax
)  |> calc_device

# Define loss function and optimizer
loss(x, y) = Flux.crossentropy(model(x), y)
accuracy(X, y) = Statistics.mean(Flux.onecold(model(X)) .== Flux.onecold(y))
optimizer = ADAM(0.001)

# Training configuration
batch_size = 64
num_epochs = 10

# Initialize variables for monitoring
total_training_time = 0.0
total_memory_usage = 0
throughputs = []

monitoring_data = Dict(
    "epoch" => [],
    "epoch_memory_usage" => [],
    "epoch_time" => [],
    "batch_processing_time" => [],
    "throughput" => [],
    "accuracy" => [],
    "loss" => [],
)

# Start training loop
for epoch in 1:num_epochs
    println("Epoch $epoch/$num_epochs")

    # Initialize variables for epoch-level monitoring
    total_batch_processing_time = 0.0
    epoch_memory_use = 0
    epoch_start_time = time()

    loss_value = 0.0
    accuracy_value = 0.0

    coef = 0

    for batch_start in 1:batch_size:size(train_X, 4)


        batch_start_time = time()

        batch_end = min(batch_start + batch_size - 1, size(train_X, 4))

        batch_x = train_X[:, :, :, batch_start:batch_end]
        batch_y = train_y[:, batch_start:batch_end]


        # Perform training step
        x, y = calc_device(batch_x), calc_device(batch_y)
        gradients = Flux.gradient(() -> loss(x, y), Flux.params(model))
        Flux.Optimise.update!(optimizer, Flux.params(model), gradients)

        # collect the loss and accuracy from each batch
        loss_value += loss(x, y)
        accuracy_value += accuracy(x, y)
        # accuracy_value += (accuracy(x, y) / 938)


        batch_processing_time = time() - batch_start_time
        total_batch_processing_time += batch_processing_time


        coef +=1
    end

    CUDA.memory_status()

    accuracy_value /= coef
    loss_value /= coef

    epoch_end_time = time()
    epoch_time = epoch_end_time - epoch_start_time
    total_training_time += epoch_time
    total_memory_usage += epoch_memory_use

    num_samples = size(train_X, 4)
    throughput = num_samples / epoch_time
    push!(throughputs, throughput)

    println(" - Memory Usage: $epoch_memory_use bytes")
    println(" - Epoch Time: $epoch_time seconds")
    println(" - Batch Processing Time: $total_batch_processing_time seconds")
    println(" - Throughput: $throughput samples/second")

    # Evaluate accuracy and convergence
    println(" - Accuracy: $accuracy_value")
    println(" - Loss: $loss_value")

    push!(monitoring_data["epoch"], epoch)
    push!(monitoring_data["epoch_memory_usage"], epoch_memory_use)
    push!(monitoring_data["epoch_time"], epoch_time)
    push!(monitoring_data["batch_processing_time"], total_batch_processing_time)
    push!(monitoring_data["throughput"], throughput)
    push!(monitoring_data["accuracy"], accuracy_value)
    push!(monitoring_data["loss"], loss_value)
end

json_filename = "./json/GPU_JULIA_epoch_data.json"

stringdata = JSON.json(monitoring_data)

# write the file with the stringdata variable information
open(json_filename, "w") do f
        write(f, stringdata)
    end
