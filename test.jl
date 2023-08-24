# import Pkg; Pkg.add("Flux")
# import Pkg; Pkg.add("MLDatasets")
# import Pkg; Pkg.add("CUDA")
using Flux, MLDatasets


# Load training data (images, labels)
x_train, y_train = MLDatasets.MNIST(split=:train)[:]
# Load test data (images, labels)
x_test, y_test = MLDatasets.MNIST(split=:test)[:]
# Convert grayscale to float
x_train = Float32.(x_train)
# Create labels batch
y_train = Flux.onehotbatch(y_train, 0:9)


model = Chain(
    Dense(784, 256, relu),
    Dense(256, 10, relu), softmax
)

loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)

optimizer = ADAM(0.0001)

parameters = Flux.params(model)
# flatten() function converts array 28x28x60000 into 784x60000 (28*28x60000)
train_data = [(Flux.flatten(x_train), Flux.flatten(y_train))]
# Range in loop can be used smaller
for i in 1:400
    Flux.train!(loss, parameters, train_data, optimizer)
end


test_data = [(Flux.flatten(x_test), y_test)]
# local accuracy = 0
# for i in 1:length(y_test)
#     if findmax(model(test_data[1][1][:, i]))[2] - 1  == y_test[i]
#         accuracy = accuracy + 1
#     end
# end
# println(accuracy / length(y_test))

correct_predictions = sum(findmax(model(test_data[1][1][:, i]))[2] - 1 == y_test[i] for i in 1:length(y_test))
accuracy = correct_predictions / length(y_test)
println(accuracy)