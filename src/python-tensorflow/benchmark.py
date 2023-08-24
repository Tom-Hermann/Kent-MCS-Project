import matplotlib.pyplot as plt
import json

CPU_JULIA = json.load(open("./json/CPU_JULIA_epoch_data.json"))
GPU_JULIA = json.load(open("./json/GPU_JULIA_epoch_data.json"))
CPU_PYTHON = json.load(open("./json/CPU_PYTHON_epoch_data.json"))
GPU_PYTHON = json.load(open("./json/GPU_PYTHON_epoch_data.json"))


fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(GPU_PYTHON["epoch"], GPU_PYTHON["epoch_time"], label='epoch_time')
axs[0, 0].plot(CPU_PYTHON["epoch"], CPU_PYTHON["epoch_time"], label='epoch_time')


axs[0, 1].plot(GPU_PYTHON["epoch"], GPU_PYTHON["batch_processing_time"], label='batch_processing_time')
axs[0, 1].plot(CPU_PYTHON["epoch"], CPU_PYTHON["batch_processing_time"], label='batch_processing_time')


axs[0, 2].plot(GPU_PYTHON["epoch"], GPU_PYTHON["throughput"], label='throughput')
axs[0, 2].plot(CPU_PYTHON["epoch"], CPU_PYTHON["throughput"], label='throughput')


axs[1, 0].plot(GPU_PYTHON["epoch"], GPU_PYTHON["accuracy"], label='accuracy')
axs[1, 0].plot(CPU_PYTHON["epoch"], CPU_PYTHON["accuracy"], label='accuracy')


axs[1, 1].plot(GPU_PYTHON["epoch"], GPU_PYTHON["loss"], label='loss')
axs[1, 1].plot(CPU_PYTHON["epoch"], CPU_PYTHON["loss"], label='loss')

for ax, lab in zip(axs.flat, list(GPU_PYTHON.keys())[2:]):
    ax.set(xlabel='Epoch', ylabel='Value')
    ax.legend()

plt.tight_layout()
plt.savefig("./image/python.png")

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(GPU_JULIA["epoch"], GPU_JULIA["epoch_time"], label='GPU')
axs[0, 0].plot(CPU_JULIA["epoch"], CPU_JULIA["epoch_time"], label='CPU')

axs[0, 1].plot(GPU_JULIA["epoch"], GPU_JULIA["batch_processing_time"], label='GPU')
axs[0, 1].plot(CPU_JULIA["epoch"], CPU_JULIA["batch_processing_time"], label='CPU')


axs[0, 2].plot(GPU_JULIA["epoch"], GPU_JULIA["throughput"], label='GPU')
axs[0, 2].plot(CPU_JULIA["epoch"], CPU_JULIA["throughput"], label='CPU')


axs[1, 0].plot(GPU_JULIA["epoch"], GPU_JULIA["accuracy"], label='GPU')
axs[1, 0].plot(CPU_JULIA["epoch"], CPU_JULIA["accuracy"], label='CPU')


axs[1, 1].plot(GPU_JULIA["epoch"], GPU_JULIA["loss"], label='GPU')
axs[1, 1].plot(CPU_JULIA["epoch"], CPU_JULIA["loss"], label='CPU')

for ax, lab in zip(axs.flat, list(GPU_JULIA.keys())[2:]):
    ax.set(xlabel='Epoch', ylabel=lab)
    ax.legend()

plt.tight_layout()
plt.savefig("./image/julia.png")

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

if "epoch_memory_usage" in GPU_JULIA.keys():
    axs[0, 0].plot(GPU_JULIA["epoch"], GPU_JULIA["epoch_memory_usage"], label='JULIA')
axs[0, 0].plot(GPU_PYTHON["epoch"], GPU_PYTHON["epoch_memory_usage"], label='PYTHON')

axs[0, 1].plot(GPU_JULIA["epoch"], GPU_JULIA["epoch_time"], label='JULIA')
axs[0, 1].plot(GPU_PYTHON["epoch"], GPU_PYTHON["epoch_time"], label='PYTHON')


axs[0, 2].plot(GPU_JULIA["epoch"], GPU_JULIA["batch_processing_time"], label='JULIA')
axs[0, 2].plot(GPU_PYTHON["epoch"], GPU_PYTHON["batch_processing_time"], label='PYTHON')


axs[1, 0].plot(GPU_JULIA["epoch"], GPU_JULIA["throughput"], label='JULIA')
axs[1, 0].plot(GPU_PYTHON["epoch"], GPU_PYTHON["throughput"], label='PYTHON')


axs[1, 1].plot(GPU_JULIA["epoch"], GPU_JULIA["accuracy"], label='JULIA')
axs[1, 1].plot(GPU_PYTHON["epoch"], GPU_PYTHON["accuracy"], label='PYTHON')


axs[1, 2].plot(GPU_JULIA["epoch"], GPU_JULIA["loss"], label='JULIA')
axs[1, 2].plot(GPU_PYTHON["epoch"], GPU_PYTHON["loss"], label='PYTHON')

for ax, lab in zip(axs.flat, list(GPU_JULIA.keys())[1:]):
    ax.set(xlabel='Epoch', ylabel=lab)
    ax.legend()

plt.tight_layout()
plt.savefig("./image/julia_python_gpu.png")

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(CPU_PYTHON["epoch"], CPU_PYTHON["epoch_time"], label='PYTHON')
axs[0, 0].plot(CPU_JULIA["epoch"], CPU_JULIA["epoch_time"], label='JULIA')


axs[0, 1].plot(CPU_PYTHON["epoch"], CPU_PYTHON["batch_processing_time"], label='PYTHON')
axs[0, 1].plot(CPU_JULIA["epoch"], CPU_JULIA["batch_processing_time"], label='JULIA')


axs[0, 2].plot(CPU_PYTHON["epoch"], CPU_PYTHON["throughput"], label='PYTHON')
axs[0, 2].plot(CPU_JULIA["epoch"], CPU_JULIA["throughput"], label='JULIA')


axs[1, 0].plot(CPU_PYTHON["epoch"], CPU_PYTHON["accuracy"], label='PYTHON')
axs[1, 0].plot(CPU_JULIA["epoch"], CPU_JULIA["accuracy"], label='JULIA')


axs[1, 1].plot(CPU_PYTHON["epoch"], CPU_PYTHON["loss"], label='PYTHON')
axs[1, 1].plot(CPU_JULIA["epoch"], CPU_JULIA["loss"], label='JULIA')

for ax, lab in zip(axs.flat, list(CPU_PYTHON.keys())[2:]):
    ax.set(xlabel='Epoch', ylabel=lab)
    ax.legend()

plt.tight_layout()
plt.savefig("./image/julia_python_cpu.png")

print("Done!")
print("Images saved in ./image/")