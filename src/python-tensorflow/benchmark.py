# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import json

CPU_JULIA = json.load(open("./json/CPU_JULIA_epoch_data.json"))
CPU_PYTHON = json.load(open("./json/CPU_PYTHON_epoch_data.json"))
SMALL_GPU_PYTHON = json.load(open("./json/SMALL_GPU_PYTHON_epoch_data.json"))
SMALL_GPU_JULIA = json.load(open("./json/SMALL_GPU_JULIA_epoch_data.json"))
MEDIUM_GPU_PYTHON = json.load(open("./json/MEDIUM_GPU_PYTHON_epoch_data.json"))
MEDIUM_GPU_JULIA = json.load(open("./json/MEDIUM_GPU_JULIA_epoch_data.json"))
BIG_GPU_PYTHON = json.load(open("./json/BIG_GPU_PYTHON_epoch_data.json"))
BIG_GPU_JULIA = json.load(open("./json/BIG_GPU_JULIA_epoch_data.json"))

print(CPU_PYTHON.keys())
print(CPU_JULIA.keys())

print(SMALL_GPU_PYTHON.keys())
print(SMALL_GPU_JULIA.keys())

print(MEDIUM_GPU_PYTHON.keys())
print(MEDIUM_GPU_JULIA.keys())

print(BIG_GPU_PYTHON.keys())
print(BIG_GPU_JULIA.keys())

"""# COMPARE PYTHON GPU CPU"""

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["epoch_time"], label='GPU')
axs[0, 0].plot(CPU_PYTHON["epoch"], CPU_PYTHON["epoch_time"], label='CPU')


axs[0, 1].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["batch_processing_time"], label='GPU')
axs[0, 1].plot(CPU_PYTHON["epoch"], CPU_PYTHON["batch_processing_time"], label='CPU')


axs[0, 2].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["throughput"], label='GPU')
axs[0, 2].plot(CPU_PYTHON["epoch"], CPU_PYTHON["throughput"], label='CPU')


axs[1, 0].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["accuracy"], label='GPU')
axs[1, 0].plot(CPU_PYTHON["epoch"], CPU_PYTHON["accuracy"], label='CPU')


axs[1, 1].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["loss"], label='GPU')
axs[1, 1].plot(CPU_PYTHON["epoch"], CPU_PYTHON["loss"], label='CPU')

for ax, lab in zip(axs.flat, list(MEDIUM_GPU_PYTHON.keys())[2:]):
    ax.set(xlabel='Epoch', ylabel='Value')
    ax.legend()

plt.tight_layout()
plt.savefig("./image/PYTHON_GPU_CPU.png")

"""# COMPARE JULIA CPU GPU"""

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["epoch_time"], label='GPU')
axs[0, 0].plot(CPU_JULIA["epoch"], CPU_JULIA["epoch_time"], label='CPU')


axs[0, 1].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["batch_processing_time"], label='GPU')
axs[0, 1].plot(CPU_JULIA["epoch"], CPU_JULIA["batch_processing_time"], label='CPU')


axs[0, 2].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["throughput"], label='GPU')
axs[0, 2].plot(CPU_JULIA["epoch"], CPU_JULIA["throughput"], label='CPU')


axs[1, 0].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["accuracy"], label='GPU')
axs[1, 0].plot(CPU_JULIA["epoch"], CPU_JULIA["accuracy"], label='CPU')


axs[1, 1].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["loss"], label='GPU')
axs[1, 1].plot(CPU_JULIA["epoch"], CPU_JULIA["loss"], label='CPU')

for ax, lab in zip(axs.flat, list(MEDIUM_GPU_PYTHON.keys())[2:]):
    ax.set(xlabel='Epoch', ylabel=lab)
    ax.legend()

plt.tight_layout()
plt.savefig("./image/JULIA_GPU_CPU.png")

"""# COMPARE CPU"""

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
plt.savefig("./image/CPU.png")

"""# COMPARE PYTHON  JULIA GPU

# SMALL
"""

fig, axs = plt.subplots(2, 3, figsize=(15, 10))


axs[0, 0].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["epoch_memory_usage"], label='JULIA')
axs[0, 0].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["epoch_memory_usage"], label='PYTHON')

axs[0, 1].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["epoch_time"], label='JULIA')
axs[0, 1].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["epoch_time"], label='PYTHON')


axs[0, 2].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["batch_processing_time"], label='JULIA')
axs[0, 2].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["batch_processing_time"], label='PYTHON')


axs[1, 0].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["throughput"], label='JULIA')
axs[1, 0].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["throughput"], label='PYTHON')


axs[1, 1].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["accuracy"], label='JULIA')
axs[1, 1].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["accuracy"], label='PYTHON')


axs[1, 2].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["loss"], label='JULIA')
axs[1, 2].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["loss"], label='PYTHON')

for ax, lab in zip(axs.flat, list(SMALL_GPU_PYTHON.keys())[1:]):
    ax.set(xlabel='Epoch', ylabel=lab)
    ax.legend()

plt.tight_layout()
plt.savefig("./image/SMALL_GPU.png")

"""# MEDIUM"""

fig, axs = plt.subplots(2, 3, figsize=(15, 10))


axs[0, 0].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["epoch_memory_usage"], label='JULIA')
axs[0, 0].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["epoch_memory_usage"], label='PYTHON')

axs[0, 1].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["epoch_time"], label='JULIA')
axs[0, 1].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["epoch_time"], label='PYTHON')


axs[0, 2].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["batch_processing_time"], label='JULIA')
axs[0, 2].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["batch_processing_time"], label='PYTHON')


axs[1, 0].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["throughput"], label='JULIA')
axs[1, 0].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["throughput"], label='PYTHON')


axs[1, 1].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["accuracy"], label='JULIA')
axs[1, 1].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["accuracy"], label='PYTHON')


axs[1, 2].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["loss"], label='JULIA')
axs[1, 2].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["loss"], label='PYTHON')

for ax, lab in zip(axs.flat, list(MEDIUM_GPU_PYTHON.keys())[1:]):
    ax.set(xlabel='Epoch', ylabel=lab)
    ax.legend()

plt.tight_layout()
plt.savefig("./image/MEDIUM_GPU.png")

"""# BIG"""

fig, axs = plt.subplots(2, 3, figsize=(15, 10))


axs[0, 0].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["epoch_memory_usage"], label='JULIA')
axs[0, 0].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["epoch_memory_usage"], label='PYTHON')

axs[0, 1].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["epoch_time"], label='JULIA')
axs[0, 1].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["epoch_time"], label='PYTHON')


axs[0, 2].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["batch_processing_time"], label='JULIA')
axs[0, 2].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["batch_processing_time"], label='PYTHON')


axs[1, 0].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["throughput"], label='JULIA')
axs[1, 0].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["throughput"], label='PYTHON')


axs[1, 1].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["accuracy"], label='JULIA')
axs[1, 1].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["accuracy"], label='PYTHON')


axs[1, 2].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["loss"], label='JULIA')
axs[1, 2].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["loss"], label='PYTHON')

for ax, lab in zip(axs.flat, list(BIG_GPU_PYTHON.keys())[1:]):
    ax.set(xlabel='Epoch', ylabel=lab)
    ax.legend()

plt.tight_layout()
plt.savefig("./image/BIG_GPU.png")

"""# VIEW EVOLUTION OF TIME WITH SIZE

# Python
"""

fig, axs = plt.subplots(2, 3, figsize=(15, 10))


axs[0, 0].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["epoch_memory_usage"], label='SMALL')
axs[0, 0].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["epoch_memory_usage"], label='MEDIUM')
axs[0, 0].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["epoch_memory_usage"], label='BIG')

axs[0, 1].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["epoch_time"], label='SMALL')
axs[0, 1].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["epoch_time"], label='MEDIUM')
axs[0, 1].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["epoch_time"], label='BIG')


axs[0, 2].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["batch_processing_time"], label='SMALL')
axs[0, 2].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["batch_processing_time"], label='MEDIUM')
axs[0, 2].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["batch_processing_time"], label='BIG')


axs[1, 0].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["throughput"], label='SMALL')
axs[1, 0].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["throughput"], label='MEDIUM')
axs[1, 0].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["throughput"], label='BIG')


axs[1, 1].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["accuracy"], label='SMALL')
axs[1, 1].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["accuracy"], label='MEDIUM')
axs[1, 1].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["accuracy"], label='BIG')


axs[1, 2].plot(SMALL_GPU_PYTHON["epoch"], SMALL_GPU_PYTHON["loss"], label='SMALL')
axs[1, 2].plot(MEDIUM_GPU_PYTHON["epoch"], MEDIUM_GPU_PYTHON["loss"], label='MEDIUM')
axs[1, 2].plot(BIG_GPU_PYTHON["epoch"], BIG_GPU_PYTHON["loss"], label='BIG')

for ax, lab in zip(axs.flat, list(BIG_GPU_PYTHON.keys())[1:]):
    ax.set(xlabel='Epoch', ylabel=lab)
    ax.legend()

plt.tight_layout()
plt.savefig("./image/ALL_PYTHON.png")

"""# JULIA"""

fig, axs = plt.subplots(2, 3, figsize=(15, 10))


axs[0, 0].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["epoch_memory_usage"], label='SMALL')
axs[0, 0].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["epoch_memory_usage"], label='MEDIUM')
axs[0, 0].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["epoch_memory_usage"], label='BIG')

axs[0, 1].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["epoch_time"], label='SMALL')
axs[0, 1].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["epoch_time"], label='MEDIUM')
axs[0, 1].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["epoch_time"], label='BIG')


axs[0, 2].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["batch_processing_time"], label='SMALL')
axs[0, 2].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["batch_processing_time"], label='MEDIUM')
axs[0, 2].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["batch_processing_time"], label='BIG')


axs[1, 0].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["throughput"], label='SMALL')
axs[1, 0].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["throughput"], label='MEDIUM')
axs[1, 0].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["throughput"], label='BIG')


axs[1, 1].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["accuracy"], label='SMALL')
axs[1, 1].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["accuracy"], label='MEDIUM')
axs[1, 1].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["accuracy"], label='BIG')


axs[1, 2].plot(SMALL_GPU_JULIA["epoch"], SMALL_GPU_JULIA["loss"], label='SMALL')
axs[1, 2].plot(MEDIUM_GPU_JULIA["epoch"], MEDIUM_GPU_JULIA["loss"], label='MEDIUM')
axs[1, 2].plot(BIG_GPU_JULIA["epoch"], BIG_GPU_JULIA["loss"], label='BIG')

for ax, lab in zip(axs.flat, list(BIG_GPU_PYTHON.keys())[1:]):
    ax.set(xlabel='Epoch', ylabel=lab)
    ax.legend()

plt.tight_layout()
plt.savefig("./image/ALL_JULIA.png")


print("Done!")
print("Images saved in ./image/")