using Pkg
Pkg.add("PackageCompiler")
using PackageCompiler
create_sysimage([:Flux, :CUDA], sysimage_path="gpu_support.so")