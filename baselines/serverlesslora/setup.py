"""
Build script for ServerlessLoRA CUDA extensions.

Extensions:
1. ext_ipc_wrap: Zero-copy tensor sharing via CUDA IPC (Section 4.4)
2. ext_stream_loader: Concurrent tensor loading with CUDA Streams (Section 5)

Usage:
    python setup.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Common compile args
cxx_args = ['-O3', '-std=c++17']
nvcc_args = ['-O3']

# Library paths (adjust as needed for your system)
library_dirs = [
    '/usr/lib64',
    '/usr/local/cuda/lib64/stubs',
    '/usr/local/cuda/lib64',
]

# Filter to existing paths
library_dirs = [d for d in library_dirs if os.path.exists(d)]

extensions = [
    # Extension 1: IPC Wrapper (existing)
    CUDAExtension(
        name='ext_ipc_wrap',
        sources=['csrc/ipc_wrapper.cpp'],
        extra_compile_args={
            'cxx': cxx_args,
            'nvcc': nvcc_args
        },
        libraries=['cuda'],
        library_dirs=library_dirs,
        runtime_library_dirs=['/usr/lib64'] if os.path.exists('/usr/lib64') else [],
    ),

    # Extension 2: Stream Loader (new - Paper Section 5)
    CUDAExtension(
        name='ext_stream_loader',
        sources=['csrc/cuda_stream_loader.cpp'],
        extra_compile_args={
            'cxx': cxx_args,
            'nvcc': nvcc_args
        },
        libraries=['cuda', 'cudart'],
        library_dirs=library_dirs,
        runtime_library_dirs=['/usr/lib64'] if os.path.exists('/usr/lib64') else [],
    ),
]

setup(
    name='serverless_lora_cuda',
    version='1.0.0',
    description='CUDA extensions for ServerlessLoRA',
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.8',
)
