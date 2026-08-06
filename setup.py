from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="ext_ipc_queue",
        sources=["src/cpp_extensions/ext_ipc_queue.cpp"],
        extra_compile_args={"cxx": ["-O3", "-DNDEBUG", "-std=c++17"]},
        extra_link_args=["-lcuda"],
    ),
    CUDAExtension(
        name="ext_aggregator",
        sources=["src/cpp_extensions/ext_aggregator.cpp"],
        extra_compile_args={"cxx": ["-O3", "-DNDEBUG", "-std=c++17"]},
        extra_link_args=["-lcuda", "-lpthread"],
    ),
    CUDAExtension(
        name="ext_unified_barrier",
        sources=["src/cpp_extensions/ext_unified_barrier.cpp"],
        extra_compile_args={"cxx": ["-O3", "-DNDEBUG", "-std=c++17"]},
        extra_link_args=["-lcuda", "-lpthread"],
    ),
]

setup(
    name="split_model_extensions",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
