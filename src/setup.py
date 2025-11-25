import os
import sys
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import multiprocessing

current_path = os.getcwd()

if len(sys.argv) > 1:
    cuda_arch = sys.argv.pop(1)
    try:
        int(cuda_arch)
    except ValueError:
        print("Error: CUDA architecture must be a number (e.g., 80, 89).")
        sys.exit(1)
else:
    cuda_arch = "80"


num_cores = multiprocessing.cpu_count()
max_jobs = min(32, num_cores * 2)


extra_compile_args = {
    "nvcc" : ["-O3", "-w", 
    "-I/usr/local/cuda/include", 
    f"-I{str(current_path)}/ops/src/include/", 
    f"-I{str(current_path)}/ops/src/include/cutlass/include",
    f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}"],
    "cxx": [
    "-fPIC",
    f"-I{str(current_path)}/ops/src/include/", 
    f"-I{str(current_path)}/ops/src/include/cutlass/include",
    ],
}
extra_link_args = []


def get_extensions():    
    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, "ops", "src")
    
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*_cuda.cu")))
    
    extension_names = set()
    for source in sources + cuda_sources:
        base_name = os.path.basename(source)
        name, _ = os.path.splitext(base_name)
        if name.endswith("_cuda"):
            name = name[:-5]
        extension_names.add(name)
            
    ext_modules = []
    for name in extension_names:
        extension_sources = [s for s in sources + cuda_sources if name in os.path.basename(s)]
        ext_modules.append(
            CUDAExtension(
                name=name,
                sources=extension_sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args
            )
        )
    
    return ext_modules

setup(
    name='Binding_Attention',     
    packages=find_packages(),
    version='0.1.1',
    author='PPoPP26 Attention-Fusion',
    
    ext_modules=get_extensions(),
    
    install_requires=["torch"],
    description="customed operator linked with cutlass implement",
    
    cmdclass={
        'build_ext': BuildExtension.with_options(
            use_ninja=True,
            max_jobs=max_jobs,
            no_python_abi_suffix=True
        )
    }
)
