import os
import shutil
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class BuildShimExt(build_ext):
  def get_ext_filename(self, ext_name):
    # Ensure canonical .so filename without cpython-312-... platform tag
    return ext_name.replace(".", "/") + ".so"

  def build_extension(self, ext):
    ext_path = self.get_ext_fullpath(ext.name)
    os.makedirs(os.path.dirname(ext_path), exist_ok=True)
    cxx = os.environ.get("CXX", "g++")
    cmd = [
        cxx,
        "-O3",
        "-shared",
        "-fPIC",
        "-std=c++17",
        "csrc/libtpu_neuron_shim.cc",
        "-o",
        ext_path,
        "-ldl",
    ]
    include_dirs = ext.include_dirs or []
    for inc in include_dirs:
      cmd.extend(["-I", inc])
    subprocess.check_call(cmd)

    target_inplace = os.path.join("torch_tpu_neuron", "libtpu_neuron_shim.so")
    if os.path.abspath(ext_path) != os.path.abspath(target_inplace):
      os.makedirs("torch_tpu_neuron", exist_ok=True)
      shutil.copy2(ext_path, target_inplace)


setup(
    name="torch-tpu-neuron",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        Extension(
            "torch_tpu_neuron.libtpu_neuron_shim",
            sources=["csrc/libtpu_neuron_shim.cc"],
            include_dirs=["csrc", "include"],
        )
    ],
    cmdclass={"build_ext": BuildShimExt},
    package_data={"torch_tpu_neuron": ["*.so"]},
)
