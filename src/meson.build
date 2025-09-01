# Need Python headers to compile the wrapper code
py_mod = import('python')
python = py_mod.find_installation()

# Also need NumPy headers
numpy_dep = dependency('numpy', required: true, fallback: ['numpy', 'numpy_dep'])

# HDF5 dependency
hdf5_dep = dependency('hdf5', required: true)

gasp_src = ['gpu_tools.cu',
            'io.c',
            'mc_cpu.c',
            'mc_gpu.cu',
	          'grid.c',
	          'mt19937ar.c',
            'bootstrap.c']

py = import('python').find_installation(pure: false)

# Add preprocessor flag for use in Python
add_project_arguments('-DPYTHON', '-w', language : 'c')
add_project_arguments('-DPYTHON', '-w', '-Wno-deprecated-gpu-targets', language : 'cuda')

# Add debug flags for C and CUDA
#add_project_arguments('-g', '-O0', language : 'c')
#add_project_arguments('-G', language : 'cuda')

# Build fortran source and wrapper code into extension module
py.extension_module(
  'gasp',     
  ['ising_ext.cu', gasp_src],
  include_directories: ['../include/'],
  dependencies: [numpy_dep, hdf5_dep],
  install: true,
)
