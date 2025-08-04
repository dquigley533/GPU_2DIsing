# Need Python headers to compile the wrapper code
py_mod = import('python')
python = py_mod.find_installation()

# Also need NumPy headers
numpy_dep = dependency('numpy', required: true, fallback: ['numpy', 'numpy_dep'])

gasp_src = ['gpu_tools.cu',
            'io.c',
            'mc_cpu.c',
            'mc_gpu.cu',
	    'grid.c',
	    'mt19937ar.c']

py = import('python').find_installation(pure: false)

# Build fortran source and wrapper code into extension module
py.extension_module(
  'gasp',     
  ['ising_ext.cu', gasp_src],
  include_directories: ['../include/'],
  install: true,
)