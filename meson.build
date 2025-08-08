project('gasp',           # The name of the project
        ['c', 'cuda'],    # The languages that Meson should initialize

        # Keyword arguments:
        version : '0.0.1',
        default_options : ['optimization=0', 'debug=true', 'warning_level=3'],
)

# List of options to pass to fortran compiler
add_project_arguments(['-Wno-deprecated-gpu-targets'], language:'cuda') 

subdir('src')
