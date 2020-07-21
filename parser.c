#include "parser.h"
#include <stdlib.h>

parser_arguments parse_cl_arguments(int argc, char **argv) {
  int c, option_index;
  
  // initalise arguments to defaults
  parser_arguments args = default_args;

  // loop through arguments
  while ((c = getopt_long(argc, argv, "s:n:d:g:b:h:t:o:i:", long_options, &option_index)) != -1) {
    switch (c) {
      case 's':
        args.nsweeps = atoi(optarg);
        break;
      case 'n':
        args.ngrids = atoi(optarg);
        break;
      case 'd':
        args.gpu_device = atoi(optarg);
        break;
      case 'g':
        args.gpu_method = atoi(optarg);
        break;
      case 'b':
        args.beta = atof(optarg);
        break;
      case 'h':
        args.h = atof(optarg);
        break;
      case 'x':
        args.mag_output_int = atoi(optarg);
        break;
      case 'y':
        args.threads_per_block = atoi(optarg);
        break;
      case 'z':
        args.grid_output_int = atoi(optarg);
        break;
      case 't':
        args.itask = atoi(optarg);
        break;
      case 'o':
        args.output_file = optarg;
        break;
      case 'i':
        args.input_file = optarg;
        break;
      case '?':
        // error should have been printed
        break;
      default:
        abort();
    } // end swithch
  } // end while 

  return args;
}

