#ifndef Parser_h
#define Parser_h

#include <getopt.h>

typedef struct {
  int nsweeps;
  int ngrids;
  int mag_output_int;
  int grid_output_int;
  int threads_per_block;
  int gpu_device;
  int gpu_method;
  double beta;
  double h;
  int itask;
  char *output_file;
  char *input_file;
} parser_arguments;

typedef struct {
  int nsweeps;
  int ngrids;
  int mag_output_int;
  int grid_output_int;
  int threads_per_block;
  int gpu_device;
  int gpu_method;
  int beta;
  int h;
  int itask;
  int output_file;
  int input_file;
} parser_flags;

// commandline options
static struct option long_options[] = {
  {"nsweeps", required_argument, 0, 's'}, // -s
  {"ngrids",  required_argument, 0, 'n'}, // -n
  {"mag_output_int", required_argument, 0, 'x'},
  {"grid_output_int", required_argument, 0, 'z'},
  {"threadsPerBlock", required_argument, 0, 'y'},
  {"gpu_device", required_argument, 0, 'd'}, // -d
  {"gpu_method", required_argument, 0, 'g'}, // -g
  {"beta", required_argument, 0, 'b'}, // -b
  {"mag", required_argument, 0, 'h'}, // -h
  {"itask", required_argument, 0, 't'}, // -t
  {"output_file", required_argument, 0, 'o'}, // -t
  {"input_file", required_argument, 0, 'i'}, // -t
  {0, 0, 0, 0}
};

parser_arguments default_args;

parser_arguments parse_cl_arguments(int argc, char **argv);

#endif

