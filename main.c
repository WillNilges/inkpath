//gcc sample.c `pkg-config --libs autotrace` `pkg-config --cflags autotrace`

/* sample.c */
#include <autotrace/autotrace.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{

  //if (sizeof(*argv) == 0) exit(1);
  //if (sizeof(*argv)/sizeof(argv[0])) exit(1);

  char * fname = argv[1];
  at_fitting_opts_type * opts = at_fitting_opts_new();
  at_input_read_func rfunc = at_input_get_handler(fname);
  at_bitmap_type * bitmap ;
  at_splines_type * splines;
  at_output_write_func wfunc = at_output_get_handler_by_suffix("svg");

  bitmap = at_bitmap_read(rfunc, fname, NULL, NULL, NULL);
  splines = at_splines_new(bitmap, opts, NULL, NULL);
  FILE *fptr;
  fptr = fopen(argv[2],"w");
  at_splines_write(wfunc, fptr, "", NULL, splines, NULL, NULL);
  fclose(fptr);
  return 0;
}
