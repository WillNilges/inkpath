//gcc sample.c `pkg-config --libs autotrace` `pkg-config --cflags autotrace`

/* sample.c */
#include <autotrace/autotrace.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
  char * fname = "data/board_square_lines.jpg";
  at_fitting_opts_type * opts = at_fitting_opts_new();
  at_input_read_func rfunc = at_input_get_handler(fname);
  at_bitmap_type * bitmap ;
  at_splines_type * splines;
  at_output_write_func wfunc = at_output_get_handler_by_suffix("eps");

  bitmap = at_bitmap_read(rfunc, fname, NULL, NULL, NULL);
  splines = at_splines_new(bitmap, opts, NULL, NULL);
  FILE *fptr;
  fptr = fopen("out2.svg","w");
  at_splines_write(wfunc, fptr, "", NULL, splines, NULL, NULL);
  fclose(fptr);
  return 0;
}
