//gcc sample.c `pkg-config --libs autotrace` `pkg-config --cflags autotrace`

/* sample.c */
#include <autotrace/autotrace.h>

int main()
{
  char * fname = "img/triangle.png";
  at_fitting_opts_type * opts = at_fitting_opts_new();
  at_input_read_func rfunc = at_input_get_handler(fname);
  at_bitmap_type * bitmap ;
  at_splines_type * splines;
  at_output_write_func wfunc = at_output_get_handler_by_suffix("eps");

  bitmap = at_bitmap_read(rfunc, fname, NULL, NULL, NULL);
  splines = at_splines_new(bitmap, opts, NULL, NULL);
  at_splines_write(wfunc, stdout, "", NULL, splines, NULL, NULL);
  return 0;
}
