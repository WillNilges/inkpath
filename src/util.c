#include "util.h"

const char* xoj_header = "<?xml version=\"1.0\" standalone=\"no\"?>\n<xournal version=\"0.4.8.2016\">\n<title>Xournal document - see http://math.mit.edu/~auroux/software/xournal/</title>\n<page width=\"612.00\" height=\"792.00\">\n<background type=\"solid\" color=\"white\" style=\"lined\" />\n<layer>\n";

const char* start_stroke = "<stroke tool=\"pen\" color=\"black\" width=\"1.41\">\n";

const char* end_stroke = "\n</stroke>\n";

const char* xoj_footer = "</layer>\n</page>\n</xournal>\n";

void invoke_autotrace(char* input_file, char* output_file, int color_count, char* background)
{
    at_fitting_opts_type* opts = at_fitting_opts_new();
    opts->color_count = color_count;
    opts->centerline = 1;
    if (background)
    {
        char s_red[3];
        char s_grn[3];
        char s_blu[3];

        strncpy(s_red, background+0, 2);
        strncpy(s_grn, background+2, 2);
        strncpy(s_blu, background+4, 2);

        opts->background_color = at_color_new(
            (char)strtol(s_red, NULL, 16),
            (char)strtol(s_grn, NULL, 16),
            (char)strtol(s_blu, NULL, 16)
        );

    }

    at_input_read_func rfunc = at_input_get_handler(input_file);
    at_bitmap_type* bitmap;
    at_splines_type* splines;

    bitmap = at_bitmap_read(rfunc, input_file, NULL, NULL, NULL);
    splines = at_splines_new(bitmap, opts, NULL, NULL);


    /* Open output file for writing */
    FILE* outptr;
    outptr = fopen(output_file, "w");
    fprintf(outptr, "%s", xoj_header);

    unsigned this_list;
    spline_list_type list;

    for (this_list = 0; this_list < SPLINE_LIST_ARRAY_LENGTH(*splines); this_list++) {
        unsigned this_spline;
        spline_type first;

        list = SPLINE_LIST_ARRAY_ELT(*splines, this_list);
        first = SPLINE_LIST_ELT(list, 0);

        double start_x = START_POINT(first).x;
        double start_y = START_POINT(first).y;

        // TODO: Figure out if it looks better or worse to separate each
        // spline as its own stroke.
        // https://github.com/WillNilges/inkpath/issues/4
        fprintf(outptr, "%s", start_stroke);
        for (this_spline = 0; this_spline < SPLINE_LIST_LENGTH(list); this_spline++) {
            spline_type s = SPLINE_LIST_ELT(list, this_spline);

            if (SPLINE_DEGREE(s) == LINEARTYPE) {
                fprintf(outptr, "%f %f %f %f ", start_x/10.0, start_y/-10.0 + 500, END_POINT(s).x/10.0, END_POINT(s).y/-10.0 + 500);
            } else {
                double x_arr[4] = {start_x, CONTROL1(s).x, CONTROL2(s).x, END_POINT(s).x};
                double y_arr[4] = {start_y, CONTROL1(s).y, CONTROL2(s).y, END_POINT(s).y};
                bezierCurve(x_arr, y_arr, outptr);
            }
            start_x = END_POINT(s).x;
            start_y = END_POINT(s).y;
        }
        fprintf(outptr, "%s", end_stroke);
    }

    fprintf(outptr, "%s", xoj_footer);
    fclose(outptr);
}

// https://www.geeksforgeeks.org/cubic-bezier-curve-implementation-in-c/
/* Function that take input as Control Point x_coordinates and
Control Point y_coordinates and draw bezier curve
Inputs: 4 x coords and 4 y coords
*/
void bezierCurve(double x[] , double y[], FILE* outptr)
{
    double xu = 0.0 , yu = 0.0 , u = 0.0;
    for(u = 0.0 ; u <= 1.0 ; u += 0.0001)
    {
        xu = (1-u)*(1-u)*(1-u)*x[0]+3*u*(1-u)*(1-u)*x[1]+3*u*u*(1-u)*x[2]
             +u*u*u*x[3];
        yu = (1-u)*(1-u)*(1-u)*y[0]+3*u*(1-u)*(1-u)*y[1]+3*u*u*(1-u)*y[2]
            +u*u*u*y[3];
        fprintf(outptr, "%f %f ", (xu/10.0), (yu/10.0)*(-1.0)+500);
    }
}

void xoj_compress(char* input_file, char* output_file) {
    gzFile outp = gzopen(output_file , "wb");

    /* Open the file for reading */
    char *line_buf = NULL;
    size_t line_buf_size = 0;
    ssize_t line_size;
    FILE *fp = fopen(input_file, "r");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", input_file);
        return;
    }

    /* Get the first line of the file. */
    line_size = getline(&line_buf, &line_buf_size, fp);

    /* Loop through until we are done with the file. */
    while (line_size >= 0)
    {
        gzwrite (outp, line_buf, line_size);

        /* Get the next line */
        line_size = getline(&line_buf, &line_buf_size, fp);
    }

    /* Free the allocated line buffer */
    free(line_buf);
    line_buf = NULL;
    fclose(fp);
    gzclose(outp);

}
