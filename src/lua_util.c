#include "lua_util.h"

/*
const char* xoj_header = "<?xml version=\"1.0\" standalone=\"no\"?>\n<xournal version=\"0.4.8.2016\">\n<title>Xournal document - see http://math.mit.edu/~auroux/software/xournal/</title>\n<page width=\"612.00\" height=\"792.00\">\n<background type=\"solid\" color=\"white\" style=\"lined\" />\n<layer>\n";

const char* start_stroke = "<stroke tool=\"pen\" color=\"black\" width=\"1.41\">\n";

const char* end_stroke = "\n</stroke>\n";

const char* xoj_footer = "</layer>\n</page>\n</xournal>\n";
*/

int transcribe_image(lua_State *L)
{
    int color_count = 2;

    char* image_path = "";
    double* points;
    process_image(image_path, color_count, "FFFFFF", points);
    return 0;
}

void process_image(char* input_file, int color_count, char* background, double* output)
{
    int output_idx;

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

    unsigned this_list;
    spline_list_type list;

    output = malloc(sizeof(double) * SPLINE_LIST_ARRAY_LENGTH(*splines) * 4);
    output_idx = 0;

    for (this_list = 0; this_list < SPLINE_LIST_ARRAY_LENGTH(*splines); this_list++) {
        unsigned this_spline;
        spline_type first;

        list = SPLINE_LIST_ARRAY_ELT(*splines, this_list);
        first = SPLINE_LIST_ELT(list, 0);

        double start_x = START_POINT(first).x;
        double start_y = START_POINT(first).y;
        for (this_spline = 0; this_spline < SPLINE_LIST_LENGTH(list); this_spline++) {
            spline_type s = SPLINE_LIST_ELT(list, this_spline);

            if (SPLINE_DEGREE(s) == LINEARTYPE) {
                // fprintf(outptr, "%f %f %f %f ", start_x/10.0, start_y/-10.0 + 500, END_POINT(s).x/10.0, END_POINT(s).y/-10.0 + 500);
                // If we can, just use a straight line
                output[output_idx] = start_x/10.0;
                output[output_idx++] = start_y/-10.0 + 500.0;
                output[output_idx++] = END_POINT(s).x/10.0;
                output[output_idx++] = END_POINT(s).y/-10.0 + 500.0;
            } else {
                // Otherwise, apply a bezier curve to the spline.
                double x_arr[4] = {start_x, CONTROL1(s).x, CONTROL2(s).x, END_POINT(s).x};
                double y_arr[4] = {start_y, CONTROL1(s).y, CONTROL2(s).y, END_POINT(s).y};
                bezierCurve(x_arr, y_arr, output, &output_idx);
            }
            start_x = END_POINT(s).x;
            start_y = END_POINT(s).y;
        }
    }
}

// https://www.geeksforgeeks.org/cubic-bezier-curve-implementation-in-c/
/* Function that take input as Control Point x_coordinates and
Control Point y_coordinates and draw bezier curve
Inputs: 4 x coords and 4 y coords
*/
void bezierCurve(double x[] , double y[], double* output, int* output_idx)
{
    output = realloc(output, sizeof(output) + sizeof(double) * 10000 * 2);
    double xu = 0.0 , yu = 0.0 , u = 0.0;
    for(u = 0.0 ; u <= 1.0 ; u += 0.0001)
    {
        xu = (1-u)*(1-u)*(1-u)*x[0]+3*u*(1-u)*(1-u)*x[1]+3*u*u*(1-u)*x[2]
             +u*u*u*x[3];
        yu = (1-u)*(1-u)*(1-u)*y[0]+3*u*(1-u)*(1-u)*y[1]+3*u*u*(1-u)*y[2]
            +u*u*u*y[3];
//        fprintf(outptr, "%f %f ", (xu/10.0), (yu/10.0)*(-1.0)+500);
        output[*output_idx] = (xu/10.0);
        output[*output_idx++] = (yu/10.0)*(-1.0)+500.0;
    }
}

