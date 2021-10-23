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
    char* image_path = luaL_checkstring(L, 1);

    //int stroke_count; // The amount of strokes
    //inkpath_stroke_point* strokes; // Array of strokes

    printf("Going to process strokes. Hold my beer.\n");
    inkpath_pointset points = invoke_autotrace(image_path, color_count, "FFFFFF");
    
    printf("%d points have been retrieved.\n", points.point_count);
    lua_pushnumber(L, 3.14);
   
    return 1;
}

inkpath_pointset invoke_autotrace(
    char* input_file,
    int color_count,
    char* background
){
    // AutoTrace Magic™
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

    // Allocate enough memory for every spline to be turned into a bezier curve
    inkpath_stroke_point* points = malloc(SPLINE_LIST_ARRAY_LENGTH(*splines) * sizeof(inkpath_stroke_point) * 10001);
    int point_count = 0;

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
                // If we can, just use a straight line

                // Start Point
                points[point_count].x = start_x/10.0;
                points[point_count].y = start_y/-10.0 + 500.0;
                (point_count)++; // Go to next point

                // End point
                points[point_count].x = END_POINT(s).x/10.0;
                points[point_count].y = END_POINT(s).y/-10.0 + 500.0;
                (point_count)++; // Go to next point
            } else {
                // Otherwise IDK man, apply a bezier curve
                double x_arr[4] = {start_x, CONTROL1(s).x, CONTROL2(s).x, END_POINT(s).x};
                double y_arr[4] = {start_y, CONTROL1(s).y, CONTROL2(s).y, END_POINT(s).y};
                double xu = 0.0 , yu = 0.0 , u = 0.0;
                for(u = 0.0 ; u <= 1.0 ; u += 0.0001)
                {
                    xu = (1-u)*(1-u)*(1-u)*x_arr[0]+3*u*(1-u)*(1-u)*x_arr[1]+3*u*u*(1-u)*x_arr[2]
                         +u*u*u*x_arr[3];
                    yu = (1-u)*(1-u)*(1-u)*y_arr[0]+3*u*(1-u)*(1-u)*y_arr[1]+3*u*u*(1-u)*y_arr[2]
                        +u*u*u*y_arr[3];
                    points[point_count].x = (xu/10.0);
                    points[point_count].y = (yu/10.0)*(-1.0)+500.0;
                    (point_count)++;
                }
            }
            // End stroke
            //(*stroke_count)++;
            start_x = END_POINT(s).x;
            start_y = END_POINT(s).y;
        }
    }
    printf("We're done!\n");
    inkpath_pointset product;
    product.points = points;
    product.point_count = point_count;
    return product;
}

// https://www.geeksforgeeks.org/cubic-bezier-curve-implementation-in-c/
/* Function that take input as Control Point x_coordinates and
Control Point y_coordinates and draw bezier curve
Inputs: 4 x coords and 4 y coords
*/
/*void bezierCurve(double x[] , double y[], inkpath_stroke** stroke, int* stroke_count, int* point_count)
{
    double xu = 0.0 , yu = 0.0 , u = 0.0;
    for(u = 0.0 ; u <= 1.0 ; u += 0.0001)
    {
        xu = (1-u)*(1-u)*(1-u)*x[0]+3*u*(1-u)*(1-u)*x[1]+3*u*u*(1-u)*x[2]
             +u*u*u*x[3];
        yu = (1-u)*(1-u)*(1-u)*y[0]+3*u*(1-u)*(1-u)*y[1]+3*u*u*(1-u)*y[2]
            +u*u*u*y[3];
        *stroke[*point_count].x = (xu/10.0);
        *stroke[*point_count].y = (yu/10.0)*(-1.0)+500.0;
        *point_count++;
    }
}*/

//library to be registered
static const struct luaL_Reg inkpath [] = {
      {"transcribe_image", transcribe_image},
      {NULL, NULL}  /* sentinel */
};

//name of this function is not flexible
int luaopen_inkpath (lua_State *L){
    luaL_newlib(L, inkpath);
    return 1;
}
