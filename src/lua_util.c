#include "lua_util.h"

int transcribe_image(lua_State *L)
{
    int color_count = 2;
    char* image_path = luaL_checkstring(L, 1);

    printf("Processing strokes...\n");
    inkpath_pointset points = invoke_autotrace(image_path, color_count, "FFFFFF");
    
    printf("%d points have been retrieved.\n", points.point_count);
    
    // Push data into lua table
    lua_newtable(L);

    for (int i = 0; i < points.point_count; i++) {
        lua_newtable(L);
        lua_pushnumber(L, points.points[i].x);
        lua_rawseti(L, -2, 1);
        lua_pushnumber(L, points.points[i].y);
        lua_rawseti(L, -2, 2);

        lua_rawseti(L, -2, i+1);
    }
    free(points.points);
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
    // *2 to account for delimiting points
    // *101 for bezier curve
    inkpath_stroke_point* points = malloc(SPLINE_LIST_ARRAY_LENGTH(*splines) * 2 * sizeof(inkpath_stroke_point) * 101);
    int point_count = 0;

    for (this_list = 0; this_list < SPLINE_LIST_ARRAY_LENGTH(*splines); this_list++) {
        unsigned this_spline;
        spline_type first;

        list = SPLINE_LIST_ARRAY_ELT(*splines, this_list);
        first = SPLINE_LIST_ELT(list, 0);

        double start_x = START_POINT(first).x;
        double start_y = START_POINT(first).y; 

        // Start stroke
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
                for(u = 0.0 ; u <= 1.0 ; u += 0.01)
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
            start_x = END_POINT(s).x;
            start_y = END_POINT(s).y;
        }
        // End stroke
        // I'm gonna say that a point with coords -1, -1 is the end of a "stroke".
        points[point_count].x = -1.0;
        points[point_count].y = -1.0;
        (point_count)++;
    }
    printf("We're done!\n");
    inkpath_pointset product;
    product.points = points;
    product.point_count = point_count;
    return product;
}

//library to be registered
static const struct luaL_Reg inkpath [] = {
      {"transcribe_image", transcribe_image},
      {NULL, NULL}  /* sentinel */
};

int luaopen_inkpath (lua_State *L){
    luaL_newlib(L, inkpath);
    return 1;
}
