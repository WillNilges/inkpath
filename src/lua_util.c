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

    int stroke_count; // The amount of strokes
    inkpath_stroke* strokes; // Array of strokes

    printf("Going to process strokes. Hold my beer.\n");
    invoke_autotrace(image_path, color_count, "FFFFFF", &strokes, &stroke_count);
    
    printf("Stroke count (%d strokes) has been retrieved.\n", stroke_count);
    for (int i = 0; i < stroke_count; i++)
    {
        for (int j = 0; j < strokes[i].point_count; i++)
        {
            printf("Coord: %f, %f", strokes[i].points[j].x, strokes[i].points[j].y);
        }
    }
/*
    // Push results to Lua stack, array by array
    for (int i = 0; i < stroke_count; i++)
    {
int arr[4] = { 5,100,-20,0 };
lua_newtable(L);              // table
for (i=0; i<4; i++) {
  lua_pushinteger(L, i+1);    // table,key
  lua_pushinteger(L, arr[i]); // table,key,value
  lua_settable(L,-3);         // table
}

        //lua_createtable(L, point_counts[i], 0);
        //lua_newtable(L);
        for(int j = 0; j < point_counts[i]; j++)
        {
            return 0;
            //printf("Stroke point: %f\n", strokes[i][j]);
            //lua_pushnumber(L, strokes[i][j]);
            //lua_rawseti(L, -2, j + 1);
        }
        //return 0;
    }*/
    lua_pushinteger(L, stroke_count);
    
    return 0;
}

void invoke_autotrace(
    char* input_file,
    int color_count,
    char* background,
    inkpath_stroke** strokes,
    int* stroke_count
){
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

    // Allocate memory for array of stroke arrays
    // We know we're gonna need N amount of arrays
    // (one array per stroke), so malloc that many right now
    strokes = malloc(sizeof(inkpath_stroke*) * SPLINE_LIST_ARRAY_LENGTH(*splines));
    *stroke_count = 0; 

    for (this_list = 0; this_list < SPLINE_LIST_ARRAY_LENGTH(*splines); this_list++) {
        unsigned this_spline;
        spline_type first;

        list = SPLINE_LIST_ARRAY_ELT(*splines, this_list);
        first = SPLINE_LIST_ELT(list, 0);

        double start_x = START_POINT(first).x;
        double start_y = START_POINT(first).y; 

        for (this_spline = 0; this_spline < SPLINE_LIST_LENGTH(list); this_spline++) {

            spline_type s = SPLINE_LIST_ELT(list, this_spline);

            // Let's see how much RAM we need.
            /*
            int RAM_required = 0;

            if (SPLINE_DEGREE(s) == LINEARTYPE) {
                RAM_required += 8; // This is doubled
            } else {
                RAM_required += 25000; // This is +5000
            }
            */

            // Start stroke
            // Initialize a stroke from the stroke array
            //inkpath_stroke test = { .point_count = 0, .points = NULL };
            //test.points = malloc(sizeof(inkpath_stroke_point) * RAM_required);
            //strokes[*stroke_count]->points = (inkpath_stroke_point*) malloc(sizeof(inkpath_stroke_point) * RAM_required);
            strokes[*stroke_count] = malloc(sizeof(inkpath_stroke));
            strokes[*stroke_count]->point_count = 0;

            if (SPLINE_DEGREE(s) == LINEARTYPE) {
                // fprintf(outptr, "%f %f %f %f ", start_x/10.0, start_y/-10.0 + 500, END_POINT(s).x/10.0, END_POINT(s).y/-10.0 + 500);
                // If we can, just use a straight line
                strokes[*stroke_count]->points = malloc(sizeof(inkpath_stroke_point) * 2);
                // Start Point
                strokes[*stroke_count]->points[strokes[*stroke_count]->point_count].x = start_x/10.0;
                strokes[*stroke_count]->points[strokes[*stroke_count]->point_count].y = start_y/-10.0 + 500.0;

                (strokes[*stroke_count]->point_count)++; // Go to next point

                // End point
                strokes[*stroke_count]->points[strokes[*stroke_count]->point_count].x = END_POINT(s).x/10.0;
                strokes[*stroke_count]->points[strokes[*stroke_count]->point_count].y = END_POINT(s).y/-10.0 + 500.0;
                (strokes[*stroke_count]->point_count)++; // Go to next point
            } else {
                strokes[*stroke_count]->points = malloc(sizeof(inkpath_stroke_point) * 11000);
                double x_arr[4] = {start_x, CONTROL1(s).x, CONTROL2(s).x, END_POINT(s).x};
                double y_arr[4] = {start_y, CONTROL1(s).y, CONTROL2(s).y, END_POINT(s).y};
                bezierCurve(x_arr, y_arr, strokes[*stroke_count]);
            }
            // End stroke
            (*stroke_count)++;
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
void bezierCurve(double x[] , double y[], inkpath_stroke* stroke)
{
    double xu = 0.0 , yu = 0.0 , u = 0.0;
    for(u = 0.0 ; u <= 1.0 ; u += 0.0001)
    {
        xu = (1-u)*(1-u)*(1-u)*x[0]+3*u*(1-u)*(1-u)*x[1]+3*u*u*(1-u)*x[2]
             +u*u*u*x[3];
        yu = (1-u)*(1-u)*(1-u)*y[0]+3*u*(1-u)*(1-u)*y[1]+3*u*u*(1-u)*y[2]
            +u*u*u*y[3];
        // fprintf(outptr, "%f %f ", (xu/10.0), (yu/10.0)*(-1.0)+500);
        stroke->points[stroke->point_count].x = (xu/10.0);
        stroke->points[stroke->point_count].y = (yu/10.0)*(-1.0)+500.0;
        stroke->point_count++;
    }
}

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
