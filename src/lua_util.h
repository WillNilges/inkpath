#ifndef UTIL_H
#define UTIL_H
#include <autotrace/autotrace.h>
#include <stdio.h>
#include <stdlib.h>
//#include <regex.h>
#include <string.h>
#include <glib.h>
//#include <zlib.h>

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#include "spline.h"

/*
extern const char* xoj_header;
extern const char* start_stroke;
extern const char* end_stroke;
extern const char* xoj_footer;
*/

int transcribe_image(lua_State *L);

void process_image(
    char* input_file,
    int color_count,
    char* background,
    double* output
);

void bezierCurve(
    double x[],
    double y[],
    double* output,
    int* output_idx
);


#endif
