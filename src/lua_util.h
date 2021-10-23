#ifndef UTIL_H
#define UTIL_H
#include <autotrace/autotrace.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#include "spline.h"

typedef struct {
    double x;
    double y;
} inkpath_stroke_point;

typedef struct {
    int point_count;
    inkpath_stroke_point* points;
} inkpath_stroke;

typedef struct {
    int point_count;
    inkpath_stroke_point* points;
} inkpath_pointset;

int transcribe_image(lua_State *L);

inkpath_pointset invoke_autotrace(
    char* input_file,
    int color_count,
    char* background
);
/*
void bezierCurve(
    double x[],
    double y[],
    inkpath_stroke** stroke,
    int* stroke_count,
    int* point_count
);
*/
#endif
