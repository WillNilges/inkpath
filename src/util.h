#ifndef UTIL_H
#define UTIL_H
#include <autotrace/autotrace.h>
#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <string.h>
#include <glib.h>
#include <zlib.h>

#include "spline.h"

extern const char* xoj_header;
extern const char* start_stroke;
extern const char* end_stroke;
extern const char* xoj_footer;

void invoke_autotrace(
    char* input_file,
    char* output_file,
    int color_count,
    char* background
);

void xoj_compress(
    char* input_file,
    char* output_file
);

void bezierCurve(
    double x[],
    double y[],
    FILE* outptr
);


#endif
