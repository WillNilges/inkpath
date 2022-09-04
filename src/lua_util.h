#ifndef UTIL_H
#define UTIL_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>

#include "autotrace/autotrace.h"
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#include "spline.h"

typedef struct {
    double x;
    double y;
} inkpath_stroke_point;

int transcribe_image(lua_State *L);

#endif
