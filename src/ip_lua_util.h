#ifndef UTIL_H
#define UTIL_H
extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

#include "cv/ipcv.h"

Shapes cv_perform_processing(const char* image_path);
static int cv_transcribe_image(lua_State* L);
int luaopen_inkpath(lua_State* L);

#endif
