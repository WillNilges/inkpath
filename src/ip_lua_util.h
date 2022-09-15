#ifndef UTIL_H
#define UTIL_H
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#include "cv/ipcv.h"

int cv_perform_processing(string image_path);
int cv_transcribe_image(lua_State *L);

#endif
