#ifndef IPCV_OBJ
#define IPCV_OBJ

#ifdef _WIN32
#define WINEXPORT __declspec(dllexport)
#else
#define WINEXPORT
#endif

/*
 * This is a barebones interface for allowing Lua to get data directly™ from
 * OpenCV
 * */

// Lua C API
#include <lua.hpp>
// C++ input/output streams
#include <iostream>
#include <opencv2/core/types.hpp>

#include "../cv/ipcv.h"

using namespace inkp;

typedef std::vector<std::vector<cv::Point>> ContourList;

// inkpath as C++ class
class Inkpath {
  private:
    ContourList contours;

  public:
    Inkpath() {}
    void set(ContourList contours) { this->contours = contours; }
    ContourList get() const { return this->contours; }
};

// XXX (wdn): Make CV object and make a new function for it
int cv_perform_processing(const char* image_path, Inkpath* data);

// inkpath identifier for the Lua metatable
#define LUA_INKPATH "Inkpath"

static int inkpath_new(lua_State* L);

// Free inkpath instance by Lua garbage collection
static int inkpath_delete(lua_State* L);

// For managing metadata about our strokes
static int inkpath_getContourCount(lua_State* L);
static int inkpath_getContourLength(lua_State* L);

// Receiving stroke data
static int inkpath_getContour(lua_State* L);

extern "C" WINEXPORT int luaopen_loadInkpath(lua_State* L);

#endif
