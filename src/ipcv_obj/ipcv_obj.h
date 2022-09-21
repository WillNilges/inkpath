#ifndef IPCV_OBJ
#define IPCV_OBJ

/*
 * This is a barebones-af interface for allowing Lua to get data directly™ from OpenCV
 * */

// Lua C API
#include <lua.hpp>
// C++ input/output streams
#include <iostream>
#include <opencv2/core/types.hpp>

#include "../cv/ipcv.h"

using namespace cv;
using namespace std;

typedef vector<vector<Point>> ContourList;
 
// IPCVObj as C++ class
class IPCVObj
{
	private:
		ContourList contours;
	public:
		IPCVObj(){}
		void set(ContourList contours){this->contours = contours;}
		ContourList get() const{return this->contours;}
};
 
int cv_perform_processing(const char* image_path, IPCVObj* data);

// IPCVObj identifier for the Lua metatable
#define LUA_IPCVOBJ "IPCVObj"
 
static int ipcvobj_new(lua_State* L);
 
// Free IPCVObj instance by Lua garbage collection
static int ipcvobj_delete(lua_State* L);
 
// Length stuff
static int ipcvobj_getLength(lua_State* L);
static int ipcvobj_getContourLength(lua_State* L);

// Receiving data
static int ipcvobj_getContour(lua_State* L);

#endif
