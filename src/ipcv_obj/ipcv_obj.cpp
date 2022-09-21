#include "ipcv_obj.h"

int cv_perform_processing(const char* image_path, IPCVObj* data)
{
    Mat img = imread(image_path, 0);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        exit(1);
    }
    Mat otsu_img = otsu(img, "");
    std::cout << "Performing otsu filtering...\n";
    Mat skel_img = skeletonize(otsu_img, "");
    std::cout << "Performing Skeletonization...\n";
    Shapes shapes = find_shapes(skel_img, "");
    std::cout << "Looking for shapes...\n";
    data->set(shapes.contours); 
    return 0;
}

// Create & return IPCVObj instance to Lua
static int ipcvobj_new(lua_State* L)
{
	//int length = luaL_checkinteger(L, 1);
	//*reinterpret_cast<IPCVObj**>(lua_newuserdata(L, sizeof(IPCVObj*))) = new IPCVObj();
	//luaL_setmetatable(L, LUA_IPCVOBJ);
	//return 1;
    const char* image_path = luaL_checkstring(L, 1);
    int tracing_scale = luaL_checkinteger(L, 2);

    IPCVObj* object; // Declare pointer
    object = new IPCVObj(); // Initialize pointer
    *reinterpret_cast<IPCVObj**>(lua_newuserdata(L, sizeof(IPCVObj*))) = object; // Make Lua aware of it
	luaL_setmetatable(L, LUA_IPCVOBJ); // Metatable magic

    cv_perform_processing(image_path, object); // do CV stuff

    std::cout << "CV Processing complete!\n";
    return 1;
}
 
// Free IPCVObj instance by Lua garbage collection
static int ipcvobj_delete(lua_State* L)
{
	delete *reinterpret_cast<IPCVObj**>(lua_touserdata(L, 1));
	return 0;
}
 
// IPCVObj member functions in Lua
//static int ipcvobj_set(lua_State* L)
//{
//	(*reinterpret_cast<IPCVObj**>(luaL_checkudata(L, 1, LUA_IPCVOBJ)))->set(luaL_checkinteger(L, 2), luaL_checkinteger(L, 3));
//	return 0;
//}

//static int ipcvobj_get(lua_State* L)
//{
//	lua_pushnumber(L, (*reinterpret_cast<IPCVObj**>(luaL_checkudata(L, 1, LUA_IPCVOBJ)))->get(luaL_checkinteger(L, 2)));
//	return 1;
//}

// Length stuff
static int ipcvobj_getLength(lua_State* L)
{
    lua_pushinteger(L, (*reinterpret_cast<IPCVObj**>(luaL_checkudata(L, 1, LUA_IPCVOBJ)))->get().size());
    return 1;
}

static int ipcvobj_getContourLength(lua_State* L)
{
    lua_pushinteger(L, (*reinterpret_cast<IPCVObj**>(luaL_checkudata(L, 1, LUA_IPCVOBJ)))->get()[luaL_checkinteger(L, 2)].size());
    return 1;  
}

// Receiving data
static int ipcvobj_getPointXInContour(lua_State* L)
{
    int contourIdx = luaL_checkinteger(L, 2);
    int pointIdx = luaL_checkinteger(L, 3);
    lua_pushnumber(L, (*reinterpret_cast<IPCVObj**>(luaL_checkudata(L, 1, LUA_IPCVOBJ)))->get()[contourIdx][pointIdx].x / 10.0);
    return 1;
}

static int ipcvobj_getPointYInContour(lua_State* L)
{
    int contourIdx = luaL_checkinteger(L, 2);
    int pointIdx = luaL_checkinteger(L, 3);
    lua_pushnumber(L, (*reinterpret_cast<IPCVObj**>(luaL_checkudata(L, 1, LUA_IPCVOBJ)))->get()[contourIdx][pointIdx].y / 10.0);
    return 1;
}

static int ipcvobj_getContour(lua_State* L)
{
    int contourIdx = luaL_checkinteger(L, 2);
    double scalingFactor = luaL_checknumber(L, 3);

    vector<Point> selectedContour = (*reinterpret_cast<IPCVObj**>(luaL_checkudata(L, 1, LUA_IPCVOBJ)))->get()[contourIdx];

    // Push all the X coords to the stack
    lua_newtable(L);
    for (int i = 0; i < selectedContour.size(); i++)
    {
        lua_pushnumber(L, selectedContour[i].x / scalingFactor);
        lua_rawseti(L, -2, i);
    }

    // Push all the Y coords to the stack
    lua_newtable(L);
    for (int i = 0; i < selectedContour.size(); i++)
    {
        lua_pushnumber(L, selectedContour[i].y / scalingFactor);
        lua_rawseti(L, -2, i);
    }

    return 2; // Returning two tables
}


// haha computor
int processImage(lua_State *L)
{
    const char* image_path = luaL_checkstring(L, 1);
    int tracing_scale = luaL_checkinteger(L, 2);

    IPCVObj* object; // Declare pointer
    object = new IPCVObj(); // Initialize pointer
    *reinterpret_cast<IPCVObj**>(lua_newuserdata(L, sizeof(IPCVObj*))) = object; // Make Lua aware of it
	luaL_setmetatable(L, LUA_IPCVOBJ); // Metatable magic

    cv_perform_processing(image_path, object); // do CV stuff

    std::cout << "CV Processing complete!\n";
    return 1;
}

// Register IPCVObj to Lua
static void register_ipcvobj(lua_State* L){
	lua_register(L, LUA_IPCVOBJ, ipcvobj_new);
	luaL_newmetatable(L, LUA_IPCVOBJ);
	lua_pushcfunction(L, ipcvobj_delete); lua_setfield(L, -2, "__gc");
	lua_pushvalue(L, -1); lua_setfield(L, -2, "__index");
	lua_pushcfunction(L, ipcvobj_getLength); lua_setfield(L, -2, "getLength");
	lua_pushcfunction(L, ipcvobj_getContourLength); lua_setfield(L, -2, "getContourLength");
	lua_pushcfunction(L, ipcvobj_getPointXInContour); lua_setfield(L, -2, "getPointXInContour");
	lua_pushcfunction(L, ipcvobj_getPointYInContour); lua_setfield(L, -2, "getPointYInContour");
	lua_pushcfunction(L, ipcvobj_getContour); lua_setfield(L, -2, "getContour");
	lua_pop(L, 1);
}
 
extern "C" {
// Program entry
int luaopen_ipcvobj(lua_State *L)
{
		luaL_openlibs(L);
		register_ipcvobj(L);
	return 1;
}
}
