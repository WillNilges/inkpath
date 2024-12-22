#include "ipcv_obj.h"

#ifdef _WIN32
#define WINEXPORT __declspec(dllexport)
#else
#define WINEXPORT
#endif

// Do OpenCV stuff
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

// Register IPCVObj to Lua
static void register_ipcvobj(lua_State* L){
	lua_register(L, LUA_IPCVOBJ, ipcvobj_new);
	luaL_newmetatable(L, LUA_IPCVOBJ);
	lua_pushcfunction(L, ipcvobj_delete); lua_setfield(L, -2, "__gc");
	lua_pushvalue(L, -1); lua_setfield(L, -2, "__index");
	lua_pushcfunction(L, ipcvobj_getLength); lua_setfield(L, -2, "getLength");
	lua_pushcfunction(L, ipcvobj_getContourLength); lua_setfield(L, -2, "getContourLength");
	lua_pushcfunction(L, ipcvobj_getContour); lua_setfield(L, -2, "getContour");
	lua_pop(L, 1);
}

extern "C" {
    // Program entry
    WINEXPORT int luaopen_ipcvobj(lua_State *L)
    {
        printf("Entered Inkpath.");
        #ifdef _WIN32
        // Add Inkpath's directory to the DLL search path
        //const wchar_t* dllDirectory = L"C:\\Program Files\\Xournal++\\share\\xournalpp\\plugins\\ImageTranscription";
        //const char* newPath = "C:\\Program Files\\Xournal++\\share\\xournalpp\\plugins\\ImageTranscription";


        //const char* inkpathLibPath = "C:\\Users\\willard\\AppData\\Local\\xournalpp\\plugins\\Inkpath\\lib";

        const char* inkpathLibPath = "C:\\Program Files\\Xournal++\\share\\xournalpp\\plugins\\ImageTranscription";


        // Get the current PATH
        char currentPath[MAX_PATH];
        GetEnvironmentVariable("PATH", currentPath, MAX_PATH);

        // Check if the new path is already in the PATH
        if (strstr(currentPath, newPath) == nullptr) {
            // Append the new path to the current PATH
            strcat(currentPath, ";");
            strcat(currentPath, newPath);

            // Set the new PATH environment variable
            SetEnvironmentVariable("PATH", currentPath);

            std::cout << "PATH updated successfully!" << std::endl;
        } else {
            std::cout << "Path is already in the PATH variable." << std::endl;
        }

        #endif

        luaL_openlibs(L);
        register_ipcvobj(L);

        return 1;
    }
}
