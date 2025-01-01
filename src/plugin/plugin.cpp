#include "plugin.h"

// Do OpenCV stuff
int cv_perform_processing(const char* image_path, Inkpath* data) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        exit(1);
    }

    // Detect a whiteboard in the image, crop, and straighten
    cv::Mat whiteboard_img = get_whiteboard(img, "");
    
    // Do it again --- this should solve the "projector problem" where you have
    // content on a projector framed by the border of the photo. Nominally this
    // should find no more squares.
    cv::Mat whiteboard_img_2 = get_whiteboard(whiteboard_img, "");

    cv::Mat otsu_img = otsu(whiteboard_img_2, "");
    std::cout << "Performing otsu filtering...\n";
    cv::Mat skel_img = skeletonize(otsu_img, "");
    std::cout << "Performing Skeletonization...\n";
    Shapes shapes = find_strokes(skel_img, "");
    std::cout << "Looking for shapes...\n";
    data->set(shapes.contours);
    return 0;
}

// Create & return inkpath instance to Lua
static int inkpath_new(lua_State* L) {
    const char* image_path = luaL_checkstring(L, 1);
    int tracing_scale = luaL_checkinteger(L, 2);

    Inkpath* object;        // Declare pointer
    object = new Inkpath(); // Initialize pointer
    *reinterpret_cast<Inkpath**>(lua_newuserdata(L, sizeof(Inkpath*))) =
        object;                        // Make Lua aware of it
    luaL_setmetatable(L, LUA_INKPATH); // Metatable magic

    cv_perform_processing(image_path, object); // do CV stuff

    std::cout << "CV Processing complete!\n";
    return 1;
}

// Free inkpath instance by Lua garbage collection
static int inkpath_delete(lua_State* L) {
    delete *reinterpret_cast<Inkpath**>(lua_touserdata(L, 1));
    return 0;
}

// Length stuff
static int inkpath_getContourCount(lua_State* L) {
    lua_pushinteger(
        L, (*reinterpret_cast<Inkpath**>(luaL_checkudata(L, 1, LUA_INKPATH)))
               ->get()
               .size());
    return 1;
}

static int inkpath_getContourLength(lua_State* L) {
    lua_pushinteger(
        L, (*reinterpret_cast<Inkpath**>(luaL_checkudata(L, 1, LUA_INKPATH)))
               ->get()[luaL_checkinteger(L, 2)]
               .size());
    return 1;
}

// Receiving data
static int inkpath_getContour(lua_State* L) {
    int contourIdx = luaL_checkinteger(L, 2);
    double scalingFactor = luaL_checknumber(L, 3);

    std::vector<cv::Point> selectedContour =
        (*reinterpret_cast<Inkpath**>(luaL_checkudata(L, 1, LUA_INKPATH)))
            ->get()[contourIdx];

    // Push all the X coords to the stack
    lua_newtable(L);
    for (int i = 0; i < selectedContour.size(); i++) {
        lua_pushnumber(L, selectedContour[i].x * scalingFactor);
        lua_rawseti(L, -2, i);
    }

    // Push all the Y coords to the stack
    lua_newtable(L);
    for (int i = 0; i < selectedContour.size(); i++) {
        lua_pushnumber(L, selectedContour[i].y * scalingFactor);
        lua_rawseti(L, -2, i);
    }

    return 2; // Returning two tables
}

// Register inkpath to Lua
static void register_inkpath(lua_State* L) {
    lua_register(L, LUA_INKPATH, inkpath_new);
    luaL_newmetatable(L, LUA_INKPATH);
    lua_pushcfunction(L, inkpath_delete);
    lua_setfield(L, -2, "__gc");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    lua_pushcfunction(L, inkpath_getContourCount);
    lua_setfield(L, -2, "getContourCount");
    lua_pushcfunction(L, inkpath_getContourLength);
    lua_setfield(L, -2, "getContourLength");
    lua_pushcfunction(L, inkpath_getContour);
    lua_setfield(L, -2, "getContour");
    lua_pop(L, 1);
}

extern "C" {
// Program entry
WINEXPORT int luaopen_loadInkpath(lua_State* L) {
    printf("Entered Inkpath.");
    luaL_openlibs(L);
    register_inkpath(L);
    return 1;
}
}
