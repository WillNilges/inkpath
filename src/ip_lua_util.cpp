#include "ip_lua_util.h"

Shapes cv_perform_processing(char* image_path) {
    Mat img = imread(image_path, 0);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        exit(1);
    }
    Mat otsu_img = otsu(img, "");
    Mat skel_img = skeletonize(otsu_img, "");
    Shapes shapes = find_shapes(skel_img, "");
    return shapes;
}

int cv_transcribe_image(lua_State* L)
{
    const char* image_path = luaL_checkstring(L, 1);
    const char* background = "FFFFFF";
    int tracing_scale = luaL_checkinteger(L, 2);

    Shapes shapes = cv_perform_processing(image_path);

    // With this method, we posess an actual raster of our stroke
    // So we can skip the spline step and just pass the coordinates raw
    // to Lua. We can probably do away with any delimiting stuff and just
    // make each stroke its own table (for now at least)

    // So anyway we create the stroke table
    lua_newtable(L);
    int point_count = 1;

    for (vector<Point> contour : shapes.contours) {
        lua_newtable(L);
        for (Point point : contour) {
            lua_newtable(L);
            lua_pushnumber(L, point.x);
            lua_rawseti(L, -2, 1);
            lua_pushnumber(L, point.y);
            lua_rawseti(L, -2, 2);
            lua_rawseti(L, -2, point_count++); // TODO: What the fuck does this do?
        } 
    }

    cout << "Image transcription complete :)\n";
    return 1;
}

//library to be registered
static const struct luaL_Reg inkpath [] = {
      {"transcribe_image", cv_transcribe_image},
      {NULL, NULL}  /* sentinel */
};

int luaopen_inkpath (lua_State *L){
    luaL_newlib(L, inkpath);
    return 1;
}
