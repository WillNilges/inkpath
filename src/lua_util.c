#include "lua_util.h"

#define LUA_PUTPOINT(x, y) \
    lua_newtable(L);       \
    lua_pushnumber(L, x);  \
    lua_rawseti(L, -2, 1); \
    lua_pushnumber(L, y);  \
    lua_rawseti(L, -2, 2); \
    lua_rawseti(L, -2, point_count++);

int transcribe_image(lua_State *L)
{
    int color_count = 2;
    char* image_path = luaL_checkstring(L, 1);
    char* background = "FFFFFF";

    printf("Processing strokes...\n");

    // Potrace Magic™
    // TODO:
    // process_file
    //  potrace_state_t *st = potrace_trace(info.param, bm);

    /*
    struct info_s info;
    info.backend = 
    potrace_bitmap_t* bm = NULL;
    bm_read(fin, info.blacklevel, &bm);
*/
    potrace_bitmap_t* bm = NULL;
    potrace_param_t* param = potrace_param_default();
    param->turdsize = 3;
    potrace_state_t* st = potrace_trace(param, bm);
    potrace_path_t* p;
    int n, i;
    potrace_dpoint_t (*c)[3];

    FILE* fin = fopen(image_path, "rb");
    bm_read(fin, 0.5, &bm);

    // So anyway we create the stroke table
    lua_newtable(L);
    int point_count = 1;
    
    // And we've got some offsets to position our points correctly on the canvas
    double offset_x = 50.0;
    double offset_y = 500.0;
    double scaling = 0.1;

    p = st->plist;
    while (p != NULL)
    {
        n = p->curve.n;
        //tag = p->curve.tag;
        c = p->curve.c;       
        
        LUA_PUTPOINT(c[n-1][2].x * scaling * offset_x, c[n-1][2].y * scaling * offset_y);
        //printf("%f %f moveto\n", c[n-1][2].x, c[n-1][2].y); // Move to the end of the last curve (the second control point of the last curve)
        for (i=0; i<n; i++) {
            LUA_PUTPOINT(c[i][0].x * scaling * offset_x, c[i][0].y * scaling * offset_y);
            LUA_PUTPOINT(c[i][1].x * scaling * offset_x, c[i][1].y * scaling * offset_y);
            LUA_PUTPOINT(c[i][2].x * scaling * offset_x, c[i][2].y * scaling * offset_y);

            /*
            switch (tag[i]) {
            case POTRACE_CORNER: // If it's a corner, create a line from the end of the last curve to the first control point, then from the first control point to the second control point
              //printf("%f %f lineto\n", c[i][1].x, c[i][1].y);
              //printf("%f %f lineto\n", c[i][2].x, c[i][2].y);
              break;
            case POTRACE_CURVETO: // Otherwise, create a bezier(?) curve from the end of the last curve to the end of this curve, using the first two points as control points. So, there's only three points per struct? Fucking weird.
              //printf("%f %f %f %f %f %f curveto\n",-
              //  c[i][0].x, c[i][0].y,
              //  c[i][1].x, c[i][1].y,    
              //  c[i][2].x, c[i][2].y);
              break;
            }
            */
        }
        LUA_PUTPOINT(-1, -1);
        /* at the end of a group of a positive path and its negative
           children, fill. */
        //if (p->next == NULL || p->next->sign == '+') {
        //  printf("0 setgray fill\n");
        //}
        p = p->next;
    }                                                  
    //printf("grestore\n");
    //printf("%%EOF\n");

//    pathlist_free(st->plist);
//    free(st);
//    free(p);

/*

  // ======================================
    // AutoTrace Magic™
    at_fitting_opts_type* opts = at_fitting_opts_new();
    opts->color_count = color_count;
    opts->centerline = 1;
    if (background)
    {
        char s_red[3];
        char s_grn[3];
        char s_blu[3];

        strncpy(s_red, background+0, 2);
        strncpy(s_grn, background+2, 2);
        strncpy(s_blu, background+4, 2);

        opts->background_color = at_color_new(
            (char)strtol(s_red, NULL, 16),
            (char)strtol(s_grn, NULL, 16),
            (char)strtol(s_blu, NULL, 16)
        );

    }

    at_input_read_func rfunc = at_input_get_handler(image_path);
    at_bitmap_type* bitmap;
    at_splines_type* splines;

    bitmap = at_bitmap_read(rfunc, image_path, NULL, NULL, NULL);
    splines = at_splines_new(bitmap, opts, NULL, NULL);

    unsigned this_list;
    spline_list_type list;

    // Each stroke is made up of a number of splines.
    // I'm going to separate out the strokes with that -1, -1 delimiter still,
    // and push each spline within that stroke one after the other. I think I still
    // want to store each point in its own Lua table, because that'll just be
    // so much easier to process after the fact
    
    // So anyway we create the stroke table
    lua_newtable(L);
    int point_count = 1;

    for (this_list = 0; this_list < SPLINE_LIST_ARRAY_LENGTH(*splines); this_list++) {
        unsigned this_spline;
        //spline_type first;

        list = SPLINE_LIST_ARRAY_ELT(*splines, this_list);
        //first = SPLINE_LIST_ELT(list, 0);

        // Dump the points of splines 4 at a time, delimit with -1, -1
        // Start "stroke"
        for (this_spline = 0; this_spline < SPLINE_LIST_LENGTH(list); this_spline++) {

            spline_type s = SPLINE_LIST_ELT(list, this_spline);

            // Push the transcribed points a little further onto the center of the page
            double offset_x = 50.0;
            double offset_y = 500.0;

            double scaling = 0.1; // Scale the points down.

            // So this is a spline.
            lua_newtable(L); // One table per point
            lua_pushnumber(L, START_POINT(s).x * scaling + offset_x);
            lua_rawseti(L, -2, 1);
            lua_pushnumber(L, START_POINT(s).y * scaling * -1.0 + offset_y);
            lua_rawseti(L, -2, 2);
            lua_rawseti(L, -2, point_count++);

            lua_newtable(L);
            lua_pushnumber(L, CONTROL1(s).x * scaling + offset_x);
            lua_rawseti(L, -2, 1);
            lua_pushnumber(L, CONTROL1(s).y * scaling * -1.0 + offset_y);
            lua_rawseti(L, -2, 2);
            lua_rawseti(L, -2, point_count++);

            lua_newtable(L);
            lua_pushnumber(L, CONTROL2(s).x * scaling + offset_x);
            lua_rawseti(L, -2, 1);
            lua_pushnumber(L, CONTROL2(s).y * scaling * -1.0 + offset_y);
            lua_rawseti(L, -2, 2);
            lua_rawseti(L, -2, point_count++);

            lua_newtable(L);
            lua_pushnumber(L, END_POINT(s).x * scaling + offset_x);
            lua_rawseti(L, -2, 1);
            lua_pushnumber(L, END_POINT(s).y * scaling * -1.0 + offset_y);
            lua_rawseti(L, -2, 2);
            lua_rawseti(L, -2, point_count++);
            // And that's it. Way less data processing and memory usage required
            // on my part.
        }
        // End "stroke". We've got between 1 and N splines.
        // I'm gonna say that a point with coords -1, -1 is the end of a "stroke".
        lua_newtable(L);
        lua_pushnumber(L, -1.0);
        lua_rawseti(L, -2, 1);
        lua_pushnumber(L, -1.0);
        lua_rawseti(L, -2, 2);
        lua_rawseti(L, -2, point_count++);
    }
*/
    printf("Image transcription complete\n");

    return 1;
}

//library to be registered
static const struct luaL_Reg inkpath [] = {
      {"transcribe_image", transcribe_image},
      {NULL, NULL}  /* sentinel */
};

int luaopen_inkpath (lua_State *L){
    luaL_newlib(L, inkpath);
    return 1;
}
