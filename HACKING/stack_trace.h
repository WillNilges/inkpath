void stackTrace(lua_State *L) 
{
    int i;
    int top = lua_gettop(L);
    printf("---- Begin Stack ----\n");
    printf("Stack size: %i\n\n", top);
    for (i = top; i >= 1; i--) 
    {
        int t = lua_type(L, i);
        switch (t) 
        { 
            case LUA_TSTRING:
            printf("%i -- (%i) string  ---- `%s'", i, i - (top + 1), lua_tostring(L, i));
            break;
    
            case LUA_TBOOLEAN:
            printf("%i -- (%i) bool    ---- %s", i, i - (top + 1), lua_toboolean(L, i) ? "true" : "false");
            break;
    
            case LUA_TNUMBER:
            printf("%i -- (%i) number  ---- %g", i, i - (top + 1), lua_tonumber(L, i));
            break;
    
            default:
            printf("%i -- (%i) default ---- %s", i, i - (top + 1), lua_typename(L, t));
            break;
        }
        printf("\n");
    }
    printf("---- End Stack ----\n");
    printf("\n\n");
}
