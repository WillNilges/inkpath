#include "util.h"

int main(int argc, char *argv[])
{
    char* temp_svg = "temp.svg";
    int color_count = 2;

    invoke_autotrace(argv[1], temp_svg, color_count, "FFFFFF");
    get_points(temp_svg, argv[2]);
    // dump_to_xoj();

    return 0;
}