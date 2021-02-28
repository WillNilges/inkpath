#include "util.h"

int main(int argc, char *argv[])
{
    char* temp_svg = "temp.svg";
    char* temp_xoj = "temp.xoj";
    int color_count = 2;

    invoke_autotrace(argv[1], temp_svg, color_count, "FFFFFF");
    svg_to_xoj(temp_svg, temp_xoj);
    xoj_compress(temp_xoj, argv[2]);
    remove(temp_svg);
    remove(temp_xoj);

    return 0;
}