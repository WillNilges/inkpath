#pragma once

using namespace std;
using namespace cv;

#ifdef __cplusplus
extern "C" {
#endif
    // That's really all we need, I suppose
    typedef struct c_Point
    {
        int x;
        int y;
    } c_Point;

    typedef struct c_vec4i { int x[4]; } c_vec4i;

    typedef struct c_Shapes
    {
        c_Point* contours;
        c_vec4i* hierarchy;
    } c_Shapes;

    c_Shapes c_perform_cv_processing(char* image_path);
#ifdef __cplusplus
}
#endif
