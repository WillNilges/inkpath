#include "ipcv_wrapper.h"
#include "ipcv.h"

#ifdef __cplusplus
extern "C" {
#endif
    c_Shapes c_perform_cv_processing(char* image_path) {
        Mat img = imread(image_path, 0);
        if(img.empty())
        {
            std::cout << "Could not read the image: " << image_path << std::endl;
            exit(1);
        }
        Mat otsu_img = otsu(img, "");
        Mat skel_img = skeletonize(otsu_img, "");
        Shapes shapes = find_shapes(skel_img, "");

        for (vector<Point> contours : shapes.contours) {
            // man fuck this.
        }
        return shapes
    }
#ifdef __cplusplus
}
#endif
