#include "otsu.h"
#include "OtsuWrapper.h"

#ifdef __cplusplus
extern "C" {
#endif
    void c_prep_otsu(char* image_path) {
        prep_otsu(image_path);
    }
#ifdef __cplusplus
}
#endif
