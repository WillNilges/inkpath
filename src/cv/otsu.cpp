#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <getopt.h>
#include <iostream>
using namespace cv;

void otsu(Mat img, bool upsampling)
{
    int k;
    // global thresholding
    
    Mat upsampled;
    if (upsampling)
    {
        // upsample BEFORE doing the thresholding
        pyrUp(img, upsampled,  Size( img.cols*2, img.rows*2 ));
        imshow("gauss_upsampled.png", upsampled);
    } else
    {
        upsampled = img;
    }
    
    /*
    Mat global_thresh;
    threshold(img, global_thresh, 127, 255, THRESH_BINARY);
    imshow("global.png", global_thresh);

    // otsu's thresholding
    Mat otsu_thresh;
    threshold(img, otsu_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("otsu.png", otsu_thresh);
    */

    /*
    //otsu's thresholding after gaussian filtering
    Mat gauss_thresh;
    Mat blur;
    GaussianBlur(upsampled, blur, Size(5, 5), 0, 0); 
    threshold(blur, gauss_thresh, 0, 255, THRESH_OTSU);
    imshow("gaussian.png", gauss_thresh);
    */

    Mat downsampled;
    pyrDown(img, downsampled, Size( img.cols/2, img.rows/2 ));
    
    Mat adaptive_thresh;
    adaptiveThreshold(downsampled, adaptive_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 51, 2);
    imshow("adaptive_thresh.png", adaptive_thresh);

    while (true) {
        k = waitKey(0); // Wait for a keystroke in the window
        if (k == 'q') {
            return;
        }
    }
}

int main(int argc, char *argv[])
{
    
   /*------------------------------------------------------------------------
     add your UI variables with sentential values and add your get_opt_long 
     code here
   ------------------------------------------------------------------------*/
  int verbose = 0;
  int order = 0;
    std::string image_path;
  int rc;
   /* getopt_long stores the option index here. */
   int option_index = 0;
  
   /* This contains the short command line parameters list */
   const char* getoptOptions = "f:";    /* add lots of stuff here */
  
   /* This contains the long command line parameter list, it should mostly 
     match the short list                                                  */
   struct option long_options[] = {
        /* These options don’t set a flag.
            We distinguish them by their indices. */
        {"file",     required_argument, 0, 'f'},
        {0, 0, 0, 0}
   };
  
   opterr = 1;           /* Enable automatic error reporting */
   while ((rc = getopt_long_only(argc, argv, getoptOptions, long_options, 
                                                    &option_index)) != -1) {
      /* Detect the end of the options. */
      switch (rc)
        {

        /* add lots of stuff here */
        case 'f':
            image_path = optarg;
            break;

       default:
          fprintf (stderr, "Internal error: undefined option %0xX\n", rc);
          exit(1);
       } // End switch 
   } /* end while */

    Mat img = imread(image_path, 0);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    otsu(img, false);
    /*imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("output.png", img);
    }*/
    return 0;
}

