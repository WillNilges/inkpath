#include <getopt.h>
#include "otsu.h"

// Test function
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
    Mat otsu_img = otsu(img);
    aggressiveXylophones(otsu_img);
    /*imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("output.png", img);
    }*/
    return 0;
}

