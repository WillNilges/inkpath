#include "arg.h"

Options::Options(int argc, char* argv[])
{
    int rc;
    /* getopt_long stores the option index here. */
    int option_index = 0;
    
    /* This contains the short command line parameters list */
    const char* getoptOptions = "d:f:o:i:u:t:vs:h";    /* add lots of stuff here */
    
    /* This contains the long command line parameter list, it should mostly 
      match the short list                                                  */
    struct option long_options[] = {
         /* These options don’t set a flag.
             We distinguish them by their indices. */
         {"device",   required_argument, 0, 'd'},
         {"file",   required_argument, 0, 'f'},
         {"output", required_argument, 0, 'o'},
         {"iterations", required_argument, 0, 'i'},
         {"upscale", required_argument, 0, 'u'},
         {"timing", required_argument, 0, 't'},
         {"short", required_argument, 0, 's'},
         {"verbose", 0, 0, 'v'},
         {"help", 0, 0, 'h'},
         {0, 0, 0, 0}
    };
    
    opterr = 1;           /* Enable automatic error reporting */
    while ((rc = getopt_long_only(
                    argc,
                    argv,
                    getoptOptions,
                    long_options,
                    &option_index)) != -1) {
       /* Detect the end of the options. */
       switch (rc)
         {
         case 'd':
             this->device = atoi(optarg);
             break;
         case 'f':
             this->image_path = optarg;
             break;
         case 'o':
             this->output_path = optarg;
             break;
         case 'i':
             this->iters = atoi(optarg);
             break;
         case 'u':
             this->artificial_upscale = atoi(optarg);
             break;
         case 't':
            this->timing = optarg;
            break;
         case 's':
            this->threshold_only = true;
            break;
         case 'v':
             this->verbose = true;
             break;
         case 'h':
             print_help();
             break;
        default:
           fprintf (stderr, "Internal error: undefined option %0xX\n", rc);
           exit(1);
        } // End switch 
    } /* end while */

    if ((optind < argc) || image_path.empty() || (iters > 1 && !output_path.empty())){
        print_help();
        exit(1);
    }
}

void Options::print_help()
{
    printf("-d : device : device to use\n");
    printf("-f : file : input file\n");
    printf("-i : number of iterations to run (CANNOT BE MORE THAN 1 WHILE USING OUTPUT FILE)\n");
    printf("-o : output : output file\n");
    printf("-s : short : only run thresholding (useful for testing)");
    printf("-t : timing : write timing data to file\n");
    printf("-u : upscale :  number of times to upscale input data\n");
    printf("-h : help : print help\n");
}
