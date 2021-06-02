#include "util.h"

#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"

const char* xoj_header = "<?xml version=\"1.0\" standalone=\"no\"?>\n<xournal version=\"0.4.8.2016\">\n<title>Xournal document - see http://math.mit.edu/~auroux/software/xournal/</title>\n<page width=\"612.00\" height=\"792.00\">\n<background type=\"solid\" color=\"white\" style=\"lined\" />\n<layer>\n";

const char* start_stroke = "<stroke tool=\"pen\" color=\"black\" width=\"1.41\">\n";

const char* end_stroke = "\n</stroke>\n";

const char* xoj_footer = "</layer>\n</page>\n</xournal>\n";

void invoke_autotrace(char* input_file, char* output_file, int color_count, char* background)
{
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
        printf("%d %d %d\n",
            (int)strtol(s_red, NULL, 16),
            (int)strtol(s_grn, NULL, 16),
            (int)strtol(s_blu, NULL, 16)
        );

        opts->background_color = at_color_new(
            (char)strtol(s_red, NULL, 16),
            (char)strtol(s_grn, NULL, 16),
            (char)strtol(s_blu, NULL, 16)
        );

        // TODO: WTF is this?
        // input_opts->background_color = at_color_copy(fitting_opts->background_color);
    }

    at_input_read_func rfunc = at_input_get_handler(input_file);
    at_bitmap_type* bitmap;
    at_splines_type* splines;
    at_output_write_func wfunc = at_output_get_handler_by_suffix("svg");

    bitmap = at_bitmap_read(rfunc, input_file, NULL, NULL, NULL);
    splines = at_splines_new(bitmap, opts, NULL, NULL);
    

    // TODO: Calculate a series of points from splines.
    // at_spline_type s = splines->data->data[0];

    // Or just dump to a file
    FILE* fptr;
    fptr = fopen(output_file,"w");
    at_splines_write(wfunc, fptr, "", NULL, splines, NULL, NULL);
    fclose(fptr);
}

// // https://www.geeksforgeeks.org/cubic-bezier-curve-implementation-in-c/
// /* Function that take input as Control Point x_coordinates and
// Control Point y_coordinates and draw bezier curve */
// void bezierCurve(int x_start , int y_start, int x_ctrl , int y_ctrl, int x_start , int y_start)
// {
//     double xu = 0.0 , yu = 0.0 , u = 0.0 ;
//     int i = 0 ;
//     for(u = 0.0 ; u <= 1.0 ; u += 0.0001)
//     {
//         xu = pow(1-u,3)*x[0]+3*u*pow(1-u,2)*x[1]+3*pow(u,2)*(1-u)*x[2]
//              +pow(u,3)*x[3];
//         yu = pow(1-u,3)*y[0]+3*u*pow(1-u,2)*y[1]+3*pow(u,2)*(1-u)*y[2]
//             +pow(u,3)*y[3];
//         SDL_RenderDrawPoint(renderer , (int)xu , (int)yu) ;
//     }
// }

// http://members.chello.at/~easyfilter/bresenham.html
void plotQuadBezierSeg(int x0, int y0, int x1, int y1, int x2, int y2)
{                            
  int sx = x2-x1, sy = y2-y1;
  long xx = x0-x1, yy = y0-y1, xy;         /* relative values for checks */
  double dx, dy, err, cur = xx*sy-yy*sx;                    /* curvature */

//   assert(xx*sx <= 0 && yy*sy <= 0);  /* sign of gradient must not change */

  if (sx*(long)sx+sy*(long)sy > xx*xx+yy*yy) { /* begin with longer part */ 
    x2 = x0; x0 = sx+x1; y2 = y0; y0 = sy+y1; cur = -cur;  /* swap P0 P2 */
  }  
  if (cur != 0) {                                    /* no straight line */
    xx += sx; xx *= sx = x0 < x2 ? 1 : -1;           /* x step direction */
    yy += sy; yy *= sy = y0 < y2 ? 1 : -1;           /* y step direction */
    xy = 2*xx*yy; xx *= xx; yy *= yy;          /* differences 2nd degree */
    if (cur*sx*sy < 0) {                           /* negated curvature? */
      xx = -xx; yy = -yy; xy = -xy; cur = -cur;
    }
    dx = 4.0*sy*cur*(x1-x0)+xx-xy;             /* differences 1st degree */
    dy = 4.0*sx*cur*(y0-y1)+yy-xy;
    xx += xx; yy += yy; err = dx+dy+xy;                /* error 1st step */    
    do {                              
    //   setPixel(x0,y0);                                     /* plot curve */
      printf("%d, %d\n", x0, y0);
      if (x0 == x2 && y0 == y2) return;  /* last pixel -> curve finished */
      y1 = 2*err < dx;                  /* save value for test of y step */
      if (2*err > dy) { x0 += sx; dx -= xy; err += dy += yy; } /* x step */
      if (    y1    ) { y0 += sy; dy -= xy; err += dx += xx; } /* y step */
    } while (dy < dx );           /* gradient negates -> algorithm fails */
  }
//   plotLine(x0,y0, x2,y2);                  /* plot remaining part to end */
}  

// https://cboard.cprogramming.com/c-programming/117525-regex-h-extracting-matches.html
char* regexp(char* string, regex_t* rgT, int* begin, int* end)
{ 
    int i, w=0, len;                  
    char *word = NULL;
    regmatch_t match;
    if ((regexec(rgT,string,1,&match,0)) == 0) {
            *begin = (int)match.rm_so;
            *end = (int)match.rm_eo;
            len = *end-*begin;
            word=malloc(len+1);
            for (i=*begin; i<*end; i++) {
                    word[w] = string[i];
                    w++; }
            word[w]=0;
    }
    // regfree(&rgT);
    return word;
}

void svg_to_xoj(char* input_file, char* output_file)
{

    // Set up the regex for svg
    // regex_t svg_regex;
    // int code;
    // if ((code = regcomp(&svg_regex,"([mMzZlLhHvVcCsSqGtTaA])(?:\\s*(-?\\d*\\.?\\d+))?(?:\\s*(-?\\d*\\.?\\d+))?(?:\\s*(-?\\d*\\.?\\d+))?(?:\\s*(-?\\d*\\.?\\d+))?(?:\\s*(-?\\d*\\.?\\d+))?(?:\\s*(-?\\d*\\.?\\d+))", REG_ICASE | REG_EXTENDED)) == 0)
    //     printf("Regular expression compiled successfully.\n");
    // else {
    //     printf("Regex compilation error: %d.\n", code);
    //     return;
    // }

    FILE* svg_in;
    svg_in = fopen(input_file, "r");

    /* Open output file for writing */
    FILE* xoj_out;
    xoj_out = fopen(output_file, "w");

    // This is stupid.
   size_t BUFF_SIZE = 20;
   char* buffer = malloc(BUFF_SIZE);
   char* p;

   long int pos;
   pos = ftell(svg_in);

    while (fgets(buffer, BUFF_SIZE, svg_in)) {
        // If the buffer is ever too small, this'll just increase it and start
        // over.
        while (strlen(buffer)+10 >= BUFF_SIZE) {
            fseek (svg_in, pos, SEEK_SET);
            BUFF_SIZE *= 10; // Maybe exponentially growing the buffer will make it faster?
            buffer = realloc(buffer, BUFF_SIZE);
            fgets(buffer, BUFF_SIZE, svg_in);
        }

    //   // Sanitize for bad characters
    //   int buffContentSize = strlen(buffer);
    //   for (int i = 0; i < buffContentSize; i++) {
    //      if (buffer[i] == '\n' || buffer[i] == '\r') {
    //         buffer[i] = '\0';
    //      }
    //   }

        int bezi_pos = 0;
        float* bezi_list = malloc(sizeof(float)*BUFF_SIZE); // Definitely too big, but fuck it.

        if (strstr(buffer, "<path") != NULL) {

            int curvecount = 0;

            // begin parsing
            p = strtok(buffer," \t");

            // Parse until no tokens left
            while ( p != NULL) {
                // Parse the next token, returns NULL for none
                p = strtok(NULL, " \t");
                if (p != NULL)
                    printf("%s\n", p);
                if (strchr(p, 'C') != NULL) {
                    p[0] = '0';
                    // bezi_list[bezi_pos] = atof()
                }
            }
        }
        // int b, e;
        // char* match = regexp(buffer, &svg_regex, &b, &e);
        // if (match) {
        //     printf("%s\n", match);
        // }
   }

    fclose(xoj_out);
    fclose(svg_in);
}

void xoj_compress(char* input_file, char* output_file) {
    gzFile outp = gzopen(output_file , "wb");

    /* Open the file for reading */
    char *line_buf = NULL;
    size_t line_buf_size = 0;
    ssize_t line_size;
    FILE *fp = fopen(input_file, "r");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", input_file);
        return;
    }

    /* Get the first line of the file. */
    line_size = getline(&line_buf, &line_buf_size, fp);

    /* Loop through until we are done with the file. */
    while (line_size >= 0)
    {
        gzwrite (outp, line_buf, line_size);

        /* Get the next line */
        line_size = getline(&line_buf, &line_buf_size, fp);
    }

    /* Free the allocated line buffer */
    free(line_buf);
    line_buf = NULL;
    fclose(fp);
    gzclose(outp);

}