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
    FILE* fptr;
    fptr = fopen(output_file,"w");
    at_splines_write(wfunc, fptr, "", NULL, splines, NULL, NULL);
    fclose(fptr);
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
    /* Open output file for writing */
    FILE* outptr;
    outptr = fopen(output_file, "w");

    NSVGimage* g_image = NULL;
    NSVGshape* shape;
	NSVGpath* path;

	g_image = nsvgParseFromFile(input_file, "px", 96.0f);
	if (g_image == NULL) {
		printf("Could not open SVG image.\n");
		return;
	}

    fprintf(outptr, "%s", xoj_header);
    for (shape = g_image->shapes; shape != NULL; shape = shape->next)
    {
		for (path = shape->paths; path != NULL; path = path->next)
        {
            fprintf(outptr, "%s", start_stroke);
            float* pts = path->pts;
            int npts = path->npts;
            for (int i = 0; i < npts-1; i += 3)
            {
                float* p = &pts[i*2];
                // glVertex2f(p[6],p[7]);
                fprintf(outptr, "%f %f ", p[6] / 10, p[7] / 10);

                // glVertex2f(p[2],p[3]);
                // glVertex2f(p[4],p[5]);
                // glVertex2f(p[6],p[7]);
            }
            fprintf(outptr, "%s", end_stroke);
		}
	}
    fprintf(outptr, "%s", xoj_footer);

    fclose(outptr);
	nsvgDelete(g_image);
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