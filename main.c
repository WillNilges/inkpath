//gcc sample.c `pkg-config --libs autotrace` `pkg-config --cflags autotrace`

/* sample.c */
#include <autotrace/autotrace.h>
// #include "color.h"
#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <string.h>
#include <glib.h>

// TODO: Move to header file
void invoke_autotrace(char* input_file, char* output_file, int color_count, char* background);
char *regexp (char* string, regex_t* rgT, int* begin, int* end);
int* get_colors(char* input_file, int max_colors);

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
char *regexp (char* string, regex_t* rgT, int* begin, int* end)
{ 
    int i, w=0, len;                  
    char *word = NULL;
    // regex_t rgT;
    regmatch_t match;
    // regcomp(&rgT,patrn,REG_EXTENDED);
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

int* get_colors(char* input_file, int max_colors)
{
    // Set up the regex for hex codes
    regex_t hex_regex;
    if (regcomp(&hex_regex,"#[0-9a-f]{6}", REG_ICASE | REG_EXTENDED) == 0)
        printf("Regular expression compiled successfully.\n");
    else {
        printf("Compilation error.\n");
        return NULL;
    }

    // Array to store our colors in
    // int max_colors = 2;
    int colors_found = 0;
    // int colors[max_colors];
    int* colors = malloc(sizeof(int)*max_colors);

    /* Open the file for reading */
    char *line_buf = NULL;
    size_t line_buf_size = 0;
    int line_count = 0;
    ssize_t line_size;
    FILE *fp = fopen(input_file, "r");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", input_file);
        return NULL;
    }

    /* Get the first line of the file. */
    line_size = getline(&line_buf, &line_buf_size, fp);

    /* Loop through until we are done with the file. */
    while (line_size >= 0)
    {
        /* Increment our line count */
        line_count++;

        /* Collect the line details */
        int b, e;
        char *match=regexp(line_buf, &hex_regex, &b, &e);
        if (match)
        {
            // printf("-> %s <-\n(b=%d e=%d)\n", match, b, e); //Debug
            char current_color[8];
            strncpy(current_color, match+1, 6);
            int i_current_color = (int)strtol(current_color, NULL, 16);
            // printf("%s\n", current_color);
            int new_color = 1;
            for (int i = 0; i < max_colors; i++)
            {
                if (i_current_color == colors[i])
                    new_color = 0;
            }
            if (new_color == 1)
                colors[colors_found++] = i_current_color;
        }

        /* Get the next line */
        line_size = getline(&line_buf, &line_buf_size, fp);
    }

    /* Free the allocated line buffer */
    free(line_buf);
    line_buf = NULL;

    regfree(&hex_regex);

    /* Close the file now that we are done with it */
    fclose(fp);
    
    return colors;
}

void dump_to_xopp() {
    char* input_file  = "data/points.txt";
    char* output_file = "out.temp";

    /* Open the file for reading */
    char *line_buf = NULL;
    size_t line_buf_size = 0;
    int line_count = 0;
    ssize_t line_size;
    FILE *fp = fopen(input_file, "r");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", input_file);
        // return NULL;
    }

    /* Open output file for writing */
    FILE* outptr;
    outptr = fopen(output_file, "w");
    // fprintf(outptr, "<stroke tool=\"pen\" ts=\"0ll\" fn=\"\" color=\"#3333ccff\" width=\"1.41000000\">");

    // File head
    fprintf(outptr, "<?xml version=\"1.0\" standalone=\"no\"?>\n<xournal version=\"0.4.8.2016\">\n<title>Xournal document - see http://math.mit.edu/~auroux/software/xournal/</title>\n<page width=\"612.00\" height=\"792.00\">\n<background type=\"solid\" color=\"white\" style=\"lined\" />\n<layer>\n<stroke tool=\"pen\" color=\"black\" width=\"1.41\">");

    /* Get the first line of the file. */
    line_size = getline(&line_buf, &line_buf_size, fp);

    /* Loop through until we are done with the file. */
    // while (line_size >= 0)
    // {
        
    // }

    /* Free the allocated line buffer */
    free(line_buf);
    line_buf = NULL;

    fprintf(outptr, "\n</stroke>\n</layer>\n</page>\n</xournal>\n");

    fclose(outptr);

    /* Close the file now that we are done with it */
    fclose(fp);

}

int main(int argc, char *argv[])
{
    int color_count = 2;

    // invoke_autotrace(argv[1], argv[2], color_count, "FFFFFF");
    dump_to_xopp();

    return 0;
}