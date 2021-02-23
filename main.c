//gcc sample.c `pkg-config --libs autotrace` `pkg-config --cflags autotrace`

/* sample.c */
#include <autotrace/autotrace.h>
#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <string.h>

void invoke_autotrace(char* input_file, char* output_file)
{
    at_fitting_opts_type * opts = at_fitting_opts_new();
    opts->color_count = 2;
    at_input_read_func rfunc = at_input_get_handler(input_file);
    at_bitmap_type * bitmap ;
    at_splines_type * splines;
    at_output_write_func wfunc = at_output_get_handler_by_suffix("svg");

    bitmap = at_bitmap_read(rfunc, input_file, NULL, NULL, NULL);
    splines = at_splines_new(bitmap, opts, NULL, NULL);
    FILE *fptr;
    fptr = fopen(output_file,"w");
    at_splines_write(wfunc, fptr, "", NULL, splines, NULL, NULL);
    fclose(fptr);
}

// https://cboard.cprogramming.com/c-programming/117525-regex-h-extracting-matches.html
char *regexp (char* string, regex_t* rgT, int* begin, int* end) {     
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

void remove_bg(char* input_file, char* output_file)
{

    // Set up the regex for hex codes
    regex_t hex_regex;
    if (regcomp(&hex_regex,"#[0-9a-f]{6}", REG_ICASE | REG_EXTENDED) == 0)
        printf("Regular expression compiled successfully.\n");
    else {
        printf("Compilation error.\n");
        return;
    }

    // Array to store our colors in
    int max_colors = 2;
    int colors_found = 0;
    char colors[max_colors][7];

    /* Open the file for reading */
    char *line_buf = NULL;
    size_t line_buf_size = 0;
    int line_count = 0;
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
            // printf("%s\n", current_color);
            int new_color = 1;
            for (int i = 0; i < max_colors; i++)
            {
                if (strcmp(current_color, colors[i]) == 0)
                    new_color = 0;
            }
            if (new_color == 1)
                strncpy(colors[colors_found++], current_color, 6);
        }

        /* Get the next line */
        line_size = getline(&line_buf, &line_buf_size, fp);
    }

    for (int i = 0; i < max_colors; i++) {
        printf("%s\n", colors[i]);
    }

    /* Free the allocated line buffer */
    free(line_buf);
    line_buf = NULL;

    regfree(&hex_regex);

    /* Close the file now that we are done with it */
    fclose(fp);
}

int main(int argc, char *argv[])
{
    // if (sizeof(*argv)/sizeof(argv[0]) != 3)
    // {
    //     printf("Bad arguments. %d\n", sizeof(*argv));
    //     return 1;
    // }

    invoke_autotrace(argv[1], argv[2]);
    remove_bg(argv[2], NULL);
    return 0;
}
