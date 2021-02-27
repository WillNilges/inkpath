#ifndef UTIL_H
#define UTIL_H
#include <autotrace/autotrace.h>
#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <string.h>
#include <glib.h>


void invoke_autotrace(char* input_file, char* output_file, int color_count, char* background);
char *regexp (char* string, regex_t* rgT, int* begin, int* end);
void get_points();

#endif