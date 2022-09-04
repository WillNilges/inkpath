/* input-png.c: PNG loader for autotrace

   Copyright (C) 2000 MenTaLguY <mental@rydia.net>

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif /* Def: HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "types.h"
#include "bitmap.h"
#include "logreport.h"
#include "xstd.h"
#include <png.h>
#include "input-png.h"

static png_bytep *read_png(png_structp png_ptr, png_infop info_ptr, at_input_opts_type * opts);

/* for pre-1.0.6 versions of libpng */
#ifndef png_jmpbuf
#	define png_jmpbuf(png_ptr) (png_ptr)->jmpbuf
#endif

static void handle_warning(png_structp png, const gchar * message)
{
  LOG("PNG warning: %s", message);
  at_exception_warning((at_exception_type *) png_get_error_ptr(png), message);
  /* at_exception_fatal((at_exception_type *)at_png->error_ptr,
     "PNG warning"); */
}

static void handle_error(png_structp png, const gchar * message)
{
  LOG("PNG error: %s", message);
  at_exception_fatal((at_exception_type *) png_get_error_ptr(png), message);
  /* at_exception_fatal((at_exception_type *)at_png->error_ptr,
     "PNG error"); */

}

static void finalize_structs(png_structp png, png_infop info, png_infop end_info)
{
  png_destroy_read_struct(png ? &png : NULL, info ? &info : NULL, end_info ? &end_info : NULL);
}

static int init_structs(png_structp * png, png_infop * info, png_infop * end_info, at_exception_type * exp)
{
  *png = NULL;
  *info = *end_info = NULL;

  *png = png_create_read_struct(PNG_LIBPNG_VER_STRING, exp, (png_error_ptr) handle_error, (png_error_ptr) handle_warning);

  if (*png) {
    *info = png_create_info_struct(*png);
    if (*info) {
      *end_info = png_create_info_struct(*png);
      if (*end_info)
        return 1;
    }
    finalize_structs(*png, *info, *end_info);
  }
  return 0;
}

#define CHECK_ERROR() 	do { if (at_exception_got_fatal(exp))	\
	  {							\
	    result = 0;						\
	    goto cleanup;					\
	  } } while (0)

static int load_image(at_bitmap * image, FILE * stream, at_input_opts_type * opts, at_exception_type * exp)
{
  png_structp png;
  png_infop info, end_info;
  png_bytep *rows;
  unsigned short width, height, row;
  int pixel_size;
  int result = 1;

  if (!init_structs(&png, &info, &end_info, exp))
    return 0;

  png_init_io(png, stream);
  CHECK_ERROR();

  rows = read_png(png, info, opts);

  width = (unsigned short)png_get_image_width(png, info);
  height = (unsigned short)png_get_image_height(png, info);
  if (png_get_color_type(png, info) == PNG_COLOR_TYPE_GRAY) {
    pixel_size = 1;
  } else {
    pixel_size = 3;
  }

  *image = at_bitmap_init(NULL, width, height, pixel_size);
  for (row = 0; row < height; row++, rows++) {
    memcpy(AT_BITMAP_PIXEL(image, row, 0), *rows, width * pixel_size * sizeof(unsigned char));
  }
cleanup:
  finalize_structs(png, info, end_info);
  return result;
}

at_bitmap input_png_reader(gchar * filename, at_input_opts_type * opts, at_msg_func msg_func, gpointer msg_data, gpointer user_data)
{
  FILE *stream;
  at_bitmap image = at_bitmap_init(0, 0, 0, 1);
  at_exception_type exp = at_exception_new(msg_func, msg_data);

  stream = fopen(filename, "rb");
  if (!stream) {
    LOG("Can't open \"%s\"\n", filename);
    at_exception_fatal(&exp, "Cannot open input png file");
    return image;
  }

  load_image(&image, stream, opts, &exp);
  fclose(stream);

  return image;
}

static png_bytep *read_image(png_structp png_ptr, png_infop info_ptr)
{
  unsigned width, height, y;
  png_bytep *rows;

  width = png_get_rowbytes(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  rows = (png_bytep *) png_malloc(png_ptr, height * sizeof(png_bytep));
  for (y = 0; y < height; y++) {
    rows[y] = (png_bytep) png_malloc(png_ptr, width);
  }

  png_read_image(png_ptr, rows);
  return rows;
}

static png_bytep *read_png(png_structp png_ptr, png_infop info_ptr, at_input_opts_type * opts)
{
  png_color_16p original_bg;
  png_color_16 my_bg;
  png_bytep *rows;

  png_read_info(png_ptr, info_ptr);

  png_set_strip_16(png_ptr);
  png_set_packing(png_ptr);
  if ((png_get_bit_depth(png_ptr, info_ptr) < 8) || (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_PALETTE) || (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)))
    png_set_expand(png_ptr);

  if (png_get_bKGD(png_ptr, info_ptr, &original_bg)) {
    /* Fill transparent region with ... */
    my_bg.index = 0;

    if (opts && opts->background_color) {
      my_bg.red = 256 * opts->background_color->r;
      my_bg.green = 256 * opts->background_color->g;
      my_bg.blue = 256 * opts->background_color->b;
      my_bg.gray = 256 * ((opts->background_color->r + opts->background_color->g + opts->background_color->b) / 3);
    } else
      /* else, use white */
      my_bg.red = my_bg.green = my_bg.blue = my_bg.gray = 0xFFFF;

    png_set_background(png_ptr, &my_bg, PNG_BACKGROUND_GAMMA_FILE, 1, 1.0);
  } else
    png_set_strip_alpha(png_ptr);
  png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  rows = read_image(png_ptr, info_ptr);
  png_read_end(png_ptr, info_ptr);
  return rows;
}
