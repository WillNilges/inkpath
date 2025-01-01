#!/bin/bash

INKPATH_LIB=libinkpath.dll
echo "Copying $INKPATH_LIB dependencies..."
ldd ImageTranscription/$INKPATH_LIB | grep mingw64 | awk '{ print $3 }' | xargs -I {} cp {} ./ImageTranscription