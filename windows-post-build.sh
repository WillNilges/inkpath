#!/bin/bash

set -e

mkdir -p ImageTranscription/

# Copy Inkpath library to plugin folder
cp libipcvobj.dll ImageTranscription

# Copy dependencies to plugin folder
ldd libipcvobj.dll | grep mingw64 | awk '{ print $3 }' | xargs -I {} cp {} ./ImageTranscription

# Copy script and manifest to plugin folder
cp ../plugin/* ./ImageTranscription