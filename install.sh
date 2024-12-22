#!/bin/bash

# Use this script to install Inkpath on a Linux system. It will copy the artifact into Xournalpp's plugin folder.

set -e
cp -r ImageTranscription /usr/share/xournalpp/plugins/
