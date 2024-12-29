#!/bin/bash
find /xopp-dev/inkpath/src/ -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i
