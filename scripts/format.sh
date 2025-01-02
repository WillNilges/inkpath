#!/bin/bash
find /xopp-dev/inkpath/src/ -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i
# TODO: Mess around with clang-tidy in checks
# find /xopp-dev/inkpath/src/ -iname '*.h' -o -iname '*.cpp' | xargs clang-tidy
