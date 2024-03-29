#!/bin/bash

# It's assumed that xournalpp will be cloned outside this repo :)
cd ..
git clone https://github.com/xournalpp/xournalpp

if [[ "$1" == "clean" ]]; then
    echo 'Deleting build dir'
    rm -rf ../xournalpp/build
    exit 0
fi

cores=4

# For compiling faster
if [ -n "$1" ]; then
    cores=$1
fi

cd xournalpp
mkdir build
cd build

cmake .. -DCMAKE_INSTALL_PREFIX=/usr
cmake --build . -j$cores --target install
./cmake/postinst configure
