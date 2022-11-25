#!/bin/bash

podman run --rm -it -v ./:/xopp-dev --name=inkpath-cuda --hooks-dir=/usr/share/containers/oci/hooks.d/ -e PKG_CONFIG_PATH=/usr/local/lib/pkgconfig inkpath-cuda
