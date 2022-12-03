#!/bin/bash

CODE_PATH=/scratch/wilnil/xopp-dev
CONTAINER_NAME='inkpath-cuda'

while getopts ":hn:c:" option; do
    case $option in
        n) # choose which container to use
            CONTAINER_NAME=$OPTARG;;
        c) # choose code path
            CODE_PATH=$OPTARG;;
        h)
            help
            exit;;
    esac
done


podman run --rm -it -v $CODE_PATH:/xopp-dev --name=$CONTAINER_NAME --hooks-dir=/usr/share/containers/oci/hooks.d/ --security-opt=label=disable -e PKG_CONFIG_PATH=/usr/local/lib/pkgconfig inkpath-cuda
