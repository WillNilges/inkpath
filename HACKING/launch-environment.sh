#!/bin/bash

set -e


function help() {
    echo "-u : Set username | -d : Set container | -h : Print help"
}

if [ "$(pwd)" == "$HOME" ]; then
    echo "DON'T FUCKING RUN THIS SCRIPT FROM YOUR FUCKING HOMEDIR"
    echo "Place it in a directory LITERALLY ANYWHERE ELSE where it can't hurt you :)"
    exit 1
fi

CODE_PATH="$HOME"/Code/xopp-dev
uname=$USER # Just trust me on this one.
container='xopp-dev'

while getopts ":hu:d:" option; do
    case $option in
        u) # change uname
            uname=$OPTARG;;
        d) # choose which container to use
            container=$OPTARG;;
        h)
            help
            exit;;
    esac
done

xauth_path=/tmp/xopp-dev-xauth

rm -rf "$xauth_path"
mkdir -p "$xauth_path"
cp "$HOME"/.Xauthority "$xauth_path"
chmod g+rwx "$xauth_path"/.Xauthority

podman run --name="$container" --rm -it              \
    -e DISPLAY="$DISPLAY"                             \
    --network=host                                    \
    --cap-add=SYS_PTRACE                              \
    --group-add keep-groups                           \
    --annotation io.crun.keep_original_groups=1       \
    -v "$xauth_path"/.Xauthority:/root/.Xauthority:Z  \
    -v "$CODE_PATH":/xopp-dev:Z                       \
    -v /scratch/wilnil:/scratch:Z                     \
    -v /tmp/.X11-unix:/tmp/.X11-unix                  \
    --env 'PKG_CONFIG_PATH=/usr/local/lib/pkgconfig'  \
    "$container"
rm -rf "$xauth_path"
