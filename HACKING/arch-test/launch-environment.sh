#!/bin/bash

set -e

if [ "$(pwd)" == "$HOME" ]; then
    echo "DON'T FUCKING RUN THIS SCRIPT FROM YOUR FUCKING HOMEDIR"
    echo "Place it in a directory LITERALLY ANYWHERE ELSE where it can't hurt you :)"
    exit 1
fi

CODE_PATH="$HOME"/Code/xopp-dev
uname=$USER # Just trust me on this one.

# For changing the group
if [ -n "$1" ]; then
    uname=$1
fi

xauth_path=/tmp/xopp-dev-xauth

rm -rf "$xauth_path"
mkdir -p "$xauth_path"
cp "$HOME"/.Xauthority "$xauth_path"
chmod g+rwx "$xauth_path"/.Xauthority
#chmod g+rwx "$CODE_PATH"
#podman unshare chown 1000:"$uname" "$xauth_path"/.Xauthority # Chom
#podman unshare chown 1000:$uname "$CODE_PATH"
#podman unshare chown 1000:$uname -R "$CODE_PATH"
#sudo chmod -R +070 "$CODE_PATH"
#sudo chown -R :$uname "$CODE_PATH"
#sudo chown -R :$uname "$xauth_path"/.Xauthority

podman run --name=xopp-dev-arch --rm -it                   \
    -e DISPLAY="$DISPLAY"                             \
    --network=host                                    \
    --cap-add=SYS_PTRACE                              \
    --group-add keep-groups                           \
    --annotation io.crun.keep_original_groups=1       \
    -v "$xauth_path"/.Xauthority:/root/.Xauthority:Z  \
    -v "$CODE_PATH":/xopp-dev:Z                       \
    -v /scratch/wilnil:/scratch:Z                     \
    -v /tmp/.X11-unix:/tmp/.X11-unix xopp-dev-arch
rm -rf "$xauth_path"
