#!/bin/bash

set -e

if [ "$(pwd)" == "$HOME" ]; then
    echo "DON'T FUCKING RUN THIS SCRIPT FROM YOUR FUCKING HOMEDIR"
    echo "Place it in a directory LITERALLY ANYWHERE ELSE where it can't hurt you :)"
    exit 1
fi

CODE_PATH=/home/$USER/Code/xopp-dev
uname=$USER # Just trust me on this one.

xauth_path=/tmp/xopp-dev-xauth

mkdir -p "$xauth_path"
cp $HOME/.Xauthority "$xauth_path"
chmod g+rwx "$xauth_path"/.Xauthority
podman unshare chown 1000:$uname "$xauth_path"/.Xauthority
#podman unshare chown 1000:$uname -R "$CODE_PATH"
#sudo chmod -R +070 "$CODE_PATH"
#sudo chown -R :$uname "$CODE_PATH"
#sudo chown -R :$uname "$xauth_path"/.Xauthority

podman run --name=xopp-dev -it -e DISPLAY=$DISPLAY --network=host --cap-add=SYS_PTRACE -v "$xauth_path"/.Xauthority:/root/.Xauthority:rw -v "$CODE_PATH":/mnt/xopp_dev -v /tmp/.X11-unix:/tmp/.X11-unix xopp-dev
rm -rf "$xauth_path"
