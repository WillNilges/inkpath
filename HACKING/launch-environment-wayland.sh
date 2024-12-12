user_uid=$UID
user_gid=$(id -rg)
user_name=$USER

CODE_PATH="$HOME"/Code/xopp-dev

cd $CODE_PATH/inkpath/HACKING/

podman run \
    --rm \
    --security-opt label=disable \
    -e XDG_RUNTIME_DIR=/tmp \
    -e "WAYLAND_DISPLAY=$WAYLAND_DISPLAY" \
    -v "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY:ro" \
    -v "/run/user/$user_uid/wayland-0:/tmp/wayland-0:ro" \
    -v "$CODE_PATH":/xopp-dev:Z \
    -it xopp-dev
