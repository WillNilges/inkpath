user_uid=$UID
user_gid=$(id -rg)
user_name=$USER

CODE_PATH="$HOME"/Code/xopp-dev

cd $CODE_PATH/inkpath/HACKING/

podman build . --tag xopp-dev-wayland --file ./wayland/Dockerfile

podman run \
    --rm \
    --security-opt label=disable \
    -e XDG_RUNTIME_DIR=/tmp \
    -e "WAYLAND_DISPLAY=$WAYLAND_DISPLAY" \
    -v "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY:ro" \
    -v "/run/user/$user_uid/wayland-0:/tmp/wayland-0:ro" \
    -v "$CODE_PATH":/xopp-dev:Z \
    -it xopp-dev-wayland
    #bash -euo pipefail -c "
    #    dnf install -y \
    #        coreutils bash bash-completion less nano sudo tree util-linux gedit
    #"
        #groupadd -g $user_gid $user_name
        #useradd -u $user_uid -g $user_gid -ms /usr/bin/bash $user_name
        #usermod -aG wheel $user_name
        #echo '$user_name ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/nopasswd
        #cd /home/$user_name
        #exec sudo -iu $user_name --preserve-env=XDG_RUNTIME_DIR,WAYLAND_DISPLAY
    #"
