
## This Dockerfile should have everything you need (sans Xforwarding, TODO) to make a development container.


We got:
    1. Packages required for the container
    2. Packages to build AutoTrace
    3. Packages to build Xournalpp and Inkpath
    (Some of these overlap)

_Note: imagemagick, pstoedit are optional._
You'll need to `xauth add <SECRET>` to make Xforwarding work. Find it with `xauth list` on your host.

Run this with:

```
podman run -dit -e DISPLAY=$DISPLAY --network=host --cap-add=SYS_PTRACE -v /home/$USER/Code/inkpath_dev:/mnt/inkpath_dev -v /tmp/.X11-unix:/tmp/.X11-unix <image_id>:latest
```

(It might be a good idea to tag this as something)
