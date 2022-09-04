
This Dockerfile should have everything you need (sans Xforwarding, TODO) to make a development container.

TODO: Not sure if I want these as separate layers. It makes sense to me at 01:36, though.

We got:
1. Packages required for the container
2. Packages to build AutoTrace
3. Packages to build Xournalpp and Inkpath
(Some of these overlap)

Notes: imagemagick, pstoedit are optional.

You'll need to `xauth add <SECRET>` to make Xforwarding work. Find it with `xauth list` on your host.

Run this with:

```
podman build . --tag xopp-dev
podman run --name=xopp-dev -dit -e DISPLAY=$DISPLAY --network=host --cap-add=SYS_PTRACE -v /home/$USER/Code/inkpath_dev:/mnt/inkpath_dev -v /tmp/.X11-unix:/tmp/.X11-unix xopp-dev
```

Or use the provided script.

## Debugging the Lua

You might find the following to be useful
- https://github.com/kikito/inspect.lua
- https://gist.github.com/lunixbochs/5b0bb27861a396ab7a86
