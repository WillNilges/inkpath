
## This Dockerfile should have everything you need (sans Xforwarding, TODO) to make a development container.


We got:
    1. Packages required for the container
    2. Packages to build AutoTrace
    3. Packages to build Xournalpp and Inkpath
    (Some of these overlap)

_Note: imagemagick, pstoedit are optional._
You'll need to `xauth add <SECRET>` to make Xforwarding work. Find it with `xauth list` on your host.

### Usage

First, open the Dockerfile and ajust the make steps to the capabilities of your CPU (Currently there's a few `-j16`'s in there.)

Then, build and tag the image:

`
podman build . --tag inkpath-dev
`

Run using this command:

```
podman run -dit -e DISPLAY=$DISPLAY --network=host --cap-add=SYS_PTRACE -v /home/$USER/Code/inkpath_dev:/mnt/inkpath_dev -v /tmp/.X11-unix:/tmp/.X11-unix inkpath-dev:latest
```

Then, get into the pod:

`
podman exec -it $(podman ps --format '{{.Image}} {{.Names}}' | grep inkpath-dev | awk '{print $2}') bash
`

Find your magic cookie outside the container

`
xauth list
`

Add it:

`
xauth add <magic_cookie>
`

Now you're ready to go.
