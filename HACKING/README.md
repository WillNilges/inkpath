# Thanks for contributing :)

You'll need to statically compile and install OpenCV (see `HACKING/build-opencv.sh`),
then install Xournalpp using your package manager of choice (or compile from source).

When you run `make lua-plugin`, it will compile inkpath and place `inkpath.so`
in the ImageTranscription directory.`make install` will copy that directory to
your Xournalpp plugins folder, and Inkpath will be installed. You can then use
it from the 'Plugins' menu from within Xournalpp.

## Setting up the development environment

This guide will walk you through everything you need to work on this project, including building OpenCV and Xournalpp.

(You should have all relevant development files (including this repo) located at `~/Code/xopp-dev`. This can be configured in `launch-environment.sh` by the `$CODE_PATH` variable)

- First, build the Dockerfile. This will download all the packages you need to compile inkpath and friends. OpenCV is the backend that now powers this project, and you might want Xournalpp as source code so you can debug more easily.
```
./HACKING/build-environment.sh
```

- Next, launch it. This container will do Xforwarding for you so that you can run Xournalpp on your desktop and do development. All subsequent build stuff should happen in here.
```
./HACKING/launch-environment.sh
```

- If you did it right, you should have an `xopp-dev` directory in your container.
```
cd /xopp-dev/inkpath/
make dev-install
```

## Using Arch
There's also an arch-based container, if you're into that sort of thing.
It is more geared towards testing the installation process, but can be
used for development as well.

- `cd HACKING/arch-test/`
- `./build-environment.sh`

To run the container:

- `./launch-environment.sh`

## Debugging Lua

You might find the following to be useful
- https://github.com/kikito/inspect.lua
- https://gist.github.com/lunixbochs/5b0bb27861a396ab7a86
