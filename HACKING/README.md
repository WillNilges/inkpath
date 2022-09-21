# Thanks for contributing :)

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

## Debugging Lua

You might find the following to be useful
- https://github.com/kikito/inspect.lua
- https://gist.github.com/lunixbochs/5b0bb27861a396ab7a86
