FROM docker.io/debian

# This Dockerfile should have everything you need (sans Xforwarding, TODO) to make a development container.

# TODO: Not sure if I want these as separate layers. It makes sense to me at 01:36, though.

# We got:
# 1. Packages required for the container
# 2. Packages to build AutoTrace
# 3. Packages to build Xournalpp and Inkpath
# (Some of these overlap)

# Notes: imagemagick, pstoedit are optional.

RUN touch /root/.Xauthority && \
apt-get update && \
apt-get install -y git gdb vim && \
apt-get install -y build-essential pkg-config autoconf intltool autopoint libtool libglib2.0-dev build-essential libmagickcore-dev libpstoedit-dev imagemagick pstoedit && \
apt-get install -y cmake libgtk-3-dev libpoppler-glib-dev portaudio19-dev libsndfile-dev \
dvipng texlive libxml2-dev liblua5.3-dev libzip-dev librsvg2-dev gettext lua-lgi

RUN cd && git clone https://github.com/autotrace/autotrace && \
git clone https://github.com/xournalpp/xournalpp && \
git clone https://gitub.com/willnilges/inkpath && \
cd autotrace && ./autogen.sh && LD_LIBRARY_PATH=/usr/local/lib ./configure --prefix=/usr && make && make install && \

# This is my fork
RUN cd && git clone https://github.com/willnilges/xournalpp && \
cd xournalpp && mkdir build && cd build && cmake .. && cmake --build .

RUN cd && git clone https://github.com/willnilges/inkpath && \
cd inkpath && make lua-plugin && cp -r ImageTranscription ../xournalpp/plugins && cp ImageTranscription/inkpath.so ../xournalpp/build/src

# You'll need to `xauth add <SECRET>` to make Xforwarding work. Find it with `xauth list` on your host.

# Run this with:
# podman run -dit -e DISPLAY=$DISPLAY --network=host --cap-add=SYS_PTRACE -v /home/$USER/Code/inkpath_dev:/mnt/inkpath_dev -v /tmp/.X11-unix:/tmp/.X11-unix <image_id>:latest

# (It might be a good idea to tag this as something)
