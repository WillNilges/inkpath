FROM docker.io/debian:bookworm

RUN touch /root/.Xauthority

# Inkpath, Xournalpp, and debugging tools
RUN apt-get -y update && apt-get install -y make liblua5.4-dev build-essential \
  pkg-config libglib2.0-dev libpng-dev cmake libgtk-3-dev libpoppler-glib-dev \
  portaudio19-dev libsndfile-dev dvipng texlive libxml2-dev libzip-dev \
  librsvg2-dev gettext lua-lgi libgtksourceview-4-dev unzip git wget tmux gdb \
  vim x11-apps tree nautilus eog clang-format

# Build and install OpenCV into the container
COPY ./build-opencv.sh .
RUN bash /build-opencv.sh 16

# Build and install Xournal++
COPY ./build-xopp.sh .
RUN bash /build-xopp.sh 16

