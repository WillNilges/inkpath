FROM docker.io/debian

RUN touch /root/.Xauthority
RUN apt-get update

# Line 10: Inkpath
# Line 11: Xournalpp
# Line 13: debugging tools
RUN apt-get install -y make liblua5.3-dev build-essential pkg-config libglib2.0-dev libpng-dev && \
  apt-get install -y cmake libgtk-3-dev libpoppler-glib-dev portaudio19-dev libsndfile-dev \
  dvipng texlive libxml2-dev libzip-dev librsvg2-dev gettext lua-lgi \
  libgtksourceview-4-dev \
  unzip git wget

# Build and install OpenCV into the container
COPY ./build-opencv.sh .
RUN bash /build-opencv.sh 48

# Build and install Xournal++
COPY ./build-xopp.sh .
RUN bash /build-xopp.sh 48

# Install dev tools
RUN apt-get install -y tmux gdb vim x11-apps tree nautilus eog 
