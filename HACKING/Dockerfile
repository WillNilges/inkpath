FROM docker.io/debian

# Line 3: debugging tools
# Line 4: Inkpath
# Line 5: Xournalpp
# Line 8: OpenCV
RUN touch /root/.Xauthority && \
  apt-get update && \
  apt-get install -y tmux git gdb vim x11-apps && \ 
  apt-get install -y make liblua5.4-dev build-essential pkg-config libglib2.0-dev libpng-dev && \
  apt-get install -y cmake libgtk-3-dev libpoppler-glib-dev portaudio19-dev libsndfile-dev \
  dvipng texlive libxml2-dev liblua5.3-dev libzip-dev librsvg2-dev gettext lua-lgi \
  libgtksourceview-4-dev

