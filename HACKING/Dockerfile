FROM docker.io/debian

RUN touch /root/.Xauthority && \
apt-get update && \
apt-get install -y git gdb vim && \
apt-get install -y build-essential pkg-config autoconf intltool autopoint libtool libglib2.0-dev build-essential libmagickcore-dev libpstoedit-dev imagemagick pstoedit && \
apt-get install -y cmake libgtk-3-dev libpoppler-glib-dev portaudio19-dev libsndfile-dev \
dvipng texlive libxml2-dev liblua5.3-dev libzip-dev librsvg2-dev gettext lua-lgi

RUN cd && git clone https://github.com/yy502/autotrace && \
cd autotrace && ./autogen.sh && LD_LIBRARY_PATH=/usr/local/lib ./configure --prefix=/usr && make && make install

# Mount these yourself.
#git clone https://github.com/xournalpp/xournalpp && \
#git clone https://gitub.com/willnilges/inkpath && \
#RUN cd && git clone https://github.com/willnilges/xournalpp && \
#cd xournalpp && mkdir build && cd build && cmake .. && cmake --build .
#
#RUN cd && git clone https://github.com/willnilges/inkpath && \
#cd inkpath && make lua-plugin && cp -r ImageTranscription ../xournalpp/plugins && cp ImageTranscription/inkpath.so ../xournalpp/build/src
