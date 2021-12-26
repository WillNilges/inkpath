# Inkpath

<div id="badges">
<img src="https://forthebadge.com/images/badges/made-with-c.svg" alt="C badge" height="30px"/>
<img src="https://forthebadge.com/images/badges/powered-by-energy-drinks.svg" alt="Coffee badge" height="30px"/>
<img src="https://forthebadge.com/images/badges/60-percent-of-the-time-works-every-time.svg" alt="60 percent badge" height="30px"/>
</div>

If you're anything like me, you're a ~~huge nerd~~ engineering major who enjoys
working out problems both on whiteboards and digitally. You might also
have six million photos of your whiteboard work trapped on your phone
or in your Google Drive with no good way to easily group them in with
your other notes.

<p align="center">
<!-- ![Makerfaire 2021 Card](https://user-images.githubusercontent.com/42927786/147401085-94773933-e4a3-4039-97e6-91cf2ea1ee6c.png) -->
  <img src="https://user-images.githubusercontent.com/42927786/147401085-94773933-e4a3-4039-97e6-91cf2ea1ee6c.png" alt="Makerfaire 2021 Card" width="400px" style="display: block; margin: 0 auto"/>

</p>
Inktrace is a project designed to crunch those whiteboard photos into easily
editable [Xournalpp](https://github.com/xournalpp) note files so that you can
drop your whiteboard scrawlings directly into your lecture notes. Convenient!

This uses [autotrace](https://github.com/autotrace/autotrace) to translate whiteboard
markings into splines. From there, it applies a bezier curve to approximate strokes as a series of points, then passes them to the Xournal++ Lua API for rendering.

## Installation and Usage

Compile and install [autotrace](https://github.com/autotrace/autotrace).

Debian:
```
# (From https://askubuntu.com/questions/1124651/how-to-install-autotrace-in-ubuntu-18-04)
# Download Autotrace Dependencies
apt-get install -y build-essential pkg-config autoconf intltool autopoint libtool libglib2.0-dev build-essential libmagickcore-dev libpstoedit-dev imagemagick pstoedit
# Download and Build Autotrace
git clone https://github.com/autotrace/autotrace.git
cd autotrace
./autogen.sh
#put everything into /usr/{bin,lib,share,include}
LD_LIBRARY_PATH=/usr/local/lib ./configure --prefix=/usr
make
sudo make install
```

<!--TODO: Change build instructions for xournalpp to point at my fork-->
Compile and install [xournalpp](https://github.com/xournalpp/xournalpp)
```
# From (https://github.com/xournalpp/xournalpp/blob/master/readme/LinuxBuild.md)
# Download Xournalpp dependencies
apt-get install -y cmake libgtk-3-dev libpoppler-glib-dev portaudio19-dev libsndfile-dev dvipng texlive libxml2-dev liblua5.3-dev libzip-dev librsvg2-dev gettext lua-lgi
# Download and Build Xournalpp
git clone http://github.com/xournalpp/xournalpp.git
cd xournalpp
mkdir build
cd build
cmake ..
cmake --build .
```

Compile and install Inkpath
```
# Download and build Inkpath (Deps included in Xournalpp)
git clone https://github.com/willnilges/inkpath.git
cd inkpath
make lua-plugin

# Copy resulting plugin files to plugin directory

```

### Plugin (recommended)

Run `make lua-plugin`, it will compile inkpath and place `inkpath.so` in the ImageTranscription directory. Copy that directory to your Xournalpp plugins folder, and Inkpath will be installed. You can then use it from the 'Plugins' menu.

### Command line

You can also run `make` to compile a CLI utilility that you can pass a source image and an output file path. This is useful for testing, or just starting a document. It will take that Bezier curve data and transcribe it to xouranalpp file syntax.

(See Makefile for dependencies)

<img src="https://forthebadge.com/images/badges/works-on-my-machine.svg" alt="C badge" height="30px"/>
