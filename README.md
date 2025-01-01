# Inkpath

<div id="badges">
<img src="https://forthebadge.com/images/badges/made-with-c.svg" alt="C badge" height="30px"/>
<img src="https://forthebadge.com/images/badges/powered-by-energy-drinks.svg" alt="Energy drink badge" height="30px"/>
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

Inkpath is a project designed to crunch those whiteboard photos into easily
editable [Xournal++](https://github.com/xournalpp/xournalpp) note files so that you can
drop your whiteboard scrawlings directly into your lecture notes. Convenient!

The project consists of a lua script and a shared object library written in C++
statically linked with some OpenCV utils. The project now uses OpenCV in place
of Autotrace to perform an Otsu Threshold. The thresholded image is then inverted,
and skeletonized, producing the centerline of each individual object in the image
(which at this point should be just markings). The resulting image is then scanned
for contours. These contours are pushed onto the lua stack, and passed to the 
Xournal++ API. Unlike the previous implementation, this operates purely on rasters.

## Installation and Usage

Requires Xournalpp >=1.2.0

Inkpath is packaged as a statically-linked `.so` file coupled with a Lua script,
so you should be able to download the release from the [releases page](https://github.com/WillNilges/inkpath/releases)
and have it Just Work™.

_As of 2022-09-24, the API changes in Xournalpp are merged, but have not made
it into the package managers. You will probably need to [build xournalpp from source](https://github.com/xournalpp/xournalpp/blob/master/readme/LinuxBuild.md)
in order to get them._

_Inkpath is coming to a package manager near you soon™!_

### Installation on Windows

1. Download the latest release of [ImageTranscription-win.zip](https://github.com/WillNilges/inkpath/releases/)

2. Unzip the archive and copy ImageTranscription/ to the plugin path:

`C:\Program Files\Xournal++\share\xournalpp\plugins\ImageTranscription`

3. Run install.ps1 to update the user Path

4. Run Xournal++ and enjoy

#### To Uninstall

1. Delete `C:\Program Files\Xournal++\share\xournalpp\plugins\ImageTranscription`

2. Remove environment variable

- Type `Win+R` and type `sysdm.cpl`
- Go to Advanced > Enviornment Variables > Path > Edit
- Delete ImageTranscription entry
- Press OK until all dialogs are closed

## Compiling

### Arch

```BASH
# Install dependencies
pacman -S \
cmake gtk3 base-devel libxml2 portaudio libsndfile \
poppler-glib texlive-bin texlive-pictures gettext libzip lua54 lua54-lgi \
gtksourceview4 wget unzip git tmux

# Build openCV
./HACKING/build-opencv.sh
```

### Debian

```BASH
# Install dependencies
apt -y install \
make liblua5.4-dev build-essential pkg-config libglib2.0-dev libpng-dev \
cmake libgtk-3-dev libpoppler-glib-dev portaudio19-dev libsndfile-dev \
dvipng texlive libxml2-dev libzip-dev librsvg2-dev gettext lua-lgi \
libgtksourceview-4-dev git gdb x11-apps wget unzip
./HACKING/build-opencv.sh $NUMBER_OF_CORES_YOU_HAVE # You'll need sudo to install this
./HACKING/build-xopp.sh $NUMBER_OF_CORES_YOU_HAVE # You'll probably need sudo to install this too
make install
```

<img src="https://forthebadge.com/images/badges/works-on-my-machine.svg" alt="C badge" height="30px"/>

### Windows

1. Install MSYS2

2. Open MSYS2 MINGW64 shell

3. Install dependencies

```
pacman -S \
mingw-w64-x86_64-cmake mingw-w64-x86_64-gtk3 base-devel libxml2 mingw-w64-x86_64-portaudio mingw-w64-x86_64-libsndfile mingw-w64-x86_64-poppler mingw-w64-x86_64-libzip mingw-w64-x86_64-lua53 mingw-w64-x86_64-lua53-lgi mingw-w64-x86_64-gtksourceview4 mingw-w64-x86_64-opencv wget unzip git tmux
```

4. Launch MSYS2 MINGW64 shell as Administrator, and use `cmake` to build and
run post-build script to gather dependencies and assemble the plugin

```
mkdir build
cd build
cmake .. # Use ${MINGW_PREFIX}/bin/cmake if normal cmake doesn't work
mingw32-make.exe install
```

6. Run scripts/win_configure_path.ps1 to update the user Path

7. Run Xournal++ and enjoy