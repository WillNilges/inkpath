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
editable [Xournalpp](https://github.com/xournalpp) note files so that you can
drop your whiteboard scrawlings directly into your lecture notes. Convenient!

This uses [autotrace](https://github.com/autotrace/autotrace) to translate
whiteboard markings into splines. Those splines are then passed directly to the
Xournal++ API and rasterized, before being placed onto the working layer.

## Installation and Usage

When you run `make lua-plugin`, it will compile inkpath and place `inkpath.so` in the ImageTranscription directory.`make install` will copy that directory to your Xournalpp plugins folder, and Inkpath will be installed. You can then use it from the 'Plugins' menu from within Xournalpp.

_Prerequisite: You must have xournalpp installed via your package manager of choice_

1. Download dependencies

**Debian:**
```
apt-get install make liblua5.4-dev build-essential pkg-config libglib2.0-dev libpng-dev
```

**Fedora:**
```
dnf install make lua-devel gcc pkg-config glib2-devel libpng-devel
```

**Arch:**
```
pacman -S base-devel pkg-config lua libpng
```

2. Compile and install Inkpath
```
# Download and build Inkpath (Deps included in Xournalpp)
git clone https://github.com/willnilges/inkpath.git
cd inkpath
make install
```

<img src="https://forthebadge.com/images/badges/works-on-my-machine.svg" alt="C badge" height="30px"/>
