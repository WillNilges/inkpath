# Inkpath

If you're anything like me, you're a ~~huge nerd~~ engineering major who enjoys
working out problems both on whiteboards and digitally. You might also
have six million photos of your whiteboard work trapped on your phone
or in your Google Drive with no good way to easily group them in with
your other notes.

![image](https://user-images.githubusercontent.com/42927786/109400114-cff24500-7914-11eb-8af2-292bfe65543e.png)

![image](https://user-images.githubusercontent.com/42927786/120553464-85c2a900-c3c6-11eb-84b9-33e9931e8190.png)

Inktrace is a project designed to crunch those whiteboard photos into easily
editable [Xournalpp](https://github.com/xournalpp) note files so that you can
drop your whiteboard scrawlings directly into your lecture notes. Convenient!

This uses [autotrace](https://github.com/autotrace/autotrace) to translate whiteboard
markings into splines. From there, it applies a bezier curve to approximate strokes as a series of points, then passes them to the Xournal++ Lua API for rendering.

## Usage

### Plugin (recommended)

Run `make lua-module`, it will compile inkpath and place `inkpath.so` in the ImageTranscription directory. Copy that directory to your Xournalpp plugins folder, and Inkpath will be installed. You can then use it from the 'Plugins' menu.

### Command line

You can also run `make` to compile a CLI utilility that you can pass a source image and an output file path. This is useful for testing, or just starting a document. It will take that Bezier curve data and transcribe it to xouranalpp file syntax.

(See Makefile for dependencies)
