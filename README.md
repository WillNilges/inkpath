# Inkpath

If you're anything like me, you're a ~~huge nerd~~ engineering major who enjoys
working out problems both on whiteboards and digitally. You might also
have six million photos of your whiteboard work trapped on your phone
or in your Google Drive with no good way to easily group them in with
your other notes. This is the problem.

![image](https://user-images.githubusercontent.com/42927786/109400114-cff24500-7914-11eb-8af2-292bfe65543e.png)

![image](https://user-images.githubusercontent.com/42927786/120553464-85c2a900-c3c6-11eb-84b9-33e9931e8190.png)

Inktrace is a project designed to crunch those whiteboard photos into easily
editable [Xournalpp](https://github.com/xournalpp) note files so that you can
drop your whiteboard scrawlings directly into your lecture notes. Convenient!

This uses [autotrace](https://github.com/autotrace/autotrace) to translate whiteboard
markings into portable SVG files. From there, it uses [nanosvg](https://github.com/memononen/nanosvg) to convert those svgs into point data that is transferred to
a .xopp file.

## Compiling

``
make
``

Yup.

(See Makefile for dependencies)
