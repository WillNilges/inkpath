-- This is an example Xournal++ Plugin - copy this to get started

-- Register all Toolbar actions and intialize all UI stuff
function initUi()
  ref = app.registerUi({["menu"] = "Transcribe Image", ["callback"] = "drawStroke", ["accelerator"] = "<Control><Alt>t"});
  print("ImageTranscription registered\n");
end

-- Callback if the menu item is executed
function drawStroke()
  inkpath = require 'inkpath'
  path = app.getFilePath({'*.ppm', '*.png', '*.pbm', '*.pnm', '*.bmp', '*.tga', '*.yuv', '*.pgm', '*.gf'}) -- Autotrace 0.40.0 supports ppm, png, pbm, pnm, bmp, tga, yuv, pgm, gf
  strokes = inkpath.transcribe_image(path)
  print("Strokes retrieved.")

  single_stroke = {} -- Each stroke will be composed of a number of splines.
  for key, value in pairs(strokes) do
      if value[1] == -1.0 and value[2] == -1.0 then -- If we get a delimiting pair, submit our stroke for processing.
        app.drawSplineStroke(single_stroke)
        single_stroke = {}
      else
        table.insert(single_stroke, value[1]) -- Y coord
        table.insert(single_stroke, value[2]) -- X coord
      end
  end
  app.refreshPage()
  print("done")
end
