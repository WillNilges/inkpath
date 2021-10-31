-- This is an example Xournal++ Plugin - copy this to get started

-- var_dump = require "var_dump"

-- Register all Toolbar actions and intialize all UI stuff
function initUi()
  print("Hello from ImageTranscription: Plugin initUi called\n");

  ref = app.registerUi({["menu"] = "Transcribe Image", ["callback"] = "drawStroke", ["accelerator"] = "<Control><Shift>t"});
  -- print("Menu reference:");
  -- var_dump(ref);

  print("ImageTranscription registered\n");
end

-- Callback if the menu item is executed
function drawStroke()
  -- app.msgbox("Processing image; Please wait.", {})
  inkpath = require 'inkpath'
  local inspect = require 'inspect'
  path = app.getFilePath()
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
