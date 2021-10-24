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
-- TODO: Place stroke on page
function drawStroke()
  -- result = app.msgbox("Test123", {[1] = "Yes", [2] = "No"});
  -- app.msgbox("Processing image; Please wait.", {})
  inkpath = require 'inkpath'
  local inspect = require 'inspect'
  path = app.getFilePath()
  -- strokes = inkpath.transcribe_image("/home/wilnil/inkpath/data/20211019_181644.jpg")
  --strokes = inkpath.transcribe_image("/home/wilnil/inkpath/data/cropped/input_fixed_01.jpg")
  strokes = inkpath.transcribe_image(path)
  print("Here are our strokes.")

  single_stroke = {}
  stroke_count = 0
  for key, value in pairs(strokes) do
      if value[1] == -1.0 and value[2] == -1.0 then
        app.drawStroke(single_stroke)
        single_stroke = {}
        stroke_count = stroke_count + 1;
      else
        table.insert(single_stroke, value[2]) -- Y coord
        table.insert(single_stroke, value[1]) -- X coord
      end
  end
  print("Stroke count is: ", stroke_count)
  app.refreshPage()
  print("done")
end
