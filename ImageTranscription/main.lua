inspect = require 'inspect'
-- Register all Toolbar actions and intialize all UI stuff
function initUi()
  ref = app.registerUi({["menu"] = "Transcribe Image", ["callback"] = "drawStroke", ["accelerator"] = "<Control><Alt>t"});
  print("ImageTranscription registered\n");
end

-- Callback if the menu item is executed
function drawStroke()
    inkpath = require "inkpath"
    -- path = app.getFilePath({'*.ppm', '*.png', '*.pbm', '*.pnm', '*.bmp', '*.tga', '*.yuv', '*.pgm', '*.gf'}) -- Autotrace 0.40.0 supports ppm, png, pbm, pnm, bmp, tga, yuv, pgm, gf
    path = app.getFilePath({'*.png', '*.bmp'}) -- The current version of Autotrace I'm using only supports PNGs.
    image_scale = app.msgbox("Select tracing scale", {[1] = "Small", [2] = "Medium", [3] = "Large"})
    output = inkpath.cv_transcribe_image(path, image_scale)
    print("Strokes retrieved.")
    strokes = {}
    single_stroke = {}
    
    print(inspect(output))
    print("Output Inspected!")
end

drawStroke()
--    for key, value in pairs(output) do
--        if value[1] == -1.0 and value[2] == -1.0 then -- If we get a delimiting pair, add our current stroke to the stroke table.
--            table.insert(strokes, {
--                ["coordinates"] = single_stroke,
--            });
--            single_stroke = {}
--        else
--            table.insert(single_stroke, value[1]) -- Y coord
--            table.insert(single_stroke, value[2]) -- X coord
--        end
--    end
--    -- When we've assembled our table of strokes, call the addSplines function
--    -- Not going to pass any options since I want to use the current tool options.
--    app.addSplines({
--        ["splines"] = strokes,
--        ["allowUndoRedoAction"] = "grouped",
--    })
--    app.refreshPage()
--    print("done")
--end
