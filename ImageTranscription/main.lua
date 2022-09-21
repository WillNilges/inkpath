-- Register all Toolbar actions and intialize all UI stuff
function initUi()
  ref = app.registerUi({["menu"] = "Transcribe Image", ["callback"] = "drawStroke", ["accelerator"] = "<Control><Alt>t"});
  print("ImageTranscription registered\n");
end

-- Callback if the menu item is executed
function drawStroke()
    print("Inkpath Activated. Transcribing image....")
    inkpath = require "ipcvobj"
    inspect = require "inspect"
    -- path = app.getFilePath({'*.ppm', '*.png', '*.pbm', '*.pnm', '*.bmp', '*.tga', '*.yuv', '*.pgm', '*.gf'}) -- Autotrace 0.40.0 supports ppm, png, pbm, pnm, bmp, tga, yuv, pgm, gf
    path = app.getFilePath({'*.jpg', '*.png', '*.bmp'}) -- The current version of Autotrace I'm using only supports PNGs.
    --image_scale = app.msgbox("Select tracing scale", {[1] = "Small", [2] = "Medium", [3] = "Large"}) -- TODO: implement this again.
    image_scale = 1
    scaling_factor = 10.0 -- THIS IS A NEW THING! HOW MUCH DO YOU WANT TO DIVIDE YOUR SHIT BY!? MUST BE FLOAT!
    local obj = IPCVObj(path, 1)
    print("Strokes retrieved.")

    contourCt = obj:getLength()
    print("Got ", contourCt, " strokes.")
    --strokes = {}
    for i = 0,contourCt-1,1 do
        pointCt = obj:getContourLength(i)
        x_points, y_points = obj:getContour(i, scaling_factor)
        app.addStrokes({
            ["strokes"] = {
                {
                    ["x"] = x_points,
                    ["y"] = y_points,
                },
            },
            ["allowUndoRedoAction"] = "grouped",
        })
    end

    print(inspect(strokes))

    -- When we've assembled our table of strokes, call the addStrokes function
    -- Not going to pass any options since I want to use the current tool options.
    --app.addStrokes({
    --    ["strokes"] = strokes,
    --    ["allowUndoRedoAction"] = "grouped",
    --})
end
