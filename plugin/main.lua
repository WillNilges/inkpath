-- Import functions from external files
getLibPath = require("getLibPath")

-- Register all Toolbar actions and intialize all UI stuff
function initUi()
  ref = app.registerUi({
    ["menu"] = "Transcribe Image",
    ["callback"] = "drawStroke",
    ["accelerator"] = "<Control><Alt>t",
    ["toolbarId"] = "TRANSCRIBE_IMAGE",
    ["iconName"] = "xopp-tool-transcribe-image"
  });
  print("ImageTranscription registered\n");
end

-- Callback if the menu item is executed
function drawStroke()
    print("Inkpath Activated.")
    -- Inkpath is installed differently depending on Windows vs Unix platforms
    local libPath = getLibPath("libinkpath")
    local load_inkpath = assert(package.loadlib(libPath, "luaopen_loadInkpath"))
    load_inkpath()
    local path = app.getFilePath({'*.jpg', '*.png', '*.bmp'})
    if path == nil then
        -- Exit if no file was selected
        return
    end
    -- Floating point value to scale stroke data coordinates. 0.1x is usually
    -- necessary to cleanly map strokes to the document
    local scaling_factor = 0.125
    local obj = Inkpath(path, 1)
    local contourCt = obj:getStrokeCount()
    print("Got ", contourCt, " strokes.")

    -- TODO: This could be much, much faster.
    local strokes = {}  -- Collect all strokes before calling `app.addStrokes`
    for i = 0,contourCt-1,1 do
        local pointCt = obj:getStrokeLength(i)
        -- We have no use for strokes that are less than two points---we can't do
        -- anything with them.
        if pointCt >= 3 then
            local x_points, y_points = obj:getStroke(i, scaling_factor)
            table.insert(strokes, {
                ["x"] = x_points,
                ["y"] = y_points,
                ["tool"] = "pen", -- Default to pen to silence warnings. TODO: Delete in the future.
            })
        end
    end
    if #strokes > 0 then
        app.addStrokes({
            ["strokes"] = strokes,
            ["allowUndoRedoAction"] = "grouped",
        })
    end
    app.refreshPage() -- Refreshes the page after inserting the strokes.
    print("Image Transcription Complete. Exiting Inkpath.")
end
