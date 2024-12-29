-- Register all Toolbar actions and intialize all UI stuff
function initUi()
  ref = app.registerUi({["menu"] = "Transcribe Image", ["callback"] = "drawStroke", ["accelerator"] = "<Control><Alt>t"});
  print("ImageTranscription registered\n");
end

-- Callback if the menu item is executed
function drawStroke()
    print("Inkpath Activated. Transcribing image....")
    local inkpath = assert(package.loadlib("/usr/share/xournalpp/plugins/ImageTranscription/ipcvobj.so", "luaopen_ipcvobj"))
    inkpath()
    local path = app.getFilePath({'*.jpg', '*.png', '*.bmp'})
    -- Floating point value to scale stroke data coordinates. 0.1x is usually
    -- necessary to cleanly map strokes to the document
    local scaling_factor = 0.1
    local obj = IPCVObj(path, 1)
    print("Strokes retrieved.")
    local contourCt = obj:getLength()
    print("Got ", contourCt, " strokes.")

    -- TODO: This could be much, MUCH faster.
    for i = 0,contourCt-1,1 do
        local pointCt = obj:getContourLength(i)
        -- We have no use for strokes that are less than two points---we can't do
        -- anything with them.
        if pointCt >= 3 then
            local x_points, y_points = obj:getContour(i, scaling_factor)
            app.addStrokes({
                ["strokes"] = {
                    {
                        ["x"] = x_points,
                        ["y"] = y_points,
                        ["tool"] = "pen", -- Default to pen to silence warnings. TODO: Delete in the future.
                    },
                },
                ["allowUndoRedoAction"] = "grouped",
            })
        end
    end
    app.refreshPage() -- Refreshes the page after inserting the strokes.
    print("Image Transcription Complete. Exiting Inkpath.")
end
