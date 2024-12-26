-- Register all Toolbar actions and intialize all UI stuff
function initUi()
  ref = app.registerUi({["menu"] = "Transcribe Image", ["callback"] = "drawStroke", ["accelerator"] = "<Control><Alt>t"});
  print("ImageTranscription registered\n");
end

-- Callback if the menu item is executed
function drawStroke()
    print("Inkpath Activated. Transcribing image....")
    inkpath = assert(package.loadlib("/usr/share/xournalpp/plugins/ImageTranscription/ipcvobj.so", "luaopen_ipcvobj"))
    inkpath()
    path = app.getFilePath({'*.jpg', '*.png', '*.bmp'}) -- The current version of Autotrace I'm using only supports PNGs.
    --image_scale = app.msgbox("Select tracing scale", {[1] = "Small", [2] = "Medium", [3] = "Large"}) -- TODO: implement this again.
    image_scale = 1
    scaling_factor = 10.0 -- how much do you want to divide your content by? must be float!
    local obj = IPCVObj(path, 1)
    print("Strokes retrieved.")
    contourCt = obj:getLength()
    print("Got ", contourCt, " strokes.")
    -- TODO: This could be much, MUCH faster.
    for i = 0,contourCt-1,1 do
        pointCt = obj:getContourLength(i)
        -- We have no use for strokes that are less than two points---we can't do
        -- anything with them.
        if pointCt >= 3 then
            x_points, y_points = obj:getContour(i, scaling_factor)
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
