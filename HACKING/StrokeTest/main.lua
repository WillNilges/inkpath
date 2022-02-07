-- Register all Toolbar actions and intialize all UI stuff
function initUi()
  ref = app.registerUi({["menu"] = "Stroke Test (points)", ["callback"] = "drawStroke", ["accelerator"] = ""});
  print("Stroke Test registered\n");
end

-- Callback if the menu item is executed
function drawStroke()
    x = {
      [1] = 110.0,
      [2] = 120.0,
      [3] = 130.0,
      [4] = 140.0,
      [5] = 150.0,
      [6] = 160.0,
      [7] = 170.0,
      [8] = 170.0
    }
    y = {
      [1] = 100.0,
      [2] = 100.0,
      [3] = 100.0,
      [4] = 150.0,
      [5] = 160.0,
      [6] = 170.0,
      [7] = 170.0,
      [8] = 180.0,
    }
    pressure = {
      [1] = 1.5,
      [2] = 1.4,
      [3] = 1.4,
      [4] = 1.4,
      [5] = 1.4,
      [6] = 1.4,
      [7] = 1.4,
      [8] = 1.4,
    }
    xBatch = {{
      [1] = 110.0,
      [2] = 120.0,
      [3] = 130.0,
      [4] = 140.0,
      [5] = 150.0,
      [6] = 160.0,
      [7] = 170.0,
      [8] = 170.0
    },
    {
      [1] = 110.0,
      [2] = 120.0,
      [3] = 130.0,
      [4] = 140.0,
      [5] = 150.0,
      [6] = 160.0,
      [7] = 170.0,
      [8] = 170.0
    }}
    yBatch = {{
      [1] = 300.0,
      [2] = 300.0,
      [3] = 300.0,
      [4] = 350.0,
      [5] = 360.0,
      [6] = 370.0,
      [7] = 370.0,
      [8] = 380.0,
    },
    {
      [1] = 100.0,
      [2] = 90.0,
      [3] = 80.0,
      [4] = 70.0,
      [5] = 60.0,
      [6] = 50.0,
      [7] = 40.0,
      [8] = 30.0,
    }}
    pressureBatch = {{
      [1] = 1.5,
      [2] = 1.4,
      [3] = 1.4,
      [4] = 1.4,
      [5] = 1.4,
      [6] = 1.4,
      [7] = 1.4,
      [8] = 1.4,
    },
    {
      [1] = 1.5,
      [2] = 1.0,
      [3] = 1.0,
      [4] = 1.0,
      [5] = 1.0,
      [6] = 1.0,
      [7] = 1.0,
      [8] = 1.4,
    }}
    stroke_completeness = app.msgbox("Stroke type?", {[1] = "Stroke Batch w/ Group Undo", [2] = "Stroke Batch w/ Individual Undo"}) --, [3] = "With highlighter", [4] = "With eraser (broken)", [5] = "No Undo Action", [6] = "Batch"})
    if stroke_completeness == 1 then
        app.addStrokes({
            ["strokes"] = {
                {
                    ["x"] = x,
                    ["y"] = y,
                    ["pressure"] = pressure,
                    ["tool"] = "pen",
                    ["width"] = 3.8,
                    ["color"] = 0x4400f0,
                    ["fill"] = 0,
                    ["lineStyle"] = "solid",
                },
                {
                    ["x"] = xBatch[2],
                    ["y"] = yBatch[2],
                    ["pressure"] = pressureBatch[2],
                    ["tool"] = "pen",
                    ["width"] = 1.8,
                    ["color"] = 0xff0000,
                    ["fill"] = 0,
                    ["lineStyle"] = "solid",
                },
                {
                    ["x"] = xBatch[1],
                    ["y"] = yBatch[1],
                    ["pressure"] = pressureBatch[1],
                    ["tool"] = "pen",
                    ["width"] = 1.0,
                    ["color"] = 0x00ff00,
                    ["fill"] = 0,
                    ["lineStyle"] = "dashdot",
                },
            },
            ["allowUndoRedoAction"] = "grouped",
        })
    elseif stroke_completeness == 2 then
        app.addStrokes({
            ["strokes"] = {
                {
                    ["x"] = x,
                    ["y"] = y,
                    ["pressure"] = pressure,
                    ["tool"] = "pen",
                    ["width"] = 3.8,
                    ["color"] = 0x4400f0,
                    ["fill"] = 0,
                    ["lineStyle"] = "solid",
                },
                {
                    ["x"] = xBatch[2],
                    ["y"] = yBatch[2],
                    ["pressure"] = pressureBatch[2],
                    ["tool"] = "pen",
                    ["width"] = 1.8,
                    ["color"] = 0xff0000,
                    ["fill"] = 0,
                    ["lineStyle"] = "solid",
                },
                {
                    ["x"] = xBatch[1],
                    ["y"] = yBatch[1],
                    ["pressure"] = pressureBatch[1],
                    ["tool"] = "pen",
                    ["width"] = 1.0,
                    ["color"] = 0x00ff00,
                    ["fill"] = 0,
                    ["lineStyle"] = "dashdot",
                },
            },
            ["allowUndoRedoAction"] = "individual",
        })
--    elseif stroke_completeness == 2 then
--        app.addStrokes({
--            ["x"] = x,
--            ["y"] = y,
--            --["pressure"] = pressure,
--            ["tool"] = "pen",
--            ["width"] = 3.8,
--            ["color"] = 0xff0000,
--            ["fill"] = 0,
--            ["lineStyle"] = "solid"
--        })
--    elseif stroke_completeness == 3 then
--        app.addStroke({
--            ["x"] = x,
--            ["y"] = y,
--            ["tool"] = "highlighter",
--        })
--    elseif stroke_completeness == 4 then
--        app.addStroke({
--            ["x"] = x,
--            ["y"] = y,
--            ["tool"] = "eraser",
--        })
--    elseif stroke_completeness == 5 then
--        app.addStroke({
--            ["x"] = x,
--            ["y"] = y,
--            ["allowUndoRedoAction"] = false,
--        })
--    elseif stroke_completeness == 6 then
--        app.addBatchStrokes({
--            ["x"] = xBatch,
--            ["y"] = yBatch,
--        })
    end
    app.refreshPage()
    print("done")
end
