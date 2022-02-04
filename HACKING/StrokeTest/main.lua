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
    stroke_completeness = app.msgbox("Stroke type?", {[1] = "No data", [2] = "With data", [3] = "With highlighter", [4] = "With eraser (broken)", [5] = "No Undo Action"})
    if stroke_completeness == 1 then
        app.addStroke({
            ["x"] = x,
            ["y"] = y,
        })
    elseif stroke_completeness == 2 then
        app.addStroke({
            ["x"] = x,
            ["y"] = y,
            --["pressure"] = pressure,
            ["tool"] = "pen",
            ["width"] = 3.8,
            ["color"] = 0xff0000,
            ["fill"] = 0,
            ["lineStyle"] = "solid"
        })
    elseif stroke_completeness == 3 then
        app.addStroke({
            ["x"] = x,
            ["y"] = y,
            ["tool"] = "highlighter",
        })
    elseif stroke_completeness == 4 then
        app.addStroke({
            ["x"] = x,
            ["y"] = y,
            ["tool"] = "eraser",
        })
    elseif stroke_completeness == 5 then
        app.addStroke({
            ["x"] = x,
            ["y"] = y,
            ["allowUndoRedoAction"] = false,
        })
    end
    app.refreshPage()
    print("done")
end
