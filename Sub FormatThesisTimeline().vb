Sub FormatThesisTimeline()
    ' Thesis Timeline Formatter - Creates professional Gantt chart matching HTML design
    ' This macro will format your thesis timeline with color coding and professional styling
    
    Dim ws As Worksheet
    Dim lastRow As Long
    Dim lastCol As Long
    Dim cell As Range
    Dim i As Long, j As Long
    
    ' Set the active worksheet
    Set ws = ActiveSheet
    
    ' Find the last row and column with data
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row
    lastCol = 25 ' 24 weeks + 1 task column
    
    ' === STEP 1: Set up basic formatting ===
    
    ' Set font for entire sheet
    With ws.Cells.Font
        .Name = "Arial"
        .Size = 11
    End With
    
    ' Set column widths
    ws.Columns("A").ColumnWidth = 35 ' Task column
    For i = 2 To lastCol
        ws.Columns(i).ColumnWidth = 4 ' Week columns
    Next i
    
    ' === STEP 2: Format header row ===
    
    ' Main header formatting
    With ws.Range(ws.Cells(1, 2), ws.Cells(1, lastCol))
        .Interior.Color = RGB(68, 114, 196) ' #4472C4
        .Font.Color = RGB(255, 255, 255) ' White
        .Font.Bold = True
        .HorizontalAlignment = xlCenter
        .VerticalAlignment = xlCenter
    End With
    
    ' Task header (first column)
    With ws.Cells(1, 1)
        .Interior.Color = RGB(47, 85, 151) ' #2F5597
        .Font.Color = RGB(255, 255, 255) ' White
        .Font.Bold = True
        .HorizontalAlignment = xlLeft
        .VerticalAlignment = xlCenter
    End With
    
    ' === STEP 3: Format task names column ===
    
    For i = 2 To lastRow
        With ws.Cells(i, 1)
            ' Check if this is a section header (contains "HEADER" or common section names)
            If InStr(.Value, "INTRODUCTION") > 0 Or _
               InStr(.Value, "CHAPTER") > 0 Or _
               InStr(.Value, "GENERAL DISCUSSION") > 0 Or _
               InStr(.Value, "COMPILATION") > 0 Then
                .Interior.Color = RGB(180, 198, 231) ' #B4C6E7
                .Font.Italic = True
                .Font.Bold = True
            Else
                .Interior.Color = RGB(217, 226, 243) ' #D9E2F3
                .Font.Bold = True
            End If
            .HorizontalAlignment = xlLeft
            .VerticalAlignment = xlCenter
        End With
    Next i
    
    ' === STEP 4: Apply color coding to timeline cells ===
    
    ' Define colors for each code
    Dim draftColor As Long, reviseColor As Long, workColor As Long
    Dim finalizeColor As Long, completeColor As Long, prepareColor As Long
    
    draftColor = RGB(255, 230, 153)    ' #FFE699 - Yellow
    reviseColor = RGB(169, 208, 142)   ' #A9D08E - Green
    workColor = RGB(155, 194, 230)     ' #9BC2E6 - Blue
    finalizeColor = RGB(244, 176, 132) ' #F4B084 - Orange
    completeColor = RGB(112, 173, 71)  ' #70AD47 - Dark Green
    prepareColor = RGB(231, 230, 230)  ' #E7E6E6 - Gray
    
    ' Apply colors based on cell values
    For i = 2 To lastRow
        For j = 2 To lastCol
            Set cell = ws.Cells(i, j)
            
            Select Case Trim(UCase(cell.Value))
                Case "D"
                    cell.Interior.Color = draftColor
                    cell.Font.Bold = True
                    cell.Font.Color = RGB(0, 0, 0)
                Case "R"
                    cell.Interior.Color = reviseColor
                    cell.Font.Bold = True
                    cell.Font.Color = RGB(0, 0, 0)
                Case "W"
                    cell.Interior.Color = workColor
                    cell.Font.Bold = True
                    cell.Font.Color = RGB(0, 0, 0)
                Case "F"
                    cell.Interior.Color = finalizeColor
                    cell.Font.Bold = True
                    cell.Font.Color = RGB(0, 0, 0)
                Case "X"
                    cell.Interior.Color = completeColor
                    cell.Font.Bold = True
                    cell.Font.Color = RGB(255, 255, 255) ' White text
                Case "P"
                    cell.Interior.Color = prepareColor
                    cell.Font.Bold = True
                    cell.Font.Color = RGB(0, 0, 0)
                Case ""
                    ' Empty cells - light gray background
                    cell.Interior.Color = RGB(248, 248, 248)
            End Select
            
            ' Center align all timeline cells
            cell.HorizontalAlignment = xlCenter
            cell.VerticalAlignment = xlCenter
        Next j
    Next i
    
    ' === STEP 5: Add borders ===
    
    With ws.Range(ws.Cells(1, 1), ws.Cells(lastRow, lastCol))
        .Borders(xlEdgeLeft).LineStyle = xlContinuous
        .Borders(xlEdgeTop).LineStyle = xlContinuous
        .Borders(xlEdgeBottom).LineStyle = xlContinuous
        .Borders(xlEdgeRight).LineStyle = xlContinuous
        .Borders(xlInsideVertical).LineStyle = xlContinuous
        .Borders(xlInsideHorizontal).LineStyle = xlContinuous
        
        ' Set border color to light gray
        .Borders.Color = RGB(221, 221, 221) ' #DDD
    End With
    
    ' === STEP 6: Set row heights for better readability ===
    
    ws.Rows("1").RowHeight = 30 ' Header row
    For i = 2 To lastRow
        ws.Rows(i).RowHeight = 25 ' Data rows
    Next i
    
    ' === STEP 7: Freeze panes (optional but useful) ===
    
    ws.Activate
    ws.Cells(2, 2).Select
    ActiveWindow.FreezePanes = True
    
    ' === STEP 8: Add page setup for printing ===
    
    With ws.PageSetup
        .Orientation = xlLandscape
        .FitToPagesWide = 1
        .FitToPagesTall = False
        .CenterHorizontally = True
        .PrintTitleRows = "$1:$1"
        .PrintTitleColumns = "$A:$A"
    End With
    
    ' Return to top-left cell
    ws.Cells(1, 1).Select
    
    ' Show completion message
    MsgBox "Timeline formatting complete!" & vbCrLf & vbCrLf & _
           "Your thesis timeline now matches the HTML design with:" & vbCrLf & _
           "• Color-coded task statuses" & vbCrLf & _
           "• Professional styling" & vbCrLf & _
           "• Frozen headers for easy navigation" & vbCrLf & _
           "• Print-ready formatting", _
           vbInformation, "Formatting Complete"
    
End Sub

' === ADDITIONAL HELPER SUBROUTINES ===

Sub AddLegend()
    ' This subroutine adds a legend below the timeline
    Dim ws As Worksheet
    Dim lastRow As Long
    Dim legendRow As Long
    
    Set ws = ActiveSheet
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row
    legendRow = lastRow + 3
    
    ' Add legend title
    ws.Cells(legendRow, 1).Value = "LEGEND:"
    ws.Cells(legendRow, 1).Font.Bold = True
    ws.Cells(legendRow, 1).Font.Size = 12
    
    ' Add legend items
    Dim legends As Variant
    legends = Array( _
        Array("D", "Draft (First writing)", RGB(255, 230, 153)), _
        Array("R", "Revise (Revision of existing)", RGB(169, 208, 142)), _
        Array("W", "Work in Progress", RGB(155, 194, 230)), _
        Array("F", "Finalize", RGB(244, 176, 132)), _
        Array("X", "Complete/Submit", RGB(112, 173, 71)), _
        Array("P", "Prepare/Plan", RGB(231, 230, 230)) _
    )
    
    Dim i As Integer
    For i = 0 To UBound(legends)
        ' Add colored box
        With ws.Cells(legendRow + 1 + i, 1)
            .Value = legends(i)(0)
            .Interior.Color = legends(i)(2)
            .Font.Bold = True
            .HorizontalAlignment = xlCenter
            .Borders.LineStyle = xlContinuous
            .Borders.Color = RGB(153, 153, 153)
        End With
        
        ' Add description
        ws.Cells(legendRow + 1 + i, 2).Value = legends(i)(1)
        ws.Range(ws.Cells(legendRow + 1 + i, 2), ws.Cells(legendRow + 1 + i, 5)).Merge
    Next i
    
End Sub

Sub HighlightCurrentWeek(weekNumber As Integer)
    ' This subroutine highlights the current week column
    Dim ws As Worksheet
    Dim lastRow As Long
    
    Set ws = ActiveSheet
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row
    
    ' Remove any existing highlighting first
    ws.Range(ws.Cells(1, 2), ws.Cells(lastRow, 25)).Borders.LineStyle = xlContinuous
    ws.Range(ws.Cells(1, 2), ws.Cells(lastRow, 25)).Borders.Weight = xlThin
    
    ' Highlight the specified week column
    If weekNumber >= 1 And weekNumber <= 24 Then
        With ws.Range(ws.Cells(1, weekNumber + 1), ws.Cells(lastRow, weekNumber + 1))
            .Borders(xlEdgeLeft).Weight = xlThick
            .Borders(xlEdgeLeft).Color = RGB(255, 0, 0) ' Red border
            .Borders(xlEdgeRight).Weight = xlThick
            .Borders(xlEdgeRight).Color = RGB(255, 0, 0) ' Red border
        End With
    End If
    
End Sub