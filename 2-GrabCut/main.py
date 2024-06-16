import sys,os
import wx
from wx.lib import buttons
from grab import algo_grabcut
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import time

idOPEN = wx.ID_OPEN
idCLEAR = wx.ID_CLEAR
idEXIT = wx.ID_EXIT


# -------------------------------------------------------------------------------------------
class DoodleWindow(wx.Window):
    # 自定义窗口部件，用于在应用程序窗口的右侧显示图像
    def __init__(self, parent, ID):
        wx.Window.__init__(self, parent, ID, style=wx.NO_FULL_REPAINT_ON_RESIZE)
        self.SetBackgroundColour("WHITE")
        self.filename = []
        self.thickness = 5
        self.drawRegion = False
        self.RegionSet = False
        self.OpenCV = True
        self.SetColour("Cyan")
        self.pos_foreground = []
        self.pos_background = []
        self.pos = wx.Point(0, 0)
        self.Regionpos1 = wx.Point(0, 0)
        self.Regionpos2 = wx.Point(0, 0)
        # 初始化缓冲区
        self.InitBuffer()
        # 设置鼠标光标为铅笔样式
        self.SetCursor(wx.Cursor(wx.CURSOR_PENCIL))

        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    # 初始化绘图缓冲区
    def InitBuffer(self):
        size = self.GetClientSize()
        self.buffer = wx.Bitmap(max(1, size.width), max(1, size.height))
        dc = wx.BufferedDC(None, self.buffer)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        self.reInitBuffer = False

    # 设置是否绘制区域
    def SetDrawRegion(self, tf):
        self.drawRegion = tf

    # 设置区域设置
    def SetRegionSet(self, tf):
        self.RegionSet = tf

    # 获取区域设置
    def GetRegionSet(self):
        return self.RegionSet

    # 设置算法OpenCV
    def SetAlgorithm(self, tf):
        self.OpenCV = tf

    # 获取算法OpenCV的值
    def GetAlgorithm(self):
        return self.OpenCV

    # 设置对象的颜色属性
    def SetColour(self, colour):
        self.colour = colour
        self.pen = wx.Pen(self.colour, self.thickness, wx.SOLID)

    # 设置文件名
    def SetFilename(self, fn):
        self.filename = fn
        self.Refresh()

    # 获取文件名
    def GetFilename(self):
        return self.filename

    # 设置窗口的线条数据
    def SetLinesData(self, lines):
        self.InitBuffer()
        self.Refresh()

    # 清空地面数据
    def ClearGroundData(self):
        self.pos_foreground = []
        self.pos_background = []

    # 获取前景数据的值
    def GetForegroundData(self):
        return self.pos_foreground

    # 获取背景数据的值
    def GetBackgroundData(self):
        return self.pos_background

    # 获取区域位置数据，返回不同的区域位置数据
    def GetRegionPos(self, x):
        if x == 1:
            return self.Regionpos1
        else:
            return self.Regionpos2

    # 当鼠标左键按下时触发的事件处理方法
    def OnLeftDown(self, event):
        self.pos = event.GetPosition()
        if self.drawRegion:
            self.Regionpos1 = self.pos
        self.CaptureMouse()

    # 当鼠标左键释放时触发的事件处理方法
    def OnLeftUp(self, event):
        if self.HasCapture():
            self.ReleaseMouse()

    # 当鼠标移动时触发的事件处理方法：如果鼠标正在拖拽并且左键是按下状态，获取鼠标位置；否则
    # 根据颜色值将当前位置的坐标添加到前景数据或背景数据中，并在画布上绘制线条
    def OnMotion(self, event):
        if event.Dragging() and event.LeftIsDown():
            pos = event.GetPosition()
            if self.drawRegion:
                self.RegionSet = True
                self.Regionpos2 = pos
                self.Refresh()
            else:
                if self.colour == 'Cyan':
                    self.pos_foreground.append((self.pos.x, self.pos.y, pos.x, pos.y))
                if self.colour == 'Red':
                    self.pos_background.append((self.pos.x, self.pos.y, pos.x, pos.y))
                dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
                # dc.BeginDrawing()
                dc.SetPen(self.pen)
                coords = (self.pos.x, self.pos.y, pos.x, pos.y)
                dc.DrawLine(*coords)
                self.pos = pos
                # dc.EndDrawing()

    # 当窗口大小改变时触发的事件处理方法，用于初始化缓冲区
    def OnSize(self, event):
        self.reInitBuffer = True

    # 当应用程序空闲时触发的事件处理方法
    def OnIdle(self, event):
        if self.reInitBuffer:
            self.InitBuffer()
            self.Refresh(False)

    # 当窗口需要重绘时触发的事件处理方法
    def OnPaint(self, event):
        self.InitBuffer()
        dc = wx.BufferedPaintDC(self, self.buffer)  # 用于在双缓冲区环境下绘制图像
        if not self.filename == []:
            img = wx.Image(self.filename)
            dc.DrawBitmap(img.ConvertToBitmap(), 0, 0, True)
            if self.RegionSet:
                dc.SetPen(wx.Pen('Yellow', 3, wx.SOLID))
                coords = (self.Regionpos1.x, self.Regionpos1.y,
                          self.Regionpos1.x, self.Regionpos2.y)
                dc.DrawLine(*coords)

                coords = (self.Regionpos2.x, self.Regionpos1.y,
                          self.Regionpos2.x, self.Regionpos2.y)
                dc.DrawLine(*coords)

                coords = (self.Regionpos1.x, self.Regionpos1.y,
                          self.Regionpos2.x, self.Regionpos1.y)
                dc.DrawLine(*coords)

                coords = (self.Regionpos1.x, self.Regionpos2.y,
                          self.Regionpos2.x, self.Regionpos2.y)
                dc.DrawLine(*coords)
            else:
                im = Image.open(self.filename)
                self.Regionpos1 = wx.Point(1, 1)
                self.Regionpos2 = wx.Point(im.size[0] - 1, im.size[1] - 1)


# ----------表示应用程序窗口左侧的控制面板，并将其添加到应用程序的主窗口中
class ControlPanel(wx.Panel):
    #	this class builds the control panel which is on the left
    #	of the application window,including several buttons.
    def __init__(self, parent, ID, doodle):
        wx.Panel.__init__(self, parent, ID, style=wx.RAISED_BORDER)
        self.doodle = doodle

        self.sizer2 = wx.BoxSizer(wx.VERTICAL)
        self.buttons = []
        self.buttons.append(wx.Button(self, -1, "Open"))
        self.buttons.append(wx.Button(self, -1, "Set Region"))
        self.buttons.append(wx.Button(self, -1, "Set Foreground"))
        self.buttons.append(wx.Button(self, -1, "Set Background"))
        self.buttons.append(wx.Button(self, -1, "Clear"))
        self.buttons.append(wx.Button(self, -1, "Run"))
        self.buttons.append(wx.CheckBox(self, -1, "OpenCV"))
        for i in range(0, 7):
            self.sizer2.Add(self.buttons[i], 1, wx.EXPAND)
        self.buttons[6].SetValue(True)

        self.Bind(wx.EVT_BUTTON, self.OnOpen, self.buttons[0])
        self.Bind(wx.EVT_BUTTON, self.OnSetRegion, self.buttons[1])
        self.Bind(wx.EVT_BUTTON, self.OnSetForeground, self.buttons[2])
        self.Bind(wx.EVT_BUTTON, self.OnSetBackground, self.buttons[3])
        self.Bind(wx.EVT_BUTTON, self.OnClear, self.buttons[4])
        self.Bind(wx.EVT_BUTTON, self.OnRun, self.buttons[5])
        self.Bind(wx.EVT_CHECKBOX, self.OnCheck, self.buttons[6])

        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(self.sizer2, 0, wx.ALL)
        self.SetSizer(box)
        self.SetAutoLayout(True)
        box.Fit(self)

    # 控制面板中“Open”按钮的事件处理程序，打开一个文件对话框，允许用户选择图像文件
    def OnOpen(self, event):
        dlg = wx.FileDialog(self, "Open image file...", os.getcwd() + r"\image",
                            style=wx.FD_OPEN,     # | wx.FD_CHANGE_DIR,
                            wildcard="Image files (*.png;*.jpeg;*.jpg)|*.png;*.jpeg;*.jpg")
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetPath()
            print(self.filename)
            self.doodle.SetFilename(self.filename)
            self.doodle.SetLinesData([])
            self.doodle.ClearGroundData()
            self.doodle.SetRegionSet(False)
        dlg.Destroy()

    # 控制面板中“Set Region”按钮的事件处理程序，用于设置GrabCut区域
    def OnSetRegion(self, event):
        fn = self.doodle.GetFilename()
        if fn == []:
            wx.MessageBox("Please open a image file before you set the GrabCut region!",
                          "oops!", style=wx.OK | wx.ICON_EXCLAMATION)
        else:
            self.doodle.SetLinesData([])
            self.doodle.ClearGroundData()
            self.doodle.SetDrawRegion(True)

    # 当用户点击前景按钮时调用的方法
    def OnSetForeground(self, event):
        self.doodle.SetDrawRegion(False)
        self.doodle.SetColour('Cyan')

    # 当用户点击背景按钮时调用的方法
    def OnSetBackground(self, event):
        self.doodle.SetDrawRegion(False)
        self.doodle.SetColour('Red')

    # 当用户点击清除按钮时调用的方法
    def OnClear(self, event):
        self.doodle.SetLinesData([])
        self.doodle.ClearGroundData()

    # 当用户点击运行按钮时调用的方法
    def OnRun(self, event):
        fn = self.doodle.GetFilename()
        if fn == []:
            wx.MessageBox("You should open a image file before you run the GrabCut algorithm!",
                          "oops!", style=wx.OK | wx.ICON_EXCLAMATION)
        else:
            pos_fore = self.doodle.GetForegroundData()
            pos_back = self.doodle.GetBackgroundData()
            pos1 = self.doodle.GetRegionPos(1)
            pos2 = self.doodle.GetRegionPos(2)
            if algo_grabcut(filename=fn,
                            foreground=pos_fore,
                            background=pos_back,
                            pos1x=pos1.x,
                            pos1y=pos1.y,
                            pos2x=pos2.x,
                            pos2y=pos2.y,
                            algo=self.doodle.GetAlgorithm()):
                im = Image.open(os.getcwd() + "/out.png")
                im.show()
            else:
                wx.MessageBox("An error occurred while running the GrabCut algorithm.",
                              "Error", style=wx.OK | wx.ICON_ERROR)

    # 处理用户勾选或取消勾选复选框时的事件
    def OnCheck(self, event):
        if event.IsChecked():
            self.doodle.SetAlgorithm(True)
        else:
            self.doodle.SetAlgorithm(False)


# ----------构建应用程序的主窗口
class GrabFrame(wx.Frame):
    #	this class builds the frame of the application,which
    #	consists of a doodle window and a control panel.
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, "GrabCut", size=(638, 512),
                          style=wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE)

        self.doodle = DoodleWindow(self, -1)  # build a doodle window object
        cPanel = ControlPanel(self, -1, self.doodle)  # build a control panel object

        # set the layout of the two objects above
        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(cPanel, 0, wx.EXPAND)
        box.Add(self.doodle, 1, wx.EXPAND)

        self.SetSizer(box)
        self.Centre()  # to make the UI on the centre of the screen


# -------------------------------------------------------------------------------------------
class GrabApp(wx.App):
    #	this class builds the whole application.
    def OnInit(self):
        frame = GrabFrame(None)
        frame.Show(True)
        return True


# -------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app = GrabApp()
    app.MainLoop()
    


