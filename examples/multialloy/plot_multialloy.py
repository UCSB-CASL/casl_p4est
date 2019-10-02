#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active source.
bialloy_6_1x10 = GetActiveSource()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [2181, 1151]

# get display properties
bialloy_6_1x10Display = GetDisplayProperties(bialloy_6_1x10, view=renderView1)

# change representation type
bialloy_6_1x10Display.SetRepresentationType('Outline')

# Properties modified on bialloy_6_1x10Display
bialloy_6_1x10Display.Opacity = 0.3

# Properties modified on bialloy_6_1x10Display
bialloy_6_1x10Display.LineWidth = 2.0

# create a new 'Clip'
clip1 = Clip(Input=bialloy_6_1x10)
clip1.ClipType = 'Plane'
clip1.Scalars = ['POINTS', 'bc_error']
clip1.Value = 2.121808448550186e-06

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.5, 0.5, 0.0]

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip1)

# Properties modified on clip1
clip1.ClipType = 'Scalar'
clip1.Scalars = ['POINTS', 'phi']
clip1.Value = -0.0
clip1.InsideOut = 1

# show data in view
clip1Display = Show(clip1, renderView1)
# trace defaults for the display properties.
clip1Display.AmbientColor = [0.0, 0.0, 0.0]
clip1Display.ColorArrayName = [None, '']
clip1Display.GlyphType = 'Arrow'
clip1Display.CubeAxesColor = [0.0, 0.0, 0.0]
clip1Display.ScalarOpacityUnitDistance = 0.05763531901097858

# set scalar coloring
ColorBy(clip1Display, ('POINTS', 'c0'))

# rescale color and/or opacity maps used to include current data range
clip1Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'c0'
c0LUT = GetColorTransferFunction('c0')
c0LUT.RGBPoints = [0.10699999999999689, 0.231373, 0.298039, 0.752941, 0.1164881848584225, 0.865003, 0.865003, 0.865003, 0.12597636971684836, 0.705882, 0.0156863, 0.14902]
c0LUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'c0'
c0PWF = GetOpacityTransferFunction('c0')
c0PWF.Points = [0.10699999999999689, 0.0, 0.5, 0.0, 0.12597636971684836, 1.0, 0.5, 0.0]
c0PWF.ScalarRangeInitialized = 1

# hide color bar/color legend
clip1Display.SetScalarBarVisibility(renderView1, False)

# get animation scene
animationScene1 = GetAnimationScene()

animationScene1.GoToLast()

# rescale color and/or opacity maps used to exactly fit the current data range
clip1Display.RescaleTransferFunctionToDataRange(False)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5, 0.5, 10000.0]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]
renderView1.CameraParallelScale = 0.7071067811865476

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).