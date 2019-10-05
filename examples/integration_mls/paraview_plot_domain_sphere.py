#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
tris_3d_quadratic_0vtu = XMLUnstructuredGridReader(FileName=[ \
'/home/dbochkov/Outputs/paper_examples/integration/vtk/3d/sphere/without/geometry/tris_3d_quadratic_0.vtu', \
'/home/dbochkov/Outputs/paper_examples/integration/vtk/3d/sphere/without/geometry/tris_3d_quadratic_1.vtu', \
'/home/dbochkov/Outputs/paper_examples/integration/vtk/3d/sphere/without/geometry/tris_3d_quadratic_2.vtu', \
'/home/dbochkov/Outputs/paper_examples/integration/vtk/3d/sphere/without/geometry/tris_3d_quadratic_3.vtu', \
'/home/dbochkov/Outputs/paper_examples/integration/vtk/3d/sphere/with/geometry/tris_3d_quadratic_0.vtu', \
'/home/dbochkov/Outputs/paper_examples/integration/vtk/3d/sphere/with/geometry/tris_3d_quadratic_1.vtu', \
'/home/dbochkov/Outputs/paper_examples/integration/vtk/3d/sphere/with/geometry/tris_3d_quadratic_2.vtu', \
'/home/dbochkov/Outputs/paper_examples/integration/vtk/3d/sphere/with/geometry/tris_3d_quadratic_3.vtu'])
tris_3d_quadratic_0vtu.CellArrayStatus = ['color', 'idx', 'simplex']
tris_3d_quadratic_0vtu.PointArrayStatus = ['scalars']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [2174, 1151]

# get color transfer function/color map for 'scalars'
scalarsLUT = GetColorTransferFunction('scalars')
scalarsLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 1.0, 0.865003, 0.865003, 0.865003, 2.0, 0.705882, 0.0156863, 0.14902]
scalarsLUT.ScalarRangeInitialized = 1.0

# show data in view
tris_3d_quadratic_0vtuDisplay = Show(tris_3d_quadratic_0vtu, renderView1)
# trace defaults for the display properties.
tris_3d_quadratic_0vtuDisplay.AmbientColor = [0.0, 0.0, 0.0]
tris_3d_quadratic_0vtuDisplay.ColorArrayName = ['POINTS', 'scalars']
tris_3d_quadratic_0vtuDisplay.LookupTable = scalarsLUT
tris_3d_quadratic_0vtuDisplay.GlyphType = 'Arrow'
tris_3d_quadratic_0vtuDisplay.CubeAxesColor = [0.0, 0.0, 0.0]
tris_3d_quadratic_0vtuDisplay.ScalarOpacityUnitDistance = 0.04745024462076715

# reset view to fit data
renderView1.ResetCamera()

# show color bar/color legend
tris_3d_quadratic_0vtuDisplay.SetScalarBarVisibility(renderView1, True)

# get opacity transfer function/opacity map for 'scalars'
scalarsPWF = GetOpacityTransferFunction('scalars')
scalarsPWF.Points = [0.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]
scalarsPWF.ScalarRangeInitialized = 1

# create a new 'Threshold'
threshold1 = Threshold(Input=tris_3d_quadratic_0vtu)
threshold1.Scalars = ['CELLS', 'color']
threshold1.ThresholdRange = [-0.5, 10.0]

# show data in view
threshold1Display = Show(threshold1, renderView1)
# trace defaults for the display properties.
threshold1Display.AmbientColor = [0.0, 0.0, 0.0]
threshold1Display.ColorArrayName = ['POINTS', 'scalars']
threshold1Display.LookupTable = scalarsLUT
threshold1Display.GlyphType = 'Arrow'
threshold1Display.CubeAxesColor = [0.0, 0.0, 0.0]
threshold1Display.ScalarOpacityUnitDistance = 0.10305768099305344

# hide data in view
Hide(tris_3d_quadratic_0vtu, renderView1)

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, False)

# turn off scalar coloring
ColorBy(threshold1Display, None)

# change representation type
threshold1Display.SetRepresentationType('Surface With Edges')

# hide data in view
Hide(threshold1, renderView1)

# set active source
SetActiveSource(threshold1)

# show data in view
threshold1Display = Show(threshold1, renderView1)

# reset view to fit data
renderView1.ResetCamera()

# Properties modified on threshold1Display
threshold1Display.LineWidth = 2.0

# set scalar coloring
ColorBy(threshold1Display, ('CELLS', 'color'))

# rescale color and/or opacity maps used to include current data range
threshold1Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, False)

# get color transfer function/color map for 'color'
colorLUT = GetColorTransferFunction('color')
colorLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 5e-17, 0.865003, 0.865003, 0.865003, 1e-16, 0.705882, 0.0156863, 0.14902]
colorLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'color'
colorPWF = GetOpacityTransferFunction('color')
colorPWF.Points = [0.0, 0.0, 0.5, 0.0, 1e-16, 1.0, 0.5, 0.0]
colorPWF.ScalarRangeInitialized = 1

# turn off scalar coloring
ColorBy(threshold1Display, None)

# change solid color
threshold1Display.DiffuseColor = [0.0, 0.6666666666666666, 1.0]

# Properties modified on threshold1Display
threshold1Display.EdgeColor = [0.0, 0.0, 0.0]

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [1.7459965577290821, 0.022556671905673942, 0.06689358889562971]
renderView1.CameraFocalPoint = [0.10563400387763966, 4.5001506805396146e-06, 4.5001506804866615e-06]
renderView1.CameraViewUp = [-0.011310172614238013, 0.9981833227973149, -0.05917883139052114]
renderView1.CameraParallelScale = 1.3336750797086063

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
