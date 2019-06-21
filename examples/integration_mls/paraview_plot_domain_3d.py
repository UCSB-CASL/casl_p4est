#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
tris_3d_quadratic_0vtu = XMLUnstructuredGridReader(FileName=['/home/dbochkov/Outputs/paper_examples/integration/vtk/3d/union/geometry/tris_3d_quadratic_0.vtu','/home/dbochkov/Outputs/paper_examples/integration/vtk/3d/difference/geometry/tris_3d_quadratic_0.vtu'])
tris_3d_quadratic_0vtu.CellArrayStatus = ['color', 'idx', 'simplex']
tris_3d_quadratic_0vtu.PointArrayStatus = ['scalars']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [2176, 1151]

# get color transfer function/color map for 'scalars'
scalarsLUT = GetColorTransferFunction('scalars')
scalarsLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 1.5, 0.865003, 0.865003, 0.865003, 3.0, 0.705882, 0.0156863, 0.14902]
scalarsLUT.ScalarRangeInitialized = 1.0

# show data in view
tris_3d_quadratic_0vtuDisplay = Show(tris_3d_quadratic_0vtu, renderView1)
# trace defaults for the display properties.
tris_3d_quadratic_0vtuDisplay.ColorArrayName = ['POINTS', 'scalars']
tris_3d_quadratic_0vtuDisplay.LookupTable = scalarsLUT
tris_3d_quadratic_0vtuDisplay.GlyphType = 'Arrow'
tris_3d_quadratic_0vtuDisplay.ScalarOpacityUnitDistance = 0.018905194931448068

# reset view to fit data
renderView1.ResetCamera()

# show color bar/color legend
tris_3d_quadratic_0vtuDisplay.SetScalarBarVisibility(renderView1, True)

# get opacity transfer function/opacity map for 'scalars'
scalarsPWF = GetOpacityTransferFunction('scalars')
scalarsPWF.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
scalarsPWF.ScalarRangeInitialized = 1

# create a new 'Threshold'
threshold1 = Threshold(Input=tris_3d_quadratic_0vtu)
threshold1.Scalars = ['CELLS', 'color']
threshold1.ThresholdRange = [-0.5, 10.0]

# show data in view
threshold1Display = Show(threshold1, renderView1)
# trace defaults for the display properties.
threshold1Display.ColorArrayName = ['POINTS', 'scalars']
threshold1Display.LookupTable = scalarsLUT
threshold1Display.GlyphType = 'Arrow'
threshold1Display.ScalarOpacityUnitDistance = 0.04264158835652916

# hide data in view
Hide(tris_3d_quadratic_0vtu, renderView1)

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, False)

# set scalar coloring
ColorBy(threshold1Display, ('CELLS', 'color'))

# rescale color and/or opacity maps used to include current data range
threshold1Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, False)

# get color transfer function/color map for 'color'
colorLUT = GetColorTransferFunction('color')
colorLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'color'
colorPWF = GetOpacityTransferFunction('color')
colorPWF.ScalarRangeInitialized = 1

# create a new 'Box'
box1 = Box()

# Properties modified on box1
box1.XLength = 2.0
box1.YLength = 2.0
box1.ZLength = 2.0

# show data in view
box1Display = Show(box1, renderView1)
# trace defaults for the display properties.
box1Display.ColorArrayName = [None, '']
box1Display.GlyphType = 'Arrow'

# Properties modified on box1Display
box1Display.Opacity = 0.1

# create a new 'Box'
box2 = Box()

# Properties modified on box2
box2.XLength = 2.0
box2.YLength = 2.0
box2.ZLength = 2.0

# show data in view
box2Display = Show(box2, renderView1)
# trace defaults for the display properties.
box2Display.ColorArrayName = [None, '']
box2Display.GlyphType = 'Arrow'

# change representation type
box2Display.SetRepresentationType('Wireframe')

# Properties modified on box2Display
box2Display.LineWidth = 2.0

# Properties modified on box2Display
box2Display.Opacity = 0.3

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [-9.621261550631559, 0.5849442339307498, 3.8966356371468254]
renderView1.CameraFocalPoint = [0.0546875, 0.03125000000000001, 0.02343750000000001]
renderView1.CameraViewUp = [0.16988467058875187, 0.9419037657437691, 0.2897524715626396]
renderView1.CameraParallelScale = 1.524819085067963

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
