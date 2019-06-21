#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
edgs_2d_quadratic_ = XMLUnstructuredGridReader(FileName=['/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/vtk/2d/circle/geometry/edgs_2d_quadratic_0.vtu', '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/vtk/2d/union/geometry/edgs_2d_quadratic_0.vtu', '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/vtk/2d/difference/geometry/edgs_2d_quadratic_0.vtu'])
edgs_2d_quadratic_.CellArrayStatus = ['location', 'c0']
edgs_2d_quadratic_.PointArrayStatus = ['location']

# create a new 'XML Partitioned Unstructured Grid Reader'
nodes_vtk = XMLPartitionedUnstructuredGridReader(FileName=['/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/vtk/2d/circle/vtu/nodes_1_1x1.0.pvtu','/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/vtk/2d/union/vtu/nodes_1_1x1.0.pvtu','/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/vtk/2d/difference/vtu/nodes_1_1x1.0.pvtu'])
nodes_vtk.CellArrayStatus = ['proc_rank', 'tree_idx', 'leaf_level']
nodes_vtk.PointArrayStatus = ['phi_tot']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [2176, 1151]

# show data in view
edgs_2d_quadratic_Display = Show(edgs_2d_quadratic_, renderView1)
# trace defaults for the display properties.
edgs_2d_quadratic_Display.ColorArrayName = [None, '']
edgs_2d_quadratic_Display.GlyphType = 'Arrow'
edgs_2d_quadratic_Display.ScalarOpacityUnitDistance = 0.4135185542000138

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.0, 10000.0]

# create a new 'Threshold'
threshold1 = Threshold(Input=edgs_2d_quadratic_)
threshold1.Scalars = ['CELLS', 'c0']
threshold1.ThresholdRange = [-0.5, 10.0]

# show data in view
threshold1Display = Show(threshold1, renderView1)
# trace defaults for the display properties.
threshold1Display.AmbientColor = [0.0, 0.0, 0.0]
threshold1Display.ColorArrayName = [None, '']
threshold1Display.GlyphType = 'Arrow'
threshold1Display.CubeAxesColor = [0.0, 0.0, 0.0]
threshold1Display.ScalarOpacityUnitDistance = 0.6309743787148683

# hide data in view
Hide(edgs_2d_quadratic_, renderView1)

# set scalar coloring
ColorBy(threshold1Display, ('CELLS', 'c0'))

# rescale color and/or opacity maps used to include current data range
threshold1Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'c0'
c0LUT = GetColorTransferFunction('c0')
c0LUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 5e-17, 0.865003, 0.865003, 0.865003, 1e-16, 0.705882, 0.0156863, 0.14902]
c0LUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'c0'
c0PWF = GetOpacityTransferFunction('c0')
c0PWF.Points = [0.0, 0.0, 0.5, 0.0, 1e-16, 1.0, 0.5, 0.0]
c0PWF.ScalarRangeInitialized = 1

# change representation type
threshold1Display.SetRepresentationType('Wireframe')

# Properties modified on threshold1Display
threshold1Display.LineWidth = 2.0

# set active source
#SetActiveSource(edgs_2d_quadratic_)

# show data in view
#edgs_2d_quadratic_Display = Show(edgs_2d_quadratic_, renderView1)

# change representation type
#edgs_2d_quadratic_Display.SetRepresentationType('Wireframe')

# create a new 'Box'
box1 = Box()

# Properties modified on box1
box1.XLength = 2.0
box1.YLength = 2.0
box1.ZLength = 2.0

# set active source
SetActiveSource(box1)

# show data in view
box1Display = Show(box1, renderView1)

# change representation type
box1Display.SetRepresentationType('Wireframe')

# Properties modified on box1Display
box1Display.LineWidth = 2.0

# Properties modified on box1Display
box1Display.Opacity = 0.5

# trace defaults for the display properties.
box1Display.AmbientColor = [0.0, 0.0, 0.0]
box1Display.ColorArrayName = [None, '']
box1Display.GlyphType = 'Arrow'
box1Display.CubeAxesColor = [0.0, 0.0, 0.0]

# show data in view
nodes_vtkDisplay = Show(nodes_vtk, renderView1)
# trace defaults for the display properties.
nodes_vtkDisplay.AmbientColor = [0.0, 0.0, 0.0]
nodes_vtkDisplay.ColorArrayName = [None, '']
nodes_vtkDisplay.GlyphType = 'Arrow'
nodes_vtkDisplay.CubeAxesColor = [0.0, 0.0, 0.0]
nodes_vtkDisplay.ScalarOpacityUnitDistance = 0.7071067811865477

# create a new 'Clip'
clip1 = Clip(Input=nodes_vtk)
clip1.ClipType = 'Scalar'
clip1.Scalars = ['POINTS', 'phi_tot']
clip1.Value = -0.0

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip1)

# show data in view
clip1Display = Show(clip1, renderView1)
# trace defaults for the display properties.
clip1Display.AmbientColor = [0.0, 0.0, 0.0]
clip1Display.ColorArrayName = [None, '']
clip1Display.GlyphType = 'Arrow'
clip1Display.CubeAxesColor = [0.0, 0.0, 0.0]
clip1Display.ScalarOpacityUnitDistance = 0.7577812914450925

# hide data in view
Hide(nodes_vtk, renderView1)

# Properties modified on clip1
clip1.InsideOut = 1

# Properties modified on clip1Display
clip1Display.Specular = 1.0

# change solid color
clip1Display.DiffuseColor = [1.0, 1.0, 0.4980392156862745]

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.0, 10000.0]
renderView1.CameraParallelScale = 2.0705500766704485

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
