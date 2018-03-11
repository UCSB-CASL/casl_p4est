#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Partitioned Unstructured Grid Reader'
nodes_1_1x1x10pvtu = XMLPartitionedUnstructuredGridReader(FileName=['/home/dbochkov/Outputs/paper_examples/poisson/3d/triangle/gradients_1st_order/vtu/nodes_1_1x1x1.0.pvtu'])
nodes_1_1x1x10pvtu.CellArrayStatus = ['proc_rank', 'tree_idx', 'leaf_level']
nodes_1_1x1x10pvtu.PointArrayStatus = ['phi', 'phi_smooth', 'sol', 'sol_ex', 'error_sl', 'error_tr', 'error_gr', 'error_ex', 'error_dd', 'mask', 'volumes']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1114, 1151]

# show data in view
nodes_1_1x1x10pvtuDisplay = Show(nodes_1_1x1x10pvtu, renderView1)
# trace defaults for the display properties.
nodes_1_1x1x10pvtuDisplay.ColorArrayName = [None, '']
nodes_1_1x1x10pvtuDisplay.GlyphType = 'Arrow'
nodes_1_1x1x10pvtuDisplay.ScalarOpacityUnitDistance = 0.054126587736527426

# reset view to fit data
renderView1.ResetCamera()

# create a new 'XML Unstructured Grid Reader'
edgs_3d_quadratic_0vtu = XMLUnstructuredGridReader(FileName=['/home/dbochkov/Outputs/paper_examples/poisson/3d/triangle/gradients_1st_order/geometry/edgs_3d_quadratic_0.vtu'])
edgs_3d_quadratic_0vtu.CellArrayStatus = ['location', 'simplex', 'c0', 'c1']
edgs_3d_quadratic_0vtu.PointArrayStatus = ['location']

# show data in view
edgs_3d_quadratic_0vtuDisplay = Show(edgs_3d_quadratic_0vtu, renderView1)
# trace defaults for the display properties.
edgs_3d_quadratic_0vtuDisplay.ColorArrayName = [None, '']
edgs_3d_quadratic_0vtuDisplay.GlyphType = 'Arrow'
edgs_3d_quadratic_0vtuDisplay.ScalarOpacityUnitDistance = 0.02920933269576832

# create a new 'XML Unstructured Grid Reader'
tris_3d_quadratic_0vtu = XMLUnstructuredGridReader(FileName=['/home/dbochkov/Outputs/paper_examples/poisson/3d/triangle/gradients_1st_order/geometry/tris_3d_quadratic_0.vtu'])
tris_3d_quadratic_0vtu.CellArrayStatus = ['color', 'idx', 'simplex']
tris_3d_quadratic_0vtu.PointArrayStatus = ['scalars']

# get color transfer function/color map for 'scalars'
scalarsLUT = GetColorTransferFunction('scalars')
scalarsLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 2.0, 0.865003, 0.865003, 0.865003, 4.0, 0.705882, 0.0156863, 0.14902]
scalarsLUT.ScalarRangeInitialized = 1.0

# show data in view
tris_3d_quadratic_0vtuDisplay = Show(tris_3d_quadratic_0vtu, renderView1)
# trace defaults for the display properties.
tris_3d_quadratic_0vtuDisplay.ColorArrayName = ['POINTS', 'scalars']
tris_3d_quadratic_0vtuDisplay.LookupTable = scalarsLUT
tris_3d_quadratic_0vtuDisplay.GlyphType = 'Arrow'
tris_3d_quadratic_0vtuDisplay.ScalarOpacityUnitDistance = 0.030460519581237435

# show color bar/color legend
tris_3d_quadratic_0vtuDisplay.SetScalarBarVisibility(renderView1, True)

# get opacity transfer function/opacity map for 'scalars'
scalarsPWF = GetOpacityTransferFunction('scalars')
scalarsPWF.Points = [0.0, 0.0, 0.5, 0.0, 4.0, 1.0, 0.5, 0.0]
scalarsPWF.ScalarRangeInitialized = 1

# get layout
viewLayout1 = GetLayout()

# split cell
viewLayout1.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [552, 1151]
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.StereoType = 0
renderView2.Background = [0.32, 0.34, 0.43]

# place view in the layout
viewLayout1.AssignView(2, renderView2)

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView2)

# split cell
viewLayout1.SplitVertical(2, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.ViewSize = [552, 560]
renderView3.AxesGrid = 'GridAxes3DActor'
renderView3.StereoType = 0
renderView3.Background = [0.32, 0.34, 0.43]

# place view in the layout
viewLayout1.AssignView(6, renderView3)

# set active view
SetActiveView(renderView1)

# split cell
viewLayout1.SplitVertical(1, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView4 = CreateView('RenderView')
renderView4.ViewSize = [553, 560]
renderView4.AxesGrid = 'GridAxes3DActor'
renderView4.StereoType = 0
renderView4.Background = [0.32, 0.34, 0.43]

# place view in the layout
viewLayout1.AssignView(4, renderView4)

# set active view
SetActiveView(renderView1)

#### saving camera placements for all active views

# current camera placement for renderView2

# current camera placement for renderView3

# current camera placement for renderView4

# current camera placement for renderView1
renderView1.CameraPosition = [0.0, 0.0, 6.899734963693014]
renderView1.CameraParallelScale = 1.7895785273893874

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).