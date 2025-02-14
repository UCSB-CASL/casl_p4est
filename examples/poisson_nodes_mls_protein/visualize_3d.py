#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1114, 1151]

# destroy renderView1
Delete(renderView1)
del renderView1

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1114, 1151]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView1.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
renderView1.StereoType = 0
renderView1.Background = [1.0, 1.0, 1.0]

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView1.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.GridColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

# get layout
viewLayout1 = GetLayout()

# place view in the layout
viewLayout1.AssignView(0, renderView1)

# split cell
viewLayout1.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [552, 1151]
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView2.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
renderView2.StereoType = 0
renderView2.Background = [1.0, 1.0, 1.0]

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView2.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.GridColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

# place view in the layout
viewLayout1.AssignView(2, renderView2)

# set active view
SetActiveView(renderView1)

# split cell
viewLayout1.SplitVertical(1, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.ViewSize = [553, 560]
renderView3.AxesGrid = 'GridAxes3DActor'
renderView3.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView3.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
renderView3.StereoType = 0
renderView3.Background = [1.0, 1.0, 1.0]

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView3.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.GridColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

# place view in the layout
viewLayout1.AssignView(4, renderView3)

# set active view
SetActiveView(renderView2)

# split cell
viewLayout1.SplitVertical(2, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView4 = CreateView('RenderView')
renderView4.ViewSize = [552, 560]
renderView4.AxesGrid = 'GridAxes3DActor'
renderView4.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView4.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
renderView4.StereoType = 0
renderView4.Background = [1.0, 1.0, 1.0]

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView4.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.GridColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

# place view in the layout
viewLayout1.AssignView(6, renderView4)

# set active view
SetActiveView(renderView2)

# set active view
SetActiveView(renderView3)

# set active view
SetActiveView(renderView1)

# create a new 'XML Partitioned Unstructured Grid Reader'
# nodes_1_1x1x10pvtu = XMLPartitionedUnstructuredGridReader(FileName=[ \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/triangle/gradients_1st_order/vtu/nodes_1_1x1x1.0.pvtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/triangle/gradients_2nd_order_c/vtu/nodes_1_1x1x1.0.pvtu', \
# \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/union/gradients_1st_order/vtu/nodes_1_1x1x1.0.pvtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/union/gradients_2nd_order_c/vtu/nodes_1_1x1x1.0.pvtu', \
# \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/difference/gradients_1st_order/vtu/nodes_1_1x1x1.0.pvtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/difference/gradients_2nd_order_c/vtu/nodes_1_1x1x1.0.pvtu', \
# \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/three_flowers/gradients_1st_order/vtu/nodes_1_1x1x1.0.pvtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/three_flowers/gradients_2nd_order_c/vtu/nodes_1_1x1x1.0.pvtu'
# ])

nodes_1_1x1 = XMLPartitionedUnstructuredGridReader(FileName=[
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.0.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.1.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.2.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.3.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.4.pvtu'
])

nodes_1_1x1x10pvtu.CellArrayStatus = ['proc_rank', 'tree_idx', 'leaf_level']
nodes_1_1x1x10pvtu.PointArrayStatus = ['phi', 'phi_smooth', 'sol', 'sol_ex', 'error_sl', 'error_tr', 'error_gr', 'error_ex', 'error_dd', 'mask', 'volumes']

# show data in view
nodes_1_1x1x10pvtuDisplay = Show(nodes_1_1x1x10pvtu, renderView1)
# trace defaults for the display properties.
nodes_1_1x1x10pvtuDisplay.AmbientColor = [0.0, 0.0, 0.0]
nodes_1_1x1x10pvtuDisplay.ColorArrayName = [None, '']
nodes_1_1x1x10pvtuDisplay.GlyphType = 'Arrow'
nodes_1_1x1x10pvtuDisplay.CubeAxesColor = [0.0, 0.0, 0.0]
nodes_1_1x1x10pvtuDisplay.ScalarOpacityUnitDistance = 0.054126587736527426

# reset view to fit data
renderView1.ResetCamera()

# create a new 'XML Unstructured Grid Reader'
# edgs_3d_quadratic_0vtu = XMLUnstructuredGridReader(FileName=[ \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/triangle/gradients_1st_order/geometry/edgs_3d_quadratic_0.vtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/triangle/gradients_2nd_order_c/geometry/edgs_3d_quadratic_0.vtu', \
# \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/union/gradients_1st_order/geometry/edgs_3d_quadratic_0.vtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/union/gradients_2nd_order_c/geometry/edgs_3d_quadratic_0.vtu', \
# \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/difference/gradients_1st_order/geometry/edgs_3d_quadratic_0.vtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/difference/gradients_2nd_order_c/geometry/edgs_3d_quadratic_0.vtu', \
# \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/three_flowers/gradients_1st_order/geometry/edgs_3d_quadratic_0.vtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d_test/three_flowers/gradients_2nd_order_c/geometry/edgs_3d_quadratic_0.vtu'
# ])

nodes_1_1x1 = XMLPartitionedUnstructuredGridReader(FileName=[
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.0.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.1.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.2.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.3.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.4.pvtu'
])

edgs_3d_quadratic_0vtu.CellArrayStatus = ['location', 'simplex', 'c0', 'c1']
edgs_3d_quadratic_0vtu.PointArrayStatus = ['location']

# show data in view
edgs_3d_quadratic_0vtuDisplay = Show(edgs_3d_quadratic_0vtu, renderView1)
# trace defaults for the display properties.
edgs_3d_quadratic_0vtuDisplay.AmbientColor = [0.0, 0.0, 0.0]
edgs_3d_quadratic_0vtuDisplay.ColorArrayName = [None, '']
edgs_3d_quadratic_0vtuDisplay.GlyphType = 'Arrow'
edgs_3d_quadratic_0vtuDisplay.CubeAxesColor = [0.0, 0.0, 0.0]
edgs_3d_quadratic_0vtuDisplay.ScalarOpacityUnitDistance = 0.02920933269576832

# create a new 'XML Unstructured Grid Reader'
# tris_3d_quadratic_0vtu = XMLUnstructuredGridReader(FileName=[ \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d/triangle/gradients_1st_order/geometry/tris_3d_quadratic_0.vtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d/triangle/gradients_2nd_order_c/geometry/tris_3d_quadratic_0.vtu', \
# \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d/union/gradients_1st_order/geometry/tris_3d_quadratic_0.vtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d/union/gradients_2nd_order_c/geometry/tris_3d_quadratic_0.vtu', \
# \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d/difference/gradients_1st_order/geometry/tris_3d_quadratic_0.vtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d/difference/gradients_2nd_order_c/geometry/tris_3d_quadratic_0.vtu', \
# \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d/three_flowers/gradients_1st_order/geometry/tris_3d_quadratic_0.vtu', \
# '/home/dbochkov/Outputs/paper_examples/poisson/3d/three_flowers/gradients_2nd_order_c/geometry/tris_3d_quadratic_0.vtu'
# ])
nodes_1_1x1 = XMLPartitionedUnstructuredGridReader(FileName=[
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.0.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.1.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.2.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.3.pvtu',
    '/home/faranak/CASL/workspace/simulations_output/poisson-nodes-mls-nick/vtu/nodes_5_1x1.4.pvtu'
])

tris_3d_quadratic_0vtu.CellArrayStatus = ['color', 'idx', 'simplex']
tris_3d_quadratic_0vtu.PointArrayStatus = ['scalars']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get color transfer function/color map for 'scalars'
scalarsLUT = GetColorTransferFunction('scalars')
scalarsLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 2.0, 0.865003, 0.865003, 0.865003, 4.0, 0.705882, 0.0156863, 0.14902]
scalarsLUT.ScalarRangeInitialized = 1.0

# show data in view
tris_3d_quadratic_0vtuDisplay = Show(tris_3d_quadratic_0vtu, renderView1)
# trace defaults for the display properties.
tris_3d_quadratic_0vtuDisplay.AmbientColor = [0.0, 0.0, 0.0]
tris_3d_quadratic_0vtuDisplay.ColorArrayName = ['POINTS', 'scalars']
tris_3d_quadratic_0vtuDisplay.LookupTable = scalarsLUT
tris_3d_quadratic_0vtuDisplay.GlyphType = 'Arrow'
tris_3d_quadratic_0vtuDisplay.CubeAxesColor = [0.0, 0.0, 0.0]
tris_3d_quadratic_0vtuDisplay.ScalarOpacityUnitDistance = 0.030460519581237435

# show color bar/color legend
tris_3d_quadratic_0vtuDisplay.SetScalarBarVisibility(renderView1, False)

# get opacity transfer function/opacity map for 'scalars'
scalarsPWF = GetOpacityTransferFunction('scalars')
scalarsPWF.Points = [0.0, 0.0, 0.5, 0.0, 4.0, 1.0, 0.5, 0.0]
scalarsPWF.ScalarRangeInitialized = 1

# create a new 'Threshold'
threshold1 = Threshold(Input=tris_3d_quadratic_0vtu)
threshold1.Scalars = ['POINTS', 'scalars']
threshold1.ThresholdRange = [0.0, 4.0]

# Properties modified on threshold1
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
threshold1Display.ScalarOpacityUnitDistance = 0.06891174239352654

# hide data in view
Hide(tris_3d_quadratic_0vtu, renderView1)

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, False)

# set scalar coloring
ColorBy(threshold1Display, ('CELLS', 'color'))

# rescale color and/or opacity maps used to include current data range
threshold1Display.RescaleTransferFunctionToDataRange(True)

# get color transfer function/color map for 'color'
colorLUT = GetColorTransferFunction('color')
colorLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 1.5, 0.865003, 0.865003, 0.865003, 3.0, 0.705882, 0.0156863, 0.14902]
colorLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'color'
colorPWF = GetOpacityTransferFunction('color')
colorPWF.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
colorPWF.ScalarRangeInitialized = 1

# Properties modified on threshold1Display
threshold1Display.Opacity = 0.5

# hide color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, False)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
colorLUT.ApplyPreset('blot', True)

# set active source
SetActiveSource(edgs_3d_quadratic_0vtu)

# create a new 'Threshold'
threshold2 = Threshold(Input=edgs_3d_quadratic_0vtu)
threshold2.Scalars = ['POINTS', 'location']
threshold2.ThresholdRange = [0.0, 4.0]

# Properties modified on threshold2
threshold2.Scalars = ['CELLS', 'location']
threshold2.ThresholdRange = [3.5, 4.0]

# show data in view
threshold2Display = Show(threshold2, renderView1)
# trace defaults for the display properties.
threshold2Display.AmbientColor = [0.0, 0.0, 0.0]
threshold2Display.ColorArrayName = [None, '']
threshold2Display.GlyphType = 'Arrow'
threshold2Display.CubeAxesColor = [0.0, 0.0, 0.0]

# hide data in view
Hide(edgs_3d_quadratic_0vtu, renderView1)

# Properties modified on threshold2Display
threshold2Display.Opacity = 0.5

# Properties modified on threshold2Display
threshold2Display.LineWidth = 2.0

# change representation type
threshold2Display.SetRepresentationType('Wireframe')

# set active source
SetActiveSource(nodes_1_1x1x10pvtu)

# change representation type
nodes_1_1x1x10pvtuDisplay.SetRepresentationType('Wireframe')

# change representation type
nodes_1_1x1x10pvtuDisplay.SetRepresentationType('Outline')

# Properties modified on nodes_1_1x1x10pvtuDisplay
nodes_1_1x1x10pvtuDisplay.Opacity = 0.5

# Properties modified on nodes_1_1x1x10pvtuDisplay
nodes_1_1x1x10pvtuDisplay.LineWidth = 2.0

# set active source
SetActiveSource(threshold2)

# hide data in view
Hide(threshold2, renderView1)

# set active source
SetActiveSource(threshold2)

# show data in view
threshold2Display = Show(threshold2, renderView1)

# Properties modified on threshold2
threshold2.ThresholdRange = [2.5, 3.0]

# create a new 'Box'
box1 = Box()

# Properties modified on box1
box1.XLength = 2.0
box1.YLength = 2.0
box1.ZLength = 2.0

# show data in view
box1Display = Show(box1, renderView1)
# trace defaults for the display properties.
box1Display.AmbientColor = [0.0, 0.0, 0.0]
box1Display.ColorArrayName = [None, '']
box1Display.GlyphType = 'Arrow'
box1Display.CubeAxesColor = [0.0, 0.0, 0.0]

# Properties modified on box1Display
box1Display.Opacity = 0.2

# set active view
SetActiveView(renderView2)

# set active source
SetActiveSource(threshold2)

# show data in view
threshold2Display_1 = Show(threshold2, renderView2)
# trace defaults for the display properties.
threshold2Display_1.AmbientColor = [0.0, 0.0, 0.0]
threshold2Display_1.ColorArrayName = [None, '']
threshold2Display_1.GlyphType = 'Arrow'
threshold2Display_1.CubeAxesColor = [0.0, 0.0, 0.0]
threshold2Display_1.ScalarOpacityUnitDistance = 0.23833873994456786

# reset view to fit data
renderView2.ResetCamera()

# change representation type
threshold2Display_1.SetRepresentationType('Wireframe')

# Properties modified on threshold2Display_1
threshold2Display_1.Opacity = 0.5

# Properties modified on threshold2Display_1
threshold2Display_1.LineWidth = 2.0

# set active source
SetActiveSource(threshold1)

# show data in view
threshold1Display_1 = Show(threshold1, renderView2)
# trace defaults for the display properties.
threshold1Display_1.AmbientColor = [0.0, 0.0, 0.0]
threshold1Display_1.ColorArrayName = ['POINTS', 'scalars']
threshold1Display_1.LookupTable = scalarsLUT
threshold1Display_1.GlyphType = 'Arrow'
threshold1Display_1.CubeAxesColor = [0.0, 0.0, 0.0]
threshold1Display_1.ScalarOpacityUnitDistance = 0.06891174239352654

# show color bar/color legend
threshold1Display_1.SetScalarBarVisibility(renderView2, False)

# Properties modified on threshold1Display_1
threshold1Display_1.Opacity = 0.1

# turn off scalar coloring
ColorBy(threshold1Display_1, None)

# set active source
SetActiveSource(nodes_1_1x1x10pvtu)

# create a new 'Slice'
slice1 = Slice(Input=nodes_1_1x1x10pvtu)
slice1.SliceType = 'Plane'
slice1.SliceOffsetValues = [0.0]

# Properties modified on slice1.SliceType
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# show data in view
slice1Display = Show(slice1, renderView2)
# trace defaults for the display properties.
slice1Display.AmbientColor = [0.0, 0.0, 0.0]
slice1Display.ColorArrayName = [None, '']
slice1Display.GlyphType = 'Arrow'
slice1Display.CubeAxesColor = [0.0, 0.0, 0.0]

# set active source
SetActiveSource(nodes_1_1x1x10pvtu)

# set active source
SetActiveSource(slice1)

# set active source
SetActiveSource(nodes_1_1x1x10pvtu)

# create a new 'Slice'
slice2 = Slice(Input=nodes_1_1x1x10pvtu)
slice2.SliceType = 'Plane'
slice2.SliceOffsetValues = [0.0]

# Properties modified on slice2.SliceType
slice2.SliceType.Origin = [0.0, 0.0, 0.5]
slice2.SliceType.Normal = [0.0, 0.0, 1.0]

# show data in view
slice2Display = Show(slice2, renderView2)
# trace defaults for the display properties.
slice2Display.AmbientColor = [0.0, 0.0, 0.0]
slice2Display.ColorArrayName = [None, '']
slice2Display.GlyphType = 'Arrow'
slice2Display.CubeAxesColor = [0.0, 0.0, 0.0]

# set active source
SetActiveSource(nodes_1_1x1x10pvtu)

# create a new 'Slice'
slice3 = Slice(Input=nodes_1_1x1x10pvtu)
slice3.SliceType = 'Plane'
slice3.SliceOffsetValues = [0.0]

# Properties modified on slice3.SliceType
slice3.SliceType.Origin = [0.0, 0.0, -0.5]
slice3.SliceType.Normal = [0.0, 0.0, 1.0]

# show data in view
slice3Display = Show(slice3, renderView2)
# trace defaults for the display properties.
slice3Display.AmbientColor = [0.0, 0.0, 0.0]
slice3Display.ColorArrayName = [None, '']
slice3Display.GlyphType = 'Arrow'
slice3Display.CubeAxesColor = [0.0, 0.0, 0.0]

# set active source
SetActiveSource(slice1)

# create a new 'Clip'
clip1 = Clip(Input=slice1)
clip1.ClipType = 'Plane'
clip1.Scalars = ['POINTS', 'error_dd']
clip1.Value = 0.4058320403459574

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip1)

# Properties modified on clip1
clip1.ClipType = 'Scalar'
clip1.Scalars = ['POINTS', 'phi']
clip1.Value = 0.0

# show data in view
clip1Display = Show(clip1, renderView2)
# trace defaults for the display properties.
clip1Display.AmbientColor = [0.0, 0.0, 0.0]
clip1Display.ColorArrayName = [None, '']
clip1Display.GlyphType = 'Arrow'
clip1Display.CubeAxesColor = [0.0, 0.0, 0.0]
clip1Display.ScalarOpacityUnitDistance = 0.15629077729024385

# hide data in view
Hide(slice1, renderView2)

# set active source
SetActiveSource(slice1)

# create a new 'Clip'
clip2 = Clip(Input=slice1)
clip2.ClipType = 'Plane'
clip2.Scalars = ['POINTS', 'error_dd']
clip2.Value = 0.4058320403459574

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip2)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip2)

# Properties modified on clip2
clip2.ClipType = 'Scalar'
clip2.Scalars = ['POINTS', 'phi']
clip2.Value = 0.0
clip2.InsideOut = 1

# show data in view
clip2Display = Show(clip2, renderView2)
# trace defaults for the display properties.
clip2Display.AmbientColor = [0.0, 0.0, 0.0]
clip2Display.ColorArrayName = [None, '']
clip2Display.GlyphType = 'Arrow'
clip2Display.CubeAxesColor = [0.0, 0.0, 0.0]
clip2Display.ScalarOpacityUnitDistance = 0.1563402902078671

# hide data in view
Hide(slice1, renderView2)

# set active source
SetActiveSource(slice2)

# create a new 'Clip'
clip3 = Clip(Input=slice2)
clip3.ClipType = 'Plane'
clip3.Scalars = ['POINTS', 'error_dd']
clip3.Value = 0.5681673152959172

# init the 'Plane' selected for 'ClipType'
clip3.ClipType.Origin = [0.0, 0.0, 0.5]

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip3)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip3)

# Properties modified on clip3
clip3.ClipType = 'Scalar'
clip3.Scalars = ['POINTS', 'phi']
clip3.Value = 0.0

# show data in view
clip3Display = Show(clip3, renderView2)
# trace defaults for the display properties.
clip3Display.AmbientColor = [0.0, 0.0, 0.0]
clip3Display.ColorArrayName = [None, '']
clip3Display.GlyphType = 'Arrow'
clip3Display.CubeAxesColor = [0.0, 0.0, 0.0]
clip3Display.ScalarOpacityUnitDistance = 0.14679646083248568

# hide data in view
Hide(slice2, renderView2)

# set active source
SetActiveSource(slice2)

# create a new 'Clip'
clip4 = Clip(Input=slice2)
clip4.ClipType = 'Plane'
clip4.Scalars = ['POINTS', 'error_dd']
clip4.Value = 0.5681673152959172

# init the 'Plane' selected for 'ClipType'
clip4.ClipType.Origin = [0.0, 0.0, 0.5]

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip4)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip4)

# Properties modified on clip4
clip4.ClipType = 'Scalar'
clip4.Scalars = ['POINTS', 'phi']
clip4.Value = 0.0
clip4.InsideOut = 1

# show data in view
clip4Display = Show(clip4, renderView2)
# trace defaults for the display properties.
clip4Display.AmbientColor = [0.0, 0.0, 0.0]
clip4Display.ColorArrayName = [None, '']
clip4Display.GlyphType = 'Arrow'
clip4Display.CubeAxesColor = [0.0, 0.0, 0.0]
clip4Display.ScalarOpacityUnitDistance = 0.18669487123288506

# hide data in view
Hide(slice2, renderView2)

# set active source
SetActiveSource(slice3)

# create a new 'Clip'
clip5 = Clip(Input=slice3)
clip5.ClipType = 'Plane'
clip5.Scalars = ['POINTS', 'error_dd']
clip5.Value = 0.31952640387781633

# init the 'Plane' selected for 'ClipType'
clip5.ClipType.Origin = [0.0, 0.0, -0.5]

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip5)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip5)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip5)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip5)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip5)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip5)

# Properties modified on clip5
clip5.ClipType = 'Scalar'
clip5.Scalars = ['POINTS', 'phi']
clip5.Value = 0.0

# show data in view
clip5Display = Show(clip5, renderView2)
# trace defaults for the display properties.
clip5Display.AmbientColor = [0.0, 0.0, 0.0]
clip5Display.ColorArrayName = [None, '']
clip5Display.GlyphType = 'Arrow'
clip5Display.CubeAxesColor = [0.0, 0.0, 0.0]
clip5Display.ScalarOpacityUnitDistance = 0.15043822982030491

# hide data in view
Hide(slice3, renderView2)

# set active source
SetActiveSource(slice3)

# create a new 'Clip'
clip6 = Clip(Input=slice3)
clip6.ClipType = 'Plane'
clip6.Scalars = ['POINTS', 'error_dd']
clip6.Value = 0.31952640387781633

# init the 'Plane' selected for 'ClipType'
clip6.ClipType.Origin = [0.0, 0.0, -0.5]

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip6)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip6)

# Properties modified on clip6
clip6.ClipType = 'Scalar'
clip6.Scalars = ['POINTS', 'phi']
clip6.Value = 0.0
clip6.InsideOut = 1

# show data in view
clip6Display = Show(clip6, renderView2)
# trace defaults for the display properties.
clip6Display.AmbientColor = [0.0, 0.0, 0.0]
clip6Display.ColorArrayName = [None, '']
clip6Display.GlyphType = 'Arrow'
clip6Display.CubeAxesColor = [0.0, 0.0, 0.0]
clip6Display.ScalarOpacityUnitDistance = 0.18080463694142054

# hide data in view
Hide(slice3, renderView2)

# set active source
SetActiveSource(slice1)

# create a new 'Contour'
contour1 = Contour(Input=slice1)
contour1.ContourBy = ['POINTS', 'error_dd']
contour1.Isosurfaces = [0.4058320403459574]
contour1.PointMergeMethod = 'Uniform Binning'

# set active source
SetActiveSource(slice2)

# set active source
SetActiveSource(contour1)

# Properties modified on contour1
contour1.ContourBy = ['POINTS', 'phi']
contour1.Isosurfaces = [0.0]

# show data in view
contour1Display = Show(contour1, renderView2)
# trace defaults for the display properties.
contour1Display.AmbientColor = [0.0, 0.0, 0.0]
contour1Display.ColorArrayName = [None, '']
contour1Display.GlyphType = 'Arrow'
contour1Display.CubeAxesColor = [0.0, 0.0, 0.0]

# hide data in view
Hide(slice1, renderView2)

# set active source
SetActiveSource(slice2)

# create a new 'Contour'
contour2 = Contour(Input=slice2)
contour2.ContourBy = ['POINTS', 'error_dd']
contour2.Isosurfaces = [0.5681673152959172]
contour2.PointMergeMethod = 'Uniform Binning'

# Properties modified on contour2
contour2.ContourBy = ['POINTS', 'phi']
contour2.Isosurfaces = [0.0]

# show data in view
contour2Display = Show(contour2, renderView2)
# trace defaults for the display properties.
contour2Display.AmbientColor = [0.0, 0.0, 0.0]
contour2Display.ColorArrayName = [None, '']
contour2Display.GlyphType = 'Arrow'
contour2Display.CubeAxesColor = [0.0, 0.0, 0.0]

# hide data in view
Hide(slice2, renderView2)

# set active source
SetActiveSource(slice2)

# create a new 'Contour'
contour3 = Contour(Input=slice2)
contour3.ContourBy = ['POINTS', 'error_dd']
contour3.Isosurfaces = [0.5681673152959172]
contour3.PointMergeMethod = 'Uniform Binning'

# Properties modified on contour3
contour3.ContourBy = ['POINTS', 'phi']
contour3.Isosurfaces = [0.0]

# show data in view
contour3Display = Show(contour3, renderView2)
# trace defaults for the display properties.
contour3Display.AmbientColor = [0.0, 0.0, 0.0]
contour3Display.ColorArrayName = [None, '']
contour3Display.GlyphType = 'Arrow'
contour3Display.CubeAxesColor = [0.0, 0.0, 0.0]

# hide data in view
Hide(slice2, renderView2)

# set active source
SetActiveSource(slice2)

# hide data in view
Hide(contour3, renderView2)

# show data in view
slice2Display = Show(slice2, renderView2)

# destroy contour3
Delete(contour3)
del contour3

# set active source
SetActiveSource(slice2)

# set active source
SetActiveSource(slice3)

# create a new 'Contour'
contour3 = Contour(Input=slice3)
contour3.ContourBy = ['POINTS', 'error_dd']
contour3.Isosurfaces = [0.31952640387781633]
contour3.PointMergeMethod = 'Uniform Binning'

# Properties modified on contour3
contour3.ContourBy = ['POINTS', 'phi']
contour3.Isosurfaces = [0.0]

# show data in view
contour3Display = Show(contour3, renderView2)
# trace defaults for the display properties.
contour3Display.AmbientColor = [0.0, 0.0, 0.0]
contour3Display.ColorArrayName = [None, '']
contour3Display.GlyphType = 'Arrow'
contour3Display.CubeAxesColor = [0.0, 0.0, 0.0]

# hide data in view
Hide(slice3, renderView2)

# set active source
SetActiveSource(clip2)

# create a new 'Warp By Scalar'
warpByScalar1 = WarpByScalar(Input=clip2)
warpByScalar1.Scalars = ['POINTS', 'error_dd']

# Properties modified on warpByScalar1
warpByScalar1.Scalars = ['POINTS', 'sol_ex']
warpByScalar1.ScaleFactor = 0.3

# show data in view
warpByScalar1Display = Show(warpByScalar1, renderView2)
# trace defaults for the display properties.
warpByScalar1Display.AmbientColor = [0.0, 0.0, 0.0]
warpByScalar1Display.ColorArrayName = [None, '']
warpByScalar1Display.GlyphType = 'Arrow'
warpByScalar1Display.CubeAxesColor = [0.0, 0.0, 0.0]
warpByScalar1Display.ScalarOpacityUnitDistance = 0.15941873548066918

# hide data in view
Hide(clip2, renderView2)

# set active source
SetActiveSource(clip4)

# create a new 'Warp By Scalar'
warpByScalar2 = WarpByScalar(Input=clip4)
warpByScalar2.Scalars = ['POINTS', 'error_dd']

# Properties modified on warpByScalar2
warpByScalar2.Scalars = ['POINTS', 'sol_ex']
warpByScalar2.ScaleFactor = 0.3

# show data in view
warpByScalar2Display = Show(warpByScalar2, renderView2)
# trace defaults for the display properties.
warpByScalar2Display.AmbientColor = [0.0, 0.0, 0.0]
warpByScalar2Display.ColorArrayName = [None, '']
warpByScalar2Display.GlyphType = 'Arrow'
warpByScalar2Display.CubeAxesColor = [0.0, 0.0, 0.0]
warpByScalar2Display.ScalarOpacityUnitDistance = 0.19599684071327753

# hide data in view
Hide(clip4, renderView2)

# set active source
SetActiveSource(clip6)

# create a new 'Warp By Scalar'
warpByScalar3 = WarpByScalar(Input=clip6)
warpByScalar3.Scalars = ['POINTS', 'error_dd']

# Properties modified on warpByScalar3
warpByScalar3.Scalars = ['POINTS', 'sol_ex']
warpByScalar3.ScaleFactor = 0.3

# show data in view
warpByScalar3Display = Show(warpByScalar3, renderView2)
# trace defaults for the display properties.
warpByScalar3Display.AmbientColor = [0.0, 0.0, 0.0]
warpByScalar3Display.ColorArrayName = [None, '']
warpByScalar3Display.GlyphType = 'Arrow'
warpByScalar3Display.CubeAxesColor = [0.0, 0.0, 0.0]
warpByScalar3Display.ScalarOpacityUnitDistance = 0.18170497407616523

# hide data in view
Hide(clip6, renderView2)

# set active source
SetActiveSource(slice1)

# show data in view
slice1Display = Show(slice1, renderView2)

# change representation type
slice1Display.SetRepresentationType('Outline')

# Properties modified on slice1Display
slice1Display.Opacity = 0.5

# Properties modified on slice1Display
slice1Display.LineWidth = 2.0

# set active source
SetActiveSource(slice2)

# change representation type
slice2Display.SetRepresentationType('Outline')

# Properties modified on slice2Display
slice2Display.Opacity = 0.5

# Properties modified on slice2Display
slice2Display.LineWidth = 2.0

# set active source
SetActiveSource(slice3)

# change representation type
slice3Display.SetRepresentationType('Outline')

# set active source
SetActiveSource(slice3)

# show data in view
slice3Display = Show(slice3, renderView2)

# Properties modified on slice3Display
slice3Display.Opacity = 0.5

# Properties modified on slice3Display
slice3Display.LineWidth = 2.0

# set active source
SetActiveSource(contour1)

# change representation type
contour1Display.SetRepresentationType('Wireframe')

# Properties modified on contour1Display
contour1Display.Opacity = 0.5

# Properties modified on contour1Display
contour1Display.LineWidth = 2.0

# set active source
SetActiveSource(contour2)

# change representation type
contour2Display.SetRepresentationType('Wireframe')

# Properties modified on contour2Display
contour2Display.Opacity = 0.5

# Properties modified on contour2Display
contour2Display.LineWidth = 2.0

# set active source
SetActiveSource(contour3)

# change representation type
contour3Display.SetRepresentationType('Wireframe')

# Properties modified on contour3Display
contour3Display.Opacity = 0.5

# Properties modified on contour3Display
contour3Display.LineWidth = 2.0

# set active source
SetActiveSource(warpByScalar1)

# set scalar coloring
ColorBy(warpByScalar1Display, ('POINTS', 'sol_ex'))

# rescale color and/or opacity maps used to include current data range
warpByScalar1Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
warpByScalar1Display.SetScalarBarVisibility(renderView2, False)

# get color transfer function/color map for 'solex'
solexLUT = GetColorTransferFunction('solex')
solexLUT.RGBPoints = [-0.7181719699721228, 0.231373, 0.298039, 0.752941, -0.007238783586908659, 0.865003, 0.865003, 0.865003, 0.7036944027983055, 0.705882, 0.0156863, 0.14902]
solexLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'solex'
solexPWF = GetOpacityTransferFunction('solex')
solexPWF.Points = [-0.7181719699721228, 0.0, 0.5, 0.0, 0.7036944027983055, 1.0, 0.5, 0.0]
solexPWF.ScalarRangeInitialized = 1

# hide color bar/color legend
warpByScalar1Display.SetScalarBarVisibility(renderView2, False)

# set active source
SetActiveSource(warpByScalar2)

# set scalar coloring
ColorBy(warpByScalar2Display, ('POINTS', 'sol_ex'))

# rescale color and/or opacity maps used to include current data range
warpByScalar2Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
warpByScalar2Display.SetScalarBarVisibility(renderView2, False)

# set active source
SetActiveSource(warpByScalar3)

# set scalar coloring
ColorBy(warpByScalar3Display, ('POINTS', 'sol_ex'))

# rescale color and/or opacity maps used to include current data range
warpByScalar3Display.RescaleTransferFunctionToDataRange(True)

# hide color bar/color legend
warpByScalar3Display.SetScalarBarVisibility(renderView2, False)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
solexLUT.ApplyPreset('Plasma (matplotlib)', True)

# set active source
SetActiveSource(nodes_1_1x1x10pvtu)

# set active view
SetActiveView(renderView3)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice3)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice3)

# set active source
SetActiveSource(threshold2)

# show data in view
threshold2Display_2 = Show(threshold2, renderView3)
# trace defaults for the display properties.
threshold2Display_2.AmbientColor = [0.0, 0.0, 0.0]
threshold2Display_2.ColorArrayName = [None, '']
threshold2Display_2.GlyphType = 'Arrow'
threshold2Display_2.CubeAxesColor = [0.0, 0.0, 0.0]
threshold2Display_2.ScalarOpacityUnitDistance = 0.23833873994456786

# reset view to fit data
renderView3.ResetCamera()

# change representation type
threshold2Display_2.SetRepresentationType('Wireframe')

# Properties modified on threshold2Display_2
threshold2Display_2.Opacity = 0.5

# Properties modified on threshold2Display_2
threshold2Display_2.LineWidth = 2.0

# set active source
SetActiveSource(threshold1)

# set active source
SetActiveSource(threshold1)

# show data in view
threshold1Display_2 = Show(threshold1, renderView3)
# trace defaults for the display properties.
threshold1Display_2.AmbientColor = [0.0, 0.0, 0.0]
threshold1Display_2.ColorArrayName = ['POINTS', 'scalars']
threshold1Display_2.LookupTable = scalarsLUT
threshold1Display_2.GlyphType = 'Arrow'
threshold1Display_2.CubeAxesColor = [0.0, 0.0, 0.0]
threshold1Display_2.ScalarOpacityUnitDistance = 0.06891174239352654

# show color bar/color legend
threshold1Display_2.SetScalarBarVisibility(renderView3, False)

# Properties modified on threshold1Display_2
threshold1Display_2.Opacity = 0.1

# turn off scalar coloring
ColorBy(threshold1Display_2, None)

# set active source
SetActiveSource(slice1)

# create a new 'Warp By Scalar'
warpByScalar4 = WarpByScalar(Input=slice1)
warpByScalar4.Scalars = ['POINTS', 'error_dd']

# Properties modified on warpByScalar4
warpByScalar4.Scalars = ['POINTS', 'error_sl']
warpByScalar4.ScaleFactor = 1000.0

# show data in view
warpByScalar4Display = Show(warpByScalar4, renderView3)
# trace defaults for the display properties.
warpByScalar4Display.AmbientColor = [0.0, 0.0, 0.0]
warpByScalar4Display.ColorArrayName = [None, '']
warpByScalar4Display.GlyphType = 'Arrow'
warpByScalar4Display.CubeAxesColor = [0.0, 0.0, 0.0]

# set scalar coloring
ColorBy(warpByScalar4Display, ('POINTS', 'error_sl'))

# rescale color and/or opacity maps used to include current data range
warpByScalar4Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
warpByScalar4Display.SetScalarBarVisibility(renderView3, False)

# get color transfer function/color map for 'errorsl'
errorslLUT = GetColorTransferFunction('errorsl')
errorslLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 0.00025067050115151757, 0.865003, 0.865003, 0.865003, 0.0005013410023030351, 0.705882, 0.0156863, 0.14902]
errorslLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'errorsl'
errorslPWF = GetOpacityTransferFunction('errorsl')
errorslPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.0005013410023030351, 1.0, 0.5, 0.0]
errorslPWF.ScalarRangeInitialized = 1

# hide color bar/color legend
warpByScalar4Display.SetScalarBarVisibility(renderView3, False)

# set active source
SetActiveSource(slice2)

# create a new 'Warp By Scalar'
warpByScalar5 = WarpByScalar(Input=slice2)
warpByScalar5.Scalars = ['POINTS', 'error_dd']

# Properties modified on warpByScalar5
warpByScalar5.Scalars = ['POINTS', 'error_sl']
warpByScalar5.ScaleFactor = 1000.0

# show data in view
warpByScalar5Display = Show(warpByScalar5, renderView3)
# trace defaults for the display properties.
warpByScalar5Display.AmbientColor = [0.0, 0.0, 0.0]
warpByScalar5Display.ColorArrayName = [None, '']
warpByScalar5Display.GlyphType = 'Arrow'
warpByScalar5Display.CubeAxesColor = [0.0, 0.0, 0.0]

# set scalar coloring
ColorBy(warpByScalar5Display, ('POINTS', 'error_sl'))

# rescale color and/or opacity maps used to include current data range
warpByScalar5Display.RescaleTransferFunctionToDataRange(True)

# hide color bar/color legend
warpByScalar5Display.SetScalarBarVisibility(renderView3, False)

# set active source
SetActiveSource(slice3)

# create a new 'Warp By Scalar'
warpByScalar6 = WarpByScalar(Input=slice3)
warpByScalar6.Scalars = ['POINTS', 'error_dd']

# Properties modified on warpByScalar6
warpByScalar6.Scalars = ['POINTS', 'error_sl']
warpByScalar6.ScaleFactor = 1000.0

# show data in view
warpByScalar6Display = Show(warpByScalar6, renderView3)
# trace defaults for the display properties.
warpByScalar6Display.AmbientColor = [0.0, 0.0, 0.0]
warpByScalar6Display.ColorArrayName = [None, '']
warpByScalar6Display.GlyphType = 'Arrow'
warpByScalar6Display.CubeAxesColor = [0.0, 0.0, 0.0]

# set scalar coloring
ColorBy(warpByScalar6Display, ('POINTS', 'error_sl'))

# rescale color and/or opacity maps used to include current data range
warpByScalar6Display.RescaleTransferFunctionToDataRange(True)

# hide color bar/color legend
warpByScalar6Display.SetScalarBarVisibility(renderView3, False)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
errorslLUT.ApplyPreset('Green Linear (9_17f)', True)

# set active source
SetActiveSource(nodes_1_1x1x10pvtu)

# set active view
SetActiveView(renderView2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice3)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice3)

# set active source
SetActiveSource(warpByScalar1)

# set active view
SetActiveView(renderView3)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice3)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice3)

# set active source
SetActiveSource(slice1)

# show data in view
slice1Display_1 = Show(slice1, renderView3)
# trace defaults for the display properties.
slice1Display_1.AmbientColor = [0.0, 0.0, 0.0]
slice1Display_1.ColorArrayName = [None, '']
slice1Display_1.GlyphType = 'Arrow'
slice1Display_1.CubeAxesColor = [0.0, 0.0, 0.0]

# change representation type
slice1Display_1.SetRepresentationType('Outline')

# Properties modified on slice1Display_1
slice1Display_1.Opacity = 0.5

# Properties modified on slice1Display_1
slice1Display_1.LineWidth = 2.0

# set active source
SetActiveSource(slice2)

# show data in view
slice2Display_1 = Show(slice2, renderView3)
# trace defaults for the display properties.
slice2Display_1.AmbientColor = [0.0, 0.0, 0.0]
slice2Display_1.ColorArrayName = [None, '']
slice2Display_1.GlyphType = 'Arrow'
slice2Display_1.CubeAxesColor = [0.0, 0.0, 0.0]

# change representation type
slice2Display_1.SetRepresentationType('Outline')

# Properties modified on slice2Display_1
slice2Display_1.Opacity = 0.5

# Properties modified on slice2Display_1
slice2Display_1.LineWidth = 2.0

# set active source
SetActiveSource(slice3)

# set active source
SetActiveSource(slice3)

# show data in view
slice3Display_1 = Show(slice3, renderView3)
# trace defaults for the display properties.
slice3Display_1.AmbientColor = [0.0, 0.0, 0.0]
slice3Display_1.ColorArrayName = [None, '']
slice3Display_1.GlyphType = 'Arrow'
slice3Display_1.CubeAxesColor = [0.0, 0.0, 0.0]

# change representation type
slice3Display_1.SetRepresentationType('Outline')

# Properties modified on slice3Display_1
slice3Display_1.Opacity = 0.5

# Properties modified on slice3Display_1
slice3Display_1.LineWidth = 2.0

# set active source
SetActiveSource(nodes_1_1x1x10pvtu)

# set active view
SetActiveView(renderView2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice3)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice3)

# set active view
SetActiveView(renderView1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice3)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice3)

# reset view to fit data
renderView1.ResetCamera()

# set active view
SetActiveView(renderView2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice3)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice3)

# reset view to fit data
renderView2.ResetCamera()

# set active view
SetActiveView(renderView3)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice3)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice3)

# reset view to fit data
renderView3.ResetCamera()

#### saving camera placements for all active views

# current camera placement for renderView3
renderView3.CameraPosition = [-6.397868539714982, -2.1112810584233523, 1.4078464643284996]
renderView3.CameraFocalPoint = [0.0, 0.0, 0.16491848230361938]
renderView3.CameraViewUp = [0.165291916890201, 0.0779074618069655, 0.983162758451295]
renderView3.CameraParallelScale = 1.7746257669161483

# current camera placement for renderView4

# current camera placement for renderView1
renderView1.CameraPosition = [-6.32342034508276, -2.0867133352535, 1.2284647676363356]
renderView1.CameraViewUp = [0.165291916890201, 0.0779074618069655, 0.983162758451295]
renderView1.CameraParallelScale = 1.7539755013355718

# current camera placement for renderView2
renderView2.CameraPosition = [-6.0529866073902365, -1.99747085951262, 1.16592702061008]
renderView2.CameraFocalPoint = [0.0, 0.0, -0.009999990463256836]
renderView2.CameraViewUp = [0.165291916890201, 0.0779074618069655, 0.983162758451295]
renderView2.CameraParallelScale = 1.6791617391740838

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
