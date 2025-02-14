#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Partitioned Unstructured Grid Reader'
nodes_1_1x1 = XMLPartitionedUnstructuredGridReader(FileName=[\
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/triangle/gradients_1st_order/vtu/nodes_1_1x1.4.pvtu', \
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/triangle/gradients_2nd_order_b/vtu/nodes_1_1x1.4.pvtu', \
\
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/union/gradients_1st_order/vtu/nodes_1_1x1.4.pvtu', \
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/union/gradients_2nd_order_b/vtu/nodes_1_1x1.4.pvtu', \
\
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/difference/gradients_1st_order/vtu/nodes_1_1x1.4.pvtu', \
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/difference/gradients_2nd_order_b/vtu/nodes_1_1x1.4.pvtu', \
\
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/three_flowers/gradients_1st_order/vtu/nodes_1_1x1.4.pvtu', \
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/three_flowers/gradients_2nd_order_b/vtu/nodes_1_1x1.4.pvtu'])
nodes_1_1x1.CellArrayStatus = ['proc_rank', 'tree_idx', 'leaf_level']
nodes_1_1x1.PointArrayStatus = ['phi', 'phi_smooth', 'sol', 'sol_ex', 'error_sl', 'error_tr', 'error_gr', 'error_ex', 'error_dd', 'mask', 'volumes']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1534, 791]

# show data in view
nodes_1_1x1Display = Show(nodes_1_1x1, renderView1)
# trace defaults for the display properties.
nodes_1_1x1Display.ColorArrayName = [None, '']
nodes_1_1x1Display.GlyphType = 'Arrow'
nodes_1_1x1Display.ScalarOpacityUnitDistance = 0.4454493590701697

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.0, 10000.0]
renderView1.Background = [1.0, 1.0, 1.0]

# create a new 'XML Unstructured Grid Reader'
edgs_2d_quadratic_ = XMLUnstructuredGridReader(FileName=[ \
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/triangle/gradients_1st_order/geometry/edgs_2d_quadratic_4.vtu', \
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/triangle/gradients_2nd_order_b/geometry/edgs_2d_quadratic_4.vtu', \
\
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/union/gradients_1st_order/geometry/edgs_2d_quadratic_4.vtu', \
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/union/gradients_2nd_order_b/geometry/edgs_2d_quadratic_4.vtu', \
\
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/difference/gradients_1st_order/geometry/edgs_2d_quadratic_4.vtu', \
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/difference/gradients_2nd_order_b/geometry/edgs_2d_quadratic_4.vtu', \
\
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/three_flowers/gradients_1st_order/geometry/edgs_2d_quadratic_4.vtu', \
'/home/dbochkov/Outputs/paper_examples/poisson/2d_test/three_flowers/gradients_2nd_order_b/geometry/edgs_2d_quadratic_4.vtu'])
edgs_2d_quadratic_.CellArrayStatus = ['location', 'c0']
edgs_2d_quadratic_.PointArrayStatus = ['location']

# show data in view
edgs_2d_quadratic_Display = Show(edgs_2d_quadratic_, renderView1)
# trace defaults for the display properties.
edgs_2d_quadratic_Display.ColorArrayName = [None, '']
edgs_2d_quadratic_Display.GlyphType = 'Arrow'
edgs_2d_quadratic_Display.ScalarOpacityUnitDistance = 0.2848459884390608

# create a new 'Threshold'
threshold1 = Threshold(Input=edgs_2d_quadratic_)
threshold1.Scalars = ['CELLS', 'c0']
threshold1.ThresholdRange = [-0.5, 10.0]

# show data in view
threshold1Display = Show(threshold1, renderView1)
# trace defaults for the display properties.
threshold1Display.ColorArrayName = [None, '']
threshold1Display.GlyphType = 'Arrow'
threshold1Display.ScalarOpacityUnitDistance = 0.5119972265761852

# hide data in view
Hide(edgs_2d_quadratic_, renderView1)

# set scalar coloring
ColorBy(threshold1Display, ('CELLS', 'c0'))

# rescale color and/or opacity maps used to include current data range
threshold1Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, False)

# get color transfer function/color map for 'c0'
c0LUT = GetColorTransferFunction('c0')
c0LUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 1.0, 0.865003, 0.865003, 0.865003, 2.0, 0.705882, 0.0156863, 0.14902]
c0LUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'c0'
c0PWF = GetOpacityTransferFunction('c0')
c0PWF.Points = [0.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]
c0PWF.ScalarRangeInitialized = 1

# Properties modified on threshold1Display
threshold1Display.LineWidth = 2.0

# set active source
SetActiveSource(nodes_1_1x1)

# create a new 'Clip'
clip1 = Clip(Input=nodes_1_1x1)
clip1.ClipType = 'Scalar'
clip1.Scalars = ['POINTS', 'phi']
clip1.Value = 0.0

# show data in view
clip1Display = Show(clip1, renderView1)
# trace defaults for the display properties.
clip1Display.ColorArrayName = [None, '']
clip1Display.GlyphType = 'Arrow'
clip1Display.ScalarOpacityUnitDistance = 0.4699584855687096

# hide data in view
Hide(nodes_1_1x1, renderView1)

# Properties modified on clip1
clip1.InsideOut = 1

# set active source
SetActiveSource(nodes_1_1x1)

# show data in view
nodes_1_1x1Display = Show(nodes_1_1x1, renderView1)

# change representation type
nodes_1_1x1Display.SetRepresentationType('Outline')

# Properties modified on nodes_1_1x1Display
nodes_1_1x1Display.LineWidth = 2.0

# Properties modified on nodes_1_1x1Display
nodes_1_1x1Display.Opacity = 0.5

# set active source
SetActiveSource(threshold1)

# rescale color and/or opacity maps used to exactly fit the current data range
threshold1Display.RescaleTransferFunctionToDataRange(False)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
c0LUT.ApplyPreset('Blue to Red Rainbow', True)

# set active source
SetActiveSource(clip1)

# change solid color
clip1Display.DiffuseColor = [1.0, 1.0, 0.4980392156862745]

# Properties modified on clip1Display
clip1Display.Specular = 1.0

# set active view
SetActiveView(None)

# set active source
SetActiveSource(nodes_1_1x1)

# create a new 'Warp By Scalar'
warpByScalar1 = WarpByScalar(Input=nodes_1_1x1)
warpByScalar1.Scalars = ['POINTS', 'error_sl']

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [957, 791]
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

# get layout
viewLayout2 = GetLayout()

# place view in the layout
viewLayout2.AssignView(0, renderView2)

# show data in view
warpByScalar1Display = Show(warpByScalar1, renderView2)
# trace defaults for the display properties.
warpByScalar1Display.AmbientColor = [0.0, 0.0, 0.0]
warpByScalar1Display.ColorArrayName = [None, '']
warpByScalar1Display.GlyphType = 'Arrow'
warpByScalar1Display.CubeAxesColor = [0.0, 0.0, 0.0]
warpByScalar1Display.ScalarOpacityUnitDistance = 0.4454500007727548

# reset view to fit data
renderView2.ResetCamera()

# Properties modified on warpByScalar1
warpByScalar1.ScaleFactor = 1000.0

# set scalar coloring
ColorBy(warpByScalar1Display, ('POINTS', 'error_sl'))

# rescale color and/or opacity maps used to include current data range
warpByScalar1Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
warpByScalar1Display.SetScalarBarVisibility(renderView2, False)

# get color transfer function/color map for 'errorsl'
errorslLUT = GetColorTransferFunction('errorsl')
errorslLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 0.0024004787953834372, 0.865003, 0.865003, 0.865003, 0.0048009575907668744, 0.705882, 0.0156863, 0.14902]
errorslLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'errorsl'
errorslPWF = GetOpacityTransferFunction('errorsl')
errorslPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.0048009575907668744, 1.0, 0.5, 0.0]
errorslPWF.ScalarRangeInitialized = 1

# rescale color and/or opacity maps used to exactly fit the current data range
warpByScalar1Display.RescaleTransferFunctionToDataRange(False)

# hide color bar/color legend
warpByScalar1Display.SetScalarBarVisibility(renderView2, False)

# set active view
SetActiveView(None)

# set active source
SetActiveSource(clip1)

# create a new 'Warp By Scalar'
warpByScalar2 = WarpByScalar(Input=clip1)
warpByScalar2.Scalars = ['POINTS', 'sol_ex']

# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.ViewSize = [957, 791]
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

# get layout
viewLayout3 = GetLayout()

# place view in the layout
viewLayout3.AssignView(0, renderView3)

# show data in view
warpByScalar2Display = Show(warpByScalar2, renderView3)
# trace defaults for the display properties.
warpByScalar2Display.AmbientColor = [0.0, 0.0, 0.0]
warpByScalar2Display.ColorArrayName = [None, '']
warpByScalar2Display.GlyphType = 'Arrow'
warpByScalar2Display.CubeAxesColor = [0.0, 0.0, 0.0]
warpByScalar2Display.ScalarOpacityUnitDistance = 0.5196672737911449

# reset view to fit data
renderView3.ResetCamera()

# set scalar coloring
ColorBy(warpByScalar2Display, ('POINTS', 'sol_ex'))

# rescale color and/or opacity maps used to include current data range
warpByScalar2Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
warpByScalar2Display.SetScalarBarVisibility(renderView3, True)

# get color transfer function/color map for 'solex'
solexLUT = GetColorTransferFunction('solex')
solexLUT.RGBPoints = [-0.7069375686632181, 0.231373, 0.298039, 0.752941, -0.07541102014450807, 0.865003, 0.865003, 0.865003, 0.556115528374202, 0.705882, 0.0156863, 0.14902]
solexLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'solex'
solexPWF = GetOpacityTransferFunction('solex')
solexPWF.Points = [-0.7069375686632181, 0.0, 0.5, 0.0, 0.556115528374202, 1.0, 0.5, 0.0]
solexPWF.ScalarRangeInitialized = 1

# set active source
SetActiveSource(nodes_1_1x1)

# create a new 'Clip'
clip2 = Clip(Input=nodes_1_1x1)
clip2.ClipType = 'Scalar'
clip2.Scalars = ['POINTS', 'phi']
clip2.Value = 0.0

# show data in view
clip2Display = Show(clip2, renderView3)
# trace defaults for the display properties.
clip2Display.AmbientColor = [0.0, 0.0, 0.0]
clip2Display.ColorArrayName = [None, '']
clip2Display.GlyphType = 'Arrow'
clip2Display.CubeAxesColor = [0.0, 0.0, 0.0]
clip2Display.ScalarOpacityUnitDistance = 0.4699584855687096

# Properties modified on clip2Display
clip2Display.Opacity = 0.5

# set active source
SetActiveSource(nodes_1_1x1)

# show data in view
nodes_1_1x1Display_1 = Show(nodes_1_1x1, renderView3)
# trace defaults for the display properties.
nodes_1_1x1Display_1.AmbientColor = [0.0, 0.0, 0.0]
nodes_1_1x1Display_1.ColorArrayName = [None, '']
nodes_1_1x1Display_1.GlyphType = 'Arrow'
nodes_1_1x1Display_1.CubeAxesColor = [0.0, 0.0, 0.0]
nodes_1_1x1Display_1.ScalarOpacityUnitDistance = 0.4454493590701697

# change representation type
nodes_1_1x1Display_1.SetRepresentationType('Outline')

# Properties modified on nodes_1_1x1Display_1
nodes_1_1x1Display_1.LineWidth = 2.0

# Properties modified on nodes_1_1x1Display_1
nodes_1_1x1Display_1.Opacity = 0.5

# set active source
SetActiveSource(threshold1)

# show data in view
threshold1Display_1 = Show(threshold1, renderView3)
# trace defaults for the display properties.
threshold1Display_1.AmbientColor = [0.0, 0.0, 0.0]
threshold1Display_1.ColorArrayName = [None, '']
threshold1Display_1.GlyphType = 'Arrow'
threshold1Display_1.CubeAxesColor = [0.0, 0.0, 0.0]
threshold1Display_1.ScalarOpacityUnitDistance = 0.5119972265761852

# set scalar coloring
ColorBy(threshold1Display_1, ('CELLS', 'c0'))

# rescale color and/or opacity maps used to include current data range
threshold1Display_1.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
threshold1Display_1.SetScalarBarVisibility(renderView3, False)

# Properties modified on threshold1Display_1
threshold1Display_1.LineWidth = 2.0

# hide color bar/color legend
threshold1Display_1.SetScalarBarVisibility(renderView3, False)

# set active source
SetActiveSource(warpByScalar2)

# hide color bar/color legend
warpByScalar2Display.SetScalarBarVisibility(renderView3, False)

# set active view
SetActiveView(renderView2)

# set active view
SetActiveView(renderView3)

#### saving camera placements for all active views

# current camera placement for renderView3
renderView3.CameraPosition = [-5.678566339844605, 3.9434105428844837, 2.266216669292239]
renderView3.CameraFocalPoint = [-0.04914581775665284, 0.020778983831405598, -0.07541102170944215]
renderView3.CameraViewUp = [0.27128563159936303, -0.1755982818683354, 0.9463452591378178]
renderView3.CameraParallelScale = 1.2816097052581197

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.0, 10000.0]
renderView1.CameraParallelScale = 1.4142135623730951

# current camera placement for renderView2
renderView2.CameraPosition = [-0.07334733180232554, -4.403521301984512, 3.2366102421409653]
renderView2.CameraFocalPoint = [0.0, 0.0, 0.0024004788137972355]
renderView2.CameraViewUp = [-0.0010788335859024543, 0.5919652773325264, 0.8059627451382099]
renderView2.CameraParallelScale = 1.4142155996518124

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
