#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
bialloy_6_1x10 = GetActiveSource()

# create a new 'Contour'
contour1 = Contour(Input=bialloy_6_1x10)
contour1.ContourBy = ['POINTS', 'bc_error']
contour1.Isosurfaces = [3.80495317330639e-06]
contour1.PointMergeMethod = 'Uniform Binning'

# Properties modified on contour1
contour1.ContourBy = ['POINTS', 'phi']
contour1.Isosurfaces = [0.0]

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [2183, 1151]

# show data in view
contour1Display = Show(contour1, renderView1)
# trace defaults for the display properties.
contour1Display.ColorArrayName = [None, '']
contour1Display.GlyphType = 'Arrow'

# hide data in view
Hide(bialloy_6_1x10, renderView1)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5, 0.5, 10000.0]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]
renderView1.CameraParallelScale = 0.7071067811865476

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).