# ##############################################################
try:
    from optparse import OptionParser
    from copy import *
    from time import *
    import os, sys
    import numpy as np
    from scipy.special import *

except ImportError as e:
    print '[Error]: ', e 

def smooth(x, b):
    a = 1.0/(2*erf(b/2.0));
    c = 0.5;
    xmin = x[0];
    xmax = x[-1];
    y = a*erf(b*(x/(xmax - xmin)-0.5))+c;
    
    return (xmax - xmin) * y + xmin;

# parse input options 
parser = OptionParser()
parser.add_option("-v", "--vtu",  action="store", dest="vtu",      type="string", default='/scratch/02032/mmirzade/proposal_g/n_256.N_16.2973336/solution.0_%04d.vtu', help="location of vtu files")
parser.add_option("-z", "--zoom", action="store", dest="zoom",     type="int",    default=56, help="proc id to zoom onto")
parser.add_option("-f", "--fps",  action="store", dest="fps",      type="int",    default=30, help="fps for the animation")
parser.add_option("-p", "--path", action="store", dest="path",     type="string", default="/scratch/02032/mmirzade/proposal_g/vis", help="path to the folder where the png images should be stored")
parser.add_option("-n", "--name", action="store", dest="img_name", type="string", default="img", help="prefix used to store the filename")

(opt, args) = parser.parse_args()
# ##############################################################

# Step 0) Prepare variables 
start = begin = clock();
from paraview.simple import *
print 'Loading paraview module and creating contours ...'

# Read the state file
paraview.simple._DisableFirstRenderCameraReset();
state = servermanager.LoadState('empty.pvsm');

z = opt.zoom;
pvtu = XMLPartitionedUnstructuredGridReader(FileName = '/scratch/02032/mmirzade/proposal_g/n_256.N_16.2973336/solution.0.pvtu');
vtu  = XMLUnstructuredGridReader(FileName = '/scratch/02032/mmirzade/proposal_g/n_256.N_16.2973336/solution.0_%04d.vtu' % z);

# Get the view
SetActiveView(GetRenderView());
view = GetActiveView();
view.ViewSize = [1280, 720];

# Create the surface 
Hide(pvtu); 
Hide(vtu);

c_pvtu  = Contour(pvtu, ContourBy='phi', Isosurfaces=0);
cr_pvtu = GetDisplayProperties(c_pvtu);
cr_pvtu.DiffuseColor = [0, 0.781, 0.574];
cr_pvtu.BackfaceRepresentation = 'Surface';
cr_pvtu.Specular = 0.5;

c_vtu = Contour(vtu, ContourBy='phi', Isosurfaces=0);
Hide(c_vtu);

outline = Outline(pvtu);
outline_r = Show(outline);
outline_r.DiffuseColor = [0,0,0];

Render();

# set animation fps;
fps = opt.fps;
mpisize = 256;

# set image filenames# create 
filename = opt.path + '/' + opt.img_name + '_s%02d_%04d.png';

# set parameters for camera movement
f0 = np.array([1.5, 1.5, 1.5]);
p0 = np.array([6.5, 5.5, 6.5]);

view.CameraFocalPoint = f0;
view.CameraPosition = p0;
view.CameraViewAngle = 30;
view.CameraViewUp = [0, 1, 0];

Render();
print '... done in %.2f sec' % (clock() - start)
# ##############################################################

# Step 1) change color type for the main object
start = clock();
sec = 1;
count = 0;

print 'Step 1:'
# fade out
cr_pvtu.ColorArrayName = '';
for d in np.linspace(0, 1, fps):
    print ' Fade out ... [%3d%%]\r' % int(100*d),
    sys.stdout.flush();

    cr_pvtu.Opacity = 1 - d;
    cr_pvtu.BackfaceOpacity = 1 - d;
    
    Render();
    WriteImage(filename % (sec, count)); count += 1;
print ' Fade out ... done!   '

# change color
rgb_points = np.linspace(0, mpisize-1, 7);
table = GetLookupTableForArray( "proc_rank", 1, NanColor=[0.5, 0.0, 0.0], 
    RGBPoints = [
    rgb_points[0], 0.0, 0.0, 0.5, 
    rgb_points[1], 0.0, 0.0, 1.0, 
    rgb_points[2], 0.0, 1.0, 1.0, 
    rgb_points[3], 0.5, 1.0, 0.5, 
    rgb_points[4], 1.0, 1.0, 0.0, 
    rgb_points[5], 1.0, 0.0, 0.0, 
    rgb_points[6], 0.5, 0.0, 0.0 ]); #  equivalent of 'jet' colorspace in paraview GUI

cr_pvtu.ColorArrayName = 'proc_rank';
cr_pvtu.LookupTable = table;

# fade in
for d in np.linspace(0, 1, fps):
    print ' Fade in ... [%3d%%]\r' % int(100*d),
    sys.stdout.flush();

    cr_pvtu.Opacity = d;
    cr_pvtu.BackfaceOpacity = d;
    
    Render();
    WriteImage(filename % (sec, count)); count += 1;
print ' Fade in ... done!   '


# pause for 5 seconds
for i in range(5*fps):
    WriteImage(filename % (sec, count)); count += 1;

print '... done in %.2f sec' % (clock() - start)
# ##############################################################

# Step 2) remove the main object, pause, and zoom into a single one
start = clock();
sec = 2;
count = 0;

print 'Step 2:'

view.CameraFocalPoint = f0;
view.CameraPosition = p0;

cr_vtu = Show(c_vtu);
cr_vtu.ColorArrayName = 'proc_rank'
cr_vtu.LookupTable = table;
outline_r.Opacity = 1;

# fade out+zoom
b = c_vtu.GetDataInformation().GetBounds();
f1 = 0.5*np.array([b[0]+b[1], b[2]+b[3], b[4]+b[5]]);
p1 = p0 + 0.8*(f1 - p0);

for d in np.linspace(0, 1, 3*fps):
    print ' Zoom in ... [%3d%%]\r' % int(100*d),
    sys.stdout.flush();

    cr_pvtu.Opacity = 1 - d;
    cr_pvtu.BackfaceOpacity = 1 - d;
    outline_r.Opacity = 1 - d;
    
    view.CameraFocalPoint = d*f1 + (1-d)*f0;
    view.CameraPosition   = d*p1 + (1-d)*p0;

    Render();
    WriteImage(filename % (sec, count)); count += 1;
print ' Zoom in ... done!    '

Hide(c_pvtu);
Hide(outline);

cl = Clip(vtu);
cl.ClipType = 'Plane';
cl.ClipType.Origin = f1;
cl.InsideOut = 1;
cl.Crinkleclip = 1;

cl_r = Show(cl);
cl_r.Representation = 'Surface With Edges'
cl_r.ColorArrayName = 'proc_rank'
cl_r.LookupTable = table;
cl_r.EdgeColor = [0,0,0];

# fade in the grid
for d in np.linspace(0, 1, 2*fps):
    print ' Fade in grid ... [%3d%%]\r' % int(100*d),
    sys.stdout.flush();

    cl_r.Opacity = d**2;
    Render();
    WriteImage(filename % (sec, count)); count += 1;
print ' Fade in grid ... done!   '

# pause for 5 sec
for i in range(5*fps):
    WriteImage(filename % (sec, count)); count += 1;

# rotate 
r = np.sqrt((p1[0]-f1[0])**2 + (p1[2]-f1[2])**2);
t0 = np.arctan2((p1[0]-f1[0]),(p1[2]-f1[2]))
for t in np.linspace(0, 2*np.pi, 10*fps):
    print ' Rotating the grid ... [%3d%%]\r' % int(100*t/(2*np.pi)),
    sys.stdout.flush();

    view.CameraPosition[0] = f1[0] + r*np.sin(t+t0);
    view.CameraPosition[2] = f1[2] + r*np.cos(t+t0);
    
    Render();
    WriteImage(filename % (sec, count)); count += 1;    
print ' Rotating the grid ... done!   '

Show(c_pvtu);
Show(outline);

# zoom out
cr_pvtu.ColorArrayName = '';
for d in np.linspace(0, 1, 3*fps):
    print ' Zooming back ... [%3d%%]\r' % int(100*d),
    sys.stdout.flush();

    cr_pvtu.Opacity = d;
    cr_pvtu.BackfaceOpacity = d;
    outline_r.Opacity = d;
    
    cr_vtu.Opacity = 1 - d;
    cl_r.Opacity = 1 - d;
    
    view.CameraFocalPoint = d*f0 + (1-d)*f1;
    view.CameraPosition   = d*p0 + (1-d)*p1;

    Render();
    WriteImage(filename % (sec, count)); count += 1;
print ' Zooming back ... done!   '

print '... done in %.2f sec' % (clock() - start)
print 'Finished in %.2f sec' % (clock() - begin)