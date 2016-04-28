# ##############################################################
try:
    from optparse import OptionParser
    from copy import *
    from time import *
    import os, sys
    import numpy as np
    from scipy import interpolate

except ImportError as e:
    print '[Error]: ', e 

# parse input options 
parser = OptionParser()
parser.add_option("-s", "--state", action="store", dest="state",    type="string", default="/home1/02032/mmirzade/state.pvsm", help="path to the paraview state file")
parser.add_option("-f", "--fps",   action="store", dest="fps",      type="int",    default=15, help="fps for the animation")
parser.add_option("-p", "--path",  action="store", dest="path",     type="string", default="/scratch/02032/mmirzade/proposal_g/vis", help="path to the folder where the png images should be stored")
parser.add_option("-n", "--name",  action="store", dest="img_name", type="string", default="img", help="prefix used to store the filename")

(opt, args) = parser.parse_args()

# ##############################################################

# Step 0) Prepare variables 
start = begin = clock();
from paraview.simple import *
print 'Loading ParaView module and state file ...'

# Read the state file
paraview.simple._DisableFirstRenderCameraReset();
state = servermanager.LoadState(opt.state);
reader = FindSource('solution.0.pvtu');

# Get the view
SetActiveView(GetRenderView());
view = GetActiveView();
view.ViewSize = [1280, 720];
Render();

# set animation fps;
fps = opt.fps;

# # set image filenames
filename = opt.path + '/' + opt.img_name + '_%04d.png';

# clip
cl = FindSource('Clip_x');
cl.ClipType.Origin[0] = 0;
cl_r = GetDisplayProperties(cl);

# slices
sx = FindSource('Slice_x');
sr = GetDisplayProperties(sx);

# outline
outline = FindSource('Outline_box');
Show(outline);

# set parameters for camera movement
f0 = np.array(view.CameraFocalPoint);
p0 = np.array(view.CameraPosition);

f1 = deepcopy(f0);
p1 = deepcopy(p0);

f2 = np.array([0,   1.5, 1.5]);
p2 = np.array([7.5, 1.5, 1.5]);

Render();
print '... done in %.2f sec' % (clock() - start)
# ##############################################################

# Step 1) evolve the geometry
start = clock()
print 'Evolving clip ... '

count = 0;

for x in np.linspace(0, 3, 10*fps):
    cl.ClipType.Origin[0] = x;
    Render();    
    WriteImage(filename % count); count += 1;

print '... done in %.2f sec' % (clock() - start)    
# ##############################################################

# Step 2) rotate around the geometry
start = clock()
print 'Rotating ... '

r0 = np.sqrt((p0[0]-f0[0])**2 + (p0[2]-f0[0])**2);
t0 = np.arctan2((p0[0]-f0[0]), (p0[2]-f0[2]))
for t in np.linspace(0, 2*np.pi, 10*fps):
    view.CameraPosition[0] = f0[0] + r0*np.sin(t + t0);
    view.CameraPosition[2] = f0[2] + r0*np.cos(t + t0);
    
    Render();
    WriteImage(filename % count); count += 1;  

print '... done in %.2f sec' % (clock() - start)
# ##############################################################

# Step 3) forward linear
start = clock()
print 'Forward zoom ... '

Hide(outline);
sx.SliceType.Origin[0] = 3;
sx.Triangulatetheslice = 0;
sr.Opacity = 1;
Show(sx);

Render();
WriteImage(filename % count); count += 1;

for t in np.linspace(0, 1, 10*fps):
    view.CameraFocalPoint = t*f2 + (1-t)*f1;
    view.CameraPosition = t*p2 + (1-t)*p1;
    
    # move the clip
    sx.SliceType.Origin[0] = 1.5*t + 3*(1-t);
    cl.ClipType.Origin[0] = 1.5*t + 3*(1-t);
    
    Render();
    WriteImage(filename % count); count += 1;

print '... done in %.2f sec' % (clock() - start)
# ##############################################################

# Step 4) zoom to a section
start = clock()
print 'Zoom to a section ... '

s_out = 1.6;
s_in  = 0.3; 
view.CameraParallelScale = s_out;
view.InteractionMode = '2D';

Render();
# WriteImage(filename % count); count += 1;

# path to move the camera 
py = np.array([0.45, 0.55, 1.44, 2.22, 2.52, 1.28, 0.50, 0.45]);
pz = np.array([2.53, 0.96, 0.45, 0.67, 2.26, 1.91, 2.65, 2.53]);

for d in np.linspace(0, 1.0, 5*fps):
    view.CameraParallelScale = d*s_in+(1-d)*s_out;
    view.CameraPosition[1] = view.CameraFocalPoint[1] = d*py[0] + (1-d)*p2[1];
    view.CameraPosition[2] = view.CameraFocalPoint[2] = d*pz[0] + (1-d)*p2[2];        
    
    Render();
    WriteImage(filename % count); count += 1;

print '... done in %.2f sec' % (clock() - start)    
# ##############################################################

# Step 5) follow the path
start = clock()
print 'Spline path ... '

interp,u = interpolate.splprep([py,pz], s=0)
t = np.linspace(0, 1., 60*fps);
s = interpolate.splev(t,interp);

for i in range(t.size):
    view.CameraPosition[1] = view.CameraFocalPoint[1] = s[0][i];
    view.CameraPosition[2] = view.CameraFocalPoint[2] = s[1][i];        
    
    Render();
    WriteImage(filename % count); count += 1;

print '... done in %.2f sec' % (clock() - start)    
# ##############################################################

# Step 6) zoom back 
start = clock()
print 'Zoom back from section... '

for d in np.linspace(1.0, 0, 5*fps):
    view.CameraParallelScale = d*s_in+(1-d)*s_out;
    view.CameraPosition[1] = view.CameraFocalPoint[1] = d*py[0] + (1-d)*p2[1];
    view.CameraPosition[2] = view.CameraFocalPoint[2] = d*pz[0] + (1-d)*p2[2];
    
    Render();
    WriteImage(filename % count); count += 1;

view.InteractionMode = '3D';

Render();
# WriteImage(filename % count); count += 1;

print '... done in %.2f sec' % (clock() - start)    
# ##############################################################

# Step 7) backward linear
start = clock()
print 'Zoom backward ... '

Show(sx);
sx.Triangulatetheslice = 0;

Render();
WriteImage(filename % count); count += 1;

for t in np.linspace(0, 1, 10*fps):
    view.CameraFocalPoint = t*f1 + (1-t)*f2;
    view.CameraPosition = t*p1 + (1-t)*p2;
    
    # move the clip back
    cl.ClipType.Origin[0] = 1.5*(1-t) + 3*t;
    sr.Opacity = 1 - t;
    
    Render();
    WriteImage(filename % count); count += 1;
    
Hide(sx);
Show(outline)
Render();

WriteImage(filename % count); count += 1;
# ##############################################################

print '... done in %.2f sec' % (clock() - start)    
print 'Finished in %.2f sec' % (clock() - begin)    

