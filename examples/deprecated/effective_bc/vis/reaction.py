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
parser.add_option("-f", "--fps",   action="store", dest="fps",      type="int",    default = 30,                                     help="fps for the animation")
parser.add_option("-p", "--path",  action="store", dest="path",     type="string", default = "/scratch/02032/mmirzade/reaction/vis", help="path to the folder where the png images should be stored")
parser.add_option("-n", "--name",  action="store", dest="img_name", type="string", default = "img",                                  help="prefix used to store the filename")
parser.add_option("-b", "--begin", action="store", dest='begin',    type='int',    default = '0',                                    help='starting pvtu file')
parser.add_option("-e", "--end",   action="store", dest='end',      type='int',    default = '500',                                  help='one beyond the last pvtu file')

(opt, args) = parser.parse_args()
# ##############################################################

# Step 0) Prepare variables 
start = begin = clock();
from paraview.simple import *
print 'Loading paraview module and creating contours ...'

# Read the state file
paraview.simple._DisableFirstRenderCameraReset();
state = servermanager.LoadState('state_reaction.pvsm');

# Get the view
SetActiveView(GetRenderView());
view = GetActiveView();
view.ViewSize = [1280, 720];

# Get access to the objects in the state file 
zclip   = FindSource('zclip');
zclip_r = GetDisplayProperties(zclip);

front   = FindSource('front');
front_r = GetDisplayProperties(front);
front_r.Opacity = 0.5;

outline = FindSource('outline');
outline_r = GetDisplayProperties(outline);

outline_r.DiffuseColor = [0,0,0];

Render();

# set animation fps;
fps = opt.fps;
animation = GetAnimationScene();

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
# Step 1)
start = clock();
count = 0;
sec   = 1;
frames  = range(opt.begin, 100); # move the front

for frame in frames:
    d = (count+1.0)/float(len(frames));
    print 'Step 1/4 ... [%3d%%]\r' % int(100*d),
    sys.stdout.flush();

    animation.AnimationTime = frame;
    WriteImage(filename % (sec,count)); count += 1;
print 'Step 1/4 ... done! [%.2f sec]' % (clock() - start)
# ##############################################################

# Step 2)
start = clock();
count = 0;
sec   = 2;
frames = range(100, 200); # move the clip + front

for frame in frames:
    d = (count+1.0)/float(len(frames));
    print 'Step 2/4 ... [%3d%%]\r' % int(100*d),
    sys.stdout.flush();

    animation.AnimationTime = frame;
    
    z = 3 - 1.5*d;
    zclip.ClipType.Origin[2] = z;

    WriteImage(filename % (sec,count)); count += 1; 
print 'Step 2/4 ... done! [%.2f sec]' % (clock() - start)
# ##############################################################

# Step 3)
start = clock();
count = 0;
sec   = 3;
frames = range(200, 600); # move the front

for frame in frames:
    d = (count+1.0)/float(len(frames));
    print 'Step 3/4 ... [%3d%%]\r' % int(100*d),
    sys.stdout.flush();

    animation.AnimationTime = frame;    
    WriteImage(filename % (sec,count)); count += 1; 
print 'Step 3/4 ... done! [%.2f sec]' % (clock() - start)
# ##############################################################

# Step 4) rotate
start = clock();
count = 0;
sec   = 4;

r0 = np.sqrt((p0[0]-f0[0])**2 + (p0[2]-f0[0])**2);
t0 = np.arctan2((p0[0]-f0[0]), (p0[2]-f0[2]))

for t in np.linspace(0, 2*np.pi, 10*fps):
    d = (count+1.0)/float(10*fps);
    print 'Step 4/4 ... [%3d%%]\r' % int(100*d),
    sys.stdout.flush();

    view.CameraPosition[0] = f0[0] + r0*np.sin(t + t0);
    view.CameraPosition[2] = f0[2] + r0*np.cos(t + t0);
    
    Render();
    WriteImage(filename % (sec,count)); count += 1; 

print 'Step 4/4 ... done! [%.2f sec]' % (clock() - start)
# ##############################################################
print 'Finished in %.2f sec' % (clock() - begin)