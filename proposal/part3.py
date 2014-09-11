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
parser.add_option("-f", "--fps",   action="store", dest="fps",      type="int",    default=30, help="fps for the animation")
parser.add_option("-p", "--path",  action="store", dest="path",     type="string", default="/scratch/02032/mmirzade/proposal_g/vis", help="path to the folder where the png images should be stored")
parser.add_option("-n", "--name",  action="store", dest="img_name", type="string", default="img",    help="prefix used to store the filename")
parser.add_option("-b", "--begin", action="store", dest='begin',    type='int',    default = '0',    help='starting pvtu file')
parser.add_option("-e", "--end",   action="store", dest='end',      type='int',    default = '5345', help='one beyond the last pvtu file')
parser.add_option("-s", "--skip",  action="store", dest='skip',     type='int',    default = '0',    help='skip every this many files when importing pvtus')

(opt, args) = parser.parse_args()
# ##############################################################

# Step 0) Prepare variables 
start = begin = clock();
from paraview.simple import *
print 'Loading paraview module and creating contours ...'

# Read the state file
paraview.simple._DisableFirstRenderCameraReset();
state = servermanager.LoadState('state_part3.pvsm');

# Get the view
SetActiveView(GetRenderView());
view = GetActiveView();
view.ViewSize = [1280, 720];

# Find existing sources
clip = FindSource('clip'); Show(clip);
front = FindSource('front');
outline = FindSource('outline');
outline_r = GetDisplayProperties(outline);

# set animation fps;
fps = opt.fps;
animation = GetAnimationScene();

# set image filenames# create 
filename = opt.path + '/' + opt.img_name + '_s%02d_%04d.png';

# set parameters for camera movement
f0 = np.array([1.5, 1.5, 1.5]);
p0 = np.array([6.5, 5.5, 6.5]);

f1 = np.array([1.5, 1.5, 0.0]);
p1 = np.array([1.5, 1.5, 8.0]);


view.CameraFocalPoint = f0;
view.CameraPosition = p0;
view.CameraViewAngle = 30;
view.CameraViewUp = [0, 1, 0];

Render();
print '... done in %.2f sec' % (clock() - start)
# ##############################################################

# # Step 1)
# start = clock();
# count = 0;
# sec   = 1;
# frames  = range(opt.begin, 1000   , opt.skip + 1); # move the front

# for frame in frames:
#     d = (count+1.0)/float(len(frames));
#     print 'Step 1/8 ... [%3d%%]\r' % int(100*d),
#     sys.stdout.flush();

#     animation.AnimationTime = frame;
#     WriteImage(filename % (sec,count)); count += 1;
# print 'Step 1/8 ... done! [%.2f sec]' % (clock() - start)
# # ##############################################################

# # Step 2)
# start = clock();
# count = 0;
# sec   = 2;
# frames = range(1000     , 1500   , opt.skip + 1); # move the clip + front

# for frame in frames:
#     d = (count+1.0)/float(len(frames));
#     print 'Step 2/8 ... [%3d%%]\r' % int(100*d),
#     sys.stdout.flush();

#     animation.AnimationTime = frame;
    
#     z = 3 - 1.5*d;
#     clip.ClipType.Origin[2] = z;

#     WriteImage(filename % (sec,count)); count += 1; 
# print 'Step 2/8 ... done! [%.2f sec]' % (clock() - start)
# # ##############################################################

# # Step 3)
# start = clock();
# count = 0;
# sec   = 3;
# frames = range(1500     , 2000   , opt.skip + 1); # move the front

# for frame in frames:
#     d = (count+1.0)/float(len(frames));
#     print 'Step 3/8 ... [%3d%%]\r' % int(100*d),
#     sys.stdout.flush();

#     animation.AnimationTime = frame;    
#     WriteImage(filename % (sec,count)); count += 1; 
# print 'Step 3/8 ... done! [%.2f sec]' % (clock() - start)
# # ##############################################################

# # Step 4)
# start = clock();
# count = 0;
# sec   = 4;
# frames = range(2000     , 2500   , opt.skip + 1); # zoom into z-plane + front

# for frame in frames:
#     d = (count+1.0)/float(len(frames));
#     print 'Step 4/8 ... [%3d%%]\r' % int(100*d),
#     sys.stdout.flush();

#     animation.AnimationTime = frame;

#     view.CameraPosition = d*p1 + (1-d)*p0;
#     view.CameraFocalPoint = d*f1 + (1-d)*f0;
#     outline_r.Opacity = 1 - d;

#     WriteImage(filename % (sec,count)); count += 1; 
# print 'Step 4/8 ... done! [%.2f sec]' % (clock() - start)
# # ##############################################################

# # Step 5)
# start = clock();
# count = 0;
# sec   = 5;
# frames = range(2500     , 4000   , opt.skip + 1); # move the front

# for frame in frames:
#     d = (count+1.0)/float(len(frames));
#     print 'Step 5/8 ... [%3d%%]\r' % int(100*d),
#     sys.stdout.flush();

#     animation.AnimationTime = frame;
#     WriteImage(filename % (sec,count)); count += 1; 
# print 'Step 5/8 ... done! [%.2f sec]' % (clock() - start)
# # ##############################################################

# # Step 6)
# start = clock();
# count = 0;
# sec   = 6;
# frames = range(4000     , 4500   , opt.skip + 1); # move out of z-plane + front

# for frame in frames:
#     d = (count+1.0)/float(len(frames));
#     print 'Step 6/8 ... [%3d%%]\r' % int(100*d),
#     sys.stdout.flush();

#     animation.AnimationTime = frame;

#     view.CameraPosition = d*p0 + (1-d)*p1;
#     view.CameraFocalPoint = d*f0 + (1-d)*f1;
#     outline_r.Opacity = d;

#     WriteImage(filename % (sec,count)); count += 1; 
# print 'Step 6/8 ... done! [%.2f sec]' % (clock() - start)
# # ##############################################################

# # Step 7)
# start = clock();
# count = 0;
# sec   = 7;
# frames = range(4500     , 5000   , opt.skip + 1); # move the clip back + front

# for frame in frames:
#     d = (count+1.0)/float(len(frames));
#     print 'Step 7/8 ... [%3d%%]\r' % int(100*d),
#     sys.stdout.flush();

#     animation.AnimationTime = frame;

#     z = 1.5 + 1.5*d;
#     clip.ClipType.Origin[2] = z;

#     WriteImage(filename % (sec,count)); count += 1; 
# print 'Step 7/8 ... done! [%.2f sec]' % (clock() - start)
# ##############################################################

# Step 8)
start = clock();
count = 0;
sec   = 8;
frames = range(5000     , opt.end, opt.skip + 1); # move the front

for frame in frames:
    d = (count+1.0)/float(len(frames));
    print 'Step 7/8 ... [%3d%%]\r' % int(100*d),
    sys.stdout.flush();

    animation.AnimationTime = frame;
    WriteImage(filename % (sec,count)); count += 1; 
print 'Step 8/8 ... done! [%.2f sec]' % (clock() - start)
# ##############################################################
print 'Finished in %.2f sec' % (clock() - begin)