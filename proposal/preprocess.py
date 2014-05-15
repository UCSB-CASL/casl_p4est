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

# parse input options 
parser = OptionParser()
parser.add_option("-v", "--vtu",  action="store", dest="vtu",      type="string", default='/scratch/02032/mmirzade/proposal_g/n_256.N_16.2973336/solution.0_%04d.vtu', help="fps for the animation")

(opt, args) = parser.parse_args()
# ##############################################################

# Step 0) Prepare variables 
start = begin = clock();
from paraview.simple import *
print 'Loading ParaView module ... ',
print 'done in %.2f sec' % (clock() - start)
# ##############################################################

# Read individual files, create a contour and save the result
start = clock();
print 'Converting formats ... ';

# load data files
mpisize = 256;

for i in range(mpisize):
    print '[%3d%%] Converting file ' % int(100 * i / float(mpisize)), ' %3d/256 \r' % (i+1),
    sys.stdout.flush();

    # read the vtu files
    reader = XMLUnstructuredGridReader(FileName = opt.vtu % i);
    reader.PointArrayStatus = 'phi';
     
    # generate contours
    contour = Contour(reader, ContourBy = 'phi', Isosurfaces = 0);
 
    # write the surface as a vtp file    
    writer = XMLPolyDataWriter(FileName = '/scratch/02032/mmirzade/proposal_g/n_256.N_16.2973336/surface_%04d.vtp' % i, Input = contour);
    writer.CompressorType = 'ZLib'
    writer.UpdatePipeline()

    # remove the old stuff
    Delete(contour);
    Delete(reader);

print '... done in %.2f sec' % (clock() - start)
# ##############################################################