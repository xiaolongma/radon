from __future__ import print_function , division
import sys
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import myImageIO as io
import myImageDisplay as dis
import myImageProcess as proc



def main():
    sino = io.readImage( sys.argv[1] )
    ctr = np.float32( sys.argv[2] )
    dis.plot( sino , 'Input sinogram' )

    sino_new = proc.sino_correct_rot_axis( sino , ctr )
    '''
    nang , npix = sino.shape
    sino_out = sino.copy()

    x = np.arange( npix )
    x_out = x - shift
    
    for i in range( nang ):
        s = InterpolatedUnivariateSpline( x , sino[i,:] )
        sino_out[i,:] = s( x_out )
    '''

    dis.plot( sino_new , 'Output sinogram' )
    filename = sys.argv[1][:len(sys.argv[1])-4] + '_interp.DMP'
    io.writeImage( filename , sino_new )



if __name__ == '__main__':
    main()
