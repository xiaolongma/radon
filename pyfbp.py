#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######                    STANDARD FILTERED BACKPROJECTION               #######
#######                                                                   #######
#######        Author: Filippo Arcadu, arcusfil@gmail.com, 01/03/2013     #######
#######                                                                   #######
#################################################################################
#################################################################################
#################################################################################




####  PYTHON MODULES
from __future__ import division,print_function
import time
import datetime
import argparse
import sys
import os
import numpy as np
import scipy as sci
import scipy.interpolate as scint



####  PYTHON PLOTTING MODULES
import pylab as py
import matplotlib.pyplot as plt
import matplotlib.cm as cm  



####  MY PYTHON MODULES
import myImageIO as io
import myPrint as pp
import myImageDisplay as dis
import myImageProcess as proc
import filters as fil  




####  MY FORMAT VARIABLES
myfloat   = np.float32
mycomplex = np.complex64




####  CONSTANTS
eps = 1e-8




##########################################################
##########################################################
####                                                  ####
####             GET INPUT ARGUMENTS                  ####
####                                                  ####
##########################################################
##########################################################

def getArgs():
    parser = argparse.ArgumentParser(description='Inverse Radon Transform -- FBP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-Di', '--pathin', dest='pathin', default='./',
                        help='Specify path to input data')    
    
    parser.add_argument('-i', '--sino', dest='sino',
                        help='Specify name of input reco')
    
    parser.add_argument('-Do', '--pathout', dest='pathout',
                        help='Specify path to output data') 
    
    parser.add_argument('-o', '--reco', dest='reco',
                        help='Specify name of output reconstruction')
    
    parser.add_argument('-n', '--nang', dest='nang', type=int, default=180,
                        help='Select the number of projection angles ( default: 180 projections )') 
    
    parser.add_argument('-g', '--geometry', dest='geometry',default='0',
                        help='Specify projection geometry; @@@@@@@@@@@@@@@@@@@@@@@'
                             +' -g 0 --> equiangular projections between 0 and 180 degrees (default);'
                             +' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
                             +' -g angles.txt use a list of angles (in degrees) saved in a text file')
    
    parser.add_argument('-c',dest='ctr', type=myfloat,
                        help='Specify the center of rotation ( default = the middle pixel of the image )') 
    
    parser.add_argument('-e',dest='edge_padding', action='store_true',
                        help='Enable edge padding for local tomography') 

    parser.add_argument('-f',dest='filt', default='ram-lak',
                        help='Select interpolation scheme: \
                              ram-lak , shepp-logan , cosine , hamming , hanning , none')  
    
    parser.add_argument('-w',dest='interp', default='lin',
                        help='''Select interpolation scheme:
                              nearest neighbour  --->  nn
                              linear  ---> lin
                              ( cubic spline  ---> spl , not working )
                             ''') 
    
    parser.add_argument('-p',dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')

    parser.add_argument('-m',dest='movie', action='store_true',
                        help='Save partial reconstructions in a folder called "fbp_movie"') 

    parser.add_argument('-dpc',dest='dpc', action='store_true',
                        help='Perform DPC reconstruction') 

    parser.add_argument('-dbp',dest='dbp', action='store_true',
                        help='Perform DBP reconstruction')        
    
    args = parser.parse_args()
    

    ##  Exit of the program in case the compulsory arguments, 
    ##  are not specified
    if args.sino is None:
        parser.print_help()
        print('ERROR: Input sino name not specified!')
        sys.exit()  
    
    
    return args




##########################################################
##########################################################
####                                                  ####
####                        IRADON                    ####
####                                                  ####
##########################################################
########################################################## 

def iradon( sino , npix , angles , ctr , filt , interp , dpc , args ):
    ##  Get number of angles
    nang = len( angles )



    ##  Pre-calculate sin and cos values
    cos = np.cos( angles )
    sin = np.sin( angles )


    
    ##  Create grid of coordinates for the reconstruction
    x = np.arange( - ( npix * 0.5  - 1 ) , npix * 0.5 + 1 )
    x = np.kron( np.ones( ( npix , 1 ) ) , x ) 
    y = np.rot90( x )


    
    ##  Filter projections
    if filt is not None:
        sino[:] = fil.filter_proj( sino , ftype=filt , dpc=dpc )
    
        if args.plot is True:
            dis.plot( sino , 'Filtered sinogram' )



    ##  Zero-pad projections to fit with the dimension of the
    ##  reoconstructing grid diagonal
    img_diag = 2 * int( np.ceil( npix / np.sqrt(2) ) ) + 1

    if npix < img_diag:
        pad = 0.5 * ( img_diag - npix )
        i1 = int( np.ceil( pad ) );  i2 =  npix + int( np.floor( pad ) )
        sino_op = np.zeros( ( nang , i1 + i2 ) , dtype=myfloat ) 
        sino_op[:, i1 : i1 + npix ] = sino[:,:]
        ctr += i1

    else:
        sino_op = np.zeros( ( nang , npix ) , dtype=myfloat )
        sino_op[:,:] = sino[:,:]
                 
    ctr = int( ctr )


    ##  Allocate memory for the reconstruction
    reco = np.zeros( ( npix , npix ) , dtype=myfloat )


    ##  Enable movie
    if args.movie is True:
        py.ion()
        im = py.imshow( x , animated=True , cmap=cm.Greys_r )

        folder = 'fbp_movie/'
        
        if not os.path.exists( folder ):
            os.makedirs( folder )
        else:
            import shutil
            shutil.rmtree( folder )
            os.makedirs( folder ) 



    ##  Filtered Backprojection
    if interp == 'spl':
        points = np.arange( i1 + i2 ) - ctr + 1

    for i in range( nang ):
        sys.stdout.write( 'Backprojecting projection number %d\r' % ( i +1 , ) );
        sys.stdout.flush()

        t = x * cos[i] + y * sin[i]

        if interp == 'nn':
            t = np.round( t ).astype( int )
            reco += sino_op[i, t + ctr - 1 ]
        
        elif interp == 'lin':
            a = np.floor( t ).astype( int )
            reco += ( t - a ) * sino_op[ i , a + ctr ] + \
                    ( a + 1 - t ) * sino_op[ i , a + ctr - 1 ]

        if args.movie is True:
            py.imshow( reco[::-1,:] , animated=True, cmap=cm.Greys_r )            
            py.draw()

            if i < 10:
                num_proj = '000' + str( i )
            elif i < 100:
                num_proj = '00' + str( i )
            elif i < 1000:
                num_proj = '0' + str( i )
            else:
                num_proj = str( i )

            sci.misc.imsave( folder + 'reco_' + num_proj + '.jpg' , reco[::-1,:] )


    
    if dpc is False:
        reco *= np.pi / ( 2.0 * nang )
    else:
        reco *= np.pi / ( 1.0 * nang ) 
    
    return reco 




##########################################################
##########################################################
####                                                  ####
####                    SAVE SINOGRAM                 ####
####                                                  ####
##########################################################
##########################################################

def save_reco( reco , args ):
    if args.pathout is None:
        pathout = args.pathin
    else:
        pathout = args.pathout
    
    if args.reco is None:
        filename = args.sino
        filename = filename[:len(filename)-4]
        filename += '_pyfbp' + '_rec.DMP'
        filename = pathout + filename
    else:
        filename = pathout + args.reco

    io.writeImage( filename , reco )




##########################################################
##########################################################
####                                                  ####
####                       MAIN                       ####
####                                                  ####
##########################################################
##########################################################

def main():
    ##  Initial print
    print('\n')
    print('#######################################')
    print('#############    PY-FBP   #############')
    print('#######################################')
    print('\n')


    ##  Get the startimg time of the reconstruction
    startTime = time.time()



    ##  Get input arguments
    args = getArgs()



    ##  Get path to input reco
    pathin = args.pathin



    ##  Get input reco
    ##  You assume the reco to be square
    sino_name = pathin + args.sino
    sino = io.readImage( sino_name ).astype( myfloat )
    npix = sino.shape[1]
    nang = sino.shape[0]

    print('\nInput sino:\n', sino_name)
    print('Number of projection angles: ', nang)
    print('Number of pixels: ', npix)

    if args.plot is True:
        dis.plot( sino , 'Sinogram' )



    ##  Get projection geometry  
    ##  1) Case of equiangular projections distributed in [0,180)
    if args.geometry == '0':
        angles = np.arange( nang )
        angles = ( angles * np.pi )/myfloat( nang )
        print('\nDealing with equally angularly spaced projections in [0,180)')

    ##  2) Case of list of projection angles in degrees
    else:
        geometryfile = pathin + args.geometry
        angles = np.fromfile( geometryfile , sep="\t" )
        angles *= np.pi/180.0
        nang = len( angles )
        print('\nReading list of projection angles: ', geometryfile)

    print('Number of projection angles: ', nang)



    ##  DPC reconstruction
    dpc = args.dpc
    print( 'DPC reconstruction: ' , dpc )


    ##  DPC reconstruction
    if args.dbp is True:
        dpc = args.dbp
        sino[:,:] = proc.diff_sino( sino ) 
        if args.plot is True:
            dis.plot( sino , 'Differential sinogram' )
    print( 'DBP reconstruction: ' , dpc )



    ##  Get center of rotation
    print('\nCenter of rotation placed at pixel: ', args.ctr) 

    if args.ctr is None:
        ctr = npix * 0.5
    else:
        ctr = args.ctr
        sino = proc.sino_correct_rot_axis( sino , ctr )
        ctr = npix * 0.5 



    ##  Enable edge padding
    if args.edge_padding is True:
        npix_old = sino.shape[1]
        sino = proc.sino_edge_padding( sino , 0.5 )
        npix = sino.shape[1]
        i1 = int( ( npix - npix_old ) * 0.5 )
        i2 = i1 + npix_old      
        ctr += i1

    

    ##  Get filter type
    filt = args.filt
    print( 'Selected filter: ', filt )



    ##  Get interpolation scheme
    interp = args.interp
    if interp == 'nn':
        print('\nSelected interpolation scheme: nearest neighbour')
    elif interp == 'lin':
        print('\nSelected interpolation scheme: linear interpolation')



    ##  Compute iradon transform
    print('\nPerforming Filtered Backprojection ....\n')
    reco = np.zeros( ( npix , npix ) , dtype=myfloat )
    reco[:,:] = iradon( sino[:,::-1] , npix , angles , ctr , filt , interp , dpc , args )



    ##  Remove edge padding
    if args.edge_padding is True:
        reco = reco[i1:i2,i1:i2]     

    
    
    ##  Show reconstruction     
    if args.plot is True:
        dis.plot( reco , 'Reconstruction' )



    ##  Save sino
    save_reco( reco , args )



    ##  Time elapsed for the computation of the radon transform
    endTime = time.time()
    print('\n\nTime elapsed: ', (endTime-startTime)/60.0)
    print('\n')




#####################################################################
#######    CALL TO MAIN
#####################################################################  
if __name__ == '__main__':
    main()
