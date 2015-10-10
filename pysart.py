#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######                          RADON TRANSFORM                          #######
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
import skimage
from skimage.transform import iradon_sart




####  PYTHON PLOTTING MODULES
import matplotlib.pyplot as plt
import matplotlib.cm as cm
       



####  MY PYTHON MODULES
sys.path.append('/home/arcusfil/tomcat/Programs/python_ambient/Common/')
import myImageIO as io
import myPrint as pp




####  MY FORMAT VARIABLES
myFloat = np.float32




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
    parser.add_argument('-g', '--geometry', dest='geometry',default='0',
                        help='Specify projection geometry; @@@@@@@@@@@@@@@@@@@@@@@'
                             +' -g 0 --> equiangular projections between 0 and 180 degrees (default);'
                             +' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
                             +' -g angles.txt use a list of angles (in degrees) saved in a text file')
    parser.add_argument('-c',dest='ctr', type=myFloat,
                        help='Specify the center of rotation ( default = the middle pixel of the image )')
    parser.add_argument('-k',dest='niter', type=int, default=3,
                        help='Choose number of iterations')
    parser.add_argument('-z',dest='edgepad', type=myFloat, default = 0.0,
                        help='Apply edge padding') 
    parser.add_argument('-p',dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')
    

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
####                    EDGE PADDING                  ####
####                                                  ####
##########################################################
##########################################################

def edgepad( sino , padding ):
    nang = sino.shape[0]
    npix_old = sino.shape[1]
        
    npix_pad = int( padding * npix_old )
        
    print("Padded pixels on one side = ", npix_pad)
        
    columnStart = np.array( sino[:,0] ).reshape( nang , 1 )
    columnEnd = np.array( sino[:,npix_old-1] ).reshape( nang , 1 )
        
    sino_aux1 = np.ones(( 1 , npix_pad )) * columnStart
    sino_aux2 = np.ones(( 1 , npix_pad )) * columnEnd
        
    sino_pad = np.concatenate(( sino_aux1 , sino , sino_aux2 ), axis=1 )
        
    return sino_pad




##########################################################
##########################################################
####                                                  ####
####                      CHECK PLOT                  ####
####                                                  ####
##########################################################
##########################################################

def checkPlot( reco , title ):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow( reco , origin="lower"  , cmap = cm.Greys_r ,
               interpolation='nearest' )
    plt.title( title )
    numrows, numcols = reco.shape 
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = reco[row,col]
            return 'x=%1.4f, y=%1.4f, value=%1.4f'%(x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y) 
    ax.format_coord = format_coord
    
    plt.show()




##########################################################
##########################################################
####                                                  ####
####                    SAVE SINOGRAM                 ####
####                                                  ####
##########################################################
##########################################################

def saveReco( reco , args ):
    if args.pathout is None:
        pathout = args.pathin
    else:
        pathout = args.pathout

    if pathout[len(pathout)-1] != '/':
        pathout += '/'
                            
    
    if args.reco is None:
        filename = args.sino
        filename = filename[:len(filename)-4]
        filename += '_rec_isart.DMP'
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
    print('########################################')
    print('#############    PY-SART   #############')
    print('########################################')
    print('\n')


    ##  Get the startimg time of the reconstruction
    startTime = time.time()


    ##  Get input arguments
    args = getArgs()


    ##  Get path to input reco
    pathin = args.pathin

    if pathin[len(pathin)-1] != '/':
        pathin += '/'  


    ##  Get input reco
    ##  You assume the reco to be square
    sino_name = pathin + args.sino
    sino = io.readImage( sino_name )
    npix = sino.shape[1]
    nang = sino.shape[0]

    print('\nInput sino:\n', sino_name)
    print('Number of projection angles: ', nang)
    print('Number of pixels: ', npix)


    ##  Show reco
    if args.plot is True:
        checkPlot( sino , 'Sinogram' )


    ##  Get projection geometry  
    ##  1) Case of equiangular projections distributed in [0,180)
    if args.geometry == '0':
        angles = np.arange( nang )
        angles = ( angles * 180.0 )/myFloat( nang )
        print('\nDealing with equally angularly spaced projections in [0,180)')

    ##  2) Case of list of projection angles in degrees
    else:
        geometryfile = pathin + args.geometry
        angles = np.fromfile( geometryfile , sep="\t" )
        nang = len( angles )
        print('\nReading list of projection angles: ', geometryfile)

    print('Number of projection angles: ', nang)


    ##  Get center of rotation
    if args.ctr is None:
        ctr = npix * 0.5
    else:
        ctr = args.ctr

    print('\nCenter of rotation placed at pixel: ', ctr)



    ##  Apply edge padding
    if args.edgepad != 0:
        npix_old = npix
        sino = edgepad( sino , args.edgepad )
        sino = sino.astype( myFloat )
        npix = sino.shape[1]
        print('\nEdge padded sinogram size: ', sino.shape[0],' X ', sino.shape[1])



    ##  Compute iradon transform
    print('\nPerforming SART iradon ....\n')
    niter = args.niter
    print('\nNumber of iterations:', niter)

    print('\nIter number  1 ....')
    reco = iradon_sart( np.rot90( sino ).astype(np.float64) , 
                        angles.astype(np.float64) )
    
    for i in range( niter - 1 ):
        print('\nIter number ', i +2,' ....')
        reco[:,:] = iradon_sart( np.rot90( sino ).astype(np.float64) ,
                                 angles.astype(np.float64) ,
                                 image=reco ) 


    
    ##  Crop image 
    if args.edgepad:
        i1 = int( 0.5 * ( npix - npix_old ))
        i2 = i1 + npix_old
        reco = reco[i1:i2,i1:i2]    

    
    
    ##  Show sino     
    if args.plot is True:
        checkPlot( reco , 'SART reconstruction' )


    ##  Save sino
    saveReco( reco , args )

    
    ##  Time elapsed for the computation of the radon transform
    endTime = time.time()
    print('\n\nTime elapsed: ', (endTime-startTime)/60.0,' min')
    print('\n')




#####################################################################
#######    CALL TO MAIN
#####################################################################  
if __name__ == '__main__':
    main()
