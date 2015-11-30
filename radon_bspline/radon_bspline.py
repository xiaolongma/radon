#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######            RADON TRANSFORM BASED ON A CUBIC B-SPLINE BASIS        #######
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




####  MY PYTHON MODULES
import myImageIO as io
import myPrint as pp
import myImageDisplay as dis
import myImageProcess as proc
import class_projectors_bspline as cpb




####  MY FORMAT VARIABLES
myfloat = np.float32




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
    parser = argparse.ArgumentParser(description='Radon Transform based on a cubic B-spline basis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-Di', '--pathin', dest='pathin', default='./',
                        help='Specify path to input data')    
    
    parser.add_argument('-i', '--image', dest='image',
                        help='Specify name of input image')
    
    parser.add_argument('-Do', '--pathout', dest='pathout',
                        help='Specify path to output data') 
    
    parser.add_argument('-o', '--sino', dest='sino',
                        help='Specify name of output sinogram')
    
    parser.add_argument('-n', '--nang', dest='nang', type=int, default=180,
                        help='Select the number of projection angles ( default: 180 projections )') 
    
    parser.add_argument('-g', '--geometry', dest='geometry',default='0',
                        help='Specify projection geometry; @@@@@@@@@@@@@@@@@@@@@@@'
                             +' -g 0 --> equiangular projections in [0,180)')
    
    parser.add_argument('-w',dest='bspline_degree', type=int , default=3,
                        help='Select B-Spline degree')

    parser.add_argument('-z',dest='lut_size', type=int , default=2048,
                        help='Select size of the B-spline look-up-table')

    parser.add_argument('-e',dest='padding', action='store_true',
                        help='Enable zero-padding if the background is not zero')  

    parser.add_argument('-p',dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')

    parser.add_argument('-dpc',dest='dpc', action='store_true',
                        help='Enable DPC forward projection')    

    args = parser.parse_args()
    

    ##  Exit of the program in case the compulsory arguments, 
    ##  are not specified
    if args.image is None:
        parser.print_help()
        print('ERROR: Input image name not specified!')
        sys.exit()  
        
    return args




##########################################################
##########################################################
####                                                  ####
####                    SAVE SINOGRAM                 ####
####                                                  ####
##########################################################
##########################################################

def save_sino( sino , angles , args ):
    ##  Save sinogram
    if args.pathout is None:
        pathout = args.pathin
    else:
        pathout = args.pathout
    
    if args.sino is None:
        filename = args.image
        filename = filename[:len(filename)-4]

        if args.nang < 10:
            str_ang = '000' + str( args.nang )
        elif args.nang < 100:
            str_ang = '00' + str( args.nang )
        elif args.nang < 1000:
            str_ang = '0' + str( args.nang )
        else:
            str_ang = str( args.nang ) 

        filename += '_ang' + str_ang + '_radon_bspline_deg' \
                    + str( args.bspline_degree )
        
        if args.geometry == '0':
            filename += '_polar'
        elif args.geometry == '1':
            filename += '_pseudo'
        else:
            filename += '_text'
        aux = filename
        filename += '_sino.DMP' 
        filename = pathout + filename
    else:
        filename = pathout + args.sino

    io.writeImage( filename , sino )





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
    print('#####################################')  
    print('#####################################')
    print('####                             ####') 
    print('####   RADON TRANSFORM BASED ON  ####') 
    print('####    A CUBIC B-SPLINE BASIS   ####')
    print('####                             ####')    
    print('#####################################')
    print('#####################################')      
    print('\n')


    
    ##  Get the startimg time of the sinonstruction
    time1 = time.time()


    
    ##  Get input arguments
    args = getArgs()


    
    ##  Get path to input sino
    pathin = args.pathin


    
    ##  Get input image
    ##  You assume the image to be square
    image_name = pathin + args.image
    image = io.readImage( image_name ).astype( myfloat )
    npix = image.shape[1]
    nang = args.nang

    print('\nInput image:\n', image_name)
    print('Number of projection angles: ', nang)
    print('Number of pixels: ', npix)


    ##  Check plot
    if args.plot is True:
        dis.plot( image , 'Input image' )



    ##  Get projection geometry  
    angles = np.arange( nang )
    angles = ( angles * np.pi )/myfloat( nang )
    print('\nDealing with equally angularly spaced projections in [0,180)')


    
    ##  Create projector class
    if args.dpc is False:
        rd = 0
    else:
        rd = 1

    deg = args.bspline_degree
    sup = deg + 1

    print('\nSelected B-spline degree: ', deg)
    print('Selected Radon degree: ', rd )

    tp = cpb.projectors( npix , angles , bspline_degree=deg , proj_support_y=sup ,
                         nsamples_y=2048 , radon_degree=rd , filt='ramp' , 
                         back = False , plot=True  )

    image[:] = image[::-1,::-1]
    sino     = tp.A( image )



    ##  Show sino     
    if args.plot is True:
        dis.plot( sino , 'Sinogram' )    


    
    ##  Save sino
    save_sino( sino , angles , args )

    
    
    ##  Time elapsed for the computation of the radon transform
    time2 = time.time()
    print('\n\nTime elapsed to run the radon: ', (time2-time1)/60.0,' min.')
    print('Time elapsed to run the radon: ', time2-time1,' sec.') 
    print('\n')




###########################################################
###########################################################
####                                                   #### 
####                      CALL TO MAIN                 ####
####                                                   ####
###########################################################
###########################################################

if __name__ == '__main__':
    main()
