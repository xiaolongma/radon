#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######                   RADON TRANSFORM WITH KAISER BESSEL              #######
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
from scipy import misc
from scipy import signal
from scipy import ndimage



####  MY PYTHON MODULES
import myImageIO as io
import myPrint as pp
import myImageDisplay as dis
import myImageProcess as proc


import ctypes as ct
grt = np.ctypeslib.load_library('generalized_radon_transform', '.')
grt.radon.argtypes = [ np.ctypeslib.ndpointer(dtype = np.float32) ,
                       np.ctypeslib.ndpointer(dtype = np.float32) ,
                       ct.c_int , np.ctypeslib.ndpointer(dtype = np.float32) ,
                       ct.c_int , ct.c_int , np.ctypeslib.ndpointer(dtype = np.float32) ,
                       ct.c_int , ct.c_float ]
grt.radon.restype  =  ct.c_int 




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
    parser = argparse.ArgumentParser(description='Radon Transform with Kaiser Bessel',
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
                             +' -g 0 --> equiangular projections in [0,180);'
                             +' -g 1 --> equally sloped projections in [0.180)'
                             +' -g angles.txt use a list of angles (in degrees) saved in a text file')
    
    parser.add_argument('-w',dest='kb_radius', type=np.float32 , default=2.0,
                        help='Select B-Spline degree')

    parser.add_argument('-d',dest='kb_degree', type=np.float32 , default=0.0,
                        help='Select B-Spline degree') 

    parser.add_argument('-z',dest='lut_size', type=int , default=100,
                        help='Select size of the B-spline look-up-table')

    parser.add_argument('-e',dest='padding', action='store_true',
                        help='Enable zero-padding if the background is not zero')  

    parser.add_argument('-p',dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')

    parser.add_argument('-l',dest='listang', action='store_true',
                        help='Save list of projection angles')     

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
####            CREATE PSEUDO POLAR ANGLES            ####
####                                                  ####
##########################################################
##########################################################

def createPseudoPolarAngles ( numAngles ):
    if numAngles % 4 != 0:
        raise Exception('\n\tError inside createPseudoPolarAngles:'
                        +'\n\t  numAngles (input) is not divisible by 4 !\n')
    n = numAngles
    pseudoAngleArr = np.zeros(n,dtype=myFloat)
    index = np.arange(int(n/4)+1,dtype=int)

    pseudoAngleArr[0:int(n/4)+1] = np.arctan(4*index[:]/myFloat(n))
    pseudoAngleArr[int(n/2):int(n/4):-1] = np.pi/2-pseudoAngleArr[0:int(n/4)]    
    pseudoAngleArr[int(n/2)+1:] = np.pi-pseudoAngleArr[int(n/2)-1:0:-1]

    return pseudoAngleArr




##########################################################
##########################################################
####                                                  ####
####          INIT KAISER-BESSEL LOOK UP TABLE        ####
####                                                  ####
##########################################################
########################################################## 

def radon_transform_kb( y , kb_radius , kb_degree ):
    a = kb_radius
    m = int( kb_degree )
    y[y > kb_radius] = 0.0
    p = 2 * a * np.power( 2 , m ) * misc.factorial( m ) \
        / misc.factorial2( 2*m + 1 ) * ( 1 - ( y / a )**2 ) ** ( m + 0.5 )
    return p
	
def init_lut_kaiser_bessel( nsamples_y , nang , kb_radius , kb_degree ):
    yrange = 2.0 * kb_radius
    yarray = np.arange( nsamples_y ).astype( np.float32 )
    yarray -= nsamples_y / 2.0
    yarray /= np.float32( nsamples_y )
    yarray *= yrange
    yarray_tile = yarray.reshape( 1 , yarray.shape[0] )
    ymatrix = np.tile( yarray_tile , ( nang , 1 ) )
    return radon_transform_kb( ymatrix , kb_radius , kb_degree )




##########################################################
##########################################################
####                                                  ####
####     CHANGE FROM KAISER BESSEL TO PIXEL BASIS     ####
####                                                  ####
##########################################################
##########################################################

def kaiser_bessel_basis_to_pixel( kb_matrix , kb_radius ):
    array_x = np.arange( -kb_radius , kb_radius + 1 )
    array_y = array_x
    x_grid, y_grid = np.meshgrid( array_x , array_y )
    r_squared = x_grid**2 + y_grid**2
    convolvent = ( ( kb_radius**2 ) - r_squared )**2
    return ndimage.filters.convolve( kb_matrix , convolvent )




##########################################################
##########################################################
####                                                  ####
####                    SAVE SINOGRAM                 ####
####                                                  ####
##########################################################
##########################################################

def saveSino( sino , angles , args ):
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

        filename += '_ang' + str_ang + '_radon_kb_r' \
                    + str( args.kb_radius ) + '_deg' \
                    + str( args.kb_degree ) 
        
        if args.geometry == '0':
            filename += '_polar'
        elif args.geometry == '1':
            filename += '_pseudo'
        else:
            filename += '_text'
        aux = filename
        filename += '_sin.DMP' 
        filename = pathout + filename
    else:
        filename = pathout + args.sino

    io.writeImage( filename , sino )


    ##  Save list of projection angles
    if args.listang is True:        
        fileang = aux + '_list_angles.txt'
        np.savetxt( pathout + fileang , angles , fmt='%.10f', delimiter='\n' )




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
    print('##############################################################')
    print('#############    KAISER-BESSEL RADON TRANSFORM   #############')
    print('##############################################################')
    print('\n')


    
    ##  Get the startimg time of the sinonstruction
    startTime = time.time()


    
    ##  Get input arguments
    args = getArgs()


    
    ##  Get path to input sino
    pathin = args.pathin


    
    ##  Get input image
    ##  You assume the image to be square
    image_name = pathin + args.image
    image = io.readImage( image_name )
    npix = image.shape[1]
    nang = args.nang

    print('\nInput image:\n', image_name)
    print('Number of projection angles: ', nang)
    print('Number of pixels: ', npix)


    ##  Check plot
    if args.plot is True:
        dis.plot( image , 'Input image' )



    ##  Get projection geometry  
    ##  1) Case of equiangular projections distributed in [0,180)
    if args.geometry == '0':
        angles = np.arange( nang )
        angles = ( angles * np.pi )/myFloat( nang )
        print('\nDealing with equally angularly spaced projections in [0,180)')

    
    ##  2) Case of equally sloped projections distributed in [0,180)
    elif args.geometry == '1':
        print('\n\nDealing with equally-sloped projections in [0,180)')
  
        if nang % 4 != 0:
            print('\n\nERROR: in case of equally-sloped projections',
                  ' the number of angles has to be a multiple of 4')

        angles = createPseudoPolarAngles( nang )    

    
    ##  3) Case of list of projection angles in degrees
    else:
        geometryfile = pathin + args.geometry
        angles = np.fromfile( geometryfile , sep="\t" )
        nang = len( angles )
        angles = np.pi * angles / 180.0
        print('\nReading list of projection angles: ', geometryfile)



    ##  Get kaiser-bessel parameters
    kb_radius = args.kb_radius
    proj_support_y = 2.0 * kb_radius
    kb_degree = args.kb_degree
    print('\nKaiser-Bessel radius selected: ', kb_radius)
    print('\nKaiser-Bessel degree selected: ', kb_degree)  



    ##  Flag to specify whether you want to perform the
    ##  radon transform or its adjoint
    flag_adj = 0



    ##  Direct transformation on the pixel image into
    ##  ita B-spline image
    #kb_image = pixel_basis_to_kaiser_bessel( image , 0 )
    kb_image = image.copy()


    
    ##  Precalculate look-up-table for B-spline
    nsamples_y = args.lut_size
    LUT = init_lut_kaiser_bessel( nsamples_y , nang , kb_radius , kb_radius )

    if args.plot is True:
        dis.plot( LUT , 'Look-up-table' )



    ##  Perform B-spline radon transform
    LUT = LUT.astype( np.float32 ) 
    angles = angles.astype( np.float32 )
    sino = np.zeros( ( nang , npix ) , dtype=np.float32 )
    exit = grt.radon( sino , kb_image , flag_adj , LUT , LUT.shape[1] ,
                      npix , angles , nang , proj_support_y )
    sino[:,:] = np.roll( sino[:,::-1] , 1 , axis=1 )   



    ##  Convert sinogram from kaiser-bessel basis to the pixel one
    #sino = kaiser_bessel_basis_to_pixel( sino , kb_radius )



    ##  Show sino     
    if args.plot is True:
        dis.plot( sino , 'Sinogram' )    


    
    ##  Save sino
    saveSino( sino , angles , args )

    
    
    ##  Time elapsed for the computation of the radon transform
    endTime = time.time()
    print('\n\nTime elapsed: ', (endTime-startTime)/60.0)
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
