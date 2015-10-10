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
from scipy import misc
from scipy import signal



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
    parser = argparse.ArgumentParser(description='Skimage Radon Transform',
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
    
    parser.add_argument('-w',dest='bspline_degree', type=int , default=3,
                        help='Select B-Spline degree')

    parser.add_argument('-z',dest='lut_size', type=int , default=2048,
                        help='Select size of the B-spline look-up-table')

    parser.add_argument('-e',dest='padding', action='store_true',
                        help='Enable zero-padding if the background is not zero')  

    parser.add_argument('-p',dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')

    parser.add_argument('-l',dest='listang', action='store_true',
                        help='Save list of projection angles') 

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
####        CHANGE FROM PIXEL TO BSPLINE BASIS        ####
####                                                  ####
##########################################################
########################################################## 

def pixel_basis_to_bspline( image , bspline_degree ):
    m = bspline_degree
    if m == 0:
        bspline_image = image
    elif m == 2:
        bspline_image = signal.qspline2d( image )
    elif m == 3:
        bspline_image = signal.cspline2d( image )
    else:
        sys.exit('\nERROR: B-spline degree not supported !!')
    return bspline_image




##########################################################
##########################################################
####                                                  ####
####  COMPUTE R^{n} OF B_SPLINE GENERAL SLOW FORMULA  ####
####                                                  ####
##########################################################
##########################################################

def positive_power( p ):
    def positive_power_p( x ):
        if np.float32( x ) > 0.0:
            return np.power( np.float32( x ) , p )
        else:
            return 0.0
    return positive_power_p 


def scalar_product(lamba, f):
    def spf(x):
        return float(lamba)*f(x)
    return spf


def finite_difference(f, h):
    def finite_difference_function(x):
        if h != 0.0:
            return ( f( x + 0.5*h ) - f( x - 0.5*h ) ) / np.float32( h )
        else:
            sys.exit('\nError in "finite_difference": h = 0 !!')
    return finite_difference_function 


def radon_n_bspline_general( y , theta , m , n ):
    exponent = 2*m - n + 1
    y_plus = positive_power( exponent )
    

    ##  Calculate  m+1 fold derivative (analytically)
    deriv_positive_power = scalar_product( ( misc.factorial( exponent ) / \
                   misc.factorial( exponent - (m +1) ) ) , positive_power( m - n ) )

    if theta == 0.0 or theta == np.pi/2.0 or theta == np.pi:
        y_plus = deriv_positive_power
        
    
    ##  Consider special case of theta = 0 , pi
    if np.abs( theta - np.pi/2.0 ) > eps:
        for i in range( m + 1 ):
            y_plus = finite_difference( y_plus, np.cos(theta) )
    

    ##  Consider special case of theta = pi/2
    if np.abs( theta ) > eps  and np.abs( theta - np.pi ) > eps:
        for i in range( m + 1 ):
            y_plus = finite_difference( y_plus , np.sin(theta) )
    
    return y_plus(y)/float( misc.factorial( exponent ) )  




##########################################################
##########################################################
####                                                  ####
####            INIT B-SPLINE LOOK UP TABLE           ####
####                                                  ####
##########################################################
########################################################## 

##  Formula n. 28 of M. Nilchian's paper:
##  "Fast iterative reconstruction of differential phase contrast
##   X-ray tomograms", M. Nilchian et al., Optics Express, 2013.
##
##  R^{n}{beta(x)}( y , theta ) = sum_{k1=0}^{m+1} sum_{k2=0}^{m+1} (-1)^{k1+k2} *
##      * comb( m+1 , k1 ) * comb( m+1 , k2 ) * ( y + ( (m+1)/2 - k1 )cos(theta) + 
##      ( (m+1)/2 - k2 )sin(theta) )_{+}^{2m-n+1} / [( 2m-n+1 )! * cos(theta)^{m+1} *
##      * sin(theta)^{m+1}]

def init_lut_bspline( nsamples_y , angles , bspline_degree , rt_degree , proj_support_y ):
    m = bspline_degree
    n = rt_degree    
    exponent = 2*m - n + 1

    
    ##  Define y-range as equally spaced points between -(m+1)/2, ... ,(m+1)/1 .
    ##  Correct is to adopt a rectangular support with length nsamples_y * sqrt(2) ,
    ##  but we do without (function values in edges are very small) .
    ##  nsamples_y should be even
    yrange = proj_support_y
    yarray = np.arange( nsamples_y ).astype( np.float32 )
    yarray -= nsamples_y / 2.0 
    yarray /= np.float32( nsamples_y )
    yarray *= yrange
    yarray_tile = yarray.reshape( 1 , yarray.shape[0] )
    
    
    ##  Define the theta-range as equally spaced points between 0 ... pi
    nsamples_theta = len( angles )
    theta_array_tile = angles.reshape( len( angles ) , 1 )
    
    
    ##  Repeat the tile yarray_tile vertically for nsamples_theta times
    y_matrix = np.tile( yarray_tile , ( nsamples_theta , 1 ) )


    ##  Repeat the tile theta_array_tile for nsamples_y times
    theta_matrix = np.tile( theta_array_tile , ( 1 , nsamples_y ) )


    ##  Precalculate sin and cos of all the angles
    sin_theta_matrix = np.sin( theta_matrix )
    cos_theta_matrix = np.cos( theta_matrix ) 

    
    ##  Prepare denominator matrix: fact( 2m-n+1 ) * cos(theta)^{m+1} * sin(theta)^{m+1}
    ##  Take care to reassign to angles 0 , pi/2 , pi values different from 0
    power_matrix = np.power( sin_theta_matrix * cos_theta_matrix , m+1 )
    divisor = np.float32( misc.factorial( exponent ) ) * power_matrix

    ind_0 = np.argwhere( np.abs( angles ) < eps )
    ind_90 = np.argwhere( np.abs( angles - np.pi/2.0 ) < eps )
    ind_180 = np.argwhere( np.abs( angles - np.pi ) < eps ) 

    if len( ind_0 ) !=0:
        divisor[ind_0, :] = 1.0

    if len( ind_90 ) != 0:
        divisor[ind_90, :] = 1.0
    
    if len( ind_180 ) != 0:
        divisor[ind_180, :] = 1.0

    
    ##  Allocate memory for the radon transform coefficients
    ##  of the B-splines functions
    result = np.zeros( ( nsamples_theta , nsamples_y ) )


    ##  Compute numerator matrix
    for k_1 in  range( 0,  m+1+1 ):
        for k_2 in  range(0, m+1+1 ):
            num = y_matrix + ( m/2.0 + 0.5 - k_1 ) * cos_theta_matrix + \
                  ( m/2.0 + 0.5 - k_2 ) * sin_theta_matrix
            num[ num < 0.0] = 0.0
            num_power = np.power( num , exponent )
            num_power *= np.power( -1.0 , k_1 + k_2 ) * misc.comb( m + 1 , k_1 ) * misc.comb( m + 1 , k_2 )
            result += num_power
    
    
    ##  Divide for the divisor which is indipendent from the sums on k_1 and k_2 
    result /= divisor


    ##  Correcting for the general values 0 , pi/2 , pi with slow general formula
    print( ind_0 )
    print( ind_90 )
    print( ind_180  )
    if len( ind_0 ) != 0 or len( ind_90 ) != 0 or len( ind_180 ) != 0:  
        for y in range( nsamples_y ):
            if len( ind_0 ) != 0:
                result[ ind_0 , y ] = radon_n_bspline_general( yarray[y] , 0.0 , m , n )
            if len( ind_90 ) != 0: 
                result[ ind_90 , y ] = radon_n_bspline_general( yarray[y] , np.pi/2.0 , m , n )
            if len( ind_180 ) != 0: 
                result[ ind_180 , y ] = radon_n_bspline_general( yarray[y] , np.pi , m , n )

    return result 




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


    ##  Save list of projection angles
    if args.listang is True:        
        fileang = aux + '_list_angles.txt'
        angles *= 180.0/np.pi
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
    print('########################################################')
    print('#############    BSPLINE RADON TRANSFORM   #############')
    print('########################################################')
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



    ##  Get bspline_degreeolation scheme
    bspline_degree = args.bspline_degree
    print('\nB-Spline degree selected: ', bspline_degree)



    ##  Flag to specify whether you want to perform the
    ##  radon transform or its adjoint
    flag_adj = 0



    ##  Direct transformation on the pixel image into
    ##  ita B-spline image
    bspline_image = pixel_basis_to_bspline( image , bspline_degree )

    if args.plot is True:
        dis.plot( bspline_image , 'B-spline image' )
    

    
    ##  Precalculate look-up-table for B-spline
    nsamples_y = args.lut_size
    proj_support_y = bspline_degree + 1
    
    if args.dpc is False:
        rt_degree = 0
    else:
        rt_degree = 1
    
    LUT = init_lut_bspline( nsamples_y , angles , bspline_degree ,
                            rt_degree , proj_support_y )

    if args.plot is True:
        dis.plot( LUT , 'Look-up-table' )



    ##  Perform B-spline radon transform
    LUT = LUT.astype( np.float32 ) 
    angles = angles.astype( np.float32 )
    sino = np.zeros( ( nang , npix ) , dtype=np.float32 )

    time1 = time.time()
    exit = grt.radon( sino , bspline_image , flag_adj , LUT , LUT.shape[1] ,
                      npix , angles , nang , proj_support_y )
    time2 = time.time()

    sino[:,:] = sino[:,::-1]
    sino[:,:] = np.roll( sino , 1 , axis=1 )



    ##  Show sino     
    if args.plot is True:
        dis.plot( sino , 'Sinogram' )    


    
    ##  Save sino
    saveSino( sino , angles , args )

    
    
    ##  Time elapsed for the computation of the radon transform
    endTime = time.time()
    print('\n\nTime elapsed to run the radon: ', (time2-time1)/60.0)
    print('\n\nTime elapsed to run the radon: ', time2-time1) 
    print('Time elapsed for the run of the program: ', (endTime-startTime)/60.0)
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
