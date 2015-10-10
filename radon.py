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




####  MY PYTHON MODULES
import myImageIO as io
import myImageDisplay as dis




####  MY FORMAT VARIABLES
myfloat = np.float64




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
    parser = argparse.ArgumentParser(description='Radon Transform',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-Di', '--pathin', dest='pathin', default='./',
                        help='Specify path to input data')    
    
    parser.add_argument('-i', '--image', dest='image',
                        help='Specify name of input image')
    
    parser.add_argument('-Do', '--pathout', dest='pathout',
                        help='Specify path to output data') 
    
    parser.add_argument('-o', '--sino', dest='sino',
                        help='Specify name of output reconstruction')
    
    parser.add_argument('-n', '--nang', dest='nang', type=int, default=180,
                        help='Select the number of projection angles') 
    
    parser.add_argument('-g', '--geometry', dest='geometry',default='0',
                        help='Specify projection geometry; @@@@@@@@@@@@@@@@@@@@@@@'
                             +' -g 0 --> equiangular projections between 0 and 180 degrees;'
                             +' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
                             +' -g 1 --> equally sloped projections between 0 and 180'
                             +' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@' 
                             +' -g angles.txt use a list of angles (in degrees) saved in a text file')
    
    parser.add_argument('-w',dest='interp', default='lin',
                        help='''Select interpolation scheme:
                              nearest neighbour  --->  nn ( default )
                              linear  ---> lin
                             ''') 
    
    parser.add_argument('-p',dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')
    
    parser.add_argument('-l',dest='list_ang', action='store_true',
                        help='Write the list of projection angles') 
    
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
####               PSEUDO POLAR VIEWS                 ####
####                                                  ####
##########################################################
##########################################################

def create_pseudo_polar_angles( nang ):
    if nang % 4 != 0:
        raise Exception('\n\tError inside createPseudoPolarAngles:'
                        +'\n\t  nang (input) is not divisible by 4 !\n')
    n = nang
    pseudo_angles = np.zeros(n,dtype=myfloat)
    pseudoAlphaArr = np.zeros(n,dtype=myfloat)
    pseudoGridIndex = np.zeros(n,dtype=int)
    index = np.arange(int(n/4)+1,dtype=int)

    pseudo_angles[0:int(n/4)+1] = np.arctan(4*index[:]/myfloat(n))
    pseudo_angles[int(n/2):int(n/4):-1] = np.pi/2-pseudo_angles[0:int(n/4)]    
    pseudo_angles[int(n/2)+1:] = np.pi-pseudo_angles[int(n/2)-1:0:-1]
    
    return pseudo_angles  




##########################################################
##########################################################
####                                                  ####
####                  FIND RAY LIMITS                 ####
####                                                  ####
##########################################################
##########################################################

def ray_limits( t , theta , n ):
    nh = n * 0.5

    limits = np.zeros( 4 , dtype=int )
    limits_aux = [] 

    if theta == 0.0:
        limits[:] = [ 0 , 0 , -nh , nh ]
                
    elif np.abs( theta - np.pi/2 ) < eps:
        limits[:] = np.array([ -nh , nh , 0 , 0 ]) 
                
    else:
        y1 = t/np.sin(theta) + nh*np.cos(theta)/np.sin(theta)
        y2 = t/np.sin(theta) - (nh-1)*np.cos(theta)/np.sin(theta) 
        x1 = t/np.cos(theta) + nh*np.sin(theta)/np.cos(theta)
        x2 = t/np.cos(theta) - (nh-1)*np.sin(theta)/np.cos(theta)

        if y1 >= -nh-eps and y1 <= nh-1+eps:
            limits_aux.append( np.array([ -nh , y1 ]) )
        if y2 >= -nh-eps and y2 <= nh-1+eps:
            limits_aux.append( np.array([ nh-1 , y2 ]) )
        if x1 >= -nh-eps and x1 <= nh-1+eps:
            limits_aux.append( np.array([ x1 , -nh ]) )
        if x2 >= -nh-eps and x2 <= nh-1+eps:
            limits_aux.append( np.array([ x2 , nh-1 ]) )

        if len( limits_aux ) == 2 and np.abs(limits_aux[0][0]-limits_aux[0][1])>eps:
            limits[0] = np.round( np.min( [ limits_aux[0][0] , limits_aux[1][0] ] ) )
            limits[1] = np.round( np.max( [ limits_aux[0][0] , limits_aux[1][0] ] ) )
            limits[2] = np.round( np.min( [ limits_aux[0][1] , limits_aux[1][1] ] ) )
            limits[3] = np.round( np.max( [ limits_aux[0][1] , limits_aux[1][1] ] ) )
        else:
            limits[0] = limits_aux[0][0]
            limits[1] = limits_aux[0][0] 
            limits[2] = limits_aux[0][1]
            limits[3] = limits_aux[0][1] 

    return limits




##########################################################
##########################################################
####                                                  ####
####             RADON TRANSFORM FUNCTION             ####
####                                                  ####
##########################################################
##########################################################

def radon_transform( image , angles , interp ):
    npix = image.shape[0]
    npixh = int( npix * 0.5 )
    nang = len( angles )
    nangh = int( nang * 0.5 )

    
    ##  Get image diagonal
    pad = int( np.ceil( npix * np.sqrt( 2 ) ) * 0.5 )
    pad = 0
    shift = npixh


    ##  Allocate memory for sinogram
    sinogram = np.zeros( ( nang , npix + 2 * pad ) , dtype=myfloat )


    ##  Array with the coordinates of the first and last pixel
    ##  to be crossed: limits = [ x_min , x_max , y_min , y_max ]
    limits = np.zeros( 4 , dtype=int )

    ##  Nearest neighbour interpolation
    if interp == 'nn':
        ##  Loop on the projection angles
        for i in range( nangh + 1 ):
            theta = angles[i]
            print('\nProjecting at theta: ', theta * 180.0/ np.pi)

            ##  Loop on the radial variable
            for j in range( -npixh , npixh ):
                ##  Pixel index for sinogram
                k = j + npixh

                ##  Find first and last pixel to be crossed
                limits[:] = ray_limits( j , theta , npix )

                ##  Angles theta |  0 =< theta =< 45  U  135 =< theta =< 180  
                if np.abs( np.sin( theta ) ) <= np.sqrt(2)/2:
                    ##  Sinogram point at angle theta and radius j
                    y = np.arange( limits[2] , limits[3] )
                    x = np.round( j/np.cos(theta ) - y*np.sin(theta)/np.cos(theta) )

                    x1 = x + shift
                    x1 = x1.astype( np.int )
                    
                    y += shift
                    y = y.astype( np.int )

                    sinogram[i,k] = 1/np.abs( np.cos( theta ) ) * np.sum( image[x1[:],y[:]] )

                    ##  Sinogram point at angle pi-theta and radius j
                    if theta != 0:
                        x2 = -x + shift
                        ind = np.argwhere( ( x2 >= 0 ) & ( x2 < npix ) )
                        x2 = x2.astype( np.int )
                        
                        sinogram[nang-i,k] = 1/np.abs( np.cos( theta ) ) * np.sum( image[x2[ind],y[ind]] )

                ##  Angles theta |  45 < theta < 135 
                else:
                    ##  Sinogram point at angle theta and radius j
                    x = np.arange( limits[0] , limits[1] )
                    y = np.round( j/np.sin(theta) - x*np.cos(theta)/np.sin(theta) )

                    x1 = x + shift
                    x1 = x1.astype( np.int )
                    
                    y += shift
                    y = y.astype( np.int )

                    sinogram[i,k] = 1/np.abs( np.sin( theta ) ) * np.sum( image[x1[:],y[:]] )
                
                    ##  Sinogram point at angle pi-theta and radius j 
                    if np.abs( theta - np.pi/2 ) > eps:
                        x2 = -x + shift
                        ind = np.argwhere( ( x2 >= 0 ) & ( x2 < npix ) )
                        x2 = x2.astype( np.int )
                        
                        sinogram[nang-i,k] = 1/np.abs( np.sin( theta ) ) * np.sum( image[x2[ind],y[ind]])


    ##  Linear interpolation
    elif interp == 'lin':
        ##  Loop on the projection angles
        for i in range( nangh + 1 ):
            theta = angles[i]
            print('Projecting at theta: ', theta * 180.0/ np.pi) 

            ##  Loop on the radial variable
            for j in range( -npixh , npixh ):
                ##  Pixel index for sinogram 
                k = j + npixh

                ##  Find first and last pixel to be crossed
                limits[:] = ray_limits( j , theta , npix )

                ##  Angles theta |  0 =< theta =< 45  U  135 =< theta =< 180 
                if np.abs( np.sin( theta ) ) <= np.sqrt(2)/2:
                    ##  Sinogram point at angle theta and radius j 
                    y = np.arange( limits[2] , limits[3] )
                    x = j/np.cos(theta ) - y*np.sin(theta)/np.cos(theta)

                    x1 = np.floor( x )
                    w = np.abs( x - x1 )
                    x1 = x1.astype( np.int ) + npixh

                    x2 = np.floor( x ) + 1 + npixh
                    ind = np.argwhere( ( x2 >= 0 ) & ( x2 < npix ) )
                    x2 = x2.astype( np.int )
                    
                    y += npixh
                    y = y.astype( np.int )

                    sinogram[i,k] = 1/np.abs( np.cos( theta ) ) * ( np.sum( ( 1 - w[ind] ) * \
                        image[x1[ind],y[ind]] + w[ind] * image[x2[ind],y[ind]] ) \
                        + image[x1[len(x1)-1],y[len(x1)-1]] )

                    ##  Sinogram point at angle pi-theta and radius j 
                    if theta != 0:
                        x[:] = -x 
                        x1 = np.floor( x )
                        w = np.abs( x - x1 )
                        x1  += npixh
                        x2 = np.floor( x ) + 1 + npixh

                        ind = np.argwhere( ( x1 >= 0 ) & ( x1 < npix ) & ( x2 >= 0 ) & ( x2 < npix ) )
                        ind_out = np.argwhere( ( x1 >= 0 ) & ( x1 < npix ) & ( ( x2 < 0 ) \
                                  | ( x2 >= npix ) ) )

                        x1 = x1.astype( np.int )
                        x2 = x2.astype( np.int )
                        
                        sinogram[nang-i,k] = 1/np.abs( np.cos( theta ) ) * ( np.sum( ( 1 - w[ind] ) * \
                            image[x1[ind],y[ind]] + w[ind] * image[x2[ind],y[ind]] ) + \
                            np.sum( image[x1[ind_out],y[ind_out]] ) )

                
                ##  Angles theta |  45 < theta < 135 
                else:
                    ##  Sinogram point at angle theta and radius j 
                    x = np.arange( limits[0] , limits[1] )
                    y = j/np.sin(theta) - x*np.cos(theta)/np.sin(theta)

                    x1 = x + npixh
                    x1 = x1.astype( np.int )

                    y1 = np.floor( y )
                    w = np.abs( y - y1 ) 
                    y1 = y1.astype( np.int ) + npixh

                    y2 = np.floor( y ) + 1 + npixh
                    y2 = y2.astype( np.int )
                    ind = np.argwhere( ( y2 >= 0 ) & ( y2 < npix ) ) 

                    sinogram[i,k] = 1/np.abs( np.sin( theta ) ) * ( np.sum( ( 1 - w[ind] ) * \
                        image[x1[ind],y1[ind]] + w[ind] * image[x1[ind],y2[ind]] ) + \
                        image[x1[len(x1)-1],y1[len(x1)-1]] )
                
                    ##  Sinogram point at angle pi-theta and radius j                     
                    if np.abs( theta - np.pi/2 ) > eps:
                        x2 = -x + npixh
                        ind = np.argwhere( ( x2 >= 0 ) & ( x2 < npix ) & ( y2 >= 0 ) & ( y2 < npix ) )
                        x2 = x2.astype( np.int )

                        sinogram[nang-i,k] = 1/np.abs( np.sin( theta ) ) * ( np.sum( ( 1 - w[ind] ) * \
                            image[x2[ind],y1[ind]] + w[ind] * image[x2[ind],y2[ind]] ) + \
                            image[x2[len(x2)-1],y1[len(x2)-1]] )

    
    return sinogram




##########################################################
##########################################################
####                                                  ####
####                    SAVE SINOGRAM                 ####
####                                                  ####
##########################################################
##########################################################

def save_sinogram( sinogram , angles , args ):
    if args.pathout is None:
        pathout = args.pathin
    else:
        pathout = args.pathout

    if pathout[len(pathout)-1] != '/':
        pathout += '/'

    
    if args.sino is None:
        filename = args.image
        extension = filename[len(filename)-4:] 
        filename = filename[:len(filename)-4]
        nang = args.nang

        if nang < 10:
            str_nang = '000' + str( nang )
        elif nang < 100:
            str_nang = '00' + str( nang )
        elif nang < 1000:
            str_nang = '0' + str( nang )
        else:
            str_nang = str( nang )

        if args.geometry != '1':
            filename += '_ang' + str_nang + '_radon_' + args.interp + '_polar_sino'
        else:
            filename += '_ang' + str_nang + '_radon_' + args.interp + '_pseudo_sino' 
        filename = pathout + filename + extension
    else:
        filename = pathout + args.sino

    io.writeImage( filename , sinogram )


    if args.list_ang is True:
        listang = filename[:len(filename)-4]

        if args.geometry != '1':
            listang += '_ang' + str( args.nang ) + '_list.txt'
        else:
            listang += '_ang' + str( args.nang ) + '_pseudopol_list.txt'

        print('\nWriting list of projection angles:\n', listang) 
        fd = open( listang , 'w' )
        for i in range( len( angles ) ):
            fd.write('%.8f\n' % angles[i] )
        fd.close()




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
    print('##############################################')
    print('#############   RADON TRANSFORM  #############')
    print('##############################################')
    print('\n')


    ##  Get the startimg time of the reconstruction
    startTime = time.time()


    ##  Get input arguments
    args = getArgs()


    ##  Get path to input image
    pathin = args.pathin

    if pathin[len(pathin)-1] != '/':
        pathin += '/'


    ##  Get input image
    ##  You assume the image to be square
    image_name = pathin + args.image
    image = io.readImage( image_name )
    npix = image.shape[0]

    if image.shape[1] != image.shape[0]:
        sys.exit('\nERROR: input image is not square!\n')

    print('\nInput image:\n', image_name)
    print('Number of pixels: ', npix)


    ##  Show image
    if args.plot is True:
        dis.plot( image , 'Image' )


    ##  Get projection geometry  
    ##  1) Case of equiangular projections distributed in [0,180)
    if args.geometry == '0':
        nang = args.nang
        angles = np.arange( nang )
        angles = ( angles * np.pi )/myfloat( nang )
        print('\nDealing with equally angularly spaced projections in [0,180)')

    ##  2) Case of equally sloped projections distributed in [0,180) 
    elif args.geometry == '1':
        nang = args.nang
        angles = create_pseudo_polar_angles( nang )
        print('\nDealing with equally sloped projections in [0,180)') 

    ##  3) Case of list of projection angles in degrees
    else:
        geometryfile = pathin + args.geometry
        angles = np.fromfile( geometryfile , sep="\t" )
        angles *= np.pi/180.0
        nang = len( angles )
        print('\nReading list of projection angles: ', geometryfile)

    print('Number of projection angles: ', nang)


    ##  Get interpolation scheme
    interp = args.interp
    if interp == 'nn':
        print('\nSelected interpolation scheme: nearest neighbour')
    elif interp == 'lin':
        print('\nSelected interpolation scheme: linear interpolation')
    else:
        print('''\nWARNING: ', interp,' does not correspond to any available
                  interpolation scheme; nearest neighbour interpolation will
                  be adopted''')
        interp = 'nn'


    ##  Compute radon transform
    sinogram = radon_transform( np.rot90( image ) , angles , interp )

    
    ##  Show sinogram     
    if args.plot is True:
        dis.plot( sinogram , 'Sinogram' )


    ##  Save sinogram
    save_sinogram( sinogram , angles , args )


    ##  Time elapsed for the computation of the radon transform
    endTime = time.time()
    print('Time elapsed: ', (endTime-startTime)/60.0)
    print('\n')




##########################################################
##########################################################
####                                                  ####
####                    CALL TO MAIN                  ####
####                                                  ####
##########################################################
##########################################################

if __name__ == '__main__':
    main()
