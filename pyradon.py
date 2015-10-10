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
import skimage
import skimage.transform



####  MY PYTHON MODULES
import myImageIO as io
import myPrint as pp
import myImageDisplay as dis
import myImageProcess as proc




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
    
    parser.add_argument('-w',dest='interp', default='lin',
                        help='''Select interpolation scheme:
                              nearest neighbour  --->  nn
                              linear  ---> lin
                             ''')        

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
####                        RADON                     ####
####                                                  ####
##########################################################
########################################################## 

def radon( image , npix , angles , ctr , interp ):

    sino = skimage.transform.radon( image , theta=angles , circle=True )
    
    return sino 




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

        filename += '_ang' + str_ang + '_radon_' + args.interp 
        
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
    print('########################################################')
    print('#############    SKIMAGE RADON TRANSFORM   #############')
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
        angles = ( angles * 180.0 )/myFloat( nang )
        print('\nDealing with equally angularly spaced projections in [0,180)')

    
    ##  2) Case of equally sloped projections distributed in [0,180)
    elif args.geometry == '1':
        print('\n\nDealing with equally-sloped projections in [0,180)')
  
        if nang % 4 != 0:
            print('\n\nERROR: in case of equally-sloped projections',
                  ' the number of angles has to be a multiple of 4')

        angles = createPseudoPolarAngles( nang ) * 180.0 / np.pi    

    
    ##  3) Case of list of projection angles in degrees
    else:
        geometryfile = pathin + args.geometry
        angles = np.fromfile( geometryfile , sep="\t" )
        nang = len( angles )
        print('\nReading list of projection angles: ', geometryfile)



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
        interp = 'lin'  



    ##  Center of rotation axis
    ctr = 0.5 * npix



    ##  Enable zero-padding
    if args.padding is True:
        npix_old = npix
        image = proc.zeroPaddingImg( image , 2 )
        npix = image.shape[0]
        dis.plot( image , 'Image zero-padded' )



    ##  Compute iradon transform
    print('\nPerforming Skimage Radon Transform ....\n')
    sino = radon( image[:,::-1] , npix , angles , ctr , interp )



    ##  Rotate sinogram
    sino = np.rot90( sino )

    

    ##  Remove edge_padding
    if args.padding is True:
        i1 = int( ( npix - npix_old ) * 0.5 )
        i2 = i1 + npix_old
        sino = sino[:,i1:i2]



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
