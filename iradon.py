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



####  PYTHON PLOTTING MODULES
import matplotlib.pyplot as plt
import matplotlib.cm as cm
       


####  MY PYTHON MODULES
sys.path.append('/home/arcusfil/tomcat/Programs/python_ambient/Common/')
import myImageIO as io



####  MY FORMAT VARIABLES
myFloat = np.float64



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
    parser = argparse.ArgumentParser(description='Inverse Radon Transform -- FBP')
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
    parser.add_argument('-w',dest='interp', default='nn',
                        help='''Select interpolation scheme:
                              nearest neighbour  --->  nn ( default )
                              linear  ---> lin
                             ''')
    parser.add_argument('-z',dest='edgepad', type=myFloat, default=0.0,
                        help='Select edge padding')
    parser.add_argument('-c',dest='ctr', type=myFloat,
                        help='Specify the center of rotation ( default = the middle pixel of the image )')          
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
####                  FIND RAY LIMITS                 ####
####                                                  ####
##########################################################
##########################################################

def rayLimits( t , theta , n ):
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

def iradonTransform( sino , angles , interp ):
    npix = sino.shape[1]
    npixh = int( npix * 0.5 )
    nang = len( angles )
    nangh = int( nang * 0.5 )

    reco = np.zeros( ( npix , npix ) , dtype=myFloat )

    
    ##  Array with the coordinates of the first and last pixel
    ##  to be crossed: limits = [ x_min , x_max , y_min , y_max ]
    limits = np.zeros( 4 , dtype=int )


    ##  Create ramp filter
    nfreq = npix
    nfreqh = int( nfreq * 0.5 )
    ramp = np.zeros( nfreq )
    ind_odd = np.arange( 1 , nfreqh , 2 )
    ramp[nfreqh+1::2] = - 1.0 / ( ind_odd * ind_odd * np.pi * np.pi * nfreqh * nfreqh  )
    ramp[nfreqh-1::-2] = ramp[nfreqh+1::2] 
    ramp[nfreqh] = 1.0 / ( 4.0 * nfreqh * nfreqh )
    #ramp[1:nfreqh] = ramp[nfreq-2:nfreqh:-1]
    #print('\nRamp filter:\n', ramp)
    #plt.plot( ramp ); plt.title('Real filter'); plt.show()
    ramp = np.abs( np.fft.fft( ramp ) )
    #plt.plot( ramp ); plt.title('FFT filter'); plt.show()

    fsino = np.zeros( 2*nfreq , dtype=np.complex64 )


    ##  Nearest neighbour interpolation
    if interp == 'nn':
        ##  Loop on the projection angles
        for i in range( nangh + 1 ):
            theta = angles[i]
            print('Back - projecting at theta: ', theta * 180.0/ np.pi)


            ##  Filter projection
            fsino[:nfreq] = np.fft.fft( sino[i,:] )
            fsino[:nfreq] *= ramp
            sino[i,:] = np.real( np.fft.ifft( fsino[:nfreq] ) )

            if theta != 0:
                fsino[:nfreq] = np.fft.fft( sino[nang-i,:] )
                fsino[:nfreq] *= ramp
                sino[nang-i,:] = np.real( np.fft.ifft( fsino[:nfreq] ) )


            ##  Loop on the radial variable
            for j in range( -npixh , npixh ):
                ##  Pixel index for sino
                k = j + npixh

                ##  Find first and last pixel to be crossed
                limits[:] = rayLimits( j , theta , npix )

                ##  Angles theta |  0 =< theta =< 45  U  135 =< theta =< 180  
                if np.abs( np.sin( theta ) ) <= np.sqrt(2)/2:
                    ##  Sinogram point at angle theta and radius j
                    y = np.arange( limits[2] , limits[3] )
                    x = np.round( j/np.cos(theta ) - y*np.sin(theta)/np.cos(theta) )
                    
                    x1 = x + npixh
                    x1 = x1.astype( np.int )
                    
                    y += npixh
                    y = y.astype( np.int )

                    reco[x1[:],y[:]] += sino[i,k] * 1/np.abs( np.cos( theta ) )  

                    ##  Sinogram point at angle pi-theta and radius j
                    if theta != 0:
                        x2 = -x + npixh
                        ind = np.argwhere( ( x2 >= 0 ) & ( x2 < npix ) )
                        x2 = x2.astype( np.int )

                        reco[x2[ind],y[ind]] += sino[nang-i,k] * 1/np.abs( np.cos( theta ) )

                ##  Angles theta |  45 < theta < 135 
                else:
                    ##  Sinogram point at angle theta and radius j
                    x = np.arange( limits[0] , limits[1] )
                    y = np.round( j/np.sin(theta) - x*np.cos(theta)/np.sin(theta) )

                    x1 = x + npixh
                    x1 = x1.astype( np.int )
                    
                    y += npixh
                    y = y.astype( np.int )

                    reco[x1[:],y[:]] += sino[i,k] * 1/np.abs( np.sin( theta ) )
                
                    ##  Sinogram point at angle pi-theta and radius j 
                    if np.abs( theta - np.pi/2 ) > eps:
                        x2 = -x + npixh
                        ind = np.argwhere( ( x2 >= 0 ) & ( x2 < npix ) )
                        x2 = x2.astype( np.int )
                        
                        reco[x2[ind],y[ind]] += sino[nang-i,k] * 1/np.abs( np.sin( theta ) )

    '''
    ##  Linear interpolation
    elif interp == 'lin':
        ##  Loop on the projection angles
        for i in range( nangh + 1 ):
            theta = angles[i]
            print('Projecting at theta: ', theta * 180.0/ np.pi) 

            ##  Loop on the radial variable
            for j in range( -npixh , npixh ):
                ##  Pixel index for sino 
                k = j + npixh

                ##  Find first and last pixel to be crossed
                limits[:] = rayLimits( j , theta , npix )

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

                    sino[i,k] = 1/np.abs( np.cos( theta ) ) * ( np.sum( ( 1 - w[ind] ) * \
                        reco[x1[ind],y[ind]] + w[ind] * reco[x2[ind],y[ind]] ) \
                        + reco[x1[len(x1)-1],y[len(x1)-1]] )

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
                        
                        sino[nang-i,k] = 1/np.abs( np.cos( theta ) ) * ( np.sum( ( 1 - w[ind] ) * \
                            reco[x1[ind],y[ind]] + w[ind] * reco[x2[ind],y[ind]] ) + \
                            np.sum( reco[x1[ind_out],y[ind_out]] ) )

                
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

                    sino[i,k] = 1/np.abs( np.sin( theta ) ) * ( np.sum( ( 1 - w[ind] ) * \
                        reco[x1[ind],y1[ind]] + w[ind] * reco[x1[ind],y2[ind]] ) + \
                        reco[x1[len(x1)-1],y1[len(x1)-1]] )
                
                    ##  Sinogram point at angle pi-theta and radius j                     
                    if np.abs( theta - np.pi/2 ) > eps:
                        x2 = -x + npixh
                        ind = np.argwhere( ( x2 >= 0 ) & ( x2 < npix ) & ( y2 >= 0 ) & ( y2 < npix ) )
                        x2 = x2.astype( np.int )

                        sino[nang-i,k] = 1/np.abs( np.sin( theta ) ) * ( np.sum( ( 1 - w[ind] ) * \
                            reco[x2[ind],y1[ind]] + w[ind] * reco[x2[ind],y2[ind]] ) + \
                            reco[x2[len(x2)-1],y1[len(x2)-1]] )
    '''
    
    return reco



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
    
    if args.reco is None:
        filename = args.sino
        filename = filename[:len(filename)-4]
        filename += '_iradon_' + args.interp + '_rec.DMP'
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
    print('##############################################')
    print('#############   RADON TRANSFORM  #############')
    print('##############################################')
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
        angles = ( angles * np.pi )/myFloat( nang )
        print('\nDealing with equally angularly spaced projections in [0,180)')

    ##  2) Case of list of projection angles in degrees
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
        ctr += int( 0.5 * ( npix - npix_old ))  
        print('\nEdge padded sinogram size: ', sino.shape[0],' X ', sino.shape[1])   




    ##  Compute radon transform
    reco = np.zeros( ( npix , npix ) , dtype=myFloat )
    reco[:,:] = iradonTransform( sino , angles , interp )


    
    ##  Crop image 
    if args.edgepad != 0:
        i1 = int( 0.5 * ( npix - npix_old ))
        i2 = i1 + npix_old
        print('\nnpix = ', npix,'  npix_old = ', npix_old,'  i1 = ', i1,
                '  i2 = ', i2)
        reco = reco[i1:i2,i1:i2]        

    
    
    ##  Show sino     
    if args.plot is True:
        checkPlot( reco , 'Sinogram' )


    
    ##  Save sino
    saveReco( reco , args )

    
    
    ##  Time elapsed for the computation of the radon transform
    endTime = time.time()
    print('Time elapsed: ', (endTime-startTime)/60.0)
    print('\n')



#####################################################################
#######    CALL TO MAIN
#####################################################################  
if __name__ == '__main__':
    main()
