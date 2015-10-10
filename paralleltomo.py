####  PYTHON MODULES
from __future__ import division , print_function
import numpy as np
from scipy.sparse import lil_matrix
import myPrint as pp
import myImageDisplay as dis
import myImageIO as io
import sys



####  MY FORMAT VARIABLE
myfloat = np.float32




####  MY CONSTANTS
eps = 1e-10
infp = 1e20
infn = -infp    



###########################################################
###########################################################
####                                                   ####
####          MATRIX REPRESENTATION FOR THE            ####
####   FORWARD PROJECTOR FOR PARALLEL BEAM GEOMETRY    ####
####                                                   ####
####   Python translation of paralleltomo.m inside     ####
####        the AIRtools package of C. Hansen          ####
####                                                   ####
###########################################################
###########################################################

####  Inputs:
####  angles  --->  array of projection angles ( given in degrees )
####  N       --->  total number of pixel of the image ( sqrt(N) x sqrt(N) ) 
####  p       --->  number of pixels for each projection; it is such that: p >= N
####
####  Output:
####  A       ---> matrix representation of the forward projector

def paralleltomo( angles , N , p , d ):
    ##  Convert angles to radiants
    theta = angles * np.pi / 180.0

    ##  Allocate memory for sparse matrix storing the forward projector
    M = len( angles )
    dim1 = M * N; dim2 = N * N
    A = lil_matrix( ( dim1 , dim2 ) , dtype=myfloat )
    #A = np.zeros( ( dim1 , dim2 ) , dtype=myfloat )


    ##  Starting values for both the x and y coordinates
    x0 = np.linspace( -0.5*d , 0.5*d , p )
    y0 = np.zeros( p , dtype=myfloat )
    
    #print('\nx0:', x0)


    ##  Intersection lines
    x = np.arange( -0.5*N , 0.5*N + 1 )
    y = x.copy()
    #print('x: ', x)


    ##  Initialize vectors that will be used inside the main loop
    x0theta = x0.copy(); y0theta = y0.copy()
    tx = x.copy(); ty = x.copy(); xy = x.copy(); yx = x.copy()
    t = np.zeros( 2 * len( x ) , dtype=myfloat )


    ##  Creating elements of the forward projector
    ##  Loop on the angles
    for i in range( M ): # M
        print('\ni = ', i)

        ##  Get all the starting points for the current angle
        s = np.sin( theta[i] ) ; c = np.cos( theta[i] )

        #print('s = ', s)
        #print('c = ', c)
        #print('x0 = ', x0)
        #print('y0 = ', y0)

        x0theta[:] = c * x0 - s * y0
        y0theta[:] = s * x0 + c * y0

        #print( '\nx0theta = ', x0theta )
        #print( '\ny0theta = ', y0theta )   


        ##  Loop on the rays
        for j in range( p ):  # p
            #print( '\nj = ' , j )

            ##  Use the parametrisation of line to get the y-coord.
            ##  of intersections with x = k, i.e. x constant
            if np.abs( s ) > eps:
                tx[:] =  - ( x - x0theta[j] ) / s
            else:
                tx[ (-( x - x0theta[j]) >= 0 ) ] = infp
                tx[ (-( x - x0theta[j] ) < 0 ) ] = infn
            yx[:] = c * tx + y0theta[j] 
                

            #print('tx = ', tx)
            #print('yx = ', yx)


            ##  Use the parametrisation of line to get the x-coord.
            ##  of intersections with y = k, i.e. y constant
            if np.abs( c ) > eps:             
                ty[:] = ( y - y0theta[j] ) / c
            else:
                ty[ ( ( y - y0theta[j] ) >= 0 ) ] = infp
                ty[ ( ( y - y0theta[j] ) < 0 ) ] = infn    
            xy[:] =  -s * ty + x0theta[j]    

            #print('ty = ', ty)
            #print('xy = ', xy)


            ##  Collect the intersections and coordinates
            t[:] = np.concatenate( ( tx , ty ) , axis=1 )
            xxy = np.concatenate( ( x , xy ) , axis=1 )
            yxy = np.concatenate( ( yx , y ) , axis=1 )

            #print('t = ', t)
            #print('xxy = ', xxy)
            #print('yxy = ', yxy)


            ##  Sort the acoordinates according to intersection time
            I = np.argsort( t )  
            t[:] = np.sort( t )
            xxy[:] = xxy[I]
            yxy[:] = yxy[I] 

            #print('t sort = ', t)
            #print('I = ', I)
    

            ##  Skip the points outside the box and double points
            #print( np.argwhere( ( xxy >= -N/2 ) & ( xxy <= N/2 ) & \
            #                   ( yxy >= -N/2 ) & ( yxy <= N/2 ) ) )
            #print( np.argwhere( np.abs( np.diff( xxy ) ) > eps ) )
            #print( np.argwhere( np.abs( np.diff( yxy ) ) > eps ) ) 
            I = np.argwhere( ( xxy >= -N/2 ) & ( xxy <= N/2 ) & \
                               ( yxy >= -N/2 ) & ( yxy <= N/2 ) )
                               #( np.abs( np.diff( xxy ) ) > eps ) & \
                               #( np.abs( np.diff( yxy ) ) > eps ) )
            xxy = xxy[I]
            yxy = yxy[I]

            #print('I = ', I)
            #print('xxy = ', xxy)
            #print('yxy = ', yxy)
            
            
            ##  Skip double points
            I = np.argwhere( ( np.abs( xxy[1:] - xxy[:-1] ) < eps ) & \
                             ( np.abs( yxy[1:] - yxy[:-1] ) < eps ) )
            if len( I ) != 0:
                np.delete( xxy , I )
                np.delete( yxy , I )

            #print('I 2nd = ', I)
            #print('xxy 2nd = ', xxy)
            #print('yxy 2nd = ', yxy) 


            ##  Calculate the length within cell and determines the
            ##  number of cells which gets hit
            d = np.sqrt( ( xxy[1:] - xxy[:-1] )**2 + ( yxy[1:] - yxy[:-1] )**2 )
            n_hit = len( d )

            #print('d = ', d)
            #print('n_hit = ', n_hit)


            ##  Store the values inside the box
            if n_hit > 0:
                ##  Neglect rays on the boundaries of the box
                cond1 = ( s == 0.0 ) and ( np.abs( x0theta[j] - N/2 ) < eps )
                cond2 = ( c == 0.0 ) and ( np.abs( y0theta[j] - N/2 ) < eps )

                #print( cond1 )
                #print( cond2 )
                
                if( not( cond1 or cond2 ) ):

                    ##  Calculate the midpoints of the line within the cells
                    xm = 0.5 * ( xxy[:-1] + xxy[1:] )  + N/2
                    ym = 0.5 * ( yxy[:-1] + yxy[1:] )  + N/2

                    #print('xm = ', xm)
                    #print('ym = ', ym)


                    ##  Translate the midpoint coordinates to index
                    cols = np.floor( xm ) * N + ( N - 1 - np.floor( ym ) )
                    cols = cols.astype( int )
                    cols = cols.reshape(-1)

                    #print('cols shape = ', cols.shape)
                    #print('cols = ', cols)


                    ##  Get rows coordinates
                    row = i * p + j
                    #print('row = ', row)

                    #print( A. shape )


                    ##  Fill the matrix
                    A[row, cols] = d
                     
    return A
    



###########################################################
###########################################################
####                                                   ####
####       Run the test for paralleltomo function      ####
####                                                   ####
###########################################################
########################################################### 

def main():
    x = io.readImage( sys.argv[1] )
    M = int( sys.argv[2] )
    dis.plot( x , 'Input image' )
    N , p = x.shape
    d = N;
    angles = np.arange( M )/myfloat( M ) * 180.0
    angles = np.fft.fftshift( angles )
    A = paralleltomo( angles , N , p , d )
    #dis.plot( A.todense() , 'Matrix A' )    
    sinogram = A.dot( x.reshape(-1) )
    sinogram = sinogram.reshape( M , N )
    dis.plot( sinogram , 'Output sinogram' )
    io.writeImage( sys.argv[1][:len(sys.argv[1])-4] + 'sino_par_tomo.DMP' , 
                   sinogram )



###########################################################
###########################################################
####                                                   ####
####                   Call to main test               ####
####                                                   ####
###########################################################
###########################################################

if __name__ == '__main__':
    main()
