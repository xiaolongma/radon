#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######             COLLECTION OF FILTERING PROCEDURE FOR                 ####### 
#######                   TOMOGRAPHIC RECONSTRUCTION                      #######
#######                                                                   #######
#######        Author: Filippo Arcadu, arcusfil@gmail.com, 05/12/2014     #######
#######                                                                   #######
#################################################################################
#################################################################################
#################################################################################




####  PYTHON MODULES
from __future__ import division , print_function
import numpy as np
from scipy import signal as sgn




####  MY FORMAT VARIABLE
myint     = np.int
myfloat   = np.float32
mycomplex = np.complex64




##########################################################
##########################################################
####                                                  ####
####     SINOGRAM FILTERING IN THE FOURIER SPACE      ####
####                                                  ####
##########################################################
##########################################################

def filter_fft( sino , filt_name , rt_degree ):
    nang , npix = sino.shape

    nfreq = 2 * int( 2**( int( np.ceil( np.log2( npix ) ) ) ) ) 

    if filt_name != 'none':
        if rt_degree:
            filtarr = np.ones( nfreq + 1 , dtype=mycomplex ) + 1.0j
        else:
            filtarr = 2 * np.arange( nfreq + 1 ) / myfloat( 2 * nfreq )
        
        w = 2 * np.pi * np.arange( nfreq + 1 ) / myfloat( 2 * nfreq )
        d = 1.0

        if filt_name == 'shepp-logan':
            filtarr[1:] *= np.sin( w[1:] ) / ( 2.0 * d * 2.0 * d * w[1:] )

        elif filt_name == 'cosine':
            filtarr[1:] *= np.cos( w[1:] ) / ( 2.0 * d * w[1:] )  

        elif filt_name == 'hamming':
            filtarr[1:] *= ( 0.54 + 0.46 * np.cos( w[1:] )/d )

        elif filt_name == 'hanning':
            filtarr[1:] *= ( 1.0 + np.cos( w[1:]/d )/2.0 )

        if rt_degree:
            aux          = filtarr.real
            print( 'aux:\n' , aux )
            filtarr.real = filtarr.imag * ( -0.5 / np.pi )
            print( 'aux:\n' , aux )   
            filtarr.imag = filtarr.real * ( +0.5 / np.pi )
            filtarr[0]   = 0.0 + 0.0j

        print( '\n\nfiltarr:\n' , filtarr )

        filtarr = np.concatenate( ( filtarr , filtarr[nfreq-1:0:-1] ) , axis=0 )
        filtarr = np.outer( np.ones( ( nang , 1 ) ) , filtarr )

        sino_filt = np.concatenate( ( sino , np.zeros( ( nang , 2*nfreq - npix ) ) ),
                                      axis=1 )

        sino_filt[:,:] = np.real( np.fft.ifft( np.fft.fft( sino_filt , axis=1 ) * \
                                  filtarr , axis=1 ) )

        sino[:,:] = sino_filt[:,:npix]        

    return sino




##########################################################
##########################################################
####                                                  ####
####      SINOGRAM FILTERING THROUGH CONVOLUTION      ####
####                                                  ####
##########################################################
##########################################################

def filter_convolve( sino ):
    nang , npix = sino.shape
    npix_h = int( npix * 0.5 )
    filtarr = np.zeros( npix_h , dtype=myfloat )

    for i in range( npix_h ):
        if i == 0:
            filtarr[i] = 0.25
        elif i % 2 != 0:
            filtarr[i] = - 1.0 / ( np.pi**2 * i**2 )

    filtarr = np.concatenate( ( filtarr[:0:-1] , filtarr ) , axis=1 )

    sino_filt = np.zeros( ( nang , 2*npix - 2 ) , dtype=myfloat )
    for i in range( nang ):
        sino_filt[i,:] = np.convolve( sino[i,:] , filtarr , mode='full' )


    sino_filt = 2 * sino_filt[:,npix_h:npix_h+npix]
    sino_filt = np.roll( sino_filt , 1 , axis=1 )

    return sino_filt

