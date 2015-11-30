##########################################################
##########################################################
####                                                  ####
####        CLASS TO HANDLE THE TOMOGRAPHIC           ####
####           FORWARD AND BACKPROJECTOR              ####
####                                                  ####
##########################################################
########################################################## 




####  PYTHON MODULES
import sys
import numpy as np




####  MY GRIDREC MODULE
sys.path.append( 'pymodule_genradon/' )
import bspline_functions as bfun
import genradon as gr
import myImageDisplay as dis
import filters as fil




####  MY FORMAT VARIABLES
myfloat = np.float32
myint = np.int




####  CLASS PROJECTORS
class projectors:

    ##  Init class projectors
    def __init__( self , npix , angles , ctr=0.0 , bspline_degree = 3 , proj_support_y=4 ,
                  nsamples_y=2048 , radon_degree=0 , filt='ramp' , plot=False ):
    
        ##  Compute regridding look-up-table and deapodizer
        self.plot = plot
        nang = len( angles )
        angles = np.arange( nang )
        angles = ( angles * 180.0 )/myfloat( nang )         
        lut = bfun.init_lut_bspline( nsamples_y , angles , 
                                     bspline_degree ,
                                     radon_degree , 
                                     proj_support_y )
        
       
        if plot is True:
            dis.plot( lut , 'Look-up-table' )          



        ##  Assign parameters
        self.lut            = lut.astype( myfloat )
        self.angles         = angles.astype( myfloat )
        self.nang           = nang
        self.bspline_degree = bspline_degree
        self.param_spline   = np.array( [ lut.shape[1] , proj_support_y ] , dtype=myfloat )
        self.filt           = filt
        self.radon_degree   = radon_degree



    
    ##  Forward projector
    def A( self , x ):
        return gr.forwproj( x.astype( myfloat ) , self.angles , self.lut , self.param_spline )


    
    ##  Backprojector
    def At( self , x ):
        return gr.backproj( x.astype( myfloat ) , self.angles , self.lut , self.param_spline )



    ##  Filtered backprojection
    def fbp( self , x ):
        ##  Option DPC
        if self.radon_degree == 0:
            dpc = False
        else:
            dpc = True

        
        ##  Filtering projection
        x[:] = fil.filter_proj( x , ftype=self.filt , dpc=dpc )
        if self.plot is True:
            dis.plot( x , 'Filtered sinogram' )
    
        
        ##  Backprojection
        reco = gr.backproj( x.astype( myfloat ) , self.angles , self.lut , self.param_spline )


        ##  Normalization
        if dpc is True:
            reco *= np.pi / ( 1.0 * self.nang )
        else:
            reco *= np.pi / ( 2.0 * self.nang )         

        return reco



    ##  Conversion to pixel basis
    def conv_to_pix( self , x ):
        print(  self.bspline_degree )
        return bfun.convert_from_bspline_to_pixel_basis( x , self.bspline_degree )     

