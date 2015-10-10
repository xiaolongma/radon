from __future__ import division , print_function
import sys
import os
import numpy as np
import glob
import multiprocessing as mproc




def function( command_base , sino_file ):
    command = command_base + ' -i ' + sino_file
    os.system( command )




def main():
    pathin = '/home/arcusfil/tomcat/Publications/forward_regridding_method_for_fast_iterative_tomographic_algorithms/forward_gridrec_paper_v2/results/forward_comparison/sins/'
    pathout = '/home/arcusfil/tomcat/Publications/forward_regridding_method_for_fast_iterative_tomographic_algorithms/forward_gridrec_paper_v2/results/forward_comparison/recs/'   

    cur_dir = os.getcwd()
    os.chdir( pathin )
    sino_list = sorted( glob.glob( '*bspline*.DMP' ) )
    n_sino = len( sino_list )
    os.chdir( cur_dir )

    command_base = 'python fbp_bspline.py -Di ' + pathin + ' -Do ' + pathout + ' -w 3'

    pool = mproc.Pool()
    for i in range( n_sino ):
        if sino_list[i].find( 'rt_anal' ) == -1:
            pool.apply_async( function , ( command_base , sino_list[i] ) )
    pool.close()
    pool.join()




if __name__ == '__main__':
    main()
