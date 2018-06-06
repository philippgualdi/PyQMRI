import numpy as np

import time
import os
import h5py
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns

import Model_Reco as Model_Reco


import IRLL_Model_new as IRLL_Model
import goldcomp
import primaldualtoolbox
import matplotlib.pyplot as plt
import ipyparallel as ipp

def main():
    DTYPE = np.complex64
    np.seterr(divide='ignore', invalid='ignore')

################################################################################
### Initiate parallel interface ################################################
################################################################################
    c = ipp.Client()
################################################################################
### Select input file ##########################################################
################################################################################

    root = Tk()
    root.withdraw()
    root.update()
    file = filedialog.askopenfilename()
    root.destroy()

    name = file.split('/')[-1]
    file = h5py.File(file)

################################################################################
### Check if file contains all necessary information ###########################
################################################################################
    test_data = ['dcf', 'fa_corr', 'imag_dat', 'imag_traj', 'real_dat',
                 'real_traj']
    test_attributes = ['image_dimensions', 'tau', 'gradient_delay',
                       'flip_angle(s)', 'time_per_slice']

    for datasets in test_data:
        if not (datasets in list(file.keys())):
            file.close()
            raise NameError("Error: '" + datasets +
                            "' data was not provided/wrongly named!")
    for attributes in test_attributes:
        if not (attributes in list(file.attrs)):
            file.close()
            raise NameError("Error: '" + attributes +
                            "' was not provided/wrongly named as an attribute!")

################################################################################
### Read Data ##################################################################
################################################################################

    data = file['real_dat'][()].astype(DTYPE) +\
         1j*file['imag_dat'][()].astype(DTYPE)

    traj = file['real_traj'][()].astype(DTYPE) + \
         1j*file['imag_traj'][()].astype(DTYPE)

    dcf = np.array(goldcomp.cmp(traj),dtype=DTYPE)

    dimX, dimY, NSlice = (file.attrs['image_dimensions']).astype(int)
    dimX = 212
    dimY = 212

############### Set number of Slices ###########################################
    reco_Slices = 3
    os_slices = 20
    class struct:
      pass
    par = struct()
################################################################################
### FA correction ##############################################################
################################################################################
    par.fa_corr = file['interpol_fa'][()].astype(DTYPE)
################################################################################
### Pick slices for reconstruction #############################################
################################################################################
    data = data[None,:,int(NSlice/2)-\
              int(np.ceil(reco_Slices/2)):int(NSlice/2)+\
              int(np.floor(reco_Slices/2)),:,:]
    if reco_Slices ==1:
        data = data[:,:,None,:,:]

    par.fa_corr = (np.flip(par.fa_corr,axis=0)[int((NSlice-os_slices)/2)-\
              int(np.ceil(reco_Slices/2)):int((NSlice-os_slices)/2)+\
              int(np.floor(reco_Slices/2)),6:-6,6:-6])

    [NScan,NC,NSlice,Nproj, N] = data.shape

################################################################################
### Set sequence related parameters ############################################
################################################################################
    par.tau         = file.attrs['tau']
    par.td          = file.attrs['gradient_delay']
    par.NC          = NC
    par.dimY        = dimY
    par.dimX        = dimX
    par.fa          = file.attrs['flip_angle(s)']/180*np.pi
    par.NSlice      = NSlice
    par.NScan       = NScan
    par.N = N
    par.Nproj = Nproj

    par.unknowns_TGV = 2
    par.unknowns_H1 = 0
    par.unknowns = 2
################################################################################
### Coil Sensitivity Estimation ################################################
################################################################################
#% Estimates sensitivities and complex image.
#%(see Martin Uecker: Image reconstruction by regularized nonlinear
#%inversion joint estimation of coil sensitivities and image content)
    nlinvNewtonSteps = 6
    nlinvRealConstr  = False

    traj_x = np.real(np.asarray(traj))
    traj_y = np.imag(np.asarray(traj))

    config = {'osf' : 2,
              'sector_width' : 8,
              'kernel_width' : 3,
              'img_dim' : dimX}

    points = (np.array([traj_x.flatten(),traj_y.flatten()]))
    op = primaldualtoolbox.mri.MriRadialOperator(config)
    op.setTrajectory(points)
    op.setDcf(np.repeat(np.sqrt(dcf),NScan,axis=0).flatten().astype(np.float32)[None,...])
    op.setCoilSens(np.ones((1,dimX,dimY),dtype=DTYPE))


    par.C = np.zeros((NC,NSlice,dimY,dimX), dtype=DTYPE)
    par.phase_map = np.zeros((NSlice,dimY,dimX), dtype=DTYPE)

    result = []
    for i in range(NSlice):
        print('deriving M(TI(1)) and coil profiles')


        ##### RADIAL PART
        combinedData = np.transpose(data[:,:,i,:,:],(1,0,2,3))
        combinedData = np.reshape(combinedData,(NC,NScan*Nproj*N))
        coilData = np.zeros((NC,dimY,dimX),dtype=DTYPE)
        for j in range(NC):
            coilData[j,:,:] = op.adjoint(combinedData[j,:]*(np.repeat(np.sqrt(dcf),NScan,axis=0).flatten())[None,...])

        combinedData = np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY)
        dview = c[int(np.floor(i*len(c)/NSlice))]
        result.append(dview.apply_async(nlinvns.nlinvns, combinedData,
                                        nlinvNewtonSteps, True, nlinvRealConstr))

        for i in range(NSlice):
            par.C[:,i,:,:] = result[i].get()[2:,-1,:,:]
        if not nlinvRealConstr:
            par.phase_map[i,:,:] = np.exp(1j * np.angle( result[i].get()[0,-1,:,:]))
            par.C[:,i,:,:] = par.C[:,i,:,:]* np.exp(1j*np.angle( result[i].get()[1,-1,:,:]))

          # standardize coil sensitivity profiles
        sumSqrC = np.sqrt(np.sum((par.C * np.conj(par.C)),0)) #4, 9, 128, 128
        if NC == 1:
            par.C = sumSqrC
        else:
            par.C = par.C / np.tile(sumSqrC, (NC,1,1,1))
################################################################################
### Reorder acquired Spokes   ##################################################
################################################################################
    if file.attrs['data_normalized_with_dcf']:
        pass
    else:
        data = data*np.sqrt(dcf)

    Nproj_new = 13
    Nproj_measured = Nproj
    NScan = np.floor_divide(Nproj,Nproj_new)
    Nproj = Nproj_new
    par.Nproj = Nproj
    par.NScan = NScan

    data = np.transpose(np.reshape(data[:,:,:,:Nproj*NScan,:],\
                                 (NC,NSlice,NScan,Nproj,N)),(2,0,1,3,4))
    traj =np.reshape(traj[:Nproj*NScan,:],(NScan,Nproj,N))
    dcf = np.array(goldcomp.cmp(traj),dtype=DTYPE)

    ################################################################################
    ### Calcualte wait time   ######################################################
    ################################################################################
    par.TR = file.attrs['time_per_slice']-(par.tau*Nproj*NScan+par.td)

    ################################################################################
    ### Standardize data norm ######################################################
    ################################################################################

    #### Close File after everything was read
    file.close()
    dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(2000))/(np.linalg.norm(data.flatten()))
    par.dscale = dscale
    ################################################################################
    ### generate nFFT for radial cases #############################################
    ################################################################################
    def gpuNUFFT(NScan,NSlice,dimX,traj,dcf,Coils):
        plan = []

        traj_x = np.real(np.asarray(traj))
        traj_y = np.imag(np.asarray(traj))

        config = {'osf' : 2,
                  'sector_width' : 8,
                  'kernel_width' : 3,
                  'img_dim' : dimX}
        for i in range(NScan):
            plan.append([])
            points = (np.array([traj_x[i,:,:].flatten(),traj_y[i,:,:].flatten()]))
            for j in range(NSlice):
                op = primaldualtoolbox.mri.MriRadialOperator(config)
                op.setTrajectory(points)
                op.setDcf(dcf.flatten().astype(np.float32)[None,...])
                op.setCoilSens(np.require(Coils[:,j,...],DTYPE,'C'))
                plan[i].append(op)
        return plan

    def nFTH_gpu(x,Coils):
        traj_x = np.real(np.asarray(traj))
        traj_y = np.imag(np.asarray(traj))

        config = {'osf' : 2,
                'sector_width' : 8,
                'kernel_width' : 3,
                'img_dim' : dimX}
        result = np.zeros((NScan,NSlice,dimX,dimY),dtype=DTYPE)
        x = np.require(np.reshape(x,(NScan,NC,NSlice,Nproj*N)))
        for scan in range(NScan):
            points = (np.array([traj_x[scan,:,:].flatten(),traj_y[scan,:,:].flatten()]))
            for islice in range(NSlice):
                  op = primaldualtoolbox.mri.MriRadialOperator(config)
                  op.setTrajectory(points)
                  op.setDcf(dcf.flatten().astype(np.float32)[None,...])
                  op.setCoilSens(np.require(Coils[:,islice,...],DTYPE,'C'))
                  result[scan,islice,...] = op.adjoint(np.require(x[scan,:,islice,...],DTYPE,'C'))
        return result

    data = data* dscale
    images= nFTH_gpu(data,par.C)
    del op


################################################################################
### IRGN - TGV Reco ############################################################
################################################################################

################################################################################
### Init forward model and initial guess #######################################
################################################################################
    model = IRLL_Model.IRLL_Model(par.fa,par.fa_corr,par.TR,par.tau,par.td,\
                                NScan,NSlice,dimY,dimX,Nproj,Nproj_measured,1,images)

    opt = Model_Reco.Model_Reco(par)

    opt.par = par
    opt.data =  data
    opt.images = images
    opt.dcf = (dcf)
    opt.dcf_flat = (dcf).flatten()
    opt.model = model
    opt.traj = traj

################################################################################
#IRGN Params
    irgn_par = struct()
    irgn_par.start_iters = 100
    irgn_par.max_iters = 300
    irgn_par.max_GN_it = 20
    irgn_par.lambd = 1e2
    irgn_par.gamma = 1e0
    irgn_par.delta = 1e-1
    irgn_par.omega = 0e0
    irgn_par.display_iterations = True
    irgn_par.gamma_min = 1e-1
    irgn_par.delta_max = 1e2
    irgn_par.tol = 5e-3
    irgn_par.stag = 1.00
    irgn_par.delta_inc = 2
    irgn_par.gamma_dec = 0.7
    opt.irgn_par = irgn_par
    opt.ratio = 1e1

    opt.execute_3D()

    result_tgv = opt.result
    res_tgv = opt.gn_res
    res_tgv = np.array(res_tgv)/(irgn_par.lambd*NSlice)
################################################################################
#### IRGN - TV referenz ########################################################
################################################################################
    result_tv = []
################################################################################
### Init forward model and initial guess #######################################
################################################################################
    model = IRLL_Model.IRLL_Model(par.fa,par.fa_corr,par.TR,par.tau,par.td,\
                                NScan,NSlice,dimY,dimX,Nproj,Nproj_measured,1,images)
    opt.model = model
################################################################################
##IRGN Params
    irgn_par = struct()
    irgn_par.max_iters = 300
    irgn_par.start_iters = 100
    irgn_par.max_GN_it = 13
    irgn_par.lambd = 1e2
    irgn_par.gamma = 1e0
    irgn_par.delta = 1e-1
    irgn_par.omega = 0e-10
    irgn_par.display_iterations = True
    irgn_par.gamma_min = 0.23
    irgn_par.delta_max = 1e2
    irgn_par.tol = 5e-3
    irgn_par.stag = 1.00
    irgn_par.delta_inc = 2
    irgn_par.gamma_dec = 0.7
    opt.irgn_par = irgn_par
    opt.ratio = 2e2

    opt.execute_2D(1)
    result_tv.append(opt.result)
    plt.close('all')
    res_tv = opt.gn_res
    res_tv = np.array(res_tv)/(irgn_par.lambd*NSlice)
################################################################################
#### IRGN - WT referenz ########################################################
################################################################################
    result_wt = []
################################################################################
### Init forward model and initial guess #######################################
################################################################################
    model = IRLL_Model.IRLL_Model(par.fa,par.fa_corr,par.TR,par.tau,par.td,\
                                NScan,NSlice,dimY,dimX,Nproj,Nproj_measured,1,images)
    opt.par = par
    opt.data =  data
    opt.images = images
    opt.dcf = (dcf)
    opt.dcf_flat = (dcf).flatten()
    opt.model = model
    opt.traj = traj

    opt.dz = 1

################################################################################
##IRGN Params
    irgn_par = struct()
    irgn_par.max_iters = 300
    irgn_par.start_iters = 100
    irgn_par.max_GN_it = 13
    irgn_par.lambd = 1e2
    irgn_par.gamma = 1e0
    irgn_par.delta = 1e-1
    irgn_par.omega = 0e-10
    irgn_par.display_iterations = True
    irgn_par.gamma_min = 0.37
    irgn_par.delta_max = 1e2
    irgn_par.tol = 5e-3
    irgn_par.stag = 1.00
    irgn_par.delta_inc = 2
    irgn_par.gamma_dec = 0.7
    opt.irgn_par = irgn_par
    opt.ratio = 2e2

    opt.execute_2D(2)
    result_wt.append(opt.result)
    plt.close('all')
    res_wt = opt.gn_res
    res_wt = np.array(res_wt)/(irgn_par.lambd*NSlice)
    del opt
################################################################################
### New .hdf5 save files #######################################################
################################################################################
    outdir = time.strftime("%Y-%m-%d  %H-%M-%S_MRI_joint_2D_"+name[:-3])
    if not os.path.exists('./output'):
        os.makedirs('./output')
    os.makedirs("output/"+ outdir)

    os.chdir("output/"+ outdir)
    f = h5py.File("output_"+name,"w")

    for i in range(len(result_tgv)):
      f.create_dataset("tgv_full_result_"+str(i),result_tgv[i].shape,\
                                   dtype=DTYPE,data=result_tgv[i])
      f.create_dataset("tv_full_result_"+str(i),result_tv[i].shape,\
                                       dtype=DTYPE,data=result_tv[i])
      f.create_dataset("wt_full_result_"+str(i),result_wt[i].shape,\
                                       dtype=DTYPE,data=result_wt[i])
      f.attrs['data_norm'] = dscale
      f.attrs['dscale'] = dscale
      f.attrs['res_tgv'] = res_tgv
      f.attrs['res_tv'] = res_tv
      f.attrs['res_wt'] = res_wt
      f.flush()
    f.close()

    os.chdir('..')
    os.chdir('..')

if __name__ == '__main__':
    main()
