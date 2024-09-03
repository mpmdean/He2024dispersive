import edrixs
import numpy as np
import scipy as sp
from scipy.sparse import csr_array



def rmat_t2o_DFT():
    """
    Get rotational matrix from trigonal to octahedral notation for face-sharing scenario 
    (coordinates used in the DFT calculations for CrI3).
    
    Returns
    -------
    rmat : 3x3 2d arrays, dtype=float.
    """
    rmat = np.zeros((3,3), dtype=float)
    rmat[:,0] = [-np.sqrt(2/3),             0,  np.sqrt(1/3)]
    rmat[:,1] = [ np.sqrt(1/6), -np.sqrt(1/2),  np.sqrt(1/3)]
    rmat[:,2] = [ np.sqrt(1/6),  np.sqrt(1/2),  np.sqrt(1/3)]

    return rmat

def prepare_slater(info, U_dd, U_dp, scale_dd_i, scale_dd_n, scale_dp):
    """
    prepare slater parameters for the Fortran solver by scaling atomic values.
    """

    F2_dd_i = info['slater_i'][1][1] * scale_dd_i
    F4_dd_i = info['slater_i'][2][1] * scale_dd_i
    F0_dd_i = U_dd + edrixs.get_F0('d', F2_dd_i, F4_dd_i)
    
    F2_dd_n = info['slater_n'][1][1] * scale_dd_n
    F4_dd_n = info['slater_n'][2][1] * scale_dd_n
    F0_dd_n = U_dd + edrixs.get_F0('d', F2_dd_n, F4_dd_n)
    
    
    F2_dp = info['slater_n'][4][1] * scale_dp
    G1_dp = info['slater_n'][5][1] * scale_dp
    G3_dp = info['slater_n'][6][1] * scale_dp
    F0_dp = U_dp + edrixs.get_F0('dp', G1_dp, G3_dp)
    
    slater = ([F0_dd_i, F2_dd_i, F4_dd_i],  # initial
              [F0_dd_n, F2_dd_n, F4_dd_n, F0_dp, F2_dp, G1_dp, G3_dp])  # with core hole
    return slater

def prepare_hopping(H_MO, ten_dq, Delta, nd, U_dd, U_dp, zeta_d_i, zeta_d_n, printParams=True):
    """
    prepare general hopping matrix for the Fortran solver. We will adjust the splitting within the d 
    orbitals by the paramter "ten_dq", and adjust the relative splitting between d and ligand orbitals 
    by the parameters "Delta", "nd", "U_dd" and "U_dp", similar to the example here:
    https://nsls-ii.github.io/edrixs/auto_examples/example_3_AIM_XAS.html#charge-transfer-energy-scales.

    Parameters
    ----------
    H_MO: 10x10 2d array
        direct output from DFT calculations.
    """
    hopping_i = np.zeros((20,20), dtype=complex)
    hopping_n = np.zeros((20,20), dtype=complex)
    
    deg = ten_dq - (H_MO[0,0] - H_MO[1,1]) # additional 10Dq on top of the DFT results.
    dt2g = 0
    Leg = 0
    Lt2g = 0
    hopping_i_temp = H_MO + np.diag([deg, dt2g, dt2g, deg, dt2g, Leg, Lt2g, Lt2g, Leg, Lt2g, ])
    hopping_n_temp = H_MO + np.diag([deg, dt2g, dt2g, deg, dt2g, Leg, Lt2g, Lt2g, Leg, Lt2g, ])

    # on-site orbital energies
    E_d, E_L = edrixs.CT_imp_bath(U_dd, Delta, nd)
    E_dc, E_Lc, E_p = edrixs.CT_imp_bath_core_hole(U_dd, U_dp, Delta, nd)
    if printParams:
        message = ("E_d = {:.3f} eV\n"
                   "E_L = {:.3f} eV\n"
                   "E_dc = {:.3f} eV\n"
                   "E_Lc = {:.3f} eV\n"
                   "E_p = {:.3f} eV\n")
        print(message.format(E_d, E_L, E_dc, E_Lc, E_p))

    d_mean = np.mean(hopping_i_temp.diagonal()[:5])
    L_mean = np.mean(hopping_i_temp.diagonal()[5:])
    # d onsite
    hopping_i_temp[:5,:5] += np.diag([E_d - d_mean]*5)
    # d onsite with a core hole
    hopping_n_temp[:5,:5] += np.diag([E_dc - d_mean]*5)
    # L onsite
    hopping_i_temp[5:,5:] += np.diag([E_L - L_mean]*5)
    # L onsite with a core hole
    hopping_n_temp[5:,5:] += np.diag([E_Lc - L_mean]*5)

    if printParams:
        np.set_printoptions(precision=3, suppress=True)
        print("hopping_i:")
        print(np.real(hopping_i_temp))
        print("hopping_n:")
        print(np.real(hopping_n_temp))

    hopping_i[0:20:2,0:20:2] += hopping_i_temp
    hopping_i[1:20:2,1:20:2] += hopping_i_temp
    hopping_n[0:20:2,0:20:2] += hopping_n_temp
    hopping_n[1:20:2,1:20:2] += hopping_n_temp
    
    ## construct SOC for d orbitals and transform into the real harmonic basis
    hopping_i[:10,:10] += edrixs.cb_op(edrixs.atom_hsoc('d', zeta_d_i), edrixs.tmat_c2r('d', True))
    hopping_n[:10,:10] += edrixs.cb_op(edrixs.atom_hsoc('d', zeta_d_n), edrixs.tmat_c2r('d', True))
    
    return hopping_i, hopping_n, E_p


def get_fortran_eigvec(eigvec_num):
    '''
    read eigenvector for nth (n = eigvec_num) eigenvalue from files produced by the Fortran solver.
    '''
    f = open('eigvec.'+str(eigvec_num), 'rb')
    dt = np.dtype(np.complex128)
    ffile  = np.fromfile( f, dtype=dt, offset = 4)
    eigval = ffile[0:1].real[0] # first complex  number is the eigenvalue
    v_for  = ffile[1:]          # the rest of complex number are the eigenvector values
    f.close()

    return v_for
    
def decimalToBinary(n, norb):
    '''
    convert a decimal number to its binary form.
    norb is the number of orbitals (i.e., the number of digits).
    '''
    # call python method "bin" and remove the prefix "0b"
    binstr = bin(int(n)).replace("0b", "")
    # convert the string to a list using the edrixs convention (see: https://nsls-ii.github.io/edrixs/reference/fock_basis.html#edrixs.fock_basis.get_fock_full_N)
    binlen = len(binstr)
    binlist = []
    for i in range(norb):
        if i < binlen:
            binlist.append(int(binstr[-1-i]))
        else:
            binlist.append(0)
    return binlist

def get_fortran_fock_i(norb):
    '''
    read fock basis from file "fock_i.in" produced by the Fortran solver, and convert to the binary form.
    '''
    dec_arr = np.loadtxt('fock_i.in', dtype='int')[1:]
    fock_i = []
    for dec in dec_arr:
        fock_i.append(decimalToBinary(dec, norb))
    return fock_i


def rixs_analysis(eval_i, denmat):
    '''
    run RIXS state analysis after ED using Fortran solver. The evaluation of <S^2> is done in the fock basis using scipy sparse matrix.

    Parameters
    ----------
    eval_i and denmat: eigenvalues of initial Hamiltonian and density matrix, respectively. They are the output of the function ed_siam_fort.
    '''
    # convert density matrix from complex spherical harmonics basis to real spherical harmonics basis
    tmat = sp.linalg.block_diag(edrixs.tmat_c2r('d',True), edrixs.tmat_c2r('d',True))
    denmat_r = edrixs.cb_op(denmat, tmat)

    neval = denmat_r.shape[0]
    evals = (eval_i-eval_i[0])[0:neval]

    # calculate electron occupations for different types of orbitals
    d_eg = denmat_r.diagonal(axis1=1,axis2=2)[:,[0,1,6,7]].sum(axis=1).real
    d_t2g = denmat_r.diagonal(axis1=1,axis2=2)[:,[2,3,4,5,8,9]].sum(axis=1).real
    L_eg = denmat_r.diagonal(axis1=1,axis2=2)[:,[10,11,16,17]].sum(axis=1).real
    L_t2g = denmat_r.diagonal(axis1=1,axis2=2)[:,[12,13,14,15,18,19]].sum(axis=1).real
   
    # check spin state
    
    # total spin operator
    spin_mom_all = np.zeros((3, 20, 20), dtype=complex)
    spin_mom_all[:, :10, :10] = edrixs.get_spin_momentum(2)
    spin_mom_all[:, 10:, 10:] = edrixs.get_spin_momentum(2)
    # spin operator for Cr orbitals only
    spin_mom_Cr = np.zeros((3, 20, 20), dtype=complex)
    spin_mom_Cr[:, :10, :10] = edrixs.get_spin_momentum(2)

    # read fock basis and eigenvectors from files
    basis_i = np.array(get_fortran_fock_i(norb=20))
    evecs_i = []
    for i in range(neval):
        evecs_i.append(get_fortran_eigvec(eigvec_num=i+1))
    evecs_i = np.array(evecs_i).T

    # convert to scipy compressed sparse row array format
    evecs_i = csr_array(evecs_i)

    Sx_all = edrixs.two_fermion(spin_mom_all[0], basis_i)
    Sx_all = csr_array(Sx_all)
    Sx2_all = Sx_all @ Sx_all
    
    Sy_all = edrixs.two_fermion(spin_mom_all[1], basis_i)
    Sy_all = csr_array(Sy_all)
    Sy2_all = Sy_all @ Sy_all
    
    Sz_all = edrixs.two_fermion(spin_mom_all[2], basis_i)
    Sz_all = csr_array(Sz_all)
    Sz2_all = Sz_all @ Sz_all
    
    S2_all = Sx2_all + Sy2_all + Sz2_all

    # calculated total <S2> in scipy sparse matrix format and then convert to regular numpy array
    S2_all_val = (np.conj(np.transpose(evecs_i)) @ S2_all @ evecs_i).toarray().diagonal().real
    Sz_all_val = (np.conj(np.transpose(evecs_i)) @ Sz_all @ evecs_i).toarray().diagonal().real

    Sx_Cr = edrixs.two_fermion(spin_mom_Cr[0], basis_i)
    Sx_Cr = csr_array(Sx_Cr)
    Sx2_Cr = Sx_Cr @ Sx_Cr
    
    Sy_Cr = edrixs.two_fermion(spin_mom_Cr[1], basis_i)
    Sy_Cr = csr_array(Sy_Cr)
    Sy2_Cr = Sy_Cr @ Sy_Cr
    
    Sz_Cr = edrixs.two_fermion(spin_mom_Cr[2], basis_i)
    Sz_Cr = csr_array(Sz_Cr)
    Sz2_Cr = Sz_Cr @ Sz_Cr
    
    S2_Cr = Sx2_Cr + Sy2_Cr + Sz2_Cr
    
    S2_Cr_val = (np.conj(np.transpose(evecs_i)) @ S2_Cr @ evecs_i).toarray().diagonal().real
    Sz_Cr_val = (np.conj(np.transpose(evecs_i)) @ Sz_Cr @ evecs_i).toarray().diagonal().real

    # store results in a dictionary
    rixs_ana_results = dict(evals=evals, S2_all_val=S2_all_val, Sz_all_val=Sz_all_val, 
                            S2_Cr_val=S2_Cr_val, Sz_Cr_val=Sz_Cr_val, 
                            d_eg=d_eg, d_t2g=d_t2g, L_eg=L_eg, L_t2g=L_t2g)   
    return rixs_ana_results