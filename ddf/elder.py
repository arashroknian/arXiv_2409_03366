import numpy as np
import porepy as pp

import ddf.common as co
from ddf.qddf import DDF, dizionario_post as dict_post


class Elder(DDF):
    def condizioni_iniziali_sd(self, sd):
        x = sd.cell_centers[0,:]
        zero = 0*x
        return { 'pressione': zero.copy(), 'concentrazione': zero.copy() }
    def condizioni_iniziali_intf(self, intf):
        x = intf.cell_centers[0,:]
        zero = 0*x
        return { 'laambda': zero.copy(), 'mu': zero.copy(), 'eta': zero.copy() }

    def bc_darcy(self, sd):
        if sd.dim == 0: return { 'bc': pp.BoundaryCondition(sd), 'bc_values': np.array([]) }
        tutte, sinistra, destra, giu, su = co.domain_boundary_faces(self.mdg, sd)
        facce_dir = np.zeros_like(giu)
        facce_dir[np.nonzero(giu)[0][0]] = True
        bc = pp.BoundaryCondition(sd, faces=facce_dir, cond='dir')

        bc_val = np.zeros(sd.num_faces)
        return { 'bc': bc, 'bc_values': bc_val }
    def bc_trasporto(self, sd):
        if sd.dim == 0: return { 'bc': pp.BoundaryCondition(sd), 'bc_values': np.array([]) }
        tutte, sinistra, destra, giu, su = co.domain_boundary_faces(self.mdg, sd)
        sopra = (150 < sd.face_centers[0,:]) & (sd.face_centers[0,:] < 450) & su
        
        if self.dati['scenario'] in ['shafabakhsh']: facce_dir = sopra
        else: facce_dir = sopra | giu
        bc = pp.BoundaryCondition(sd, faces=facce_dir, cond='dir')

        bc_val = np.zeros(sd.num_faces)
        bc_val[sopra] = 1

        return { 'bc': bc, 'bc_values': bc_val }

    def converged(self, soluzione, incrementi):
        if self.theta._value == 1 and len(incrementi) > 0: return 1

        dof_w = self.dof['concentrazione']
        incremento = incrementi[-1]
        ok = np.linalg.norm(incremento[dof_w], np.inf) < 1e-8
        
        return ok


dati_comuni = {
    'viscosita': 1e-3, 
    'g': 9.81, 
    'permeabilita_matrice': 4.845e-13, 
    'alpha': 0.2, 
    'diffusivita_matrice': 3.565e-6,
    'porosita_matrice': 0.1,
    'rho_w': 1000,
    'concentrazione_max': 1,
    
    'apertura': 1e-10,
    'permeabilita_fratture': 1,
    'diffusivita_fratture': 1,
}
pd = (300, 150)

def diersch02(l):
    Nx = 2**(l + 1)
    mdg = pp.meshing.cart_grid( [], nx=np.array([Nx, Nx//2]), physdims=pd )
    dati = dati_comuni.copy() | { 'scenario': 'diersch02' }
    return mdg, dati

def shafabakhsh(fig):
    def orizzontale(y, da, a): return np.array([[da, a], [y, y]])
    def verticale(x, da, a):   return np.array([[x, x],  [da, a]])

    dati = dati_comuni.copy() | { 'scenario': 'shafabakhsh'}

    if 'fig8' in fig: mdg = pp.meshing.cart_grid( [], nx=np.array([70, 35]), physdims=pd )
    elif 'fig9' in fig:
        dati['diffusivita_fratture'] = dati['diffusivita_matrice']

        mdg = pp.meshing.cart_grid( [
            orizzontale(150/8 + 0*150/4, 0, 300),
            orizzontale(150/8 + 1*150/4, 0, 300),
            orizzontale(150/8 + 2*150/4, 0, 300),
            orizzontale(150/8 + 3*150/4, 0, 300),

            verticale(300/8 + 0*300/4, 0, 150),
            verticale(300/8 + 1*300/4, 0, 150),
            verticale(300/8 + 2*300/4, 0, 150),
            verticale(300/8 + 3*300/4, 0, 150)
        ], nx=np.array([80, 40]), physdims=pd )
    else: raise SystemError

    if fig == 'fig8 ra60': dati['diffusivita_matrice'] = 2.376e-5
    elif fig == 'fig8 ra40': dati['diffusivita_matrice'] = 3.565e-5
    elif fig == 'fig8 ra20': dati['diffusivita_matrice'] = 7.129e-5
    elif fig == 'fig9 d1': dati['apertura'] = 0.8e-4
    else: raise SystemError

    dati['permeabilita_fratture'] = dati['apertura']**2/12

    return mdg, dati



def norma_concentrazione(self):
    init = not ('norma_concentrazione' in self.scalari.keys())
    if init: self.scalari['norma_concentrazione'] = []

    w = self.stato(dof_key='concentrazione')
    norma_concentrazione = np.sqrt(np.sum( w*(self._M@w) ))
    self.scalari['norma_concentrazione'].append(norma_concentrazione)

    return []

def sherwood_shafa(self):
    init = not ('sherwood' in self.scalari.keys())
    if init: self.scalari['sherwood'] = []

    subdomains = self.mdg.subdomains()
    D = self.mdg.dim_max()
    # TODO: con le fratture che toccano il bordo, dovrei considerare anche gli altri sd
    sD = [sd for sd in subdomains if sd.dim == D][0]

    _, _, _, _, su = co.domain_boundary_faces(self.mdg, sD)
    I = self.ops_velocita['I'].evaluate(self.dof_manager).val[self.fdof[sD]][su]

    sherwood = np.abs( np.sum(I * self.dati['concentrazione_max']/self.car['L']) ) # equazione 8 di Shafabakhsh
    self.scalari['sherwood'].append(sherwood)
    
    return []

dizionario_post = dict_post | { 'norma_concentrazione': norma_concentrazione, 'sherwood_shafa': sherwood_shafa }

