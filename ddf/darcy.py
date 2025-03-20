
import numpy as np
import porepy as pp

import common as co
from boussinesq import Boussinesq

class Darcy(Boussinesq):
    def condizioni_iniziali_sd(self, sd):
        x,y,z = sd.cell_centers[0,:], sd.cell_centers[1,:], sd.cell_centers[2,:]
        W,H,P = co.grid_size(self.mdg)
        return { 'pressione': 0*x, 'concentrazione': 0*x }

    def condizioni_iniziali_intf(self, intf):
        x = intf.cell_centers[0,:]
        return { 'eta': 0*x, 'mu': 0*x, 'laambda': 0*x }

    def bc_darcy(self, sd):
        if sd.dim == 0: return { 'bc': pp.BoundaryCondition(sd), 'bc_values': np.array([]) }
        tutte, sinistra, destra, giu, su = co.domain_boundary_faces(self.mdg, sd)

        bc_val = np.zeros(sd.num_faces)

        if '_darcy_' in self.impostazioni['scenario']:
            facce_dir = sinistra | destra
            bc_val[sinistra] = 1
        elif '_diff_' in self.impostazioni['scenario']:
            facce_dir = tutte
        elif '_avv_' in self.impostazioni['scenario']:
            facce_dir = sinistra | destra
            bc_val[sinistra] = 1
        elif 'hrl' in self.impostazioni['scenario']:
            facce_dir = giu | su
            bc_val[su] = 1
        else: raise SystemError

        bc = pp.BoundaryCondition(sd, faces=facce_dir, cond='dir')
        return { 'bc': bc, 'bc_values': bc_val }

    def bc_trasporto(self, sd):
        if sd.dim == 0: return { 'bc': pp.BoundaryCondition(sd), 'bc_values': np.array([]) }
        tutte, sinistra, destra, giu, su = co.domain_boundary_faces(self.mdg, sd)
        bc_val = np.zeros(sd.num_faces)

        if '_darcy_' in self.impostazioni['scenario']:
            facce_dir = tutte
        elif '_diff_' in self.impostazioni['scenario']:
            facce_dir = sinistra | destra
            bc_val[sinistra] = 1
        elif '_avv_' in self.impostazioni['scenario']:
            facce_dir = sinistra | destra
            bc_val[sinistra] = 1
        elif 'hrl' in self.impostazioni['scenario']:
            facce_dir = tutte
        else: raise SystemError

        bc = pp.BoundaryCondition(sd, faces=facce_dir, cond='dir')
        return { 'bc': bc, 'bc_values': bc_val }

    def converged(self, soluzione, incrementi):
        incremento = incrementi[-1]
        max_incremento = np.linalg.norm(incremento, np.inf)
        # max_incremento_pressione = np.linalg.norm(incremento[self.dof['pressione']], np.inf)
        ok = max_incremento < 1e-10
        return ok

    def init(self):
        super().init()
        # self.dizionario_post['permeabilita_bulk'] = permeabilita_bulk
        # print(self.Ra)


    # def info(self):
    #     print(self.dati)
    # def post(self):     
    #     self.out_file.close()
        
    #     permeabilita = self.stima_permeabilita()
    #     print(f'stima permeabilita: {permeabilita}')

    # def stima_permeabilita(self):
    #     subdomains = self.mdg.subdomains()
    #     sd = [sd for sd in subdomains if sd.dim == self.D][0] # TODO: con le fratture che toccano il bordo, dovrei considerare anche gli altri sd
    #     _, _, _, giu, su = co.domain_boundary_faces(self.mdg, sd)
        
    #     V_d = co.distribuisci( self.ops['V'].evaluate(self.dof_manager).val, subdomains, facce=True )
    #     V = V_d[sd][giu] * self.ops['V0']
    #     V_avg = np.abs(np.mean(V / sd.face_areas[giu]))

    #     d = self.dati
    #     mult = d['viscosita']*d['porosita_matrice']/(d['rho_w']*d['g'])
    #     permeabilita = V_avg*mult
        
    #     return permeabilita        


    # def __init__(self, lettera, numero, dati_ext):
    #     mdg, dati_vg = vg14(lettera, numero, dati_ext)
    #     dati_darcy = {
    #         'nome': f'darcy_{lettera}{numero}',
    #         'dt': 0.1*secolo,
    #         'T': np.inf,
    #         'unita_tempo': secolo,
    #         'L': dati_vg['H'],
    #     }
    #     dati = dati_vg
    #     dati.update(dati_darcy)
    #     dati.update(dati_ext)

    #     self.mdg = mdg
    #     self.dati = dati
        
    #     self.dbg = True
    #     self.esporta_newton = False

    #     self.init()
    #     co.salva_script(self, ['common.py', 'media.py', 'ddf.py', 'darcy.py'])
    #     self.info()

    #     self.steady(True)

def scenario(nome, nx=np.array([60, 30])):
    dati = {
        'viscosita': 1, 'g': 0, 'permeabilita_matrice': 0.1, 'alpha': 1, 
        'diffusivita_matrice': 1, 'porosita_matrice': 0.1, 'rho_w': 1, 'concentrazione_max': 1, 
        'diffusivita_fratture': 1
    }

    assert ('_avv_' in nome) or ('_diff_' in nome) or ('_darcy_' in nome)
    assert ('_nofrac' in nome) or ('_1frac' in nome) or ('_2frac' in nome) or ('_qfrac' in nome)

    pd = (20, 10)
    def orizzontale(y, da, a): return np.array([[da, a], [y, y]])
    def verticale(x, da, a):   return np.array([[x, x],  [da, a]])

    impostazioni = {
        'scenario': nome,
        'nome': nome,
        'esporta': 1,
        'lista_post': ['tempo', 'velocita']
    }

    if '_nofrac' in nome:
        dati['apertura'] = 1
        dati['permeabilita_fratture'] = 1
    else:
        dati['permeabilita_matrice'] = 1e-16
        dati['apertura'] = 1e-4
        dati['permeabilita_fratture'] = dati['apertura']**2/12

    if '_avv_' in nome:
        dati['diffusivita_matrice'] = 1e-4
        dati['diffusivita_fratture'] = 1e-4

    if '_nofrac' in nome: lista_fratture = []
    elif '_1frac' in nome: lista_fratture = [orizzontale(5, 5, 15)]
    elif '_2frac' in nome: lista_fratture = [orizzontale(5, 5, 10), orizzontale(5, 10, 15)]
    elif '_qfrac' in nome: lista_fratture = [orizzontale(7, 5, 15), orizzontale(3, 5, 15), verticale(5, 3, 7), verticale(15, 3, 7)]
    else: raise SystemError

    mdg = pp.meshing.cart_grid(lista_fratture, nx=nx, physdims=pd )

    return mdg, dati, impostazioni


def permeabilita_bulk(self):
    pass


# class TDarcy(Darcy):
#     def post(self):     
#         self.out_file.close()

#     def converged(self, incrementi):
#         incremento = incrementi[-1]
#         max_incremento_concentrazione = np.linalg.norm(incremento[self.dof['concentrazione']], np.inf)
#         max_incremento_quota = np.linalg.norm(incremento[self.dof['quota']], np.inf)
#         ok = max_incremento_quota < 1e-6 and max_incremento_concentrazione < 1e-6
#         return ok

#     def bc_trasporto(self, sd):
#         if sd.dim == 0: return { 'bc': pp.BoundaryCondition(sd), 'bc_values': np.array([]) }

#         _, sinistra, destra, giu, su = co.domain_boundary_faces(self.mdg, sd)
#         facce_dir = giu | su
#         bc = pp.BoundaryCondition(sd, faces=facce_dir, cond='dir')
#         bc_val = np.zeros(sd.num_faces)
#         bc_val[su] = self.dati['concentrazione_max']

#         return { 'bc': bc, 'bc_values': bc_val }

#     def __init__(self, lettera, numero, dati_ext):
#         mdg, dati_vg = vg14(lettera, numero, dati_ext)
#         dati_darcy = {
#             'nome': f'darcy_{lettera}{numero}',
#             'dt': 0.1*secolo,
#             'T': 3*secolo,
#             'unita_tempo': secolo,
#             'L': dati_vg['H'],
#         }
#         dati = dati_vg
#         dati.update(dati_darcy)
#         dati.update(dati_ext)

#         self.mdg = mdg
#         self.dati = dati

#         co.init_output(self)
#         co.salva_script(self, ['common.py', 'media.py', 'ddf.py', 'darcy.py'])

#         self.init()
#         self.info()

# if __name__ == '__main__':
#     pb = Darcy('0', 3)

#     globals().update(pb.locali)
#     pb.risolvi()
#     pb.post()
