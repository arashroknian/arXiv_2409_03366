import numpy as np
import porepy as pp

import ddf.common as co
from ddf.qddf import DDF, dizionario_post as dict_post

class HRL(DDF):
    def condizioni_iniziali_sd(self, sd):
        x,y,z = sd.cell_centers[0,:], sd.cell_centers[1,:], sd.cell_centers[2,:]
        D = self.mdg.dim_max()
        W,H,P = co.grid_size(self.mdg)

        pressione_0 = self.car['Ra']/2 * (1 - (y/H)**2)

        concentrazione_0 = y/H
        distanze = np.linalg.norm( sd.cell_centers - np.array([ W/2, H/2, P/2 ]).reshape((3,1)), axis=0 )
        vicinissimo = np.argmin(distanze)

        imp_c0 = self.dati.get('concentrazione_iniziale', 'lineare')
        if imp_c0 == 'zero': concentrazione_0 = 0*y
        elif imp_c0 == 'nodo':
            concentrazione_0 = y/H
            if sd.dim == D: concentrazione_0[vicinissimo] = 1
        elif imp_c0 == 'lineare': concentrazione_0 = y/H
        else: assert False

        return { 'pressione': pressione_0, 'concentrazione': concentrazione_0 }
    def condizioni_iniziali_intf(self, intf):
        x = intf.cell_centers[0,:]
        return { 'laambda': 0*x, 'mu': 0*x, 'eta': 0*x }
    def perturba_cella(self):
        W,H,P = co.grid_size(self.mdg)
        sD = self.mdg.subdomains()[0]

        distanze = np.linalg.norm( sD.cell_centers - np.array([ W/2, H/2, P/2 ]).reshape((3,1)), axis=0 )
        vicinissimo = np.argmin(distanze)

        nuovo_stato = self.stato()
        nuovo_stato[ self.dof[(sD, 'concentrazione')][vicinissimo] ] = 1 - 1e-4
        self.carica_stato(nuovo_stato)


    def bc_darcy(self, sd):
        if sd.dim == 0: 
            return { 'bc': pp.BoundaryCondition(sd), 'bc_values': np.array([]) }
        
        if sd.dim < self.mdg.dim_max():
            bc_val = np.zeros(sd.num_faces)
            bc = pp.BoundaryCondition(sd)
            return { 'bc': bc, 'bc_values': bc_val }

        tutte, sinistra, destra, giu, su = co.domain_boundary_faces(self.mdg, sd)
        facce_dir = np.zeros_like(su)
        facce_dir[np.nonzero(su)[0][0]] = True
        bc = pp.BoundaryCondition(sd, faces=facce_dir, cond='dir')
        bc_val = np.zeros(sd.num_faces)
        return { 'bc': bc, 'bc_values': bc_val }
    
    def bc_trasporto(self, sd):
        if sd.dim == 0: return { 'bc': pp.BoundaryCondition(sd), 'bc_values': np.array([]) }
        tutte, sinistra, destra, giu, su = co.domain_boundary_faces(self.mdg, sd)
        facce_dir = giu | su
        bc = pp.BoundaryCondition(sd, faces=facce_dir, cond='dir')
        bc_val = np.zeros(sd.num_faces)
        bc_val[su] = 1
        return { 'bc': bc, 'bc_values': bc_val }

import traceback
def sherwood(self):
    init = not ('sherwood' in self.scalari.keys())
    if init: self.scalari['sherwood'] = []

    subdomains = self.mdg.subdomains()
    D = self.mdg.dim_max()
    # TODO: con le fratture che toccano il bordo, dovrei considerare anche gli altri sd. credo si possano ignorare...
    sD = [sd for sd in subdomains if sd.dim == D][0]

    _, _, _, _, su = co.domain_boundary_faces(self.mdg, sD)
    flusso_diff = self.ops_velocita['I'] if self.impostazioni.get('boussinesq', 1) else self.ops_velocita['rhoI']
    I = flusso_diff.evaluate(self.dof_manager).val[self.fdof[sD]][su]

    _,H,_ = co.grid_size(self.mdg)
    sh = np.abs(np.sum(I)) * H/( self.car['L']*self.dati['intrusion_area'] )

    self.scalari['sherwood'].append( sh )
    return []

dizionario_post = dict_post | { 'sherwood': sherwood }


def artificiali(b, Nx=60):
    pd = np.array([2, 1])
    dati = { 
        'g': 1, 'concentrazione_max': 1, 'alpha': 1, 'rho_w': 1, 'viscosita': 1, 'porosita_matrice': 1, 'diffusivita_matrice': 1, 
        'permeabilita_matrice': 1, 'permeabilita_fratture': 1, 'diffusivita_fratture': 1, 'apertura': b
    }
    fratture = [ np.array([[0.5, 1.5], [0.5, 0.5]]) ]
    mdg = pp.meshing.cart_grid(fratture, nx=np.array([Nx, Nx//2]), physdims=pd )
    return mdg, dati


sh_vg14 = {
    '0A': 1.00, '0B': 1.00, '0C': 1.00, '0D': 1.00, # non ci sono nel paper, li tengo perche' mi fanno comodo
    'A1': 1.00, 'A2': 1.36, 'A3': 1.56, 'A4': 1.75,
    'B1': 1.00, 'B2': 1.13, 'B3': 1.49, 'B4': 1.32, 
    'C1': 1.00, 'C2': 1.17, 'C3': 1.21, 'C4': 1.21, 
    'D1': 1.08, 'D2': 1.00, 'D3': 1.00, 'D4': 1.01, 
    'D5': 1.07, 'D6': 1.08, 'D7': 1.28, 'D8': 1.15, 
    'D9': 1.15, 'D10': 1.45, 'D11': 1.37, 'D12': 1.39,
    'E9a': 1.06, 'E9b': 2.69,
}
def vg14(lettera, numero, grid_scale=1, tipo_griglia=None):
    H = 10
    pd = np.array([2*H, H])
    def orizzontale(y, da, a): return np.array([[da, a], [y, y]])  * np.array([2*H, H]).reshape((2, 1))
    def verticale(x, da, a):   return np.array([[x, x],  [da, a]]) * np.array([2*H, H]).reshape((2, 1))
    
    dati = {
        'g': 9.81,
        'concentrazione_max': 0.1,
        # 'concentrazione_min': 0.0,
        'alpha': 0.7, # il paper lo chiama beta
        # 'rho_max': 1070, # si puo' ricavare da rho_w, alpha, concentrazione_max
        'rho_w': 1000,
        'viscosita': 1.1e-3,
        'porosita_matrice': 0.1,
        'diffusivita_matrice': 1e-9, # diffusivita_molecolare * tortuosita
        # 'dispersivita_long': 0.1, 
        # 'dispersivita_trans': 0.005, # TODO: si riesce a implementare? e' importante farlo?

        # questi due li prendo = ai valori nella matrice
        # 'porosita_fratture': 0.1,
        'diffusivita_fratture': 1e-9,
        
        'intrusion_area': 2*H
    }

    if lettera == '0':
        fratture = []

        permeabilita_matrice = 1.6e-16 * numero
        apertura = 1e10 # qualcosa dev'esserci

    elif lettera == '1':
        fratture = []
        permeabilita_matrice = 1
        apertura = np.sqrt(12 * 1e-6) # -> permeabilita_frattura = 1e-3
        dati = dati | {
            'g': 1, 'concentrazione_max': 1, 'alpha': 1, 'rho_max': 1, 'rho_w': 1, 'viscosita': 1, 'porosita_matrice': 1, 
            'diffusivita_matrice': 1, 'diffusivita_fratture': 1,
        }

    elif lettera[0] == 'A':
        spacing = 10/2**(numero - 1)
        x_fr = np.arange(spacing/2, 2*H - spacing/2 + 0.1*spacing, spacing)
        y_fr = np.arange(spacing/2, H - spacing/2 + 0.1*spacing, spacing)
        fratture_verticali = [ np.array([[ x, x ], [0, H ]]) for x in x_fr ]
        fratture_orizzontali = [ np.array([[ 0, 2*H ], [y, y]]) for y in y_fr ]
        fratture = fratture_orizzontali + fratture_verticali

        if len(lettera) == 1:
            permeabilita_matrice = 1e-16
            apertura = [ np.nan, 46.9, 37.2, 29.6, 23.5, 18.6, 14.8][numero] * 1e-6 # tabella 4 in VG14
        elif lettera[1] == '*':
            permeabilita_matrice = 9*1e-16
            apertura = [ np.nan, 19.4, 15.4, 12.2, 9.7, 7.7, 6.1][numero] * 1e-6
        else: raise SystemError

        km = permeabilita_matrice
        permabilita_fratture = apertura**2 / 12
        kf = permabilita_fratture
        permeabilita_bulk = 1/( 1/(km + apertura*kf/spacing) + 1/( kf*spacing/apertura ) )

        # print(f'apertura: {apertura:.2e}, spacing: {spacing}, kf/km: {kf/km:.2e} permeabilita bulk: {permeabilita_bulk:.2e}')
        
    elif lettera[0] == 'B':
        if numero == 1:
            vs = [37, 51.8]
            hs = [59.8]
            fratture = [ verticale(v/200, 0, 1) for v in vs ] + [ orizzontale(h/100, 0, 1) for h in hs ]
        elif numero == 2:
            vs = [52.1, 105.9, 124, 158.2]
            hs = [37.6, 77.7]
            fratture = [ verticale(v/200, 0, 1) for v in vs ] + [ orizzontale(h/100, 0, 1) for h in hs ]
        elif numero == 3: fratture = [ orizzontale(0.73, 0, 1), orizzontale(0.62, 0, 1), orizzontale(0.22, 0, 1), verticale(0.01, 0, 1), verticale(0.12, 0, 1), verticale(0.25, 0, 1), verticale(0.33, 0, 1), verticale(0.49, 0, 1), verticale(0.52, 0, 1), verticale(0.73, 0, 1), verticale(0.93, 0, 1), ]
        elif numero == 4:
            vs = [10, 32.2, 51.6, 64, 67.8, 85.4, 98.1, 105.2, 117.6, 127.2, 134.9, 139.1, 144.9, 161.4, 187.3, 190]
            hs = [4.6, 27.2, 32, 37.5, 55.5, 71.7, 77.2]
            fratture = [ verticale(v/200, 0, 1) for v in vs ] + [ orizzontale(h/100, 0, 1) for h in hs ]
        else: raise SystemError

        permeabilita_matrice = 1e-16
        apertura = [ np.nan, 46.9, 37.2, 29.6, 23.5, 18.6, 14.8][numero] * 1e-6 # tabella 4 in VG14

    elif lettera[0] == 'C':
        spacing = 10/2**(numero - 1)
        x_fr = np.arange(spacing/2, 2*H + H + 0.1*spacing, spacing)
        y_fr = np.arange(spacing/2, H - spacing/2 + 0.1*spacing, spacing)

        fratture_diagonali = [ np.array([[ 0, x ], [x, 0]]) for x in x_fr ]
        fratture_orizzontali = [ np.array([[ 0, 2*H ], [y, y]]) for y in y_fr ]
        fratture = fratture_orizzontali + fratture_diagonali

        permeabilita_matrice = 1e-16
        apertura = [ np.nan, 46.9, 37.2, 29.6, 23.5, 18.6, 14.8][numero] * 1e-6 # tabella 4 in VG14

    elif lettera == 'D':
        numero = str(numero)
        if numero == '1': fratture = [ verticale( 0.35, 0.35, 0.65 ), verticale( 0.65, 0.35, 0.65 ), orizzontale( 0.35, 0.35, 0.65 ), orizzontale( 0.65, 0.35, 0.65 ) ]
        elif numero == '2': fratture = [ verticale( 0.35, 0.35, 0.65 ), verticale( 0.65, 0.35, 0.65 ), orizzontale( 0.35, 0.35, 0.65 ), orizzontale( 0.65, 0.38, 0.65 ) ]
        elif numero == '2a': fratture = [ verticale( 0.35, 0.35, 0.65 ), verticale( 0.65, 0.35, 0.65 ), orizzontale( 0.35, 0.35, 0.65 ), orizzontale( 0.65, 0.36, 0.65 ) ]
        elif numero == '2b': fratture = [ verticale( 0.35, 0.35, 0.65 ), verticale( 0.65, 0.35, 0.65 ), orizzontale( 0.35, 0.35, 0.65 ), orizzontale( 0.65, 0.36, 0.65 ), verticale( 0.36, 0.45, 0.65 ) ] 
        elif numero == '3': fratture = [ verticale( 0.35, 0.35, 0.65 ), verticale( 0.65, 0.35, 0.65 ), orizzontale( 0.35, 0.35, 0.65 ), orizzontale( 0.65, 0.35, 0.65 ) ]
        elif numero == '3a': fratture = [ verticale( 0.35, 0.34, 0.66 ), verticale( 0.65, 0.34, 0.66 ), orizzontale( 0.35, 0.34, 0.66 ), orizzontale( 0.65, 0.34, 0.66 ) ]
        elif numero == '4': fratture = [ verticale( 0.40, 0.40, 0.60 ), verticale( 0.60, 0.40, 0.60 ), orizzontale( 0.40, 0.40, 0.60 ), orizzontale( 0.60, 0.40, 0.60 ) ]
        elif numero == '5': fratture = [ verticale( 0.35, 0.17, 0.46 ), verticale( 0.65, 0.17, 0.46 ), orizzontale( 0.17, 0.35, 0.65 ), orizzontale( 0.46, 0.35, 0.65 ) ]
        elif numero == '6': fratture = [ verticale( 0.35, 0.35, 1.00 ), verticale( 0.65, 0.00, 0.65 ), orizzontale( 0.35, 0.00, 0.65 ), orizzontale( 0.65, 0.35, 1.00 ) ]
        elif numero == '7': fratture = [ orizzontale( 0.8, 0.43, 0.57), verticale( 0.43, 0.2, 0.8), orizzontale( 0.2, 0.43, 0.57), verticale( 0.57, 0.2, 0.8) ]
        elif numero == '7*': fratture = [ verticale( 0.7, 0.38, 0.62), orizzontale( 0.38, 0.3, 0.7), verticale( 0.3, 0.38, 0.62), orizzontale( 0.62, 0.3, 0.7) ]
        elif numero == '8': fratture = [ verticale(0.05, 0.1, 0.4), verticale(0.35, 0.1, 0.4), verticale(0.65, 0.6, 0.9), verticale(0.95, 0.6, 0.9), orizzontale(0.1, 0.05, 0.35), orizzontale(0.4, 0.05, 0.35), orizzontale(0.6, 0.65, 0.95), orizzontale(0.9, 0.65, 0.95) ]
        elif numero == '9': fratture = [ verticale(0.2, 0.35, 0.65), verticale(0.5, 0.35, 0.65), verticale(0.8, 0.35, 0.65), orizzontale(0.35, 0.2, 0.8), orizzontale(0.65, 0.2, 0.8) ]
        elif numero == '10': fratture = [ orizzontale(0.2, 0.35, 0.65), orizzontale(0.5, 0.35, 0.65), orizzontale(0.8, 0.35, 0.65), verticale(0.35, 0.2, 0.8), verticale(0.65, 0.2, 0.8) ] 
        elif numero == '11': fratture = [ verticale( 0.25, 0.25, 0.75 ), verticale( 0.75, 0.25, 0.75 ), orizzontale( 0.25, 0.25, 0.75 ), orizzontale( 0.75, 0.25, 0.75 ) ]
        elif numero == '11a': fratture = [ verticale( 0.25, 0.25, 0.75 ), verticale( 0.75, 0.25, 0.75 ), orizzontale( 0.25, 0.25, 0.75 ), orizzontale( 0.75, 0.25, 0.75 ), orizzontale( 0.5, 0.25, 0.75 ) ]
        elif numero == '11b': fratture = [ verticale( 0.25, 0.25, 0.75 ), verticale( 0.75, 0.25, 0.75 ), orizzontale( 0.25, 0.25, 0.75 ), orizzontale( 0.75, 0.25, 0.75 ), verticale( 0.50, 0.25, 0.75 ) ] 
        elif numero == '11c': fratture = [ verticale( 0.25, 0.25, 0.75 ), verticale( 0.75, 0.25, 0.75 ), orizzontale( 0.25, 0.25, 0.75 ), orizzontale( 0.75, 0.25, 0.75 ), orizzontale( 0.4, 0.25, 0.75 ), orizzontale( 0.6, 0.25, 0.75 ) ]
        elif numero == '12': fratture = [ verticale( 0.35, 0.35, 0.65 ), verticale( 0.65, 0.35, 0.65 ), orizzontale( 0.35, 0.35, 0.65 ), orizzontale( 0.65, 0.35, 0.65 ),  verticale( 0.25, 0.25, 0.75 ), verticale( 0.75, 0.25, 0.75 ), orizzontale( 0.25, 0.25, 0.75 ), orizzontale( 0.75, 0.25, 0.75 ) ]
        else: raise SystemError
        
        # pag.71 in Vujevic Graf
        permeabilita_matrice = 1e-16
        apertura = 46.9 * 1e-6
        if numero == '3': apertura = 37.2 * 1e-6

    elif lettera == 'E':
        X, Y = 27.5, 13.7

        if numero == '9a':
            def h(y, da, a): return np.array([[da, a], [519-y,  519-y]]) * np.array([2*H/1032, H/519]).reshape((2, 1))
            def v(x, da, a): return np.array([[x,  x], [519-da, 519-a]]) * np.array([2*H/1032, H/519]).reshape((2, 1))
            fratture = [
                h(30, 43, 464), v(123, 0, 269), v(219, 0, 259), h(136, 0, 305), 
                v(373, 52, 475), v(53, 160, 519), v(413, 222, 519), v(613, 155, 519),
                v(628, 218, 519), v(687, 0, 315), h(131, 558, 982), h(49, 640, 1031), 
                v(785, 44, 468), h(13, 784, 1032), h(279, 700, 1032), h(320, 775, 1032), 
                v(744, 0, 300), h(433, 640, 1032), h(488, 807, 1032), h(172, 769, 1032)
            ]


        elif numero == '9b':
            def forizzontale(y, x1, x2): return orizzontale(y/Y, x1/X, x2/X)
            def fverticale(x, y1, y2): return verticale(x/X, y1/Y, y2/Y)
            fratture = [
                forizzontale( 11.2, 0, 6), forizzontale( 7.3, 0, 11.2), forizzontale( 10.6, 1.1, 12.5), forizzontale( 8.1, 1.3, 12.7), fverticale( 2.7, 0.7, 11.9), 
                fverticale( 5.1, 0.1, 11.4), fverticale( 7.4, 4.1, 13.7), forizzontale( 1.9, 10.8,   22.1), fverticale( 14.1, 5, 13.7), 
                fverticale( 15.3, 0, 7.2), forizzontale( 11.5, 15.7,   27.1), forizzontale( 9.3, 17.1,   27.5), forizzontale( 5.4, 20.3,   27.5), fverticale( 25.2, 0.4, 11.8), 
                fverticale( 3.4, 0, 6), fverticale( 18.7, 4, 13.7), forizzontale( 13.5, 8.5, 19.9), forizzontale( 0.81, 2.6, 13.8),
                fverticale( 11.6, 0, 10.1), fverticale( 21.8, 0.8, 12.2), 
            ]

        elif numero == '9bu':
            def forizzontale(y, x1, x2): return orizzontale(y/Y, x1/X, x2/X)
            def fverticale(x, y1, y2): return verticale(x/X, y1/Y, y2/Y)
            fratture = [
                forizzontale( 11.2, 0, 6), forizzontale( 7.3, 0, 11.2), forizzontale( 10.6, 1.1, 12.5), forizzontale( 8.1, 1.3, 12.7), fverticale( 2.7, 0.7, 11.9), 
                fverticale( 5.1, 0.1, 11.4), fverticale( 7.4, 4.1, 13.7), forizzontale( 1.9, 10.8,   22.1), fverticale( 14.1, 5, 13.7), 
                fverticale( 15.3, 0, 7.2), forizzontale( 11.5, 15.7,   27.1), forizzontale( 9.3, 17.1,   27.5), forizzontale( 5.4, 20.3,   27.5), fverticale( 25.2, 0.4, 11.8), 
                fverticale( 3.4, 0, 6), fverticale( 18.7, 4, 13.7), forizzontale( 13.5, 8.5, 19.9), forizzontale( 0.81, 2.6, 13.8),
                fverticale( 11.6, 0, 3.3), fverticale( 21.8, 0.8, 6), 
            ]


        else: raise

        # pag.71 in Vujevic Graf
        permeabilita_matrice = 1e-16
        apertura = 46.9 * 1e-6
        if numero == 3: apertura = 1e10

    elif lettera == 'P':
        if numero == 1: fratture = [verticale(0.5, 0.2, 0.8)]
        elif numero == 2: fratture = [ verticale(0.5, 0.2, 0.5 + 1e-10), verticale(0.5, 0.5 - 1e-10, 0.8), ]
        else: raise SystemError

        apertura = 1e-3
        permeabilita_matrice = 1e-17

    elif lettera == 'U':
        if numero == 1: fratture = [ orizzontale(0.4, 0.2, 0.8) ]
        elif numero == 2: fratture = [ orizzontale(0.4, 0.2, 0.8), orizzontale(0.6, 0.0, 0.5),  ]
        elif numero == 3: fratture = [ orizzontale(0.4, 0.2, 0.8), orizzontale(0.6, 0.0, 0.5), orizzontale(0.5, 0.4, 1.0) ]
        else: raise SystemError
        
        permeabilita_matrice = 1e10
        apertura = 1e10

    else:
        fratture = []

    permabilita_fratture = apertura**2 / 12
    dati['permeabilita_matrice'] = permeabilita_matrice
    dati['apertura'] = apertura
    dati['permeabilita_fratture'] = permabilita_fratture

    if lettera == 'D': Nx, Ny = 100, 50 # pagina 71
    else: Nx, Ny = 128, 64
    
    Nx = int(Nx*grid_scale)
    Ny = int(Ny*grid_scale)

    if tipo_griglia is None:
        if lettera in ['E', 'B', 'C']: tipo_griglia = 'triangolare'
        else: tipo_griglia = 'quadrata'

    if tipo_griglia == 'triangolare': mdg = co.mesh_triangolare(Nx, pd, fratture_q=fratture)
    elif tipo_griglia == 'quadrata': mdg = pp.meshing.cart_grid( fratture, nx=np.array([Nx, Ny]), physdims=pd )
    else: raise SystemError

    return mdg, dati


def vg15(figura, lettera, grid_scale=1.0, tipo_griglia='triangolare'):
    dati = {
        'g': 9.81, 'concentrazione_max': 1, 'alpha': 0.07, 'rho_w': 1000, 'viscosita': 1.1e-3, 
        'porosita_matrice': 0.1, 'diffusivita_matrice': 1e-10, 
        'diffusivita_fratture': 1e-10, 'permeabilita_matrice': 1e-18
    }

    Nx = grid_scale*15

    if figura == '5':
        physdims = (10, 10, 6)
        Nq = np.int_(grid_scale * np.array([ 20, 20, 12 ]))

        xmin,xmax = 2,8
        ymin,ymax = 2,8
        zmin,zmax = 1,5
        fratture = [
            np.array([[xmin,ymin,zmin], [xmin,ymin,zmax], [xmin,ymax,zmax], [xmin,ymax,zmin]]).T,
            np.array([[xmax,ymin,zmin], [xmax,ymin,zmax], [xmax,ymax,zmax], [xmax,ymax,zmin]]).T,
        ]

        # apertura critica = 1.8e-5 (pagina 24)
        if lettera == 'A': dati['apertura'] = 1.8e-5*(1 - 0.05)
        elif lettera == 'B': dati['apertura'] = 1.8e-5*(1 + 0.05)
        elif lettera == 'C': dati['apertura'] = 2.0e-5*(1 + 0.05)
        elif lettera == 'D': dati['apertura'] = 5.0e-5
        else: raise 

    elif figura == '6':
        physdims = (10, 10, 6)
        Nq = np.int_(grid_scale * np.array([ 20, 20, 12 ]))
    
        xmin,xmax = 2,8
        ymin,ymax = 2,8
        zmin,zmax = 0,6
        fratture = [
            np.array([[xmin,ymin,zmin], [xmin,ymin,zmax], [xmin,ymax,zmax], [xmin,ymax,zmin]]).T,
            np.array([[xmax,ymin,zmin], [xmax,ymin,zmax], [xmax,ymax,zmax], [xmax,ymax,zmin]]).T,
            np.array([[xmin,ymin,zmin], [xmin,ymin,zmax], [xmax,ymin,zmax], [xmax,ymin,zmin]]).T,
            np.array([[xmin,ymax,zmin], [xmin,ymax,zmax], [xmax,ymax,zmax], [xmax,ymax,zmin]]).T
        ]

        if lettera == 'A': dati['apertura'] = 1.4e-5*(1 - 0.05)
        elif lettera == 'B': dati['apertura'] = 1.4e-5*(1 + 0.05)
        elif lettera == 'B^': dati['apertura'] = 1.6e-5
        elif lettera == 'B^^': dati['apertura'] = 1.8e-5
        elif lettera == 'C': dati['apertura'] = 2.0e-5*(1 + 0.05)
        elif lettera == 'D': dati['apertura'] = 5.0e-5*(1 + 0.05)
        else: raise 

    elif figura == '9a' or figura == '9b':

        physdims = (10, 10, 4)
        Nq = np.int_(grid_scale * np.array([ 20, 20, 8 ]))

        zmin,zmax = 0,6
        def verticale(xc, yc):   return np.array([[xc, yc-2, zmin], [xc, yc-2, zmax], [xc, yc+2, zmax], [xc, yc+2, zmin]]).T
        def orizzontale(xc, yc): return np.array([[xc-2, yc, zmin], [xc-2, yc, zmax], [xc+2, yc, zmax], [xc+2, yc, zmin]]).T

        fratture = [ verticale(4.6, 4.6), verticale(5.8, 7.2), orizzontale(3, 5), orizzontale(6, 6) ]
        if figura[1] == 'a': fratture += [ verticale(2.2, 6.6), orizzontale(4, 8) ]
        elif figura[1] == 'b': fratture += [ verticale(2.2, 7), orizzontale(4, 9) ]
        else: raise SystemError

        if lettera == 'A':   dati['apertura'] = 1.5e-5
        elif lettera == 'B': dati['apertura'] = 1.6e-5
        elif lettera == 'C': dati['apertura'] = 1.7e-5
        elif lettera == 'D': dati['apertura'] = 1.8e-5
        elif lettera == 'E': dati['apertura'] = 1.9e-5
        elif lettera == 'F': dati['apertura'] = 2.0e-5
        else: raise

    else: raise SystemError

    if tipo_griglia == 'triangolare': mdg = co.mesh_tetra(Nx, physdims, [ pp.PlaneFracture(f) for f in fratture ])
    elif tipo_griglia == 'quadrata': mdg = pp.meshing.cart_grid(fratture, Nq, physdims=physdims)
    else: raise

    dati['intrusion_area'] = physdims[0]*physdims[2]
    dati['permeabilita_fratture'] = dati['apertura']**2/12
    return mdg, dati



