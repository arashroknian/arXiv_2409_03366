
import numpy as np
import porepy as pp
import scipy.sparse as sps
import os
import subprocess
import shutil
import pickle
import gzip
import shutil

anno = 60*60*24*365

#~ ...

try: import pypardiso
except ImportError: pass
def spsolve(A, b):
    # return sps.linalg.spsolve(A, b)
    try: sol = pypardiso.spsolve(A, b)
    except : sol = sps.linalg.spsolve(A, b)
    return sol

#~ ...

def grid_data(mdg, grid):
    if isinstance(grid, pp.MortarGrid): return mdg.interface_data(grid)
    else: return mdg.subdomain_data(grid)

def stato(pb, zero=False, dof_key=None):
    vec = np.zeros(pb.dof.num)
    if zero:
        for var,dof in pb.dof.var_dof():
            source = pb.variabili_0[var]._values
            assert vec[dof].shape == source.shape
            vec[dof] = source.copy()
    else:
        for (grid, var),dof in pb.dof.gridvar_dof():
            source = grid_data(pb.mdg, grid)[pp.STATE][var]
            assert source.shape == vec[dof].shape
            vec[dof] = source.copy()
    if dof_key is not None: return vec[pb.dof[dof_key]] 
    return vec

def carica_stato(pb, vec, zero=False):
    assert pb.dof.num == vec.shape[0]
    if zero:
        for var,dof in pb.dof.var_dof():
            dest = pb.variabili_0[var]._values
            assert vec[dof].shape == dest.shape
            dest[:] = vec[dof]
    else:
        for (grid, var),dof in pb.dof.gridvar_dof():
            dest = grid_data(pb.mdg, grid)[pp.STATE][var]
            assert dest.shape == vec[dof].shape
            dest[:] = vec[dof]

def carica_stato_f(mdg, grids, fun):
    if grids == 'subdomains': gds = mdg.subdomains(return_data=True)
    if grids == 'interfaces': gds = mdg.interfaces(return_data=True)

    for grid, grid_data in gds:
        dizionario = fun(grid)
        assert isinstance(dizionario, dict)
        for val in dizionario.values(): assert val.shape[-1] == grid.num_cells
        pp.set_state(grid_data, dizionario)

def griglie(mdg, nome):
    campi_sd = mdg.subdomain_data(mdg.subdomains()[0])[pp.STATE].keys()
    campi_intf = mdg.interface_data(mdg.interfaces()[0])[pp.STATE].keys() if len(mdg.subdomains()) > 1 else []
    if nome in campi_sd: gds = mdg.subdomains(return_data=True)
    elif nome in campi_intf: gds = mdg.interfaces(return_data=True)
    else: raise SystemError # {nome} non trovato
    return gds

def raccogli_stato(mdg, nome, griglia=None):
    gds = griglie(mdg, nome)
    if griglia is None: return { grid: grid_data[pp.STATE][nome].copy() for (grid, grid_data) in gds }
    for grid, grid_data in gds:
        if grid == griglia: return grid_data[pp.STATE][nome].copy()
def raccogli_stato_vec(mdg, nome):
    gds = griglie(mdg, nome)
    d = raccogli_stato(mdg, nome)
    vec = np.concatenate([ d[g] for (g,_) in gds ], axis=-1)
    return vec

def parametro(mdg, grids, chiave, p_chiave, fun):
    if grids == 'subdomains': gds = mdg.subdomains(return_data=True)
    if grids == 'interfaces': gds = mdg.interfaces(return_data=True)

    for g,g_data in gds:
        if p_chiave is None:
            dizionario = fun(g)
            assert isinstance(dizionario, dict)
        else: dizionario = { p_chiave: fun(g) }
        pp.initialize_data(g, g_data, chiave, dizionario)

# originale = co.stato(pb)
# co.carica_stato(pb, originale)
# nuovo = co.stato(pb)
# assert np.all(originale == nuovo)
# assert not (originale is nuovo)

# originale = co.stato(pb, zero=True)
# co.carica_stato(pb, originale, zero=True)
# nuovo = co.stato(pb, zero=True)
# assert np.all(originale == nuovo)
# assert not (originale is nuovo)

def ricava_eq_dof(dof_manager, eq_manager):
    subdomains = dof_manager.mdg.subdomains()
    interfaces = dof_manager.mdg.interfaces()

    eq_dof = {}
    idx = 0
    for nome,equazione in eq_manager.equations.items():
        num_dof = equazione.evaluate(dof_manager).val.shape[0]
        eq_dof[nome] = np.arange(idx, idx + num_dof)

        idx += num_dof
    return eq_dof

class dof_dict(dict): 
    def gridvar_dof(self):
        gen = (x for x in self.items() if isinstance(x[0], tuple) and hasattr(x[0][0], 'num_cells'))
        return gen
    def var_dof(self):
        gen = (x for x in self.items() if isinstance(x[0], str))
        return gen
def ricava_dof(dof_manager):
    # per quando devo pescare i valori di una particolare griglia e/o variabile da un vettore di stato

    res = dof_dict()

    idx = 0
    for key,n_dof in zip(dof_manager.block_dof.keys(), dof_manager.full_dof):
        dof = np.arange(idx, idx+n_dof, dtype=np.int32)
        
        griglia, variabile = key
        res[(griglia, variabile)] = dof
        
        if not variabile in res.keys(): res[variabile] = np.array([], dtype=np.int32)
        res[variabile] = np.concatenate(( res[variabile], dof ))

        idx += n_dof
    res.num = idx

    return res
def ricava_vecdof(mdg):
    # per quando devo pescare i valori di una particolare griglia da un vettore multi-griglia

    cdof = {}
    fdof = {}
    idof = {}

    idx = 0
    for g in mdg.subdomains():
        n = g.num_cells
        cdof[g] = np.arange(idx, idx+n)
        idx += n
    assert idx == np.sum([ g.num_cells for g in mdg.subdomains() ])

    idx = 0
    for g in mdg.subdomains():
        n = g.num_faces
        fdof[g] = np.arange(idx, idx+n)
        idx += n
    assert idx == np.sum([ g.num_faces for g in mdg.subdomains() ])

    idx = 0
    for g in mdg.interfaces():
        n = g.num_cells
        idof[g] = np.arange(idx, idx+n)
        idx += n
    assert idx == np.sum([ g.num_cells for g in mdg.interfaces() ])

    return cdof, fdof, idof

#~ ...

def controlla_ci(mdg, dof):
    lista_id = []
    for (grid, var),_ in dof.gridvar_dof():
        state = grid_data(mdg, grid)[pp.STATE]
        _id = id(state[var])
        assert not(_id in lista_id)
        lista_id.append(_id)
    del lista_id


#~ ...

def norma(dof, vec, arr=0):
    formatter = "{:>8.3g}".format
    norme = [np.linalg.norm(vec[_dof],np.inf) for _,_dof in dof.var_dof()]
    if arr: return np.array(norme)
    res = '[' + ' '.join([ formatter(n) for n in norme ]) + ']'
    return res

#~ ...

def domain_boundary_faces(mdg, sd):
    if sd.dim == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty

    D = max([ sd.dim for sd in mdg.subdomains() ])
    sd_D = mdg.subdomains(dim=D)[0]
    x_max = np.max( sd_D.face_centers[0,:] )
    y_max = np.max( sd_D.face_centers[1,:] )
    
    eps = 1e-10
    tutte = sd.tags['domain_boundary_faces']
    west =  tutte & ( sd.face_centers[0] < eps )
    east =  tutte & ( sd.face_centers[0] > x_max - eps )
    south = tutte & ( sd.face_centers[1] < eps )
    north = tutte & ( sd.face_centers[1] > y_max - eps )

    return tutte, west, east, south, north


#~ ...

def grid_size(mdg):
    sd = mdg.subdomains()[0]
    _min = np.min( sd.nodes , axis=1)
    _max = np.max( sd.nodes , axis=1)
    return _max - _min


#~ ...

def repeat_mat(grids, D):
    if len(grids) == 0: return 0 # per domini non fratturati

    def _repeat(g):
        rows = np.arange(g.num_cells * D)
        cols = np.repeat( np.arange(g.num_cells), D)
        vals = np.ones(g.num_cells * D)
        mat = sps.coo_matrix((vals, (rows, cols)))
        return mat
    
    mat = sps.block_diag([_repeat(g) for g in grids]).tocsr()
    mat = pp.ad.Matrix(mat, 'per accontentare la gravità')
    return mat

def is_dir(mdg, sd, key): return mdg.subdomain_data(sd)[pp.PARAMETERS][key]['bc'].is_dir * 1
def is_neu(mdg, sd, key): return mdg.subdomain_data(sd)[pp.PARAMETERS][key]['bc'].is_neu * 1


#~ ...

cartella_script = os.path.dirname(os.path.abspath(__file__))
cartella_tmp = os.path.join(cartella_script, '..', 'tmp')

def prepara_cartella(cartella):
    if os.path.exists(cartella): shutil.rmtree(cartella)
    os.makedirs(cartella)
    os.makedirs(f'{cartella}/script/')
    os.makedirs(f'{cartella}/txt/')
    
    files = [f for f in os.listdir(cartella_script) if os.path.isfile(f'{cartella_script}/{f}')]
    for f in files:
        shutil.copy(f'{cartella_script}/{f}', f'{cartella}/script/')
        # cmd = f'cp "{cartella_script}/{f}" "{cartella}/script/"'
        # subprocess.call(cmd, shell=True)

def salva(pb, eig=0):
    if eig:
        with open(f'{pb.cartella_out}/eig.pkl', 'wb') as f: pickle.dump(pb._eig, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'{pb.cartella_out}/sim.pkl', 'wb') as f: pickle.dump(pb, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with gzip.open(f'{pb.cartella_out}/sim.gz', 'wb') as f: pickle.dump(pb, f, protocol=pickle.HIGHEST_PROTOCOL)
    
def pkl(impostazioni, eig=0):
    nome = impostazioni['nome']
    cartella_out = impostazioni.get('cartella', cartella_tmp) + '/' + nome

    try:
        if eig:
            if os.path.isfile(f'{cartella_out}/eig.pkl'):
                with open(f'{cartella_out}/eig.pkl', 'rb') as f: return pickle.load(f)
            else: return None

        if os.path.isfile(f'{cartella_out}/sim.pkl'):
            with open(f'{cartella_out}/sim.pkl', 'rb') as f: pb = pickle.load(f)
        elif os.path.isfile(f'{cartella_out}/sim.gz'):
            with gzip.open(f'{cartella_out}/sim.gz', 'rb') as f: pb = pickle.load(f)
        else: return None
        pb.cartella_out = cartella_out
        return pb
    
    except Exception as e:
        print(e)
        return None

def line(s):
    CSI = "\x1b["
    CL = CSI + "2K"
    print(CL, end='\r')
    print(s, end='')


#~ ...

def riscrivi_fratture(lista):
    if len(lista) == 0: return None, None

    xs, ys, edges = [], [], []
    i = 0
    for f in lista:
        xs += list(f[0])
        ys += list(f[1])
        edges += [i, i+1]
        i += 2
    pts = np.vstack((xs, ys))
    edges = np.array(edges).reshape((-1, 2)).T
    return pts, edges

def mesh_triangolare(Nx, pd, fratture_q=None, fratture_t=None):
    mesh_size = pd[0]/Nx * np.sqrt(2)
    domain = {'xmin': 0, 'xmax': pd[0], 'ymin': 0, 'ymax': pd[1]}
    mesh_args = {'mesh_size_frac': mesh_size, 'mesh_size_bound': mesh_size}

    if fratture_q is None and fratture_t is None: 
        network = pp.FractureNetwork2d(domain=domain)
    elif fratture_q is not None:
        fratture_t = riscrivi_fratture(fratture_q)
        pts, edges = fratture_t
        network = pp.FractureNetwork2d(pts=pts, edges=edges, domain=domain)
    elif fratture_t is not None:
        pts, edges = fratture_t
        network = pp.FractureNetwork2d(pts=pts, edges=edges, domain=domain)
    else:
        raise SystemError

    mdg = network.mesh(mesh_args, write_geo=False)
    mdg.compute_geometry()
    return mdg

def mesh_tetra(Nx, pd, fratture=None):
    mesh_size = pd[0]/Nx
    domain = {'xmin': 0, 'xmax': pd[0], 'ymin': 0, 'ymax': pd[1], 'zmin': 0, 'zmax': pd[2] }
    mesh_args = {'mesh_size_frac': mesh_size, 'mesh_size_bound': mesh_size, 'mesh_size_min': mesh_size/2}

    network = pp.FractureNetwork3d(fratture, domain=domain)
    mdg = network.mesh(mesh_args, write_geo=False)
    mdg.compute_geometry()
    return mdg
