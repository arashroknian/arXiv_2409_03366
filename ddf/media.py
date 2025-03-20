from __future__ import annotations
from typing import List, Any
import numpy as np
import scipy.sparse as sps
import porepy as pp
from porepy.numerics.ad._ad_utils import MergedOperator, wrap_discretization
from porepy.numerics.ad.discretizations import Discretization

class Media(Discretization):

    def __init__(self, keyword: str = "transport") -> None:
        self.keyword = keyword

        self.interna_matrix_key = "media_interna"
        self.dir_matrix_key = "media_dir"
        self.neu_matrix_key = "media_neu"
        self.intf_matrix_key = "media_intf"

    def ndof(self, sd): return sd.num_cells
    def assemble_matrix_rhs(self, sd, data): pass

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        parameter_dictionary: dict[str, Any] = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        if sd.dim == 0:
            matrix_dictionary[self.interna_matrix_key] = sps.csr_matrix((0, 1))
            matrix_dictionary[self.dir_matrix_key] = sps.csr_matrix((0, 0))
            matrix_dictionary[self.neu_matrix_key] = sps.csr_matrix((0, 1))
            matrix_dictionary[self.intf_matrix_key] = sps.csr_matrix((0, 0))
            return

        bc = parameter_dictionary["bc"]
        faccia_interna = np.ones(sd.num_faces)
        faccia_interna[bc.bf] = 0

        # celle -> facce pesando con 0.5
        # weight_array = 0.5 * np.ones(sd.num_faces)
        # weight_array[bc.bf] = 0.0
        # weights = sps.dia_matrix((weight_array, 0), shape=(sd.num_faces, sd.num_faces))
        # media_interna = weights * np.abs(sd.cell_faces)

        # # celle -> facce pesando la distanza tra i centri
        # coo = sps.coo_matrix(sd.cell_faces)
        # centri_facce = sd.face_centers[:, coo.row]
        # centri_celle = sd.cell_centers[:, coo.col]
        # distanze = np.linalg.norm(centri_facce - centri_celle, axis=0)
        # vicinanza = sps.coo_matrix((1/distanze, (coo.row, coo.col)))
        # somma = np.array(vicinanza.sum(axis=1)).reshape((-1))
        # media_interna = vicinanza.multiply( (1/somma * faccia_interna).reshape((-1, 1)) )

        # # celle -> nodi -> facce pesando i volumi
        # coo = sps.coo_matrix(sd.cell_nodes())
        # pesi1 = sd.cell_volumes[coo.col]
        # media1 = sps.coo_matrix((pesi1, (coo.row, coo.col)))
        # somma = np.array(media1.sum(axis=1)).reshape((-1))
        # media1 = sps.diags(1/somma) * media1
        # media2 = sd.face_nodes.T
        # somma = np.array(media2.sum(axis=1)).reshape((-1))
        # media2 = sps.diags(1/somma) * media2
        # media = media2 @ media1
        # media_interna = sps.diags(faccia_interna) * media

        # celle -> facce proiettando 
        c_to_f = sps.coo_matrix(sd.cell_faces)
        f_to_hf = sps.coo_matrix((1*(c_to_f.data!=0), (np.arange(c_to_f.data.shape[0]), c_to_f.row)))
        hf_to_hf = f_to_hf @ f_to_hf.T
        hf_to_hf.setdiag(0)
        hf_to_hf = abs(hf_to_hf)
        centri_facce = sd.face_centers[:, c_to_f.row]
        centri_celle = sd.cell_centers[:, c_to_f.col]
        centri_celle_bar = (hf_to_hf @ centri_celle.T).T
        dist_centri_celle = centri_celle_bar - centri_celle
        alpha = np.sum((centri_facce - centri_celle) * dist_centri_celle, axis=0) / np.sum(dist_centri_celle * dist_centri_celle, axis=0)
        alpha_bar = (c_to_f.data < 0) * (hf_to_hf @ alpha)
        pesi = (c_to_f.data > 0) * (1-alpha) + (c_to_f.data < 0) * alpha_bar
        media_interna = sps.diags(faccia_interna) * sps.coo_matrix(( pesi, (c_to_f.row, c_to_f.col)))
        # breakpoint()
        if not np.all(np.array(media_interna.sum(axis=1))[:,0][faccia_interna > 0] == 1): raise SystemError
        if np.any(pesi < 0): raise SystemError
        matrix_dictionary[self.interna_matrix_key] = media_interna


        #| sto solo portando da mortar alla griglia sopra (e aggiustando i segni)
        sgn_div = pp.fvutils.scalar_divergence(sd).sum(axis=0).A.squeeze()
        media_intf = sps.dia_matrix((bc.is_internal*sgn_div, 0), shape=(sd.num_faces, sd.num_faces))
        matrix_dictionary[self.intf_matrix_key] = media_intf.tocsr()

        # #| su neumann riporto semplicemente la quantita' delle cella sulla faccia. 
        # #| si potrebbe anche fare qualcosa un po' meglio: sui bordi neumann conosco la derivata normale 
        # #| => potrei correggere linearmente il valore che porto dalla cella alla faccia
        # #| sto facendo una cosa simile quando ricostruisco la pressione sulla traccia
        # # TODO: magari in ddf aggiungere il controllo che non ho inflow su neumann (a dir la verita' non sono sicuro di volerlo evitare per forza... pensa a quello che c'e' scritto alla riga sopra)

        # weight_array_neu = np.zeros(sd.num_faces)
        # weight_array_neu[bc.bf] = 1 # solo bordo
        # weight_array_neu[bc.is_internal | bc.is_dir] = 0.0 # solo bordo neumann
        # weights_neu = sps.dia_matrix((weight_array_neu, 0), shape=(sd.num_faces, sd.num_faces))
        # media_neu = weights_neu * np.abs(sd.cell_faces)
        # matrix_dictionary[self.neu_matrix_key] = media_neu.tocsr()

        #| lo tengo solo perche' a volte porepy fa casino quando moltiplica due vettori (di cui uno ad e l'altro np)
        media_dir = sps.dia_matrix((1*bc.is_dir, 0), shape=(sd.num_faces, sd.num_faces))
        matrix_dictionary[self.dir_matrix_key] = media_dir.tocsr()

class MediaAd(Discretization):
    def __init__(self, keyword: str, subdomains: List[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = Media(keyword)
        self._name = "Media"
        self.keyword = keyword

        self.media_interna: MergedOperator
        self.media_dir: MergedOperator
        self.media_intf: MergedOperator
        wrap_discretization(self, self._discretization, subdomains=subdomains)



# # mdg = pp.meshing.cart_grid([], nx=np.array([20, 10, 10]), physdims=(20, 10, 10))
# # mdg = pp.meshing.cart_grid([], nx=np.array([20, 10]), physdims=(20, 10))
# # mdg = co.mesh_triangolare(10, (20, 10))
# mdg = co.mesh_tetra(10, (20, 10, 10))
# sd = mdg.subdomains()[0]

# # pb = hrl.HRL(mdg, hrl.vg14('D',1)[1], {})
# # pb.init()
# # a = np.zeros(sd.num_cells); a[0] = 1
# # plot_cells(pb, a)

# def angolo3d(a, b, c):
#     n = np.linalg.norm; dot = np.dot
#     num = np.abs(dot(a, np.cross(b, c)))
#     den = n(a)*n(b)*n(c) + dot(a,b)*n(c) + dot(a,c)*n(b) + dot(b,c)*n(a)
#     res = 2*np.arctan(num/den)
#     assert res > 0
#     return res

# def angoli_(d, nodi):
#     n = nodi.shape[1]
#     angoli = np.zeros(n)
    
#     if d == 2 and n == 3:
#         a, b, c = nodi[:,0], nodi[:,1], nodi[:,2]
#         angoli[0] = np.arccos( np.dot(b-a, c-a) / (np.linalg.norm(b-a)*np.linalg.norm(c-a)) )
#         angoli[1] = np.arccos( np.dot(c-b, a-b) / (np.linalg.norm(c-b)*np.linalg.norm(a-b)) )
#         angoli[2] = np.arccos( np.dot(a-c, b-c) / (np.linalg.norm(a-c)*np.linalg.norm(b-c)) )
#     elif d ==3 and n == 4:
#         a, b, c, d = nodi[:,0], nodi[:,1], nodi[:,2], nodi[:,3]
#         angoli[0] = angolo3d(b-a, c-a, d-a)
#         angoli[1] = angolo3d(a-b, c-b, d-b)
#         angoli[2] = angolo3d(a-c, b-c, d-c)
#         angoli[3] = angolo3d(a-d, b-d, c-d)
#     else: raise
        
#     return angoli

# def angoli(sd, coo):
#     nodi = sd.nodes[:, coo.row]
#     angolo = np.zeros(nodi.shape[1])
#     if sd.name == 'TriangleGrid':
#         for j in range(0, nodi.shape[1], 3): angolo[j:j+3] = angoli_( 2, nodi[:,j:j+3] )
#         for j in range(0, angolo.shape[0], 3): assert np.isclose(np.sum(angolo[j:j+3]), np.pi)
#         for n in range(sd.num_nodes): assert np.any(np.isclose(np.sum(angolo[coo.row == n]), 2*np.pi*np.array([0.25, 0.5, 1])))
#     elif sd.name == 'CartGrid' and sd.dim == 2:
#         angolo[:] = np.pi/2
#     elif sd.name == 'CartGrid' and sd.dim == 3:
#         angolo[:] = 4*np.pi**2/8
#     elif sd.name == 'TetrahedralGrid':
#         for j in range(0, nodi.shape[1], 4): angolo[j:j+4] = angoli_( 3, nodi[:,j:j+4] )
#     #     for j in range(0, angolo.shape[0], 4): assert np.isclose(np.sum(angolo[j:j+4]), np.pi)
#         for n in range(sd.num_nodes): assert np.any(np.isclose(np.sum(angolo[coo.row == n]) / (4*np.pi), np.array([1/8, 1/4, 1/2, 1]) ))
#     else:
#         print(sd.name)
#         raise

# cn = sd.cell_nodes()
# coo = sps.coo_matrix(cn)

# # pesi = angoli(sd, coo)
# pesi = sd.cell_volumes[coo.col]

# media1 = sps.coo_matrix((peso, (coo.row, coo.col)))
# somma = np.array(media1.sum(axis=1)).reshape((-1))
# media1 = sps.diags(1/somma) * media1

# media2 = sd.face_nodes.T
# somma = np.array(media2.sum(axis=1)).reshape((-1))
# media2 = sps.diags(1/somma) * media2

# media = media2 @ media1
# assert np.allclose(np.array(media.sum(axis=1)).reshape((-1)), 1)

# sps.find(media)