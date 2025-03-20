import numpy as np
import scipy
import scipy.sparse as sps
import porepy as pp

from types import SimpleNamespace

import ddf.common as co
from ddf.media import MediaAd
from ddf.dks import DKS

def volume_specifico(apertura, D, dim): return apertura**(D - dim)

def quantita_caratteristiche(mdg, dati):
    d = dati
    _,H,_ = co.grid_size(mdg)
    Ra = (
        d['permeabilita_matrice']*d['rho_w']*d['alpha']
        * d['concentrazione_max']*d['g']*H
        / (d['porosita_matrice']*d['viscosita']*d['diffusivita_matrice'])
    )
    car = { 
        'L': H, 'T': H**2/d['diffusivita_matrice'], 
        'U': d['diffusivita_matrice'] / H,
        'P': d['porosita_matrice']*d['viscosita']*d['diffusivita_matrice']/d['permeabilita_matrice'],
        'concentrazione': d['concentrazione_max'],
        'Ra': Ra,
    }
    return car

class DDF:
    def __init__(self, mdg, dati, impostazioni):
        # TODO: da mettere un controllo che mdg non sia gia' "pieno"?
        self.mdg = mdg
        self.dati = dati
        self.impostazioni = impostazioni
        self.random = np.random.default_rng(2023)

    def stato(self, **kw): return co.stato(self, **kw)
    def carica_stato(self, vec, **kw): return co.carica_stato(self, vec, **kw)

    def converged(self, soluzione, incrementi):
        if self.theta._value == 1 and len(incrementi) > 0: return 1

        incremento = incrementi[-1]
        max_incremento = np.linalg.norm(incremento, np.inf)
        # max_incremento = np.linalg.norm(incremento[self.dof['concentrazione']], np.inf)
        ok = (max_incremento < self.dati.get('newton_atol', 1e-8)) and (len(incrementi) >= self.dati.get('newton_miniter', 0))
        return ok

    def init(self):
        global diffusivita_darcy, diffusivita_n_darcy, diffusivita_trasporto, diffusivita_n_trasporto, flussi, gravita, correzione_interfaccia, diffondi, direzione_gravita, get_bc, F_trasporto, F_darcy, trasporta_media

        subdomains = self.mdg.subdomains(); interfaces = self.mdg.interfaces()
        fratturato = len(subdomains) > 1
        D = self.mdg.dim_max()
        b = self.dati['apertura']

        #~ ...

        for d in (self.mdg.subdomain_data(sd) for sd in subdomains): d[pp.PRIMARY_VARIABLES] = {
            'pressione': {'cells': 1},
            'concentrazione': {'cells': 1},
        }
        for d in (self.mdg.interface_data(sd) for sd in interfaces): d[pp.PRIMARY_VARIABLES] = {
            'laambda': {'cells': 1},
            'mu': {'cells': 1}
        }

        dof_manager = pp.DofManager(self.mdg)
        eq_manager = pp.ad.EquationManager(self.mdg, dof_manager)
        dof = co.ricava_dof(dof_manager)
        self.cdof, self.fdof, self.idof = co.ricava_vecdof(self.mdg)
        self.dof_manager = dof_manager; self.eq_manager = eq_manager; self.dof = dof;

        #~ ...

        pressione = eq_manager.merge_variables( [(sd, 'pressione') for sd in subdomains ])
        concentrazione = eq_manager.merge_variables( [(sd, 'concentrazione') for sd in subdomains ])
        laambda = eq_manager.merge_variables( [(intf, 'laambda') for intf in interfaces ])
        mu = eq_manager.merge_variables( [(intf, 'mu') for intf in interfaces ])
        variabili = { 'pressione': pressione, 'concentrazione': concentrazione, 'laambda': laambda, 'mu': mu }
        self.variabili = variabili

        # [pressione] = 1
        # [concentrazione] = 1
        # [laambda] = m^(D-1)
        # [mu] = m^(D-1)

        ndof_sd = dof['pressione'].shape[0]
        ndof_intf = dof['laambda'].shape[0] if fratturato else 0
        pressione_0 = pp.ad.Array( np.zeros(ndof_sd) )
        concentrazione_0 = pp.ad.Array( np.zeros(ndof_sd) )
        laambda_0 = pp.ad.Array(np.zeros( ndof_intf ))
        mu_0 = pp.ad.Array(np.zeros( ndof_intf ))
        variabili_0 = { 'pressione': pressione_0, 'concentrazione': concentrazione_0, 'laambda': laambda_0, 'mu': mu_0 }
        self.variabili_0 = variabili_0


        #~ ...

        car = quantita_caratteristiche(self.mdg, self.dati)
        Ra = car['Ra']
        L = car['L']
        self.car = car

        co.carica_stato_f(self.mdg, 'subdomains', lambda sd: { 'cc': sd.cell_centers })
        co.carica_stato_f(self.mdg, 'subdomains', lambda sd: { 'dim': sd.dim*np.ones(sd.num_cells) })
        co.carica_stato_f(self.mdg, 'subdomains', lambda sd: self.condizioni_iniziali_sd(sd))
        co.carica_stato_f(self.mdg, 'interfaces', lambda intf: self.condizioni_iniziali_intf(intf))
        co.controlla_ci(self.mdg, dof)

        #~ ...

        proj = pp.ad.MortarProjections(self.mdg, interfaces=interfaces, subdomains=subdomains)
        trace = pp.ad.Trace(subdomains=subdomains)
        div = pp.ad.Divergence(subdomains)

        co.parametro(self.mdg, 'subdomains', 'darcy', 'mass_weight', lambda sd: volume_specifico(b, D, sd.dim) )
        massa = pp.ad.MassMatrixAd('darcy', subdomains)
        self.massa = massa

        k_m, k_f = self.dati['permeabilita_matrice'], self.dati['permeabilita_fratture']
        def diffusivita_darcy(sd):
            k_rel = 1 if sd.dim == D else k_f/k_m
            diffusivita = k_rel * volume_specifico(b, D, sd.dim) * np.ones(sd.num_cells)
            return pp.SecondOrderTensor( diffusivita )
        def diffusivita_n_darcy(intf):
            diffusivita_n = k_f/k_m * volume_specifico(b, D, intf.dim+1) * np.ones(intf.num_cells)
            return diffusivita_n
        co.parametro(self.mdg, 'subdomains', 'darcy', 'second_order_tensor', diffusivita_darcy)
        co.parametro(self.mdg, 'interfaces', 'darcy', 'normal_diffusivity', diffusivita_n_darcy)
        co.parametro(self.mdg, 'subdomains', 'darcy', None, lambda sd: self.bc_darcy(sd))
        co.parametro(self.mdg, 'subdomains', 'darcy', None, lambda sd: { 'is_dir': co.is_dir(self.mdg, sd, 'darcy'), 'is_neu': co.is_neu(self.mdg, sd, 'darcy') })
        op_darcy = {
            'diff': pp.ad.MpfaAd('darcy', subdomains),
            'robin': pp.ad.RobinCouplingAd('darcy', interfaces)
        }
        p_dir = pp.ad.ParameterArray('darcy', 'is_dir', subdomains) * pp.ad.BoundaryCondition('darcy', subdomains)
        U_neu = pp.ad.ParameterArray('darcy', 'is_neu', subdomains) * (pp.ad.BoundaryCondition('darcy', subdomains) + proj.mortar_to_primary_int * laambda)


        sd_D = [ sd for sd in self.mdg.subdomains() if sd.dim == D ][0]
        bc_D = self.mdg.subdomain_data(sd_D)[pp.PARAMETERS]['darcy']['bc']
        darcy_tutto_neu = np.all( bc_D.is_neu[bc_D.bf] )
        if darcy_tutto_neu: raise SystemError()

        d_m, d_f = self.dati['diffusivita_matrice'], self.dati['diffusivita_fratture']
        def diffusivita_trasporto(sd):
            d_rel = 1 if sd.dim == D else d_f/d_m
            diffusivita = d_rel * volume_specifico(b, D, sd.dim) * np.ones(sd.num_cells)
            return pp.SecondOrderTensor( diffusivita )
        def diffusivita_n_trasporto(intf):
            diffusivita_n = d_f/d_m * volume_specifico(b, D, intf.dim+1) * np.ones(intf.num_cells)
            return diffusivita_n
        co.parametro(self.mdg, 'subdomains', 'trasporto', 'second_order_tensor', diffusivita_trasporto)
        co.parametro(self.mdg, 'interfaces', 'trasporto', 'normal_diffusivity', diffusivita_n_trasporto)
        co.parametro(self.mdg, 'subdomains', 'trasporto', None, lambda sd: self.bc_trasporto(sd))
        co.parametro(self.mdg, 'subdomains', 'trasporto', None, lambda sd: { 'is_dir': co.is_dir(self.mdg, sd, 'trasporto'), 'is_neu': co.is_neu(self.mdg, sd, 'trasporto') })
        op_trasporto = {
            'diff': pp.ad.MpfaAd('trasporto', subdomains),
            'robin': pp.ad.RobinCouplingAd('trasporto', interfaces),
            'media': MediaAd('trasporto', subdomains),
        }
        c_dir = pp.ad.ParameterArray('trasporto', 'is_dir', subdomains) * pp.ad.BoundaryCondition('trasporto', subdomains)
        I_neu = pp.ad.ParameterArray('trasporto', 'is_neu', subdomains) * (pp.ad.BoundaryCondition('trasporto', subdomains) + proj.mortar_to_primary_int * mu)


        #~ ...

        def direzione_gravita(grid):
            idx = (0 if D == 1 else 1)
            g = np.zeros((D, grid.num_cells)); g[idx, :] = -1
            vec = g.ravel('F')
            return vec
        co.parametro(self.mdg, 'subdomains', 'darcy', 'direzione_gravita', lambda sd: direzione_gravita(sd))
        co.parametro(self.mdg, 'interfaces', 'darcy', 'direzione_gravita', lambda intf: direzione_gravita(intf))
        co.parametro(self.mdg, 'subdomains', 'darcy', 'ambient_dimension', lambda _: D)
        co.parametro(self.mdg, 'subdomains', 'trasporto', 'ambient_dimension', lambda _: D) 
        def gravita(concentrazione_sd, concentrazione_intf):
            sd_vettore = (co.repeat_mat(subdomains,D) * concentrazione_sd * Ra) * pp.ad.ParameterArray('darcy', 'direzione_gravita', subdomains=subdomains)
            intf_vettore = (co.repeat_mat(interfaces, D) * concentrazione_intf * Ra) * pp.ad.ParameterArray('darcy', 'direzione_gravita', interfaces=interfaces)
            return {'sd': sd_vettore, 'intf': intf_vettore}

        def diffondi(op, quantita, dir_, neu, gravita=None):
            if gravita is None: gravita = { 
                'sd': 0*pp.ad.ParameterArray('darcy', 'direzione_gravita', subdomains=subdomains), 
                'intf': 0*pp.ad.ParameterArray('darcy', 'direzione_gravita', interfaces=interfaces) 
            }

            portata_sd = (
                L*op['diff'].flux * quantita
                + L*op['diff'].bound_flux * dir_
                + op['diff'].bound_flux * neu #| potrei anche togliere bound_flux dato che su neu e' = 1
                + op['diff'].vector_source * gravita['sd']
            )

            traccia_quantita = (
                op['diff'].bound_pressure_cell * quantita
                + op['diff'].bound_pressure_face * dir_ # potrei anche togliere bound_pressure_face dato che su dir e' = 1
                + (1/L)*op['diff'].bound_pressure_face * neu
                + (1/L)*op['diff'].bound_pressure_vector_source * gravita['sd']
            )
            fuori = proj.primary_to_mortar_avg * traccia_quantita
            dentro = proj.secondary_to_mortar_avg * quantita
            portata_intf = -1*op['robin'].mortar_discr * (
                (2*L/b)*(fuori - dentro)
                + op['robin'].mortar_vector_source*gravita['intf'] # contact_mechanics_biot_model.py:914
            )

            return portata_sd, portata_intf, traccia_quantita

        def trasporta_media(media, velocita, velocita_intf, quantita, quantita_tr, quantita_dir):
            quantita_intf = 0.5 * (proj.secondary_to_mortar_avg*quantita + proj.primary_to_mortar_avg*quantita_tr)
            flusso_intf = velocita_intf * quantita_intf
            flusso = (
                velocita * (media.interna * quantita)
                + velocita * (media.dir * quantita_dir)
                + media.intf * (proj.mortar_to_primary_int * flusso_intf)
            )
            return flusso, flusso_intf

        def flussi(var):
            _c, _p, _l, _m = map(var.get, ['concentrazione', 'pressione', 'laambda', 'mu'])

            _I, _M, _tr_c = diffondi(op_trasporto, _c, c_dir, I_neu)
            _c_intf = 0.5 * (proj.secondary_to_mortar_avg*_c + proj.primary_to_mortar_avg*_tr_c)
            _U, _Q, _tr_p = diffondi(op_darcy, _p, p_dir, U_neu, gravita(_c, _c_intf))

            alpa = self.dati['alpha']*self.dati['concentrazione_max']
            _rho = 1 + alpa*_c
            _tr_rho = 1 + alpa*_tr_c
            rho_dir = 1 + alpa*c_dir

            _cU, _cl =       trasporta_media(op_trasporto['media'], _U, _l, _c, _tr_c, c_dir)
            _rhoU, _rhol =   trasporta_media(op_trasporto['media'], _U, _l, _rho, _tr_rho, rho_dir)
            _rhocU, _rhocl = trasporta_media(op_trasporto['media'], _U, _l, _rho*_c, _tr_rho*_tr_c, rho_dir*c_dir)
            _rhoI, _rhom =   trasporta_media(op_trasporto['media'], _I, _m, _rho, _tr_rho, rho_dir)

            return _rho, SimpleNamespace(
                U=_U, Q=_Q, l=_l,
                I=_I, M=_M, m=_m,
                cU=_cU, rhocU=_rhocU, rhoU=_rhoU,
                cl=_cl, rhocl=_rhocl, rhol=_rhol,
                rhoI=_rhoI, rhom=_rhom
            )


        #~ ...

        rho, v = flussi(variabili)
        rho0, v0 = flussi(variabili_0)

        dt = pp.ad.Scalar(np.nan)
        steady = pp.ad.Scalar(0)
        theta = pp.ad.Scalar( self.dati.get('euler_theta', 0) ) # theta = 0: eulero implicito
        self._dt = dt; self.steady = steady
        self.t = 0
        self.theta = theta

        if self.impostazioni.get('boussinesq', 1):
            def F_darcy(_v): return L*( div*_v.U - proj.mortar_to_secondary_int*_v.l )
            def F_trasporto(_v): return L*( 
                div*_v.cU + div*_v.I 
                - proj.mortar_to_secondary_int*_v.cl - proj.mortar_to_secondary_int*_v.m
            )
            darcy = F_darcy(v)
            trasporto = (
                (1-steady)*massa.mass/dt*(concentrazione - concentrazione_0) 
                + (1-theta) * F_trasporto(v)
                + theta     * F_trasporto(v0)
            )

        else:
            def F_darcy(_v): return L*( div*_v.rhoU - proj.mortar_to_secondary_int*_v.rhol )
            def F_trasporto(_v): return L*( 
                div*_v.rhocU + div*_v.rhoI 
                - proj.mortar_to_secondary_int*_v.rhocl - proj.mortar_to_secondary_int*_v.rhom
            )
            darcy = (1-steady)*massa.mass/dt*(rho - rho0) + (1-theta)*F_darcy(v) + theta*F_darcy(v0)
            trasporto = (
                (1-steady)*massa.mass/dt*(rho*concentrazione - rho0*concentrazione_0) 
                + (1-theta) * F_trasporto(v) 
                + theta     * F_trasporto(v0)
            )

        darcy_int = L*(laambda - v.Q)
        trasporto_diff_int = L*(mu - v.M)

        eq_manager.equations.update({ 
            'darcy': darcy, 'trasporto': trasporto, 
            'darcy_int': darcy_int, 'trasporto_diff_int': trasporto_diff_int
        })

        #~ ...

        mvem_u = pp.MVEM('mvem_u'); mvem_i = pp.MVEM('mvem_i')
        co.parametro(self.mdg, 'subdomains', 'mvem_u', 'second_order_tensor', diffusivita_darcy)
        co.parametro(self.mdg, 'subdomains', 'mvem_i', 'second_order_tensor', diffusivita_darcy)
        self.ops_velocita = { 'U': v.U, 'I': v.I, 'rhoI': v.rhoI, 'mvem_u': mvem_u, 'mvem_i': mvem_i }

        mu_sd = div*op_trasporto['media'].intf*(proj.mortar_to_primary_int*mu)
        laambda_sd = div*op_trasporto['media'].intf*(proj.mortar_to_primary_int*laambda)
        self.flussi_intf = { 'mu_sd': mu_sd, 'laambda_sd': laambda_sd }

        _scalari_post = ['tempo', 'l2_concentrazione'] + self.impostazioni.get('scalari_post', [])
        self._scalari_post = set([ (dizionario_post[fun] if isinstance(fun, str) else fun) for fun in _scalari_post ])
        self.scalari = {}

        _campi_post = self.impostazioni.get('campi_post', ['velocita'])
        self._campi_post = set([ (dizionario_post[fun] if isinstance(fun, str) else fun) for fun in _campi_post ])
        self.campi = set(['pressione', 'concentrazione'])
        if fratturato: self.campi.update([ 'laambda', 'mu' ])


        self.sistema_lineare() # discretizzo subito gli operatori

        #~ ...

        if True:
            mdg = self.mdg
            dof_manager = self.dof_manager
            self.locali = {}; self.locali.update(locals())

    
    def sistema_lineare(self):
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        primo_giro = not ('bound_flux' in sd_data[pp.DISCRETIZATION_MATRICES]['darcy'].keys())  # una a caso
        if primo_giro:
            self.eq_manager.discretize(self.mdg)
            self._M = self.massa.mass.evaluate(self.dof_manager)
            self.eq_dof = co.ricava_eq_dof(self.dof_manager, self.eq_manager)

        A, b = self.eq_manager.assemble()
        return A, b
    
    def newton(self):
        residuali = []
        incrementi = []

        A, b = self.sistema_lineare()
        residuali.append(b)

        soluzione = self.stato()
        p = [ lambda s:0, lambda s:0, lambda s:0, co.line, print ][ self.impostazioni.get('parla', 0) ]
        p('{:.2f}: ({:d}) sol: {} theta: {:.1f}'.format(self.t, 0, co.norma(self.dof, soluzione), self.theta._value))

        alpha = self.dati.get('newton_alpha', 1)
        k_max = self.dati.get('newton_iter', 20)

        _converged = False
        for k in range(k_max):
            incremento = alpha * co.spsolve(A, b)
            incrementi.append(incremento)

            self.dof_manager.distribute_variable(incremento, additive=True)

            A, b = self.sistema_lineare()
            residuali.append(b)

            soluzione = self.stato()
            p( '{:.2f}: ({:d}) sol: {}  inc: {}'.format(self.t, k+1, co.norma(self.dof, soluzione), co.norma(self.dof, incremento)) )
            
            if np.linalg.norm(incremento[self.dof['concentrazione']], np.inf) > 4: break
            if np.any(np.isnan(incremento)): break
            _converged = self.converged(soluzione, incrementi)
            if _converged: break
        
        if self.impostazioni.get('parla', 0) >= 3: print('')

        res = SimpleNamespace(converged=_converged, iterations=k+1, incrementi=incrementi, residuali=residuali)
        self._ultimo_newton = res
        return res


    def eig_ops(self):
        assert np.all(self.stato() == self._ultimo_steady_state) # gli autovalori si calcolano solo attorno a uno steady state!
        assert self.dati.get('boussinesq', 1) == 1 # il metodo agli autovalori funziona solo con l'approssimazione di boussinesq
        J,_ = self.sistema_lineare()
        return EigOps(J, self._M, self.eq_dof, self.dof)

    def norma_M(self, funs):
        assert funs.shape[0] == sum([ sd.num_cells for sd in self.mdg.subdomains() ])
        return np.sqrt(np.sum( funs * (self._M @ funs), axis=0 ))

    def eig(self, k=3, which='LR', tol=0.01, post=1, ks=0, **kwargs):
        ops = self.eig_ops()
        M, S = ops.M, ops.S_op()

        if ks == 0:
            self._matvec_count = 0
            eigs_kwargs = dict(k=k, which=which, tol=tol, v0=self.random.standard_normal(M.shape[0])) | kwargs
            vals, funs = sps.linalg.eigs(S, **eigs_kwargs)
            # vals, funs = sps.linalg.eigs( L, M=M, k=k, which=which, tol=tol, **kwargs )
            matvec_count = self._matvec_count
        elif ks == 1:
            kw = dict(m=64) | kwargs
            ks = DKS(ops.S, M.shape[0], l=k, b=M.diagonal(), tol=tol, **kw); self._ks = ks
            ks.itera(maxmv=M.shape[0]*10, log_freq=(-1 if self.impostazioni.get('parla', 0) < 4 else 1))
            vals, funs = ks.eig(completi=1)
            vals = vals[:k]; funs = funs[:,:k]
            matvec_count = ks.mv_count
        else: raise SystemError

        tutto_reale = np.all(0 == np.imag(vals)) and np.all(0 == np.imag(funs))
        if not tutto_reale:
            print('autovalori immaginari...')
            return SimpleNamespace(vals=vals, funs=funs, errori=None)

        vals = np.real(vals)
        funs = np.real(funs)

        sort = np.argsort(-vals)
        vals = vals[sort]
        funs = funs[:,sort]

        funs /= self.norma_M(funs)
        funs *= np.sign( funs[0,:] )

        errori = S @ funs - funs@np.diag(vals)
        err_rel = self.norma_M(errori) / vals

        if post: self.eig_post(funs)

        res = SimpleNamespace(vals=vals, funs=funs, errori=errori, err_rel=err_rel, matvec_count=matvec_count)
        self._eig = res
        return res

    def eig_post(self, funs):
        A = self.eig_ops().A
        Ayw = A[1][0]; Ayy = A[1][1]

        if len(funs.shape) == 1: funs = funs.reshape((-1, 1))
        k = funs.shape[1]

        dof_w = np.full(self.dof.num, False); dof_w[self.dof['concentrazione']] = True
        dof_y = np.logical_not(dof_w)

        eig_stati = np.zeros(( self.dof.num, k ))
        eig_stati[dof_w, :] = funs
        eig_stati[dof_y, :] = -co.spsolve(Ayy, Ayw@funs).reshape((-1, k))

        for j,fun in enumerate(funs.T):
            co.carica_stato_f(self.mdg, 'subdomains', lambda sd: { f'eig{j}': fun[self.cdof[sd]] })
            self.campi.add(f'eig{j}')

        stato_originale = self.stato()
        for j,fun in enumerate(funs.T):
            self.carica_stato(eig_stati[:,j])
            velocita(self)
            for _,data in self.mdg.subdomains(return_data=True): data[pp.STATE][f'eig_uvel{j}'] = data[pp.STATE]['vu'].copy()
            self.campi.add(f'eig_uvel{j}')

        self.carica_stato(stato_originale)
        velocita(self)

    def dt(self, dt=None):
        if dt == np.inf:
            self.steady._value = 1
            self._dt._value = 100 # un valore a caso...
        elif not dt is None:
            self.steady._value = 0
            self._dt._value = dt
        else: return self._dt._value

    def step(self):
        self.carica_stato(self.stato(), zero=True)
        newton = self.newton()
        return newton

    def steady_state(self):
        self.dt(np.inf)
        newton = self.step()
        
        if not newton.converged: 
            print('newton not converged')
            raise SystemError
        
        self._ultimo_steady_state = self.stato()
        if not hasattr(self, '_primo_steady_state'): self._primo_steady_state = self._ultimo_steady_state
        if self.impostazioni.get('esporta', False): self.esporta(stato=1, pkl=1)

        return newton

    def avanza(self, dt, T):
        self.dt(dt)

        p = [ lambda s:0, co.line, print, print, print ][ self.impostazioni.get('parla', 0) ]
        esporta = self.impostazioni.get('esporta', False)
        if esporta: self.esporta()

        while True:
            newton = self.step()
            if not newton.converged: 
                self.t += dt/10
                if esporta: self.esporta(stato=1)
                raise SystemError

            self.t += self.dt()

            sherwood = ''
            if 'sherwood' in self.scalari.keys(): sherwood = f'sherwood-1 = {self.scalari["sherwood"][-1]-1:<10.4g}'
            p(f't = {self.t:<8.3g} #iterazioni = {newton.iterations:<15} ' + sherwood)

            if esporta: self.esporta()
            if self.t >= T: break
        
        if esporta: self.esporta(stato=1, pkl=1)
        

    def adaptive(self, dt_init, dt_max=None, T=None, sherwood_min=None):
        POCHE_ITER = 3; TROPPE_ITER = 6; ACC = 1.05; MAX_DCONC = 1e-2; DEC = 2 # parametri adattivita

        assert not (dt_max is None and T is None)
        if dt_max is None: dt_max = np.inf
        if T is None: T = np.inf

        p = [ lambda s:0, co.line, print, print, print ][ self.impostazioni.get('parla', 0) ]
        esporta = self.impostazioni.get('esporta', False)

        self.dt(dt_init)
        prev_dconc = 0

        while True:
            dt = self.dt()
            newton = self.step()
            if (not newton.converged):
                self.carica_stato(self.stato(zero=True))
                self.dt(dt / DEC)
                continue

            self.t += dt

            dconcentrazione = self.stato(dof_key='concentrazione') - self.stato(dof_key='concentrazione', zero=1)
            max_dconc = np.linalg.norm(dconcentrazione, np.inf)

            sherwood = ''
            if 'sherwood' in self.scalari.keys(): sherwood = f'sherwood-1 = {self.scalari["sherwood"][-1]-1:<8.3g}'
            p(f't = {self.t:<8.3g} dt = {dt:<8.2g} #iterazioni = {newton.iterations:<15} dconc = {max_dconc:<10.2g} ' + sherwood)

            if esporta: self.esporta()
            if max_dconc < MAX_DCONC and (dt >= dt_max or self.t >= T): break
            if sherwood_min is not None: 
                assert 'sherwood' in self.scalari.keys()
                if self.scalari["sherwood"][-1] >= sherwood_min: break


            accelera = newton.iterations <= POCHE_ITER and (
                (max_dconc < 1e-6*MAX_DCONC) or (max_dconc < MAX_DCONC and max_dconc < ACC**2*prev_dconc)
            )
            if accelera: self.dt(dt * ACC)
            elif newton.iterations >= TROPPE_ITER: self.dt( dt / ACC )

            prev_dconc = max_dconc

        if esporta: self.esporta(stato=1, pkl=1)


    def esporta(self, stato=None, pkl=None):
        nome = self.impostazioni['nome']

        init = not hasattr(self, 'exporter')
        if init: 
            self.i = 0
            self.cartella_out = self.impostazioni.get('cartella', co.cartella_tmp) + '/' + nome
            co.prepara_cartella(self.cartella_out)
            
            self.vtu_ts = []
            self.vtu_is = []
            self.exporter = pp.Exporter(self.mdg, nome, folder_name=self.cartella_out + '/pvd')

        if stato is None: stato = (self.i % self.impostazioni.get('stato_freq', 20) == 0)
        if pkl is None: pkl = (self.i % self.impostazioni.get('pkl_freq', 50) == 0)

        if stato:
            self.vtu_ts.append(self.t)
            self.vtu_is.append(self.i)
            self.exporter.write_vtu(list(self.campi) + self.campi_post(), time_step=self.i)
            self.exporter.write_pvd(np.array(self.vtu_ts))
            _stato = self.stato()
            np.save(f'{self.cartella_out}/txt/soluzione_{self.i}.npy', _stato)

        if pkl:
            co.salva(self)
            if hasattr(self, '_eig'): co.salva(self, eig=1)

        self.scalari_post()
        scipy.io.savemat(f'{self.cartella_out}/scalari.mat', self.scalari, oned_as='column')

        self.i += 1

    def scalari_post(self):
        for fun in self._scalari_post: fun(self)
    def campi_post(self):
        lista_campi = []
        for fun in self._campi_post: 
            lista = fun(self)
            lista_campi += lista
        return lista_campi

    def chiudi_scalari(self):
        scalari = {}
        for key,val in self.scalari.items():
            arr = np.vstack(val)
            if arr.shape[1] == 1: arr = arr.reshape(-1)
            scalari[key] = arr
        return scalari

    def importa_i(self, idx):
        txt = f'{self.cartella_out}/txt/soluzione_{idx}.npy'
        return np.load(txt)
    def importa_t(self, t): 
        i = self.vtu_is[ np.abs(np.array(self.vtu_ts) - t).argmin() ]
        return self.importa_i(i)



        # ; m_inv = 1/M.diagonal()
        # self._matvec_count = 0
        # def L_matvec(v): 
        #     self._matvec_count += 1
        #     return Awy@Ayy_inv(Ayw@v) - Aww@v
        # L = sps.linalg.LinearOperator(shape=M.shape, matvec=L_matvec, matmat=L_matvec)
        
        # def S_matvec(v): return (m_inv*L_matvec(v).T).T
        # S = sps.linalg.LinearOperator(shape=M.shape, matvec=S_matvec, matmat=S_matvec)

        # return SimpleNamespace(A=[[Aww, Awy], [Ayw, Ayy]], M=M, L=L, S=S)

class EigOps:
    def __init__(self, J, M, eq_dof, dof):
        eqdof_w = np.full(dof.num, False); eqdof_w[eq_dof['trasporto']] = True
        eqdof_y = np.logical_not(eqdof_w)
        dof_w = np.full(dof.num, False); dof_w[dof['concentrazione']] = True
        dof_y = np.logical_not(dof_w)

        Aww = J[eqdof_w, :][:, dof_w]; Awy = J[eqdof_w, :][:, dof_y]
        Ayw = J[eqdof_y, :][:, dof_w]; Ayy = J[eqdof_y, :][:, dof_y]
        Ayy_inv = sps.linalg.factorized(Ayy.tocsc())        

        self.A = (( Aww, Awy ), ( Ayw, Ayy ))
        self.M = M
        self.m_inv = 1/M.diagonal()
        self.Ayy_inv = Ayy_inv

    def __getstate__(self): return self.A, self.M
    def __setstate__(self, state): 
        A, M = state
        Ayy = A[1][1]; Ayy_inv = sps.linalg.factorized(Ayy.tocsc())
        self.A = A; self.M = M
        self.m_inv = 1/M.diagonal()
        self.Ayy_inv = Ayy_inv

    def L(self, v):
        ((Aww, Awy), (Ayw, Ayy)) = self.A
        Ayy_inv = self.Ayy_inv
        Lv = Awy@Ayy_inv(Ayw@v) - Aww@v 
        return Lv
    def L_op(self): return sps.linalg.LinearOperator(shape=self.M.shape, matvec=self.L)

    def S(self, v):
        ((Aww, Awy), (Ayw, Ayy)) = self.A
        Ayy_inv = self.Ayy_inv
        Lv = Awy@Ayy_inv(Ayw@v) - Aww@v 
        return (self.m_inv * Lv.T).T
    def S_op(self): return sps.linalg.LinearOperator(shape=self.M.shape, matvec=self.S)



def tempo(self):
    init = not ('t' in self.scalari.keys())
    if init: self.scalari['t'] = []
    self.scalari['t'].append(self.t)

def l2_concentrazione(self):
    init = not ('l2_concentrazione' in self.scalari.keys())
    if init: self.scalari['l2_concentrazione'] = []
    self.scalari['l2_concentrazione'].append(0) # TODO


def velocita(self):
    subdomains = self.mdg.subdomains()

    init = not ('vu' in self.mdg.subdomain_data(subdomains[0])[pp.STATE] )
    if init:
        for sd, sd_data in self.mdg.subdomains(return_data=True): 
            self.ops_velocita['mvem_u'].discretize(sd, sd_data)
            self.ops_velocita['mvem_i'].discretize(sd, sd_data)

    U = self.ops_velocita['U'].evaluate(self.dof_manager).val
    I = self.ops_velocita['I'].evaluate(self.dof_manager).val

    # NOTE: da simulazioni su darcy nudo sembra che project_flux trasformi flussi [m^D/s] in velocita [m/s]
    co.carica_stato_f(self.mdg, 'subdomains', lambda sd: { 
        'vu': self.ops_velocita['mvem_u'].project_flux(sd, U[self.fdof[sd]], self.mdg.subdomain_data(sd)),
        'vi': self.ops_velocita['mvem_i'].project_flux(sd, I[self.fdof[sd]], self.mdg.subdomain_data(sd)),
    })

    return ['vu', 'vi']

def flussi_intf(self):
    co.carica_stato_f(self.mdg, 'subdomains', lambda sd: { 
        'xmu': self.flussi_intf['mu_sd'].evaluate(self.dof_manager).val[self.cdof[sd]], 
        'xlaambda': self.flussi_intf['laambda_sd'].evaluate(self.dof_manager).val[self.cdof[sd]] 
    })

    return ['xmu', 'xlaambda']

dizionario_post = { 'velocita': velocita, 'tempo': tempo, 'l2_concentrazione': l2_concentrazione, 'flussi_intf': flussi_intf }

