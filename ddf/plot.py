import numpy as np
import ddf.common as co
import porepy as pp

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib

def plot_cells(pb, var, Nx=1024, ax=None, fratture_lw=0, solo_fratture=0, fill=1, method='cubic', **kwargs):
    if ax is None: fig,ax = plt.subplots()

    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

    mdg = pb.mdg
    sD = mdg.subdomains()[0]

    if fratture_lw > 0:
        for sd in mdg.subdomains()[1:]:
            x,y,_ = sd.nodes
            ax.plot(x,y,c='white' if fill else 'black', linewidth=fratture_lw)
        if solo_fratture: return ax
    
    if isinstance(var, str): f = mdg.subdomain_data(sD)[pp.STATE][var]
    elif hasattr(var, 'shape'):
        if var.shape[0] == sD.num_cells: f = var
        elif var.shape[0] == sum([ sd.num_cells for sd in mdg.subdomains() ]): f = var[pb.cdof[sD]]
        else: raise SystemError
    else: raise SystemError
    
    W,H,_ = co.grid_size(mdg)
    X,Y = np.meshgrid( np.linspace(0,W,Nx), np.linspace(0,H,int(Nx*H/W)) )
    x,y,_ = sD.cell_centers

    F = griddata((x,y), f, (X,Y), method=method)
    if fill: ax.contourf(X, Y, F, **kwargs)
    # if fill: ax.pcolormesh(X, Y, F, **kwargs)
    else: ax.contour(X, Y, F, **kwargs)

    return ax

def quiver(pb, var, subdomains=None, ax=None, soglia=0.05, skip=None, **kwargs):
    if ax is None: fig,ax = plt.subplots()

    mdg = pb.mdg
    if subdomains is None: subdomains = mdg.subdomains()
    
    if isinstance(var, str):
        dist = co.raccogli_stato(mdg, var)
        dist = { sd: dist[sd] for sd in subdomains }
    elif hasattr(var, 'shape'): 
        if var.shape[0] != 3: raise SystemError
        dist = { sd: var[:, pb.cdof[sd]] for sd in subdomains  }
    elif isinstance(var, dict): dist = { sd: var[sd] for sd in subdomains }
    else: raise SystemError

    xs, ys, _ = np.concatenate( [sd.cell_centers for sd in subdomains], axis=1 )
    us, vs, _ = np.concatenate( [dist[sd] for sd in subdomains], axis=1 )
    
    speed = np.sqrt(us**2 + vs**2)
    max_speed = np.linalg.norm(speed, np.inf)
    mask = speed > soglia*max_speed
    xs = xs[mask]; ys = ys[mask]
    us = us[mask]; vs = vs[mask]

    if skip is not None:
        xs = xs[::skip]; ys = ys[::skip]
        us = us[::skip]; vs = vs[::skip]
    ax.quiver(xs,ys,us,vs, pivot='mid', **kwargs)

    return ax

def quiver2(pb, var, subdomains=None, ax=None, soglia=0.05, skip=None, **kwargs):
    if ax is None: fig,ax = plt.subplots()

    mdg = pb.mdg
    if subdomains is None: subdomains = mdg.subdomains()
    
    if isinstance(var, str):
        dist = co.raccogli_stato(mdg, var)
        dist = { sd: dist[sd] for sd in subdomains }
    elif hasattr(var, 'shape'): 
        if var.shape[0] != 3: raise SystemError
        dist = { sd: var[:, pb.cdof[sd]] for sd in subdomains  }
    elif isinstance(var, dict): dist = { sd: var[sd] for sd in subdomains }
    else: raise SystemError

    xs, ys, _ = np.concatenate( [sd.cell_centers for sd in subdomains], axis=1 )
    us, vs, _ = np.concatenate( [dist[sd] for sd in subdomains], axis=1 )
    
    speed = np.sqrt(us**2 + vs**2)
    max_speed = np.linalg.norm(speed, np.inf)
    mask = speed > soglia*max_speed
    xs = xs[mask]; ys = ys[mask]
    us = us[mask]; vs = vs[mask]
    speed = speed[mask]

    if skip is not None:
        xs = xs[::skip]; ys = ys[::skip]
        us = us[::skip]; vs = vs[::skip]
    ax.quiver(xs,ys,us/speed,vs/speed, np.log(speed), pivot='mid', **kwargs)

    return xs,ys,us,vs,speed

def streamplot(pb, var, ax=None, N=256, soglia=0.05, max_lw=1.5, density=1.5, arrowsize=0.6, max_speed=None, **kwargs):
    if ax is None: fig,ax = plt.subplots()

    mdg = pb.mdg
    sD = mdg.subdomains()[0]

    if isinstance(var, str): f = mdg.subdomain_data(sD)[pp.STATE][var]
    elif hasattr(var, 'shape'):
        if var.shape[1] == sD.num_cells: f = var
        elif var.shape[1] == sum([ sd.num_cells for sd in mdg.subdomains() ]): f = var[:,pb.cdof[sD]]
        else: raise SystemError
    else: raise SystemError

    W,H,_ = co.grid_size(pb.mdg)
    X,Y = np.meshgrid(np.linspace(0,W,N), np.linspace(0,H,int(N*H/W)) )
    x,y,_ = sD.cell_centers

    U = griddata((x,y), f[0,:], (X,Y), method='cubic')
    V = griddata((x,y), f[1,:], (X,Y), method='cubic')

    ccr = co.raccogli_stato(pb.mdg, 'cc')
    xf, yf, _ = np.concatenate([ccr[sd] for sd in mdg.subdomains()], axis=1)
    f = np.zeros_like(xf); f[sD.num_cells:] = 1
    F = griddata((xf, yf), f, (X, Y), method='nearest')

    speed = np.sqrt(U**2 + V**2)

    _max_speed = np.max(speed[~np.isnan(speed)])
    if max_speed is None: max_speed = _max_speed
    if max_speed < _max_speed: raise SystemError
    
    aspeed = speed/max_speed

    color = np.ones_like(aspeed); color[aspeed < soglia] = 0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('asdf', [(0,0,0,0),(0,0,0,1)], N=2)
    lw = max_lw*aspeed*(aspeed>soglia)
    
    ax.streamplot(
        X, Y, np.ma.array(U, mask=F), V, 
        color=color, cmap=cmap, linewidth=lw, 
        density=density, arrowsize=arrowsize, **kwargs
    )

