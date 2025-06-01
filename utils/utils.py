# Helper functions

import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib import gridspec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from geopy.distance import distance


# Function to create a cubic bounding box for 3D plots
def cubic_box(X,Y,Z,ax):
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
        
# Function to format the z-axis labels from degrees to km     
def z_fmt(z, pos):
    return round(z*111.1)

def fig_setup(freqs):
    fig = plt.figure(figsize=(8,12))
    spec = gridspec.GridSpec(nrows=len(freqs), ncols=3,
                        wspace=0.1,
                        hspace=0.3, width_ratios = [1.25,1.25,1.25], 
                        height_ratios= [1]*len(freqs))
    return fig, spec

def fig_axes_config(fig, spec, i, extent, region, im=None, gdfs=None):
    index_base = i * 3
    axes = [fig.add_subplot(spec[index_base+j], projection=ccrs.PlateCarree()) for j in range(3)]
    for ax in axes:
        ax.set_extent(extent)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.set(xlim=(region[0], region[1]), ylim=(region[2], region[3]))
        ax.imshow(im, extent=region, cmap='gray')
        
        if gdfs:
            for i, color in zip([0, 1, 2], ['white', 'aqua', 'k']):
                gdfs[i].plot(ax=ax, color=color, zorder=i+1)
                  
    if i == 0:
        axes[0].set_yticks(np.linspace(7,7.5,2), crs=ccrs.PlateCarree())
    else:
        axes[0].set_xticks(np.linspace(-74,-73,3), crs=ccrs.PlateCarree())
        axes[0].set_yticks(np.linspace(7,7.5,2), crs=ccrs.PlateCarree())
        axes[1].set_xticks(np.linspace(-74,-73,3), crs=ccrs.PlateCarree())
        axes[2].set_xticks(np.linspace(-74,-73,3), crs=ccrs.PlateCarree())        
    return axes 
      
def reshape_and_flip(array, shape=(13, 17)):
    return np.flip(np.reshape(array, shape), 0)


def matrix_rot_trans(x, y, z, x_angle, y_angle, z_angle, x_pos, y_pos, z_pos):
    ''' 
    Return the rotation and translation coordinates of arbitrary points in regards to x,y,z axes
    Works with scalars or arrays of the same shape
    '''
    # Rotation matrices
    cx, sx = np.cos(x_angle), np.sin(x_angle)
    cy, sy = np.cos(y_angle), np.sin(y_angle)
    cz, sz = np.cos(z_angle), np.sin(z_angle)
    
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])
    
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
    
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix   
    m_r = Rz @ Ry @ Rx  # shape (3, 3)
    
    # Stack x, y, z into shape (..., 3)
    v = np.stack((x,y,z), axis=-1)

    # Rotation to each 3-vector
    v_r = v @ m_r.T  # shape (256, 256, 3)

    # Translation to each 3-vector
    xrt = v_r[..., 0] + x_pos
    yrt = v_r[..., 1] + y_pos
    zrt = v_r[..., 2] + z_pos

    return xrt, yrt, zrt

def translate_and_rotate_points(theta, phi, x0, y0, z0, xv, yv, zv):
        # Precompute rotation matrices (note: Rx is identity matrix since angle is 0)
    Rx = np.eye(3)  
    Ry = np.array([
        [np.cos(-phi), 0, np.sin(-phi)],
        [0, 1, 0],
        [-np.sin(-phi), 0, np.cos(-phi)]
    ])
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # The order of multiplication matters: R = Rx @ Ry @ Rz. First rotate around z, then y, then x
    R = Rx @ Ry @ Rz

    # Stack the coordinates into a 3D grid (nx, ny, nz, 3)
    points = np.stack([xv, yv, zv], axis=-1)

    # Subtract translation
    points -= np.array([x0, y0, z0])

    # Apply rotation
    rotated_points = np.tensordot(points, R.T, axes=1)  # shape: (nx, ny, nz, 3)

    # Unpack rotated points into x, y, z 
    x_rot, y_rot, z_rot = rotated_points[..., 0], rotated_points[..., 1], rotated_points[..., 2]

    return x_rot, y_rot, z_rot
    
    

def parameterize_ellipsoid(a, b, c, sin_beta, cos_beta, cos_gamma, sin_gamma, ones_gamma):
    x = (a / 111) * sin_beta * cos_gamma
    y = (b / 111) * sin_beta * sin_gamma
    z = (c / 111) * cos_beta * ones_gamma
    return x, y, z

def compute_ellipsoid_parameters(x1, y1, x2, y2, z1, z2, v, t):
    
    r = np.hypot(distance((y1,x1),(y2,x2)).km, (z1-z2)*111.1)
    
    a = (v*t/2)
    
    if a**2 - r**2/4 < 0 or np.iscomplex((a**2 - r**2/4)**0.5):
        b = 0 # Set b to 0 if the number is negative or complex           
    else:
        b = (a**2 - r**2/4)**0.5
      
    slope_xy = (y2-y1)/(x2-x1)
    theta = -np.arctan(slope_xy)
    x1_p = x1/np.cos(theta)
    x2_p = x2/np.cos(theta)
    slope_xz = (z2-z1)/(x2_p-x1_p)
    phi = -np.arctan(slope_xz)
    
    return a, b, theta, phi