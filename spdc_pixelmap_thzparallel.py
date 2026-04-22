import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm

#Units
mm = 1e-3
nm = 1e-9
um = 1e-6
thz = 1e12
ps = 1e-12
c = 3e8

pi4 = np.pow(np.pi,4)
pi2 = np.pow(np.pi,2)
pi = np.pi

eps0 = 8.854e-12


#Experimental Parameters ===================================================================================================================================================

#Crystal parameters

chi_eff = 327*1e-12 #pm/V 
pp = 180*um #Poling period
T = 27 #degree C
a = np.array([0,0,1]) #Crystal axis
L = 10*mm


#pump parameters
lam_p = 659.58*nm 
om_p = 2*np.pi*c/lam_p
w_p = 43*um
T_I = 75*ps
pow = 200*1e-3

#THz range
nu_thz_max = 3.5*thz
nu_thz_min = 0.1*thz

theta_max = 0.01*np.pi

#Refractive Index functions

def n_o_thz(nu):
    nu = nu*1e-12 #frequency in thz
    A = 6.5
    B = 8.2e-2
    C = 6e-3
    n = A + B*nu*nu + C*nu*nu*nu*nu
    return(n)

def n_e_thz(nu):
    nu = nu*1e-12
    A = 5
    B = 2.5e-2
    C = 3e-3
    n = A + B*nu*nu + C*nu*nu*nu*nu
    return(n)

def n_ge_thz(nu):
    nu = nu*1e-12
    A = 5
    B = 2.5e-2
    C = 3e-3
    n_p = 2*B*nu + 4*C*nu*nu*nu*(1e-12/(2*pi)) #differenciating wrt omega (chain rule)
    ng = n_e_thz(nu) + 2*pi*nu*n_p*1e12
    return(ng)

def n_go_thz(nu):
    nu = nu*1e-12
    A = 6.5
    B = 8.2e-2
    C = 6e-3
    n_p = (2*B*nu + 4*C*nu*nu*nu)*(1e-12/(2*pi))
    ng = n_o_thz(nu) + 2*pi*nu*n_p*1e12
    return(ng)

def n_e_ir(lam_m):
    lam = lam_m*1e6
    a1 = 5.756
    a2 = 0.0983
    a3 = 0.2020 
    a4 = 189.32
    a5 = 12.52
    a6 = 1.32e-2
    b1 = 2.86e-6
    b2 = 4.7e-8
    b3 = 6.113e-8
    b4 = 1.5616e-4
    f = (T-24.5)*(T+570.82)
    nsq = a1 + b1*f + (a2+b2*f)/((lam*lam)-(a3 + b3*f)*(a3+b3*f)) + (a4+b4*f)/(lam*lam-a5*a5) - a6*lam*lam
    return(np.sqrt(nsq))

def n_ge_ir(lam_m):
    lam = lam_m*1e6
    a1 = 5.756
    a2 = 0.0983
    a3 = 0.2020 
    a4 = 189.32
    a5 = 12.52
    a6 = 1.32e-2
    b1 = 2.86e-6
    b2 = 4.7e-8
    b3 = 6.113e-8
    b4 = 1.5616e-4
    f = (T-24.5)*(T+570.82)
    n_p = (1e6*(lam**3)/(2*pi*c*n_e_ir(lam_m)))*((a3+b3*f)/(((lam**2)-(a3+b3*f)**2)**2) + (a4+b4*f)/(((lam**2)-(a5**2))**2) + a6)
    ng = n_e_ir(lam_m) + (2*pi*c/lam_m)*n_p
    return(ng)


def n_o_ir(lam_m):
    lam = lam_m*1e6
    a1 = 5.653
    a2 = 0.1185
    a3 = 0.2091
    a4 = 89.61
    a5 = 10.85
    a6 = 1.97e-2
    b1 = 7.941e-7
    b2 = 3.134e-8
    b3 = -4.641e-9
    b4 = -2.188e-6
    f = (T-24.5)*(T+570.82)
    nsq = a1 + b1*f + (a2+b2*f)/((lam**2)-(a3 + b3*f)**2) + (a4+b4*f)/((lam**2)-a5**2) - a6*lam**2
    return(np.sqrt(nsq))

def n_go_ir(lam_m):
    lam = lam_m*1e6
    a1 = 5.653
    a2 = 0.1185
    a3 = 0.2091
    a4 = 89.61
    a5 = 10.85
    a6 = 1.97e-2
    b1 = 7.941e-7
    b2 = 3.134e-8
    b3 = -4.641e-9
    b4 = -2.188e-6
    f = (T-24.5)*(T+570.82)
    n_p = (1e6*(lam**3)/(2*pi*c*n_o_ir(lam_m)))*((a3+b3*f)/(((lam**2)-(a3+b3*f)**2)**2) + (a4+b4*f)/(((lam**2)-(a5**2))**2) + a6)
    ng = n_o_ir(lam_m) + (2*pi*c/lam_m)*n_p
    return(ng)


def n_e_eff_thz(nu,theta):
    nsq = 1/((np.cos(theta)**2/(n_e_thz(nu)**2)+(np.sin(theta)*np.sin(theta)**2)/n_o_thz(nu)**2))
    return(np.sqrt(nsq))


#Imaging ===================================================================================================================================================

#Imaging
f2 = 120.0      # mm
f3 = 400.0      # mm
slit_width = 1 # mm

# Camera
pixel_size = 0.065 # mm
camera_x = 1920
camera_y = 1080
image = np.zeros((camera_y, camera_x))

# Grating
lines_per_mm = 1908
d = 1/ (1000*lines_per_mm)
order = 1
rot = 0.681 #Grating rotation angle

def matrix_free_space(L):
    return np.array([[1, L], [0, 1]])

def matrix_lens(f):
    return np.array([[1, 0], [-1/f, 1]])

ts =[]
ws =[]

d1 = f2 #Before Lens f2
d2 = f2 #After Lens f2 to slit
d3 = f3 #After Slit to Lens f3
d4 = f3 #After Lens f3 to Grating
d5 = 2000 #After Grating to Camera

M_to_slit = matrix_free_space(d2)@ matrix_lens(f2)@ matrix_free_space(d1)
M_slit_to_grating = matrix_free_space(d4)@ matrix_lens(f3)@ matrix_free_space(d3)
M_grating_to_camera = matrix_free_space(d5)

def pixelmap_vec(kst_x,kst_y,ks_z,om_s):
    
    lambda_s = 2*pi*c/om_s
    
    # Initial ray angles from k_s components
    theta_x = np.arctan2(kst_x,ks_z)  # angle in x-z plane
    theta_y = np.arctan2(kst_y,ks_z)  # angle in y-z plane

    ray_x = np.vstack((np.zeros(np.shape(theta_x)),theta_x))
    ray_y = np.vstack((np.zeros(np.shape(theta_y)),theta_y))


    mask = np.ones(len(theta_x),dtype = bool)
    
    ray_x = M_to_slit @ ray_x
    ray_y = M_to_slit @ ray_y


    mask = np.abs(ray_x[0])<=(slit_width/2)
    ray_x[:,~mask] = 0

    
    ray_x = M_slit_to_grating @ ray_x
    ray_y = M_slit_to_grating @ ray_y
    

    # Apply grating in x

    
    sintheta = np.sin(ray_x[1]-rot) + order*(lambda_s/d)
    theta_x_after_grating = np.arcsin(sintheta)-rot

    ray_x = np.vstack((ray_x[0],theta_x_after_grating))
    
    # theta = ray_x[1]-rot + order*(lambda_s/d)
    # theta_x_after_grating = theta -rot

    # ray_x = np.vstack((ray_x[0],theta_x_after_grating))


    # Propagate after grating
    ray_x = M_grating_to_camera @ ray_x
    ray_y = M_grating_to_camera @ ray_y
    

    # Convert position to pixel index
    i = (camera_x // 2 + (ray_x[0] / pixel_size).astype(int))
    j = (camera_y // 2 + (ray_y[0] / pixel_size).astype(int))
    
    camera_mask = (i>=0) & (i<camera_x) & (j>=0) & (j<camera_y)

    i_out = np.where(camera_mask,i,0)
    j_out = np.where(camera_mask,j,0)


    total_mask = camera_mask & mask

    return i_out,j_out,total_mask

#=====================================================================================================================================================================

#Simulation parameters

n_samples = 10000

#Range for signal
om_s_min = om_p - (2*pi*nu_thz_max)
om_s_max = om_p - (2*pi*nu_thz_min)

# om_s_min = 2*pi*c/(665*1e-9)
# om_s_max = 2*pi*c/(659.72*1e-9)

delta_om_s = (2*np.pi*c) / (2 * 2.2 *L)
N_omega_s = int((om_s_max - om_s_min)/delta_om_s)  +  100


#Range for signal angle(theta_s)
theta_max = 3 * np.pi/180 
delta_theta = (lam_p) / (5 * np.pi * w_p)
N_theta = int(theta_max / delta_theta) +100


Z = (16*pow*(w_p**2)*(L**2)*T_I)/(((2*np.pi)**7)*eps0*n_o_ir(lam_p)*c)


#Range for signal angle(theta_s)
N_phi = 100

om_s_grid = np.linspace(om_s_min, om_s_max, N_omega_s)
cos_theta_grid = np.linspace(np.cos(theta_max),1, N_theta)
phi_s_grid = np.linspace(0, 2*np.pi, N_phi, endpoint=False)

d_omega = om_s_grid[1] - om_s_grid[0]
d_cos_theta = cos_theta_grid[1] - cos_theta_grid[0]  # uniform in cos
d_phi = phi_s_grid[1] - phi_s_grid[0]

A1, B1 = 0.97718851, 0.40371072
A2, B2 = 0.03285665, 0.40371451
A3, B3 = 0.01294228, 0.00518248


a_coeffs = np.array([A1,A2,A3])
b_coeffs = np.array([B1,B2,B3])

weights = a_coeffs/a_coeffs.sum()
sigmas = 1/np.sqrt(2*b_coeffs)

k_p = n_o_ir(lam_p)*om_p/c

def sample_sinc(N):

    components = np.random.choice(3,size=N,p=weights)
    sig = sigmas[components]
    x_samples = np.random.normal(0,sig)

    return(x_samples)

def sinc_pdf(x):
    w = weights[:,None]
    sig = sigmas[:,None]
    

    pdf = w*norm.pdf(x[None,:],0,sig)

    q = pdf.sum(axis = 0)

    return q

#Monte Carlo Loop
def gamma(ks_x,ks_z,om_s,m,pid,ps,n_s):

    #sampling omega
    om_i_center = om_p - om_s #(scalar)
    x_omega = sample_sinc(n_s) #(ns)
    om_i = om_i_center + x_omega[None,:]*(2/T_I) #(1,ns)

    if pid=='o':
        n_i = n_o_thz(2*np.pi*c/om_i)  #(1,ns)
        n_gi = n_go_thz(2*np.pi*c/om_i) #(1,ns)
        k_i = n_i*om_i/c #(1,ns)
        k_i_star = n_i*om_i_center/c
        dk_domega = n_gi/c #(1,ns)
    elif pid=='e':
        n_i = n_e_thz(2*np.pi*c/om_i)  #(1,ns)
        n_gi = n_ge_thz(2*np.pi*c/om_i) #(1,ns)
        k_i = n_i*om_i/c #(1,ns)
        k_i_star = n_i*om_i_center/c
        dk_domega = n_gi/c #(1,ns)

    #sampling theta
    cos_theta_i_star = np.clip((k_p - ks_z[:,None] + (2*pi*m/pp))/k_i_star, -1,1) #(Ntheta,ns)

    theta_i_star = np.arccos(cos_theta_i_star) #(Ntheta)
    sin_theta_i_star = np.sin(theta_i_star) #(Ntheta)
    
    sigma_theta = 2.0/(L*k_i*np.where(sin_theta_i_star>1e-6,sin_theta_i_star,1e-6)) #(Ntheta)

    eps_theta = np.random.normal(0,1,n_s) #(ns)
    theta_i = theta_i_star + sigma_theta*eps_theta[None,:] #(Ntheta,ns)
    theta_i = np.clip(theta_i,0,np.pi)
    sin_theta_i = np.sin(theta_i)
    cos_theta_i = np.cos(theta_i) #(Ntheta,ns)

    #sample phi

    k_perp_i = k_i*sin_theta_i #(Ntheta,ns)
    sigma_phi = 1/(w_p*np.where(k_perp_i>1e-10,k_perp_i,1e-10))

    phi_i_star = np.arccos(np.clip(-ks_x[:,None]/np.where(k_perp_i > 1e-10, k_perp_i, 1e-10), -1, 1))

    eps_phi = np.random.normal(0,1,n_s)
    phi_i = phi_i_star + sigma_phi*eps_phi[None,:]

    # Reconstruct Cartesian k_i components
    k_ix = k_perp_i * np.cos(phi_i)
    k_iy = k_perp_i * np.sin(phi_i)
    k_iz = k_i * cos_theta_i 
    
    #Phase mismatches
    
    delta_kz    = k_p - ks_z[:,None] - k_iz + (2*pi/pi)
    delta_kx    = ks_x[:,None] + k_ix
    delta_ky    = k_iy  
    delta_omega = om_p - om_s - om_i

    #Integrand
    if ps =='o':
        n_s = n_o_ir(2*np.pi*c/om_s)
    elif ps=='e':
        n_s = n_e_ir(2*np.pi*c/om_s)

    if(pid == 'e'):
        n_i = n_e_eff_thz(om_i/(2*np.pi),theta_i)

    prefactor = (chi_eff**2)*om_i*om_s/(n_s**2 * n_i**2)
    
    sinc_z     = np.sinc(delta_kz    * L   / (2*np.pi))**2
    sinc_t     = np.sinc(delta_omega * T_I / (2*np.pi))**2
    gauss_perp = np.exp(-0.5*(delta_kx**2 + delta_ky**2)*w_p**2)

    jacobian = (k_i**2)* dk_domega*sin_theta_i

    f = prefactor * sinc_z * gauss_perp * sinc_t * jacobian


    #PDFs for Monte carlo
    q_omega = sinc_pdf(x_omega) * (T_I/2)
    q_theta = norm.pdf(eps_theta[None,:], 0, 1) / sigma_theta
    q_phi   = norm.pdf(eps_phi[None,:], 0, 1) / sigma_phi
    
    q = q_omega * q_theta * q_phi
    w = f/q
    return (w.mean(axis = 1))

sim_len = len(om_s_grid)*len(cos_theta_grid)*len(phi_s_grid)
om_len = len(om_s_grid)
print("Total Iterations = ",sim_len)

#Signal pixel Loop

ws = []
ins = []

def process_omega(om_s):
    
    theta = np.arccos(cos_theta_grid)
    
    print((2*pi*c/om_s)*1e9,"nm")


    m = [1] 
    intensity = np.zeros(len(cos_theta_grid))
    
    for order in m:

        # #Signal ordinary
        # k = n_o_ir(2*pi*c/om_s)*(om_s/c)
        # ks_x = k*np.sin(theta)
        # ks_z = k*np.cos(theta)
        
        # Tk = gamma(ks_x,ks_z,om_s,order,"o","o",n_samples) #idler ordinary signal ordinary
        # intensity += Z*Tk*k*k*(n_go_ir(2*pi*c/om_s)/c)*d_omega*d_cos_theta*d_phi
        
        # Tk = gamma(ks_x,ks_z,om_s,order,"e","o",n_samples) #idler eordinary signal ordinary
        # intensity += Z*Tk*k*k*(n_ge_ir(2*pi*c/om_s)/c)*d_omega*d_cos_theta*d_phi
        

        # Signal Eordinary 
        k = n_e_ir(2*pi*c/om_s)*(om_s/c)
        ks_x = k*np.sin(theta)
        ks_z = k*np.cos(theta)

        print("Ir",n_e_ir(2*pi*c/om_s))
        
        Tk = gamma(ks_x,ks_z,om_s,order,"o","e",n_samples) #idler ordinary signal eordinary
        intensity += Z*Tk*k*k*(n_go_ir(2*pi*c/om_s)/c)*d_omega*d_cos_theta*d_phi
        
        Tk = gamma(ks_x,ks_z,om_s,order,"e","e",n_samples) #idler eordinary signal eordinary
        intensity += Z*Tk*k*k*(n_ge_ir(2*pi*c/om_s)/c)*d_omega*d_cos_theta*d_phi
    
    

    xs,ys,vals = [],[],[]

    for phi in phi_s_grid:
        
        kst_x = ks_x*np.cos(phi)
        kst_y = ks_x*np.sin(phi)

        X,Y,P = pixelmap_vec(kst_x,kst_y,ks_z,om_s)

        valid = P.astype(bool)

        if valid.any():

            xs.append(X[valid].astype(int))
            ys.append(Y[valid].astype(int))
            vals.append(intensity[valid])
    
    if(xs):
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        v = np.concatenate(vals)

        return(x,y,v)

    

results = Parallel(n_jobs=-1, verbose=10, return_as='generator')(
    delayed(process_omega)(om_s)
    for om_s in om_s_grid
)

image = np.zeros((camera_y, camera_x))
for result in results:
    if result is not None:
        x, y, v = result
        np.add.at(image, (y, x), v)   
    

plt.figure(figsize=(10,6))

image_masked = np.where(image>1e6,image,np.nan)

plt.imshow(image_masked, cmap='magma', norm=LogNorm())
plt.title("SPDC Spectrum Simulation (Matrix Optics + Grating + Camera)")
plt.xlabel("Pixel X ")
plt.ylabel("Pixel Y ")
plt.colorbar(label="counts")
#plt.plot(ws,ins)
#plt.savefig("pixelmapmasked.png")

plt.show()