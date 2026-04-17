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


#Experimental Parameters ===================================================================================================================================================

#Crystal parameters

chi_eff = 327 #pm/V 
pp = 170*um #Poling period
T = 27 #degree C
a = np.array([0,0,1]) #Crystal axis
L = 10*mm


#pump parameters
lam_p = 659.58*nm 
om_p = 2*np.pi*c/lam_p
w_p = 43*um
T_I = 75*ps

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
#No imaging done here

#=====================================================================================================================================================================

#Simulation parameters

n_samples = 100000

#Range for signal
# om_s_min = om_p - (2*pi*nu_thz_max)
# om_s_max = om_p - (2*pi*nu_thz_min)


om_s_min = 2*pi*c/(661*1e-9)
om_s_max = 2*pi*c/(660*1e-9)

delta_om_s = (2*np.pi*c) / (2 * 2.2 *L)
N_omega_s = int((om_s_max - om_s_min)/delta_om_s)+1


#Range for signal angle(theta_s)
theta_max = 0.5 * np.pi/180 
delta_theta = (lam_p) / (5 * np.pi * w_p)
N_theta = int(theta_max / delta_theta) + 50



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

    return pdf.sum(axis=0)

#Monte Carlo Loop
def gamma(ks_x,ks_z,om_s,n_s):

    #Sampling peaks
    om_i_center = om_p - om_s
    k_iz_center = k_p - ks_z + (2*pi/pp)

    #sampling omega
    x_omega = sample_sinc(n_s)
    om_i = om_i_center + x_omega[None,:]*(2/T_I)
    k_i = n_o_thz(om_i/(2*pi))*(om_i/c)

    #sampling k_iz
    x_kz = sample_sinc(n_s)
    k_iz = k_iz_center[:,None] + x_kz[None,:]*(2/L)

    #k_perp

    kp_sq = k_i**2 - k_iz**2
    
    if(kp_sq<0).any():
        return np.zeros(len(ks_x))

    k_perp = np.sqrt(k_i**2 - k_iz**2)
    sig_phi = 1/(w_p*k_perp)
    phi_i = np.random.normal(0,sig_phi)

    k_ix = k_perp*np.cos(phi_i)
    k_iy = k_perp*np.sin(phi_i)

    #phase mismatches
    
    delta_kz = x_kz[None,:]*(2/L)
    delta_om = x_omega[None,:]*(2/T_I)
    delta_kx = ks_x[:,None]+k_ix
    delta_ky = k_iy

    #Integrand

    prefactor = chi_eff*om_i*om_s/((n_o_ir(2*pi*c/om_s)**2)*(n_o_thz(om_i/(2*pi))**2))

    sinc_z = np.sinc(delta_kz*L/2)**2

    sinc_t = np.sinc(delta_om*T_I/2)**2
    

    gauss_perp = np.exp(-0.5*((delta_kx**2) + (delta_ky**2))*(w_p**2))

    jacobian = k_i*n_go_thz(om_i/(2*pi))/c

    f = prefactor*sinc_z*sinc_t*gauss_perp*jacobian

    #PDFs for Monte Carlo

    q_om = sinc_pdf(x_omega)[None,:]*(T_I/2)
    q_kz = sinc_pdf(x_kz)[None,:]*(L/2)
    q_phi = norm.pdf(phi_i,0,sig_phi)


    q = q_om*q_kz*q_phi


    w = f/q

    return (w.mean(axis = -1))

sim_len = len(om_s_grid)*len(cos_theta_grid)*len(phi_s_grid)
om_len = len(om_s_grid)
print("Total Iterations = ",sim_len)

#Signal pixel Loop


image = np.zeros((len(om_s_grid),len(cos_theta_grid)))
theta = np.arccos(cos_theta_grid)

for i,om_s in enumerate(om_s_grid):
    
    print((2*pi*c/om_s)*1e9,"nm")
    k = n_o_ir(2*pi*c/om_s)*(om_s/c)
    
    ks_x = k*np.sin(theta)
    ks_z = k*np.cos(theta)

    
    
    Tk = gamma(ks_x,ks_z,om_s,n_samples)
    #Tk = np.ones(len(theta))
    
    intensity = Tk*k*k*(n_go_ir(2*pi*c/om_s)/c)*d_omega*d_cos_theta*2*pi
    image[i] += intensity

x,y = np.meshgrid((2*pi*c/(om_s_grid))*1e9,theta*(180/pi))

    
    


plt.figure(figsize=(10,6))

plt.pcolormesh(x,y,image.T,shading='auto',cmap ='viridis')
plt.title("SPDC Spectrum Simulation (Matrix Optics + Grating + Camera)")
plt.xlabel("Theta (degrees)")
plt.ylabel("Lambda (nm) ")
plt.colorbar(label="counts")
plt.show()
