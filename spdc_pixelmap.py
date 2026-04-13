import numpy as np
import matplotlib.pyplot as plt

#Imaging
f2 = 125.0      # mm
f3 = 400.0      # mm
slit_width = 1 # mm

# Camera
pixel_size = 1  # mm
camera_x = 1000
camera_y = 1000
image = np.zeros((camera_y, camera_x))

#Units
mm = 1e-3
nm = 1e-9
um = 1e-6
thz = 1e12
c = 3e8

pi4 = np.pow(np.pi,4)
pi2 = np.pow(np.pi,2)
pi = np.pi


#Crystal parameters
chi_eff = 327 #pm/V 
pp = 170*um #Poling period
T = 27 #degree C
a = np.array([0,0,1]) #Crystal axis
L = 10*mm

# Grating
lines_per_mm = 1908
d = 1 / lines_per_mm
m = 1
L_grating_camera = 1000  # mm

#pump parameters
lam_p = 659.58*nm 
om_p = 2*np.pi*c/lam_p
w_p = 43*um

#THz range
nu_thz_max = 3.5*thz
nu_thz_min = 0.1*thz

theta_max = 0.01*np.pi

def sinc_gauss(x):
    A1, B1 = 0.97718851, 0.40371072
    A2, B2 = 0.03285665, 0.40371451
    A3, B3 = 0.01294228, 0.00518248
    g = A1 * np.exp(-B1 * x**2) + A2 * np.exp(-B2 * x**2) + A3 * np.exp(-B3 * x**2)
    return(g)
    

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
    np = 2*B*nu + 4*C*nu*nu*nu*(1e-12/2*pi) #differenciating wrt omega (chain rule)
    ng = n_e_thz(nu) + 2*pi*nu*np
    return(ng)

def n_go_thz(nu):
    nu = nu*1e-12
    A = 6.5
    B = 8.2e-2
    C = 6e-3
    np = (2*B*nu + 4*C*nu*nu*nu)*(1e-12/2*pi)
    ng = n_e_thz(nu) + 2*pi*nu*np
    return(ng)

def n_e_ir(lam):
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


def n_o_ir(lam):
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
    nsq = a1 + b1*f + (a2+b2*f)/((lam*lam)-(a3 + b3*f)*(a3+b3*f)) + (a4+b4*f)/(lam*lam-a5*a5) - a6*lam*lam
    return(np.sqrt(nsq))

def n_e_eff_thz(nu,theta):
    nsq = 1/((np.cos(theta)**2/(n_e_thz(nu)**2)+(np.sin(theta)*np.sin(theta)**2)/n_o_thz(nu)**2))
    return(np.sqrt(nsq))

def pixelmap(k_x,k_y,k_z):
    #Image propagation

    theta_x = k_i_x/k_i_z 
    theta_y = k_i_y/k_i_z 
 
    ray_x = np.array([0, theta_x])
    ray_y = np.array([0, theta_y])

    # ---- Crystal -> Slit ----
    ray_x_slit = M_crystal_to_slit @ ray_x
    ray_y_slit = M_crystal_to_slit @ ray_y

    # ---- Slit filtering ----
    if abs(ray_x_slit[0]) > slit_width/2:
        return 0,0

    # ---- Slit -> Lens f3 ----
    ray_x_lens = M_slit_to_lens @ ray_x_slit
    ray_y_lens = M_slit_to_lens @ ray_y_slit

    # ---- Through Lens f3 ----
    ray_x_after_lens = M_lens_f3 @ ray_x_lens
    ray_y_after_lens = M_lens_f3 @ ray_y_lens

    # ---- Lens -> Grating ----
    ray_x_grating = M_lens_to_grating @ ray_x_after_lens
    ray_y_grating = M_lens_to_grating @ ray_y_after_lens

    # ---- Grating diffraction ----
    ray_x_grating[1] = ray_x_grating[1] + (m * lam_s / d)

    # ---- Grating -> Camera ----
    x_cam = ray_x_grating[0] + L_grating_camera * ray_x_grating[1]
    y_cam = ray_y_grating[0] + L_grating_camera * ray_y_grating[1]

    # Convert to pixel
    pix_i = int(x_cam / pixel_size) + camera_x//2
    pix_j = int(y_cam / pixel_size) + camera_y//2
    return pix_i, pix_j


def matrix_free_space(L):
    return np.array([[1, L], [0, 1]])

def matrix_lens(f):
    return np.array([[1, 0], [-1/f, 1]])

# Crystal -> Slit (2f Fourier system)
M_crystal_to_slit = matrix_free_space(f2) @ matrix_lens(f2) @ matrix_free_space(f2)

# Slit -> Lens f3
M_slit_to_lens = matrix_free_space(f3)

# Lens f3
M_lens_f3 = matrix_lens(f3)

# Lens -> Grating
M_lens_to_grating = matrix_free_space(f3)

N_samples = 100000

#Pump phase
k_p = 2*pi*n_o_ir(lam_p)/lam_p

for i in range(N_samples):

    #sampling a uniform cuboidal region
    u1 = np.random.uniform(0,1)
    u2 = np.random.uniform(0,1)
    u3 = np.random.uniform(0,1)

    #Thz idler
    nu_i = nu_thz_min + (nu_thz_max-nu_thz_min)*u1
    om_i = 2*np.pi*nu_i
    
    om_s = om_p - om_i #Energy conservation strictly assumed
    lam_s = 2*pi*c/om_s

    #Ordinary THz + Ordinary Signal
    n_i = n_o_thz(nu_i)
    n_s = n_o_ir(lam_s)

    #theta
    theta_i = np.arccos(1-u2*(1-np.cos(theta_max)))
    
    #phi
    phi_i = 2*pi*u3

    #Propagation direction of THz
    k_i_hat = np.array([np.sin(theta_i)*np.cos(phi_i),np.sin(theta_i)*np.sin(phi_i),np.cos(theta_i)])

    #Progation angle of THz
    theta_n = np.arccos(np.dot(k_i_hat,a))

    #wave vector
    k_i_amp = n_i*om_i/c
    print(2*pi*c/om_s)
    k_i = k_i_hat*k_i_amp

    k_i_x = k_i[0]
    k_i_y = k_i[1]
    k_i_z = k_i[2]


    #sampling for signal transverse wave vectors

    sigma = 0.001
    delta_k_x = np.random.normal(0,sigma)
    delta_k_y = np.random.normal(0,sigma)

    k_s_x = -k_i_x + delta_k_x 
    k_s_y = -k_i_y + delta_k_y 

    #Paraxial approximation for signal wavevector
    
    
    k_s_amp = n_s*om_s/c
    
    k_s_z = np.sqrt(k_s_amp*k_s_amp - k_s_x*k_s_x - k_s_y*k_s_y)
    
    k_s = np.array([k_s_x,k_s_y,k_s_z])

    delta_k_z = k_p - k_s_z - k_i_z + 2*pi/pp


    #Contributed weight
    A_sq = (om_s*om_i/n_s*n_s*n_i*n_i)*sinc_gauss(delta_k_z*L/2)*(n_go_thz(nu_i)/c)*2*pi*(nu_thz_max-nu_thz_min)*(2*pi*(1-np.cos(theta_max)))

    pix_i, pix_j = pixelmap(k_s_x,k_s_y,k_s_z)
    
    if 0 <= pix_i < camera_x and 0 <= pix_j < camera_y:
        image[pix_j, pix_i] += A_sq



plt.figure(figsize=(10,6))
plt.imshow(image, cmap = 'hot')
plt.title("SPDC Spectrum Simulation (Matrix Optics + Grating + Camera)")
plt.xlabel("Pixel X (Wavelength)")
plt.ylabel("Pixel Y (Angle)")
plt.colorbar(label="counts")
plt.show()