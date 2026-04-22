'''
Developed by Aparna R (MSc Student 2026 TeraQuaNTA Lab )
'''

import numpy as np
import matplotlib.pyplot as plt

# Units
nm = 1e-9
mm = 1e-3
um = 1e-6
thz = 1e12
c = 3e8

# Pump
lam_p = 659.58 * nm
w_p = (2*np.pi*c)/lam_p

#Idler wavelength range
lam_i1 = 85.714*um
lam_i2 = 3*mm

# poling period range
pp_1 = 50*um
pp_2 = 200*um

# Meshgrid
lam_i = np.linspace(lam_i1, lam_i2, 100)
pp = np.linspace(pp_1, pp_2, 80)
LAM_I, PP = np.meshgrid(lam_i, pp)

# Computing Idler frequency
nu_i = c / LAM_I
w_i = 2*np.pi * nu_i

#Signal
w_s = w_p - w_i
lam_s = (2*np.pi*c)/w_s

T = 25 #in celcius

# for 5% doped MgO in PPLN

# Refractive index functions for ordinary and extraordinary cases
#for optical pump and signal wavelengths
#Using Sellmeier equations

# n_e : extraordinary refractive index
def n_e(lam_1):
  lam_1 = lam_1*1e6
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
  nsq_1 = a1 + b1*f + (a2 + b2*f)/((lam_1**2)-(a3 + b3*f)**2) + (a4 + b4*f)/(lam_1**2 - a5**2) - a6*lam_1**2
  return(np.sqrt(nsq_1))

# n_o : ordinary refractive index
def n_o(lam_2):
  lam_2 = lam_2*1e6
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
  nsq_2 = a1 + b1*f + (a2 + b2*f)/((lam_2**2)-(a3 + b3*f)**2) + (a4 + b4*f)/((lam_2**2)-a5**2) - a6*lam_2**2
  return(np.sqrt(nsq_2))

#refractive index functions for idler in terahertz frequency

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

# Type I phase matching condition
# pump - extraordinary, signal & idler - ordinary
k_pe = (n_o(lam_p)*w_p)/c
k_so = (n_o(lam_s)*w_s)/c
k_io = (n_o_thz(nu_i)*w_i)/c
k_pp = (2*np.pi*1)/PP
#equation for delta_k [wave vector mismatch]
Z = k_pe - k_so - k_io + k_pp


#Type II phase matching condition
#Case 1

cs = plt.contour((w_i*1e-12)/(2*np.pi), PP/um, Z, levels=50)
plt.clabel(cs, inline=True, fontsize=8)
plt.xlabel("Idler frequency")
plt.ylabel("Poling Period")
plt.colorbar()
plt.show()


# highlight phase matching

plt.show()