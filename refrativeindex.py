import numpy as np
import matplotlib.pyplot as plt

mm = 1e-3
nm = 1e-9
um = 1e-6
thz = 1e12
ps = 1e-12
c = 3e8

pi4 = np.pow(np.pi,4)
pi2 = np.pow(np.pi,2)
pi = np.pi


# Temperature (set this as needed)
T = 25  # Celsius

thz = 1e12

lam_p = 659.58*nm 
om_p = 2*np.pi*c/lam_p

nu_thz_max = 3.5*thz
nu_thz_min = 0.1*thz

om_s_min = om_p - (2*pi*nu_thz_max)
om_s_max = om_p - (2*pi*nu_thz_min)

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

# Wavelength range around 600 nm

om_s = np.linspace(om_s_min, om_s_max, 500)  # nm

n_vals = n_o_ir(2*pi*c/om_s)

test = 2853946818074781.5

print(n_o_ir(2*pi*c/test))

# Plot
plt.figure()
plt.plot(om_s, n_vals)

plt.xlabel("Angular Freq(Hz)")
plt.ylabel("Refractive Index n")
plt.title("Refractive Index vs Wavelength")
plt.grid()
plt.show()