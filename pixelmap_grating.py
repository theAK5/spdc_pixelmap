import numpy as np
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
f2 = 125.0      # mm
f3 = 400.0      # mm
slit_width = 1.1  # mm

# Camera
pixel_size = 0.0065  # mm
camera_x = 1920
camera_y = 1080
image = np.zeros((camera_y, camera_x))

# Grating
lines_per_mm = 1908
d = 1 / lines_per_mm
m = 1
L_grating_camera = 365  # mm

# Detector
QE = 0.55
exposure_time = 0.2
dark_rate = 20
background_rate = 160
read_noise_sigma = 1

# Photon simulation
n_photons = 150000

# Wavelength distribution (within camera spectral range)
lambda_center = 0.00065958
lambda_sigma = 0.000007

# ---------------- ABCD MATRICES ----------------
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

# ---------------- PHOTON LOOP ----------------
for i in range(n_photons):

    # Generate wavelength
    lam = np.random.normal(lambda_center, lambda_sigma)

    # SPDC angle vs wavelength (simple model)
    theta0 = 0.02
    theta = theta0 + 0.5*(lam - lambda_center)/lambda_center

    phi = np.random.uniform(0, 2*np.pi)
    theta_x = theta * np.cos(phi)
    theta_y = theta * np.sin(phi)

    ray_x = np.array([0, theta_x])
    ray_y = np.array([0, theta_y])

    # ---- Crystal -> Slit ----
    ray_x_slit = M_crystal_to_slit @ ray_x
    ray_y_slit = M_crystal_to_slit @ ray_y

    # ---- Slit filtering ----
    if abs(ray_x_slit[0]) > slit_width/2:
        continue

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
    ray_x_grating[1] = ray_x_grating[1] + (m * lam / d)

    # ---- Grating -> Camera ----
    x_cam = ray_x_grating[0] + L_grating_camera * ray_x_grating[1]
    y_cam = ray_y_grating[0] + L_grating_camera * ray_y_grating[1]

    # Convert to pixel
    pix_i = int(x_cam / pixel_size) + camera_x//2
    pix_j = int(y_cam / pixel_size) + camera_y//2

    if 0 <= pix_i < camera_x and 0 <= pix_j < camera_y:
        image[pix_j, pix_i] += 1

# ---------------- DETECTOR MODEL ----------------
# Quantum efficiency
image = QE * image

# Photon shot noise
image = np.random.poisson(image)

# Dark + background noise
dark_counts = dark_rate * exposure_time
background_counts = background_rate * exposure_time
noise_poisson = np.random.poisson(dark_counts + background_counts, image.shape)
image = image + noise_poisson

# Readout noise
read_noise = np.random.normal(0, read_noise_sigma, image.shape)
image = image + read_noise

# ---------------- DISPLAY ----------------
plt.figure(figsize=(10,6))
plt.imshow(np.log(image + 1), cmap = 'hot')
plt.title("SPDC Spectrum Simulation (Matrix Optics + Grating + Camera)")
plt.xlabel("Pixel X (Wavelength)")
plt.ylabel("Pixel Y (Angle)")
plt.colorbar(label="log(counts)")
plt.show()
