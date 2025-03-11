
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Definir la función de ajuste exponencial A * exp(-t / T) + B (considerando fondo B)
def exponential_fit(x, A, T, B):
    return A * np.exp(-x / T) + B

#Definir la función de ajuste exponencial sin B
def exponential_fit_no_B(x, A, T):
    return A * np.exp(-x / T)

# Cargar datos
def read_data(path):
    with open(path, 'r') as file:
        data = [int(line.split(' ')[0]) for line in file]
    return data
  
datos = read_data("Física de muones\datos.data")

filtered_data = [d for d in datos if 80 <= d <= 20000]

# Constantes físicas
hbar_GeV_s = 6.582119569e-25  # hbar en GeV·s
m_mu_GeV = 0.1056583755  # masa del muón en GeV/c^2
hbar_c_GeV2 = 1.973269804e-14  # hbar * c en GeV·cm

# Crear histogramas con diferentes números de bins
num_bins_list = [65]  # Diferentes bin sizes para analizar su impacto en tau
tau_values = []  # Guardar los valores de tau obtenidos
B_values = []  # Guardar los valores de B obtenidos
A_values = []  # Guardar los valores de A obtenidos

for num_bins in num_bins_list:
    bins = np.linspace(80, 20000, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_counts, _ = np.histogram(filtered_data, bins=bins)

    # Filtrar bins con datos válidos para el ajuste con B
    mask = bin_counts > 0
    x_data = bin_centers[mask]
    y_data = bin_counts[mask]

    # Ajustar la curva exponencial con fondo
    initial_guess = [max(y_data), 2000, min(y_data)]  # A, T, B
    params, cov_matrix = curve_fit(exponential_fit, x_data, y_data, p0=initial_guess)
    A_fit, tau_fit, B_fit = params
    B_values.append((B_fit, np.sqrt(cov_matrix[2, 2])))
    A_values.append((A_fit, np.sqrt(cov_matrix[0, 0])))

    # Calcular incertidumbre en los parámetros
    errors = np.sqrt(np.diag(cov_matrix))
    tau_error = errors[1]
    B_error = errors[2]

    if len(errors) == 3:
        B_error = errors[2]
        B_values.append((params[2], B_error))
    else:
        print("Advertencia: No se pudo calcular B correctamente.")
    
    # Guardar el valor de tau obtenido
    tau_values.append((params[1], tau_error))

    # Crear figura con dos subgráficos
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Graficar el ajuste con barras de error
    axs[0].errorbar(x_data, y_data, yerr=np.sqrt(y_data), fmt='o', label="Datos", color="blue", ecolor='gray', capsize=3)
    axs[0].plot(x_data, exponential_fit(x_data, *params), label=f"Ajuste: τ = {params[1]:.2f} ± {tau_error:.2f} ns", color="red")
    axs[0].set_xlabel("Tiempo de decaimiento (ns)")
    axs[0].set_ylabel("Frecuencia")
    axs[0].set_title(f"Ajuste exponencial con {num_bins} bins")
    axs[0].legend()

    # Calcular residuales normalizados
    residuals = (y_data - exponential_fit(x_data, *params)) / np.sqrt(y_data)
    
    # Graficar residuales normalizados
    axs[1].scatter(x_data, residuals, color="purple", label="Residuales Normalizados")
    axs[1].axhline(0, color="black", linestyle="--")
    axs[1].set_xlabel("Tiempo de decaimiento (ns)")
    axs[1].set_ylabel("Residuales normalizados")
    axs[1].set_title(f"Residuales normalizados para {num_bins} bins")
    axs[1].legend()
    
    plt.tight_layout()

     # Ajuste sin B
    initial_guess_no_B = [max(y_data), 2000]  # A, T
    params_no_B, cov_matrix_no_B = curve_fit(exponential_fit_no_B, x_data, y_data, p0=initial_guess_no_B)
    A_fit, tau_fit, B_fit = params
    tau_no_B_error = np.sqrt(np.diag(cov_matrix_no_B))[1]
    
    # Crear figura con ajuste sin B

    y_data_no_B = y_data - B_fit  # Restar B a los datos
    params_no_B, cov_matrix_no_B = curve_fit(exponential_fit_no_B, x_data, y_data_no_B, p0=[A_fit, tau_fit])
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    axs[0].errorbar(x_data, y_data, yerr=np.sqrt(y_data), fmt='o', label="Datos", color="#c411da", ecolor='#26baec', capsize=3)
    axs[0].plot(x_data, exponential_fit_no_B(x_data, *params_no_B), label=f"Ajuste sin B: τ = {params_no_B[1]:.2f} ± {tau_no_B_error:.2f} ns", color="green")
    axs[0].set_xlabel("Tiempo de decaimiento (ns)")
    axs[0].set_ylabel("Frecuencia")
    axs[0].set_title(f"Ajuste exponencial sin B con {num_bins} bins")
    axs[0].legend()
    
    residuals_no_B = (y_data - exponential_fit_no_B(x_data, *params_no_B)) / np.sqrt(y_data)
    
    axs[1].scatter(x_data, residuals_no_B, color="orange", label="Residuales Normalizados")
    axs[1].axhline(0, color="black", linestyle="--")
    axs[1].set_xlabel("Tiempo de decaimiento (ns)")
    axs[1].set_ylabel("Residuales normalizados")
    axs[1].set_title(f"Residuales normalizados sin B para {num_bins} bins")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

# Calcular G_F / (hbar * c)^3 con el valor de tau obtenido
tau_mu_s = tau_values[-1][0] * 1e-9  # Convertir ns a s
tau_mu_error_s = tau_values[-1][1] * 1e-9  # Convertir ns a s

numerador = 192 * np.pi**3 * hbar_GeV_s
denominador = (m_mu_GeV)**5 * tau_mu_s
G_F_over_hbar_c3 = np.sqrt(numerador / denominador)

# Propagación de incertidumbre
G_F_over_hbar_c3_error = 0.5 * G_F_over_hbar_c3 * (tau_mu_error_s / tau_mu_s)

print(f"G_F / (hbar * c)^3 = ({G_F_over_hbar_c3} ± {G_F_over_hbar_c3_error}) GeV^-2")

# Muones en el ambiente
print(f"Muones en el ambiente = {B_values[-1][0]} ± {B_values[-1][1]}")

# Calcular número total de muones
A_final = A_values[-1][0]
B_final = B_values[-1][0]
N_mu_total = A_final + B_final

# Calcular incertidumbre en N_mu_total
A_error = A_values[-1][1]
B_error = B_values[-1][1]
N_mu_total_error = np.sqrt(A_error**2 + B_error**2)

# Calcular la fracción de antimuones
fraction_antimuons = B_final / N_mu_total
fraction_antimuons_error = fraction_antimuons * np.sqrt((B_error / B_final) ** 2 + (N_mu_total_error / N_mu_total) ** 2)
print(f"Número total de muones y antimuones = {N_mu_total} ± {N_mu_total_error}")
print(f"Fracción de antimuones = {fraction_antimuons:.4f} ± {fraction_antimuons_error:.4f}")
