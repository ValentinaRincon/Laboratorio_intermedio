import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
from scipy.optimize import curve_fit

def format_number(num, precision=2):
    if abs(num) < 1e-2 or abs(num) > 1e3: 
        return f'{num:.{precision}e}'
    else:  
        return f'{num:.{precision}f}'

d = 2.014e-10

Aldens = 2.7
Zndens = 7.14

# Nombres de archivos (asegúrate de que incluyan las extensiones correctas si las tienen)
NameB = np.array(['nobarrera.txt'])  # Cambia la extensión si es necesario
NamesA = np.array(['al002.txt','al004.txt','al008.txt','al01.txt'])
NamesB = np.array(['zn0025.txt','zn005.txt','zn0075.txt','zn01.txt'])
thiknA = np.array([0.02, 0.04, 0.08, 0.1])
thiknB = np.array([0.025, 0.05, 0.075, 0.1])

# Path usando os.path.join para compatibilidad entre sistemas
PPath = os.path.join('', 'Rayos X', '')

# Funciones de regresión (se mantienen igual)
def Lineal_reg(x, A):
    return A*x

def lineal_regb(x, a, b):
    return a*x + b

def cuadratic_reg(x, a, b, c):
    return a*x**2 + b*x + c

def cub_reg(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def cuatro_reg(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def cinco_reg(x, a, b, c, d, e, f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

def seis_reg(x, a, b, c, d, e, f, g):
    return a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x + g

def read_data(path, Names, thikn, skip_rows=3, BName=NameB):
    print(f"\nIntentando leer datos desde: {path}")
    print(f"Archivo de referencia: {BName[0]}")
    print(f"Archivos de muestra: {Names}")
    
    # Verificar que el directorio existe
    if not os.path.exists(path):
        raise FileNotFoundError(f"El directorio no existe: {path}")
    
    # Verificar archivo de barrera
    barrier_path = os.path.join(path, BName[0])
    if not os.path.exists(barrier_path):
        available_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        raise FileNotFoundError(
            f"Archivo de barrera no encontrado: {barrier_path}\n"
            f"Archivos disponibles: {available_files}"
        )
    
    # Leer datos de referencia (sin barrera)
    with open(barrier_path, 'r') as fileb:
        lines = islice(fileb, skip_rows, None)
        DataB = [list(map(float, line.replace(',', '.').split())) for line in lines]
    
    angles = {}
    
    # Procesar cada archivo de muestra
    for i in range(len(Names)):
        sample_path = os.path.join(path, Names[i])
        print(f"\nProcesando muestra {i+1}/{len(Names)}: {Names[i]}")
        
        if not os.path.exists(sample_path):
            print(f"¡Advertencia! Archivo no encontrado: {sample_path}")
            continue
            
        try:
            with open(sample_path, 'r') as file:
                lines = islice(file, skip_rows, None)
                for j, line in enumerate(lines):
                    try:
                        D = list(map(float, line.replace(',', '.').split()))
                        angle_key = str(D[0])
                        
                        if angle_key not in angles:
                            angles[angle_key] = np.array([[thikn[i], D[1]/DataB[j][1]]])
                        else:
                            angles[angle_key] = np.append(angles[angle_key], [[thikn[i], D[1]/DataB[j][1]]], axis=0)
                    except Exception as e:
                        print(f"Error procesando línea {j+1} en {Names[i]}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error procesando archivo {Names[i]}: {e}")
            continue
    
    print("\nLectura de datos completada")
    return angles

# Leer datos con manejo de errores
try:
    print("Leyendo datos de Aluminio...")
    Datos1 = read_data(PPath, NamesA, thiknA)
    print("Leyendo datos de Zinc...")
    Datos2 = read_data(PPath, NamesB, thiknB)
except Exception as e:
    print(f"Error al leer los datos: {e}")
    exit()

# Configuración de gráficos
plt.style.use('seaborn')
colores = plt.cm.tab10(np.linspace(0, 1, 10))
marcadores = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X']

# Función para graficar los datos
def plot_data(Datos, material, color_map):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), 
                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
    
    fig.suptitle(f'Intensidad normalizada vs Espesor ({material})', y=0.98)
    
    lambs = np.array([])
    mus = np.array([])
    sigmus = np.array([])
    
    for j, (angle, data) in enumerate(Datos.items()):
        # Calcular longitud de onda
        lamb = 2*d*np.sin(np.radians(float(angle)))*(10**9)
        lambs = np.append(lambs, float(lamb))
        lamb_fmt = format_number(lamb, 4)
        
        # Preparar datos
        x = [data[k][0] for k in range(len(data))]
        y = [np.log(data[k][1]) for k in range(len(data))]
        
        # Ajuste lineal
        try:
            param, cov = curve_fit(Lineal_reg, x, y)
            mu = -param[0]
            sigmu = np.sqrt(cov[0][0])
            
            mus = np.append(mus, mu)
            sigmus = np.append(sigmus, sigmu)
            
            # Formatear para mostrar
            mu_fmt = format_number(mu, 3)
            sigmu_fmt = format_number(sigmu, 1)
            
            # Curva de ajuste
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = Lineal_reg(x_fit, param)
            
            # Residuales
            res = (y - Lineal_reg(x, param))/np.std(y)
            
            # Gráfico principal
            axs[0].scatter(x, y, color=color_map[j], marker=marcadores[j], 
                          label=f'{lamb_fmt} nm; μ = {mu_fmt} ± {sigmu_fmt}')
            axs[0].plot(x_fit, y_fit, color=color_map[j], alpha=0.7)
            
            # Gráfico de residuales
            axs[1].scatter(x, res, color=color_map[j], marker=marcadores[j])
            
        except Exception as e:
            print(f"Error en ajuste para ángulo {angle}: {e}")
            continue
    
    # Configurar ejes y leyenda
    axs[0].set_ylabel(r'$\ln(I/I_0)$')
    axs[0].legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.22, 1))
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    axs[1].axhline(0, color='gray', linestyle='--')
    axs[1].set_xlabel('Espesor (mm)')
    axs[1].set_ylabel('Residuales')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Guardar figura
    save_path = os.path.join(PPath, f'IntensidadEspesor_{material}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {save_path}")
    
    plt.close()
    
    return lambs, mus, sigmus

# Procesar y graficar datos de Aluminio
print("\nProcesando datos de Aluminio...")
lambsAl, muAl, sigmuAl = plot_data(Datos1, 'Al', colores[:len(Datos1)])

# Procesar y graficar datos de Zinc
print("\nProcesando datos de Zinc...")
lambsZn, muZn, sigmuZn = plot_data(Datos2, 'Zn', colores[:len(Datos2)])

# Calcular coeficientes másicos
muAlY = (muAl/Aldens)*10  # Convertir a cm²/g
muZnY = (muZn/Zndens)*10
lambAlX = lambsAl**3
lambZnX = lambsZn**3
sigmuAlN = (1/Aldens)*sigmuAl*10
sigmuZnN = (1/Zndens)*sigmuZn*10

# Gráfico final de coeficientes de absorción
fig, axs = plt.subplots(2, 1, figsize=(12, 8), 
                       gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})

fig.suptitle('Coeficiente de absorción másico vs λ³', y=0.98)

# Ajustes cúbicos
try:
    # Ajuste para Aluminio
    param_Al, cov_Al = curve_fit(cub_reg, lambAlX, muAlY, 
                                p0=[5.12e-6, -1.1912e-1, 3.01e1, 30])
    x_fit_Al = np.linspace(min(lambAlX), max(lambAlX), 100)
    y_fit_Al = cub_reg(x_fit_Al, *param_Al)
    
    # Ajuste para Zinc
    param_Zn, cov_Zn = curve_fit(cub_reg, lambZnX, muZnY, 
                                p0=[5.12e-6, -1.1912e-1, 3.01e1, 30])
    x_fit_Zn = np.linspace(min(lambZnX), max(lambZnX), 100)
    y_fit_Zn = cub_reg(x_fit_Zn, *param_Zn)
    
    # Gráfico principal
    axs[0].errorbar(lambAlX, muAlY, yerr=sigmuAlN, fmt='o', color=colores[0], 
                   label='Al', capsize=3)
    axs[0].plot(x_fit_Al, y_fit_Al, color=colores[0], label='Ajuste Al')
    
    axs[0].errorbar(lambZnX, muZnY, yerr=sigmuZnN, fmt='o', color=colores[1], 
                   label='Zn', capsize=3)
    axs[0].plot(x_fit_Zn, y_fit_Zn, color=colores[1], label='Ajuste Zn')
    
    axs[0].set_ylabel(r'$\mu/\rho$ (cm²/g)')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Residuales
    res_Al = (muAlY - cub_reg(lambAlX, *param_Al))/sigmuAlN
    res_Zn = (muZnY - cub_reg(lambZnX, *param_Zn))/sigmuZnN
    
    axs[1].scatter(lambAlX, res_Al, color=colores[0], marker='x', label='Res. Al')
    axs[1].scatter(lambZnX, res_Zn, color=colores[1], marker='p', label='Res. Zn')
    axs[1].axhline(0, color='gray', linestyle='--')
    axs[1].set_xlabel('λ³ (nm³)')
    axs[1].set_ylabel('Residuales')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Guardar figura
    save_path = os.path.join(PPath, 'Absorcion_vs_LambdaCubica.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico final guardado en: {save_path}")
    
    plt.close()
    
except Exception as e:
    print(f"Error en ajustes cúbicos: {e}")
    plt.close()

print("\nAnálisis completado exitosamente!")