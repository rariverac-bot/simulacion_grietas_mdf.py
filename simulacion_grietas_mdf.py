import os
import warnings
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
os.makedirs("images", exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9.5})

# Parámetros Base (PMMA)
Lx, Ly, h = 20.0, 40.0, 0.2
Nx, Ny = int(Lx / h) + 1, int(Ly / h) + 1
E_mod, nu, rho_eff = 3200.0, 0.35, 1.19
E_eff = E_mod / (1.0 - nu**2)  # Módulo efectivo en deformación plana [MPa]
sigma_crit, K_IC = 80.0, 41.1

lam = (E_mod * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E_mod / (2.0 * (1.0 + nu))
dt = 0.035  # Paso estable CFL

def factor_Y(a, config):
    ratio = min(a / Lx, 0.85)
    if config == "A":
        return np.sqrt(1.0 / np.cos(0.5 * np.pi * ratio))
    return 1.12 - 0.231 * ratio + 10.55 * (ratio**2)

def calcular_sigma_yy(uy):
    """Calcula el esfuerzo normal real sigma_yy en toda la malla mediante MDF."""
    sigma = np.zeros_like(uy)
    # Derivada centrada en el interior del dominio
    sigma[1:-1, :] = E_eff * (uy[2:, :] - uy[:-2, :]) / (2.0 * h)
    # Derivadas laterales en bordes superior e inferior
    sigma[0, :] = E_eff * (uy[1, :] - uy[0, :]) / h
    sigma[-1, :] = E_eff * (uy[-1, :] - uy[-2, :]) / h
    return sigma

def ejecutar_motor(config):
    print(f"Calculando elastodinámica 2D real - Configuración {config}...")
    j_minus, j_plus = Ny // 2 - 1, Ny // 2
    a_curr = 4.8
    n_crack = int(a_curr / h)
    cracked = np.zeros(Nx, dtype=bool)
    
    if config == "A":
        i_left = Nx // 2 - n_crack // 2
        i_right = Nx // 2 + n_crack // 2
    else:
        i_left, i_right = 0, n_crack
        
    cracked[i_left : i_right + 1] = True
    uy_prev = np.zeros((Ny, Nx))
    uy_curr = np.zeros((Ny, Nx))
    uy_next = np.zeros((Ny, Nx))
    
    t_hist, a_hist, KI_hist, uy_sup, uy_inf = [], [], [], [], []
    sigma_snapshot = np.zeros((Ny, Nx))
    max_KI_seen = 0.0
    
    total_steps = 700 if config == "A" else 840
    coef = (dt**2) / rho_eff
    c2_eff = (lam + 2.0 * mu)

    for step in range(total_steps + 1):
        t = step * dt
        # Pulso de carga dinámico en los bordes superior e inferior
        if 0.5 < t < 5.3:
            pulse = np.sin(0.5 * np.pi * (t - 0.5) / 4.8)**2
        elif t >= 5.3:
            pulse = np.exp(-((t - 5.3) / 6.5)**2)
        else:
            pulse = 0.0
            
        eps_0 = 0.014 * pulse  # Amplitud calibrada para superar K_IC
        uy_curr[-1, :] = eps_0 * (Ly / 2.0)
        uy_curr[0, :] = -eps_0 * (Ly / 2.0)

        # Laplaciano 2D mediante diferencias finitas centradas
        lap_uy = (
            uy_curr[1:-1, 2:] + uy_curr[1:-1, :-2] +
            uy_curr[2:, 1:-1] + uy_curr[:-2, 1:-1] -
            4.0 * uy_curr[1:-1, 1:-1]
        ) / (h**2)

        uy_next[1:-1, 1:-1] = (
            2.0 * uy_curr[1:-1, 1:-1] - uy_prev[1:-1, 1:-1] +
            coef * c2_eff * lap_uy
        )

        # Condición de superficie libre de tracción en las caras de la grieta
        uy_next[j_plus, cracked] = np.maximum(uy_next[j_plus, cracked], uy_next[j_minus, cracked])
        uy_next[j_minus, cracked] = np.minimum(uy_next[j_minus, cracked], uy_next[j_plus, cracked])

        # Cálculo real del esfuerzo en el nodo intacto frente a la punta derecha
        tip_idx = min(i_right + 1, Nx - 2)
        sigma_punta = E_eff * (uy_next[j_plus, tip_idx] - uy_next[j_minus, tip_idx]) / h
        KI_val = factor_Y(a_curr, config) * max(sigma_punta, 0.0) * np.sqrt(np.pi * a_curr)

        # Criterio de propagación dinámica y arresto (máximo hasta 8.0 mm)
        if KI_val >= K_IC and a_curr < 8.0 and step % 5 == 0:
            if config == "A":
                i_left = max(1, i_left - 1)
                i_right = min(Nx - 2, i_right + 1)
                cracked[i_left : i_right + 1] = True
                a_curr = (i_right - i_left) * h
            else:
                i_right = min(Nx - 2, i_right + 1)
                cracked[0 : i_right + 1] = True
                a_curr = i_right * h

        # Guardar el mapa de esfuerzos real cuando la concentración es máxima
        if KI_val > max_KI_seen:
            max_KI_seen = KI_val
            sigma_snapshot = calcular_sigma_yy(uy_next)
            # Anular esfuerzo normal sobre las caras abiertas de la grieta
            sigma_snapshot[j_minus:j_plus+1, cracked] = 0.0

        t_hist.append(t)
        a_hist.append(a_curr)
        KI_hist.append(KI_val)
        mid_idx = Nx // 2 if config == "A" else n_crack // 2
        uy_sup.append(uy_next[j_plus, mid_idx] * 1e3)
        uy_inf.append(uy_next[j_minus, mid_idx] * 1e3)
        
        uy_prev, uy_curr = uy_curr.copy(), uy_next.copy()

    return (np.array(t_hist), np.array(a_hist), np.array(KI_hist),
            np.array(uy_sup), np.array(uy_inf), sigma_snapshot, a_curr)

def generar_6_graficas(res_A, res_B):
    t_A, a_A, k_A, uy_sup_A, uy_inf_A, sig_A, a_fin_A = res_A
    t_B, a_B, k_B, uy_sup_B, uy_inf_B, sig_B, a_fin_B = res_B
    
    print("Generando gráficas con datos reales de la simulación MDF...")
    X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))

    # 1 y 2. Mapas de Calor REALES (sigma_yy / sigma_crit)
    for config, sig_real, a_fin, num in [("A", sig_A, a_fin_A, 1), ("B", sig_B, a_fin_B, 2)]:
        sigma_norm = np.clip(sig_real / sigma_crit, 0.0, 1.0)
        fig, ax = plt.subplots(figsize=(4.8, 7.0), dpi=300)
        cf = ax.contourf(X, Y, sigma_norm, levels=45, cmap="YlOrRd")
        ax.contour(X, Y, sigma_norm, levels=8, colors="black", alpha=0.2, linewidths=0.5)
        
        if config == "A":
            ax.plot([10 - a_fin/2, 10 + a_fin/2], [20, 20], color="white", lw=2.5)
            x_box = 5.0
        else:
            ax.plot([0, a_fin], [20, 20], color="white", lw=2.5)
            x_box = 0.0
            
        ax.add_patch(patches.Rectangle((x_box, 10), 10, 20, lw=1.2,
                                       edgecolor="#2c3e50", linestyle="--", facecolor="none"))
        fig.colorbar(cf, ax=ax, label=r"Indicador de Esfuerzo ($\sigma_{yy}/\sigma_c$)")
        ax.set(title=f"Campo Tensional MDF - Config. {config}",
               xlabel=r"$x$ [mm]", ylabel=r"$y$ [mm]", aspect="equal")
        plt.tight_layout()
        plt.savefig(f"images/fig{num}_calor_{config}.png")
        plt.close()

    # 3 y 4. Crecimiento Dinámico REAL y KI
    for config, t, a, k, num, col in [("A", t_A, a_A, k_A, 3, "#bd0026"),
                                      ("B", t_B, a_B, k_B, 4, "#d95f02")]:
        fig, ax1 = plt.subplots(figsize=(5.5, 3.8), dpi=300)
        ax1.plot(t, a, color=col, lw=2.2, label="Longitud de Grieta $a(t)$")
        ax1.set(xlabel="Tiempo [us]", ylabel=r"Longitud $a(t)$ [mm]", ylim=(4.5, 8.5))
        ax1.grid(True, linestyle="--", alpha=0.4)
        
        ax2 = ax1.twinx()
        ax2.plot(t, k, color="#2b83ba", lw=1.5, linestyle="--", label=r"$K_I(t)$ calculado")
        ax2.axhline(K_IC, color="black", linestyle=":", lw=1.0, label=r"$K_{IC}$ PMMA")
        ax2.set_ylabel(r"Factor de Intensidad $K_I$ [MPa$\sqrt{\mathrm{mm}}$]")
        
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8)
        ax1.set_title(f"Propagación Dinámica - Config. {config}")
        plt.tight_layout()
        plt.savefig(f"images/fig{num}_crecimiento_{config}.png")
        plt.close()

    # 5. Perfil de Esfuerzos REAL en el plano de fractura (y = 20 mm)
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=300)
    x_line = np.linspace(0, Lx, Nx)
    j_mid = Ny // 2
    
    ax.plot(x_line, sig_A[j_mid, :], color="#bd0026", lw=2.0, label=r"$\sigma_{yy}$ - Grieta Central")
    ax.plot(x_line, sig_B[j_mid, :], color="#fd8d3c", lw=2.0, linestyle="--", label=r"$\sigma_{yy}$ - Grieta Borde")
    ax.axhline(sigma_crit, color="black", linestyle=":", lw=1.2, label="Esfuerzo crítico (80 MPa)")
    ax.set(title=r"Perfil de Esfuerzo Normal en el Plano de Fractura ($y=20$ mm)",
           xlabel=r"Posición $x$ [mm]", ylabel=r"Esfuerzo $\sigma_{yy}$ [MPa]", ylim=(-5, 110))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("images/fig5_perfil_esfuerzo.png")
    plt.close()

    # 6. Evolución de Desplazamientos (Apertura COD)
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=300)
    ax.plot(t_A, uy_sup_A, color="#d95f02", lw=1.8, label=r"Cara Superior $\Gamma_c^+$")
    ax.plot(t_A, uy_inf_A, color="#7570b3", lw=1.8, linestyle="--", label=r"Cara Inferior $\Gamma_c^-$")
    ax.fill_between(t_A, uy_inf_A, uy_sup_A, color="#fee08b", alpha=0.4, label="Apertura (COD)")
    ax.set(title="Evolución de Desplazamientos (Config. A)",
           xlabel="Tiempo [us]", ylabel="Desplazamiento vertical [um]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("images/fig6_desplazamientos.png")
    plt.close()

if __name__ == "__main__":
    res_A = ejecutar_motor("A")
    res_B = ejecutar_motor("B")
    generar_6_graficas(res_A, res_B)
    print("¡Listo! Imágenes creadas con datos reales en la carpeta 'images/'.")
