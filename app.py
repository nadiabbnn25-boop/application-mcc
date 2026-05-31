# 1. Les importations
import streamlit as st
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd  
from scipy.integrate import solve_ivp  

# 2. L'UNIQUE configuration de la page
st.set_page_config(page_title="PFE MCC", layout="wide")

# 3. En-tête et Titre

st.markdown("<div style='text-align: center; font-size: 26px; color: #2C3E50; margin-bottom: 10px;'>Université Batna 2<br>Département d'Électromécanique<br><span style='font-size: 22px; color: #E74C3C; font-weight: bold;'>Projet de fin de cycle (Licence 3)</span></div>", unsafe_allow_html=True)

st.title("Étude théorique et simulation des méthodes de commande de la Machine à Courant Continu (MCC) ")

st.markdown("<h3 style='text-align: center; color: #555555; font-weight: normal; margin-top: -15px;'>Développement d’une application web pédagogique interactive</h3>", unsafe_allow_html=True)

# 4. Menu déroulant "À propos"
with st.expander("ℹ️ À propos de ce projet (Crédits)"):
    st.write("Ce simulateur a été développé dans le cadre des travaux pratiques de Licence 3 Électromécanique.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🎓 Réalisé par :**")
        st.markdown("""
        - M.A.A BACHA
        - Y. BAASSOU
        - A. BENAMMAR
        """)
    with col2:
        st.markdown("**👨‍🏫 Encadré par :**")
        st.markdown("- Dr. N. Benbouza")
    st.markdown("**📅 Année universitaire :** 2025/2026")

# 5. Introduction
st.markdown("""
Cette application pédagogique permet d’étudier une Machine à Courant Continu en **régime permanent**
et en **régime dynamique**. Elle permet de modifier les paramètres de la machine et d’observer
les courbes principales.
""")

# =========================================================
# 6. Barre latérale : paramètres
# =========================================================
st.sidebar.header("Paramètres de la MCC")

R = st.sidebar.number_input("Résistance d’induit R (Ω)", value=0.5, step=0.1)
L = st.sidebar.number_input("Inductance d’induit L (H)", value=0.01, step=0.001, format="%.4f")
K = st.sidebar.number_input("Constante K (Nm/A ou V.s/rad)", value=0.1, step=0.01)
J = st.sidebar.number_input("Moment d’inertie J (kg.m²)", value=0.01, step=0.001, format="%.4f")
f = st.sidebar.number_input("Frottement visqueux f (Nm.s/rad)", value=0.001, step=0.0005, format="%.4f")

U_nom = st.sidebar.number_input("Tension nominale U (V)", value=24.0, step=1.0)
Cr = st.sidebar.number_input("Couple résistant Cr (N.m)", value=0.1, step=0.05)
omega_ref = st.sidebar.number_input("Vitesse de référence Ωref (rad/s)", value=150.0, step=10.0)

st.sidebar.header("Régulateurs")
Kp = st.sidebar.number_input("Gain proportionnel Kp", value=1.0, step=0.1)
Ki = st.sidebar.number_input("Gain intégral Ki", value=10.0, step=1.0)

Umax = st.sidebar.number_input("Limite de tension Umax (V)", value=24.0, step=1.0)
t_final = st.sidebar.number_input("Durée de simulation (s)", value=4.0, step=0.5)

# =========================================================
# Fonctions mathématiques (UNIQUES ET CORRIGÉES)
# =========================================================
def limiter(x, xmin, xmax):
    return np.minimum(np.maximum(x, xmin), xmax)

def modele_boucle_ouverte(t, y):
    I, omega = y
    dI = (U_nom - R * I - K * omega) / L
    domega = (K * I - Cr - f * omega) / J
    return [dI, domega]

def modele_P(t, y):
    I, omega = y
    e = omega_ref - omega
    Ucmd = limiter(Kp * e, -Umax, Umax)
    dI = (Ucmd - R * I - K * omega) / L
    domega = (K * I - Cr - f * omega) / J
    return [dI, domega]

def modele_PI(t, y):
    I, omega, integ = y
    e = omega_ref - omega
    Ucmd = limiter(Kp * e + Ki * integ, -Umax, Umax)
    dI = (Ucmd - R * I - K * omega) / L
    domega = (K * I - Cr - f * omega) / J
    dinteg = e
    return [dI, domega, dinteg]

def calcul_metriques(t, omega, ref):
    omega_max = np.max(omega)
    dep = max(0, (omega_max - ref) / ref * 100) if ref != 0 else 0
    err_stat = abs(ref - omega[-1]) / ref * 100 if ref != 0 else 0
    seuil = 0.95 * ref
    indices = np.where(omega >= seuil)[0]
    tr95 = t[indices[0]] if len(indices) > 0 else np.nan
    return omega_max, dep, err_stat, tr95

def afficher_figure(fig):
    st.pyplot(fig)
    plt.close(fig)

# Création des onglets
tab1, tab2, tab3 = st.tabs(["1. Régime permanent", "2. Régime dynamique", "3. Conclusion"])

# =========================================================
# Onglet 1 : régime permanent
# =========================================================
with tab1:
    st.header("Étude en régime permanent")
    I = np.linspace(0, 40, 300)
    U_values = np.linspace(0, 30, 300)
    phi_vals = [1.0, 0.8, 0.6]
    Rs_vals = [0.0, 0.2, 0.5]

    col1, col2 = st.columns(2)
    with col1:
        C = K * I
        fig, ax = plt.subplots()
        ax.plot(I, C, linewidth=2)
        ax.set_xlabel("Courant I (A)")
        ax.set_ylabel("Couple Cem (N.m)")
        ax.set_title("Couple en fonction du courant : Cem = f(I)")
        ax.grid(True)
        afficher_figure(fig)
        st.markdown("Le couple électromagnétique est proportionnel au courant d’induit.")

    with col2:
        omega_I = (U_nom - R * I) / K
        fig, ax = plt.subplots()
        ax.plot(I, omega_I, linewidth=2)
        ax.set_xlabel("Courant I (A)")
        ax.set_ylabel("Vitesse Ω (rad/s)")
        ax.set_title("Vitesse en fonction du courant : Ω = f(I)")
        ax.grid(True)
        afficher_figure(fig)
        st.markdown("Lorsque le courant augmente, la chute de tension interne augmente et la vitesse diminue.")

    col3, col4 = st.columns(2)
    with col3:
        I_const = 10
        omega_U = (U_values - R * I_const) / K
        fig, ax = plt.subplots()
        ax.plot(U_values, omega_U, linewidth=2)
        ax.set_xlabel("Tension U (V)")
        ax.set_ylabel("Vitesse Ω (rad/s)")
        ax.set_title("Vitesse en fonction de la tension : Ω = f(U)")
        ax.grid(True)
        afficher_figure(fig)
        st.markdown("La vitesse augmente avec la tension d’induit.")

    with col4:
        C = K * I
        omega_C = (U_nom - R * I) / K
        fig, ax = plt.subplots()
        ax.plot(C, omega_C, linewidth=2)
        ax.set_xlabel("Couple Cem (N.m)")
        ax.set_ylabel("Vitesse Ω (rad/s)")
        ax.set_title("Caractéristique mécanique : Ω = f(Cem)")
        ax.grid(True)
        afficher_figure(fig)
        st.markdown("La vitesse diminue lorsque le couple demandé augmente.")

    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots()
        for phi in phi_vals:
            Kphi = K * phi
            omega_phi = (U_nom - R * I) / Kphi
            ax.plot(I, omega_phi, linewidth=2, label=f"φ = {phi}")
        ax.set_xlabel("Courant I (A)")
        ax.set_ylabel("Vitesse Ω (rad/s)")
        ax.set_title("Influence du flux sur la vitesse")
        ax.legend()
        ax.grid(True)
        afficher_figure(fig)
        st.markdown("Lorsque le flux diminue, la vitesse augmente, mais le couple disponible diminue.")

    with col6:
        fig, ax = plt.subplots()
        for Rs in Rs_vals:
            omega_Rs = (U_nom - (R + Rs) * I) / K
            ax.plot(I, omega_Rs, linewidth=2, label=f"Rs = {Rs} Ω")
        ax.set_xlabel("Courant I (A)")
        ax.set_ylabel("Vitesse Ω (rad/s)")
        ax.set_title("Influence de la résistance série")
        ax.legend()
        ax.grid(True)
        afficher_figure(fig)
        st.markdown("Lorsque la résistance série augmente, la vitesse diminue à cause de la chute de tension.")

# =========================================================
# Onglet 2 : régime dynamique
# =========================================================
with tab2:
    st.header("Étude dynamique")
    
    # ─────────────────────────────────────────
    # CORRECTION : Importation locale forcée de solve_ivp
    # ─────────────────────────────────────────
    from scipy.integrate import solve_ivp

    t_eval = np.linspace(0, t_final, 1000)

    # Appel corrigé de la boucle ouverte
    sol_bo = solve_ivp(
        modele_boucle_ouverte,
        [0, t_final], [0, 0], t_eval=t_eval, rtol=1e-6, atol=1e-8
    )

    sol_p = solve_ivp(
        modele_P,
        [0, t_final], [0, 0], t_eval=t_eval, rtol=1e-6, atol=1e-8
    )

    sol_pi = solve_ivp(
        modele_PI,
        [0, t_final], [0, 0, 0], t_eval=t_eval, rtol=1e-6, atol=1e-8
    )

    t = t_eval
    I_bo, w_bo = sol_bo.y[0], sol_bo.y[1]
    I_p, w_p = sol_p.y[0], sol_p.y[1]
    I_pi, w_pi = sol_pi.y[0], sol_pi.y[1]

    Cem_bo = K * I_bo
    Cem_p = K * I_p
    Cem_pi = K * I_pi

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(t, w_bo, linewidth=2, label="Boucle ouverte")
        ax.plot(t, w_p, linewidth=2, label="Commande P")
        ax.plot(t, w_pi, linewidth=2, label="Commande PI")
        ax.axhline(omega_ref, linestyle="--", linewidth=1.5, label="Consigne")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Vitesse Ω (rad/s)")
        ax.set_title("Comparaison des vitesses")
        ax.legend()
        ax.grid(True)
        afficher_figure(fig)
        st.markdown("La boucle fermée améliore le suivi de la consigne par rapport à la boucle ouverte.")

    with col2:
        fig, ax = plt.subplots()
        ax.plot(t, I_bo, linewidth=2, label="Boucle ouverte")
        ax.plot(t, I_p, linewidth=2, label="Commande P")
        ax.plot(t, I_pi, linewidth=2, label="Commande PI")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Courant I (A)")
        ax.set_title("Courant d’induit")
        ax.legend()
        ax.grid(True)
        afficher_figure(fig)
        st.markdown("Le courant est élevé au démarrage puis évolue selon la commande appliquée.")

    fig, ax = plt.subplots()
    ax.plot(t, Cem_bo, linewidth=2, label="Boucle ouverte")
    ax.plot(t, Cem_p, linewidth=2, label="Commande P")
    ax.plot(t, Cem_pi, linewidth=2, label="Commande PI")
    ax.axhline(Cr, linestyle="--", linewidth=1.5, label="Cr")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Couple (N.m)")
    ax.set_title("Couple électromagnétique et couple résistant")
    ax.legend()
    ax.grid(True)
    afficher_figure(fig)

    omega_max, dep, err_stat, tr95 = calcul_metriques(t, w_pi, omega_ref)
    st.subheader("Métriques de performance pour la commande PI")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ω max", f"{omega_max:.2f} rad/s")
    c2.metric("Dépassement", f"{dep:.2f} %")
    c3.metric("Erreur statique", f"{err_stat:.2f} %")
    if np.isnan(tr95):
        c4.metric("Temps à 95 %", "Non atteint")
    else:
        c4.metric("Temps à 95 %", f"{tr95:.3f} s")
# =========================================================
# Onglet 3 : Conclusion et Perspectives
# =========================================================
with tab3:
    st.header("Conclusion et Perspectives")
    st.markdown("""
    L’étude en régime permanent de la Machine à Courant Continu a permis de mettre en évidence l’influence 
    directe de la tension d’induit, du flux d’excitation et du couple résistant sur la vitesse de rotation.

    L’analyse en régime dynamique démontre le comportement transitoire du moteur. Si la commande en boucle 
    ouverte s'avère insuffisante face aux perturbations, l'intégration d'un correcteur Proportionnel (P) 
    améliore la réactivité. L'ajout de l'action intégrale (Correcteur PI) est quant à lui indispensable 
    pour annuler l'erreur statique et garantir un suivi parfait de la consigne.

    **Perspectives pédagogiques :**
    La conception de cet outil web interactif, présentée avec succès lors de la soutenance des posters, valide l'intérêt majeur des simulations numériques pour la modernisation de l'enseignement. 
    À l'instar des travaux menés en parallèle sur le transformateur de puissance, ce simulateur a vocation à être intégré de manière pérenne comme support de travaux pratiques. Il permettra aux futures promotions de Licence 3 d'assimiler les stratégies de commande complexes grâce à une approche visuelle et interactive.
    """)
