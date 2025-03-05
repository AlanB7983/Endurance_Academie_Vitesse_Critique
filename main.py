# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:41:28 2025

@author: alanb
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

import pandas as pd
import os
from scipy.stats import linregress
from plotly.subplots import make_subplots
from PIL import Image
    
# Importations nécessaires pour la fonction de création de pdf

import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.pdfgen import canvas
from io import BytesIO
import pandas as pd
from PIL import Image as PILImage

from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os

from fontTools.ttLib import TTCollection



# =============================================================================
# FONCTIONS
# =============================================================================

# Fonction pour calculer la vitesse critique avec le modèle hyperbolique
def calculate_critical_speed(distances, times):
    times = np.array(times)
    speeds = np.array(distances) / times  # Vitesse moyenne pour chaque test
    inverse_times = 1 / times  # Transformation en 1/t

    # Régression linéaire : V = CS + D'/t
    slope, intercept, _, _, _ = linregress(inverse_times, speeds)

    CS = intercept  # Ordonnée à l'origine = vitesse critique
    D_prime_0 = slope  # Pente = D'

    return CS, D_prime_0, speeds

# Convertisseurs
def speed_to_pace(speed_m_s):
    """Convertit une vitesse en m/s en allure min/km."""
    pace_sec_km = 1000 / speed_m_s if speed_m_s > 0 else float("inf") # Temps en secondes pour 1 km
    min_per_km = int(pace_sec_km // 60)
    sec_per_km = int(pace_sec_km % 60)
    return f"{min_per_km}:{sec_per_km:02d} min/km"

def speed_m_s_to_kmh(speed_m_s):
    """Convertit une vitesse en m/s en km/h."""
    return speed_m_s * 3.6



# Fonction pour calculer l'évolution de D' en fonction du temps
def compute_D_prime_evolution(CS, D_prime_0, session, dt=1) :
    """
    Simule l'évolution de D' au fil du temps en fonction des blocs d'effort.

    CS : Vitesse critique (m/s)
    D_prime_0 : Capacité anaérobie initiale (m)
    session : Liste des blocs [(durée en s, %CS)]
    k : Facteur de récupération de D' (entre 0.1 et 0.3)
    dt : Pas de temps (s)

    Retourne : 
    - time : Temps écoulé en secondes
    - D_prime : Valeur de D' au fil du temps
    - velocities : Vitesse appliquée à chaque instant
    """
    
    total_duration = sum([bloc[0] for bloc in session]) # + (len(session) - 1) * dt  
    time = np.arange(0, total_duration + dt, dt) # Liste numpy qui va de 0 à total_duration avec un espace entre les valeurs de dt
    D_prime = np.full_like(time, D_prime_0, dtype=float) # Liste numpy calquée sur "time" remplie de la valeur D_prime 
    velocities = np.zeros_like(time, dtype=float) # Liste numpy calquée sur "time" remplie de 0

    t_index = 0
    tau = 450 # Constante de temps
    
    for i, (duration, percent_CS) in enumerate(session):
        V = CS * (percent_CS / 100)  
        
        for _ in range(int(duration / dt)):
            if t_index >= len(time):  
                break

            velocities[t_index] = V  
            
            if V > CS :
                D_prime[t_index] = max(0, D_prime[t_index-1] - (V - CS) * dt)  
            else :
                #D_prime[t_index] = min(D_prime_0, D_prime[t_index-1] + 0.1*(CS - V) * dt)  
                D_prime[t_index] = D_prime_0 - (D_prime_0 - D_prime[t_index - 1]) * np.exp(-1 / tau)

            
            t_index += 1
            
    # la dernière valeur est prise égale à la dernière valeur calculée, sinon c'est Dprime0 qui est affiché
    D_prime[-1] = D_prime[-2]

        # if i < len(session) - 1:  
        #     last_D_prime = D_prime[t_index - 1]  
        #     for _ in range(int(dt)):
        #         if t_index >= len(time):
        #             break
        #         velocities[t_index] = 0  
        #         D_prime[t_index] = last_D_prime  
        #         t_index += 1

    return time, D_prime, velocities


def afficher_blocs(L):
    """
    Parcourt la liste L et détecte les répétitions de couples d'éléments successifs.
    Affiche les répétitions sous forme condensée avec Streamlit.
    """
    i = 0
    while i < len(L):
        st.write(L[i])  # Toujours afficher le premier élément d'un groupe
        i += 1
        
        # Vérification des répétitions
        if i < len(L) - 1:
            bloc1, bloc2 = L[i], L[i + 1]
            nb_rep = 1
            
            while i + 2 < len(L) and L[i + 2] == bloc1 and L[i + 3] == bloc2:
                nb_rep += 1
                i += 2
            
            if nb_rep > 1:
                st.write(f"Répétition {nb_rep}x :")
                # st.write(f"     {bloc1}")
                # st.write(f"     {bloc2}")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{bloc1}")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{bloc2}")
                i += 2  # Avancer après la répétition détectée
            else:
                st.write(bloc1)
                i += 1
  
                
  
    
def create_table(data, col_widths):
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),  # Couleur de la première ligne
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Texte en blanc pour contraste
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Athelas'),
        ('FONTNAME', (0, 1), (-1, -1), 'Athelas'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        #('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BOX', (0, 0), (-1, -1), 0, colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    return table
  # HexColor("#aa3d00")

def header_footer(canvas, doc):
    # Dimensions du document
    page_width, page_height = doc.pagesize  # (595, 842) pour du A4 en points
  
    # Récupérer la date du jour
    date = datetime.datetime.now()
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    full_date = str(day) + "/" + str(month) + "/" + str(year)
    
    # Dessiner l'en-tête
    canvas.saveState()
    canvas.setFont('Athelas', 10)
    canvas.drawString(6.95 * inch, 10.5 * inch, full_date)
    
    # Ajouter le logo en haut à gauche 
    logo_path_full_name_noir = "Pictures/__LOGO_EA_NOIR.png"  # Le chemin vers votre logo
    logo_path_badge_orange = "Pictures/__LOGO_EABADGE_ORANGE.png" # Le chemin vers votre logo
    canvas.drawImage(logo_path_badge_orange, 0.5 * inch, 9.9 * inch, width=1.0 * inch, height=1.0 * inch, preserveAspectRatio=True)
    
    # Dessiner le pied de page avec le numéro de page
    canvas.drawString(7.2 * inch, 0.75 * inch, f"Page {doc.page}")
    #canvas.drawString(3 * inch, 0.75 * inch, "Document powered by Endurance Académie")
    logo_width = 1.1 * inch
    center_x = (page_width - logo_width) / 2  # Centrage horizontal
    canvas.drawImage(logo_path_full_name_noir, center_x, 0.25 * inch, width=logo_width, height=1.1 * inch, preserveAspectRatio=True)

    canvas.restoreState()
    
    
    
def create_pdf_template(df_test, CS_pace, CS_kmh, D_prime_0, CS_graph_path) :
    buffer = BytesIO()
    
    
    # Configuration du document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    page_width = letter[0] # Largeur de la page

    # Récupérer la feuille de style par défaut
    styles = getSampleStyleSheet()
    
    # Créer un style personnalisé pour les titres
    title_style = styles['Title']
    title_style.fontName = 'StretchPro'  # Appliquer la police StretchPro
    title_style.fontSize = 16  # Définir la taille de police souhaitée
    
    # Exemple pour les sous-titres
    subtitle2_style = styles['Heading2']
    subtitle2_style.fontName = 'StretchPro'  # Appliquer la police StretchPro
    subtitle2_style.fontSize = 11  # Définir la taille de police souhaitée

    subtitle3_style = styles['Heading3']
    subtitle3_style.fontName = 'StretchPro'  # Appliquer la police StretchPro
    subtitle3_style.fontSize = 12  # Définir la taille de police souhaitée
    
    
    normal_style = ParagraphStyle(
      'BodyTextCustom',
      parent=styles['BodyText'],
      fontName="Athelas",  # Utiliser la police Athelas
      fontSize = 9,  # Set the desired font size
      alignment=TA_JUSTIFY  # Justification du texte
    )
    
    # Legend
    legend_style = ParagraphStyle(
      'Legend',
      parent=styles['Normal'],
      fontName="Athelas-Italic",  # Utiliser la police Athelas
      fontSize=8,
      textColor=colors.grey,
      spaceBefore=6,
      alignment=1, # Centrer la légende
    )

    
    
    
    # Ajout du titre
    title = Paragraph("BILAN DU TEST DE VITESSE CRITIQUE", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))  # Ajouter un espace après le titre

    
    # =============================================================================
    #     PROFIL DE L'ATHLETE
    # =============================================================================
    
    subtitle_1 = Paragraph("Résultats des tests", subtitle2_style)
    elements.append(subtitle_1)

    elements.append(Spacer(1, 6))  # Ajouter un espace après le texte

    # TABLEAU DES DONNEES DE L'ATHLETE

    # nouveaux_noms_athlete_profile = [
    #     Paragraph('Distance [m]', normal_style),
    #     Paragraph('Temps [s]', normal_style),
    # ]
    
    # df_athlete_profile.columns = nouveaux_noms_athlete_profile
  
    # Convertir le DataFrame en une liste de listes
    L_test = [df_test.columns.tolist()] + df_test.values.tolist()    
    
    col_widths = [120, 120]
    
    #table_athlete_profile = Table(L_athlete_profile, colWidths=col_widths)
    table_test = create_table(L_test, col_widths)

    
    elements.append(table_test)
    legend = Paragraph("Tableau 1 : Résumé des valeurs de test utilisées pour déterminer la vitesse critique", legend_style)
    elements.append(legend)
    elements.append(Spacer(1, 12))  # Ajouter un espace après le texte
    


    # =============================================================================
    #     RESULTATS - COURBE VITESSE CRITIQUE
    # =============================================================================
    # Saut de page
    #elements.append(PageBreak())
    elements.append(Spacer(1, 12))  # Ajouter un espace après le texte
    subtitle_2 = Paragraph("Courbe de vitesse critique", subtitle2_style)
    elements.append(subtitle_2)
    #elements.append(Spacer(1, 6))  # Ajouter un espace entre les graphes
  
    # On affiche le graphe d'évolution de l'effort dans la liaison boulonnée en fonction de l'effort extérieur dans le cas où la thermique est prise en compte
    # scale_factor = 0.8  # Réduction de 5%

    # Ajustement de la largeur et de la hauteur du graphe 
    graph_width = (page_width - 2 * inch) # * scale_factor
    CS_graph = Image(CS_graph_path)
    CS_graph.drawHeight = graph_width * CS_graph.drawHeight / CS_graph.drawWidth
    CS_graph.drawWidth = graph_width

    elements.append(CS_graph)
    
    # Centrage du graphe via un tableau
    # table = Table([[CS_graph]], colWidths=[page_width - 2 * inch])
    # table.setStyle(TableStyle([
    #     ('ALIGN', (0, 0), (-1, -1), 'CENTER')  # Centrer le graphe
    # ]))
    
    # elements.append(table)  # Ajout du graphe centré

    legend = Paragraph("Figure 1 : Courbe de vitesse critique", legend_style)
    elements.append(legend)
    
    elements.append(Spacer(1, 12))  # Ajouter un espace entre les graphes
    elements.append(Spacer(1, 12))  # Ajouter un espace après le texte


    
    # Génération du PDF
    doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)

    
    buffer.seek(0)
    return buffer



# =============================================================================
# PREREQUIS POUR LA CREATION DU PDF
# =============================================================================

# Enregistrer la police sous le nom "Athelas"
pdfmetrics.registerFont(TTFont("Athelas", "Polices/Athelas-Regular-01.ttf"))
pdfmetrics.registerFont(TTFont("Athelas-Italic", "Polices/Athelas-Italic-02.ttf"))

# Enregistrer ta police StretchPro
pdfmetrics.registerFont(TTFont('StretchPro', 'Polices/StretchPro.ttf'))




# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

# LOGO
######

col1_logo, col2_logo, col3_logo = st.columns([1, 2, 1])  # Ajuste les proportions

with col1_logo:
    st.image("Pictures/__LOGO_EA_ORANGE.png", use_container_width=True)



st.title("Vitesse Critique (Critical Speed)")

with st.expander("Définition et hypothèses") :
    st.write(r"""
La vitesse critique (VC) est un modèle mathématique utilisé pour estimer la frontière entre deux domaines d'intensité d'exercice : l’intensité élevée (où l'homéostasie est maintenue) et l’intensité sévère (où l'homéostasie est rompue et l'épuisement est inévitable). Ce modèle repose sur une relation hyperbolique entre la vitesse et la durée de l’effort, définissant une vitesse seuil soutenable sur une longue période.

Le paramètre $D'$ représente une quantité de travail pouvant être effectuée au-delà de la vitesse critique avant d'atteindre l'épuisement. Plutôt que d’être une simple "réserve anaérobie", $D'$ est mieux décrit comme une capacité de travail finie issue de processus métaboliques non soutenables sur le long terme. Il inclut l’utilisation des phosphagènes (ATP-PCr), de la glycolyse anaérobie et potentiellement de contributions aérobie transitoires.

Lorsque la vitesse excède la vitesse critique, $D'$ est consommé proportionnellement à l’intensité de l’effort. Plus l’athlète court vite, plus $D'$ se vide rapidement. À l’inverse, lorsqu’il ralentit sous la vitesse critique, $D'$ se reconstitue progressivement selon une dynamique exponentielle, dont la vitesse dépend d’une constante de temps $\tau$.

L'utilisation de $D'$ permet de calibrer les séances d'entraînement en haute intensité (HIT) en quantifiant précisément la tolérance à l'effort au-dessus de la vitesse critique. Cela aide à structurer les répétitions et les périodes de récupération. Cependant, il s'agit d'un modèle, et la cinétique de récupération de $D'$ varie d'un individu à l'autre, notamment en fonction de l'entraînement et des caractéristiques physiologiques. Avec l'accumulation des séances, $D$ peut être ajusté progressivement, permettant ainsi d'affiner la programmation et d'offrir une ligne directrice fiable pour optimiser la performance.

Pertinence de la Vitesse Critique pour Déterminer le MMSS
L’article de Jones et al. (2019)​ met en lumière l'importance de la vitesse critique pour identifier le Maximal Metabolic Steady State (MMSS), qui correspond à la plus haute intensité d'exercice où l'homéostasie physiologique peut être maintenue. Contrairement au concept traditionnel de Maximal Lactate Steady State (MLSS), qui repose sur l’accumulation du lactate sanguin, l'étude montre que la vitesse critique est un indicateur plus robuste du MMSS, car elle permet de mieux distinguer :

L’intensité élevée, où la consommation d’oxygène et les niveaux de lactate atteignent un état stable.
L’intensité sévère, où la demande énergétique excède les capacités oxydatives, entraînant une dérive progressive du métabolisme et l’épuisement.

Ce programme permet de calculer la **vitesse critique (CS)** et la **capacité anaérobie ($D'$)** à partir de tests de course.
""")

    st.latex(r"V(t) = CS + \frac{D'}{t}")




# =============================================================================
# CALCUL DE LA VITESSE CRITIQUE (CS)
# =============================================================================

st.write("")
st.subheader("CALCUL DE LA VITESSE CRITIQUE (CS)")

# Sélection du nombre de tests
num_points = st.radio("Nombre de tests à entrer :", [2, 3], horizontal = True)

# Entrée utilisateur
distances = []
times = []

for i in range(num_points):
    col1, col2 = st.columns(2)
    with col1:
        d = st.number_input(f"Distance {i+1} (m)", min_value=0, step=100, key=f"d_{i}")
    with col2:
        t = st.number_input(f"Temps {i+1} (s)", min_value=1, step=1, key=f"t_{i}")
    
    distances.append(d)
    times.append(t)


# Vérifier si les variables existent dans session_state
if "CS" not in st.session_state:
    st.session_state.CS = None
    st.session_state.D_prime_0 = None
    st.session_state.fig = None
    

# Calcul de la vitesse critique
CS, D_prime_0, speeds = calculate_critical_speed(distances, times)

# Transformer les valeurs de test en tableau dataframe pour pouvoir l'afficher dans le rapport ensuite
L_speeds = speeds.tolist()
L_allures = []
for i in range(0, len(L_speeds)) :
    L_allures.append(speed_to_pace(float(L_speeds[i])))
df_test = pd.DataFrame({"Distance [m]": distances, "Temps [s]": times, "Allure moyenne [min/km]" : L_allures})

# Conversion de CS en km/h et en allure min/km
CS_kmh = speed_m_s_to_kmh(CS)
CS_pace = speed_to_pace(CS)

# Bouton pour calculer la vitesse critique
if st.button("Calculer la Vitesse Critique"):
    if len(distances) >= 2 and len(times) >= 2:
        

        

        # Stocker les valeurs dans session_state
        st.session_state.CS = CS
        st.session_state.D_prime_0 = D_prime_0

        # Génération des données pour la courbe
        time_range = np.linspace(20, 2000, 200)  # Étendre jusqu'à 2500 s
        speed_pred = CS + (D_prime_0 / time_range)  # Courbe hyperbolique

        # Création du graphique avec Plotly
        fig = go.Figure()
        
        # Ajout des zones de couleur
        fig.add_trace(go.Scatter(
            x=[0, 2000, 2000, 0],
            y=[0.8*CS, 0.8*CS, 0.5*CS, 0.5*CS],
            fill='toself', fillcolor='rgba(168, 198, 134, 0.3)', #'rgba(168, 198, 134, 0.3)'
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
        
        # fig.add_trace(go.Scatter(
        #     x=[0, 2000, 2000, 0],
        #     y=[CS, CS, 0.8*CS, 0.8*CS],
        #     fill='toself', fillcolor='rgba(255, 179, 71, 0.3)',
        #     line=dict(color='rgba(0,0,0,0)'),
        #     showlegend=False
        # ))
        
        # fig.add_trace(go.Scatter(
        #     x=[0, 2000, 2000, 0],
        #     y=[max(speed_pred), max(speed_pred), CS, CS],
        #     fill='toself', fillcolor='rgba(170, 61, 0, 0.2)',
        #     line=dict(color='rgba(0,0,0,0)'),
        #     showlegend=False
        # ))
        
        # Calcul du point CS_5min
        CS_5min = CS + (D_prime_0 / 300) 
        
        # Ajout de la zone représentant D' sous forme de rectangle
        fig.add_trace(go.Scatter(
            x=[0, 300, 300, 0, 0],
            y=[CS, CS, CS_5min, CS_5min, CS],
            fill='toself', fillcolor='rgba(69, 62, 59, 0.3)',
            line=dict(color='#453E3B', width=1),
            name=f"Réserve anaérobie (D') = {round(D_prime_0, 2)} m"
        ))

        # Points expérimentaux
        fig.add_trace(go.Scatter(
            x=times, y=np.array(distances) / np.array(times),
            mode='markers', marker=dict(color='#AA3D00', size=8),
            name="Données expérimentales"
        ))

        # Courbe hyperbolique ajustée
        fig.add_trace(go.Scatter(
            x=time_range, y=speed_pred,
            mode='lines', line=dict(color='#453E3B', width=2),
            name="Modèle hyperbolique ajusté"
        ))


        # Asymptote horizontale (CS)
        fig.add_trace(go.Scatter(
            x=[0, 2000], y=[CS, CS],
            mode='lines', line=dict(color='#A8C686', width=2, dash='dash'),
            name=f"Vitesse Critique (CS) = {CS_pace}"
        ))
        
        

        # Mise en forme du graphique
        fig.update_layout(
        margin=dict(t=40, b=0),  # Supprime l'espace réservé au titre et en bas du graphe
        xaxis=dict(
            title = dict(text = "Temps (s)", font = dict(color = 'black')),
            showline=True,  # Afficher la barre de l'axe X
            linecolor='black',  # Couleur de la barre de l'axe X
            linewidth=1,  # Largeur de la barre de l'axe X
            range=[0, 2000], 
            showgrid=False,
            tickformat='.0f',
            tickfont=dict(color='black')
            ),
        yaxis=dict(
            title = dict(text = "Vitesse (m/s)", font = dict(color = 'black')),
            showline=True,  # Afficher la barre de l'axe Y
            linecolor='black',  # Couleur de la barre de l'axe Y
            linewidth=1,  # Largeur de la barre de l'axe Y
            range=[0.5*CS, max(speed_pred)], 
            showgrid=False,
            tickformat='.1f',
            tickfont=dict(color='black')
            ),
        legend=dict(
            x=0.95, y=0.95, xanchor='right', yanchor='top',
            bordercolor='#453E3B', borderwidth=0.5
            ),
        template="simple_white"
        )

        

        # Stocker le graphique dans session_state
        st.session_state.fig = fig
        

    else:
        st.error("❌ Veuillez entrer au moins deux points pour le calcul.")
        
        

# Affichage des résultats enregistrés
if st.session_state.CS is not None:
    st.success(f"✅ Vitesse Critique estimée : {speed_m_s_to_kmh(st.session_state.CS):.2f} km/h")
    st.write(f"📌 Allure correspondante : {speed_to_pace(st.session_state.CS)}")
    st.write(f"📌 D' (capacité anaérobie) estimée : {st.session_state.D_prime_0:.2f} m")

if st.session_state.fig is not None:
    figure = st.session_state.fig
    st.plotly_chart(figure)
    
    # On affiche la légende du graphe
    st.markdown(
        "<p style='text-align: center; font-size:15px; color:darkgray; font-style:italic;'>"
        "Modèle de vitesse critique"
        "</p>",
        unsafe_allow_html=True
    )
    st.write("\n\n")  # Deux lignes vides
    
    # Sauvegarder le graphe en tant qu'image
    CS_graph_path = "Temp/CS_graph.png"
    save_dir = os.path.dirname(CS_graph_path)
    
    figure.write_image(CS_graph_path, scale=4) #
    


# =============================================================================
# TELECHARGER LE RAPPORT PDF
# =============================================================================

st.subheader("TELECHARGER LE RAPPORT PDF") # Partie


# Bouton télécharger
pdf_buffer = create_pdf_template(df_test, CS_pace, CS_kmh, D_prime_0, CS_graph_path)


# Champ pour le nom du fichier
file_name = st.text_input("Nom du fichier PDF :", placeholder="Bilan Profil Force-Vitesse.pdf")

# Définir un état pour détecter le téléchargement
if "pdf_downloaded" not in st.session_state:
    st.session_state.pdf_downloaded = False

# Bouton de téléchargement
if st.download_button(
    label="Télécharger le bilan PDF",
    data=pdf_buffer,
    file_name=file_name,
    mime="application/pdf" # utilisé pour spécifier le type de fichier que l'utilisateur peut télécharger. Ici, application/pdf signifie qu'il s'agit d'un document pdf
):
    st.session_state.pdf_downloaded = True

# Afficher le message seulement après le téléchargement
if st.session_state.pdf_downloaded:
    st.success("PDF exporté avec succès")
else : # Il n'y a pas assez de lignes pour tracer un graphe
    st.error("Le fichier PDF ne peut pas être exporté.")
 





# =============================================================================
# CALIBRAGE D'UNE SEANCE HIT (> CS)
# =============================================================================

st.write("")
st.subheader("CALIBRAGE D'UNE SEANCE HIT (> CS)")

st.write("""
Ajoutez des blocs d'entraînement ou des répétitions (enchaînement de 2 blocs répétés plusieurs fois).  
Le graphique se mettra à jour au fur et à mesure.
""")


# Initialisation de la séance
if "session" not in st.session_state:
    st.session_state.session = []

# Choix entre ajout d'un bloc ou d'une répétition
choice = st.radio("Ajouter :", ["Bloc simple", "Répétition"], horizontal = True)

with st.form("Ajouter un bloc ou une répétition"):
    if choice == "Bloc simple":
        duration = st.number_input("Durée du bloc (min) :", min_value=1, step=1, value=5) * 60
        percent_CS = st.slider("Intensité en % de CS :", min_value=50, max_value=150, step=5, value=100)
        submitted = st.form_submit_button("Ajouter ce bloc")

        if submitted:
            st.session_state.session.append((duration, percent_CS))
            st.success(f"Bloc ajouté : {duration//60} min à {percent_CS}% de CS")
            st.rerun()

    elif choice == "Répétition":
        repetitions = st.number_input("Nombre de répétitions :", min_value=1, step=1, value=3)
        
        st.write("")
        st.subheader("Bloc 1")
        duration1 = st.number_input("Durée du 1er bloc (min) :", min_value=1, step=1, value=2) * 60
        percent_CS1 = st.slider("Intensité du 1er bloc (% de CS) :", min_value=50, max_value=150, step=5, value=100)
        
        st.write("")
        st.subheader("Bloc 2")
        duration2 = st.number_input("Durée du 2ème bloc (min) :", min_value=1, step=1, value=1) * 60
        percent_CS2 = st.slider("Intensité du 2ème bloc (% de CS) :", min_value=50, max_value=150, step=5, value=75)

        
        submitted = st.form_submit_button("Ajouter cette répétition")

        if submitted:
            for _ in range(repetitions):
                st.session_state.session.append((duration1, percent_CS1))
                st.session_state.session.append((duration2, percent_CS2))
            st.success(f"Répétition ajoutée : {repetitions} x [{duration1//60} min à {percent_CS1}% + {duration2//60} min à {percent_CS2}%]")
            st.rerun()
            

# Affichage de la séance actuelle
st.subheader("Séance saisie")

if st.session_state.session:
    L_saisie_seance = []
    
    for i, (dur, percent_CS) in enumerate(st.session_state.session):
        speed_m_s = CS * (percent_CS / 100)
        speed_kmh = speed_m_s_to_kmh(speed_m_s)
        pace = speed_to_pace(speed_m_s)
        if percent_CS > 100 :
            L_saisie_seance.append(f"🔴 {dur//60} min à {percent_CS}% de CS → {speed_kmh:.2f} km/h ({pace} min/km)")
        else :
            L_saisie_seance.append(f"🟢 {dur//60} min à {percent_CS}% de CS → {speed_kmh:.2f} km/h ({pace} min/km)")
    
    afficher_blocs(L_saisie_seance)
    # for i in range(len(L_saisie_seance)) :
    #     st.write(L_saisie_seance[i])

    # Calcul de l'évolution de D'
    time, D_prime, velocities = compute_D_prime_evolution(CS, D_prime_0, st.session_state.session)

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Courbe D'(t) en #453e3b
    fig.add_trace(go.Scatter(
        x=time, y=D_prime,
        mode='lines', line=dict(color='#453e3b', width=2),
        name="D' restant",
        showlegend=False,
        yaxis="y1"
    ))

    # Blocs d'effort affichés sur l'axe secondaire (droite)
    start_time = 0
    height_max = 0
    for i, (dur, percent_CS) in enumerate(st.session_state.session):
        end_time = start_time + dur
        height = CS * (percent_CS / 100)  
        
        if height > height_max :
            height_max = height

        if percent_CS < 75:
            color = 'rgba(168, 198, 134, 0.5)'  
        elif percent_CS < 100:
            color = 'rgba(255, 179, 71, 0.5)'  
        else:
            color = 'rgba(170, 61, 0, 0.5)'  

        fig.add_trace(go.Scatter(
            x=[start_time, end_time, end_time, start_time, start_time],
            y=[0, 0, height, height, 0],
            fill='toself', fillcolor=color,
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            yaxis="y2"
        ))
        start_time = end_time # + 5 # espace de 5 secondes entre les blocs pour le rendu visuel

    fig.update_layout(
        #title="Évolution de D' et structure de la séance",
        xaxis_title="Temps (s)",
        yaxis=dict(title="D' restant (m)", side="left", range=[0, D_prime_0], showgrid=False),
        yaxis2=dict(title="Vitesse (m/s)", side="right", overlaying="y", range=[0, height_max], showgrid=False),
    )

    st.plotly_chart(fig)

    if st.button("Réinitialiser la séance"):
        st.session_state.session = []
        st.rerun()
