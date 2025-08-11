# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:41:28 2025

@author: alanb
"""
#import plotly.io as pio
#pio.kaleido.scope.chromium_executable = "/usr/bin/chromium"


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

from fontTools.ttLib import TTCollection

# --- Kaleido/Chromium setup for Streamlit Cloud (headless) ---
#import shutil
#import plotly.io as pio
#chromium = shutil.which("chromium") or "/usr/bin/chromium"
#os.environ.setdefault("BROWSER_PATH", chromium)
#pio.defaults.default_format = "png"
#pio.defaults.default_scale = 3


# =============================================================================
# FONCTIONS
# =============================================================================

# Fonction pour calculer la vitesse critique avec le modèle hyperbolique
def calculate_critical_speed(distances, times, use_power_data, powers):
    times = np.array(times)
    speeds = np.array(distances) / times  # Vitesse moyenne pour chaque test
    inverse_times = 1 / times  # Transformation en 1/t
    if use_power_data :
        powers = np.array(powers)

    # Régression linéaire : V = CS + D'/t s'il y a assez de valeur
    if len(inverse_times) > 1 and times[0] != times[1] :
        slope, intercept, _, _, _ = linregress(inverse_times, speeds)
    
        CS = intercept  # Ordonnée à l'origine = vitesse critique
        D_prime_0 = slope  # Pente = D'

        # Si on utilise la puissance
        if use_power_data :
            # Régression linéaire : P = CP + W'/t
            slope, intercept, _, _, _ = linregress(inverse_times, powers)
        
            CP = intercept  # Ordonnée à l'origine = vitesse critique
            W_prime_0 = slope  # Pente = D'
        else :
            CP = 0
            W_prime_0 = 0

    # Sinon on met des valeurs random
    else :
        CS = 3
        D_prime_0 = 100
        CP = 0
        W_prime_0 = 0
    return CS, D_prime_0, speeds, CP, W_prime_0

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

def generate_training_zone_graph(pace_values):
    """
    Génère un graphique des zones d'entraînement avec allure (min/km),
    fréquence cardiaque et échelle RPE.

    Parameters:
    - pace_values (dict): Dictionnaire contenant les valeurs des seuils
      Exemple:
      {
          "seuil_1": "5:45",
          "seuil_bas": "5:00",
          "seuil_haut": "4:35",
          "zone1_fc": 150
      }
    """
    
    fig = go.Figure()

    # Ajout des zones colorées (Z1, Z2, Z3, Z4)
    fig.add_trace(go.Scatter(
        x=[0, 4, 4, 0], y=[0, 0, 1, 1],
        fill="toself", fillcolor="rgba(168, 198, 134, 0.8)", line=dict(color="rgba(0,0,0,0)"),
        name="Domaine modéré", showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[4, 7, 7, 4], y=[0, 0, 1, 1],
        fill="toself", fillcolor="rgba(248, 201, 0, 0.8)", line=dict(color="rgba(0,0,0,0)"),
        name="Domaine élevé", showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[7, 7.5, 7.5, 7], y=[0, 0, 1, 1],
        fill="toself", fillcolor="rgba(242, 123, 0, 0.8)", line=dict(color="rgba(0,0,0,0)"), # ou rgba(236, 182, 0, 0.8)
        name="Domaine très élevé", showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[7.5, 9, 9, 7.5], y=[0, 0, 1, 1],
        fill="toself", fillcolor="rgba(170, 61, 0, 0.8)", line=dict(color="rgba(0,0,0,0)"),
        name="Domaine sevère", showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[9, 10, 10, 9], y=[0, 0, 1, 1],
        fill="toself", fillcolor="rgba(69, 62, 59, 0.8)", line=dict(color="rgba(0,0,0,0)"),
        name="Domaine extrême", showlegend=False
    ))

    # Ajout des lignes verticales pour les seuils
    L_i = [4, 7, 7.5]
    for i, (label, pace) in enumerate(pace_values.items()):
        indice = L_i[i]
        fig.add_trace(go.Scatter(
            x=[indice, indice], y=[-0.08, 1.08],
            mode="lines", line=dict(color="#453E3B", dash="dot", width=0.5),
            name=label
        ))
        fig.add_annotation(
            x=indice, y=1.25, text=f"{label}<br>{pace}", showarrow=False,
            font=dict(size=10, color="#453E3B")
        )


    # Ajout de l'échelle RPE en bas
    rpe_values = ["0-4", "4-7", "7-7.5", "7.5-9", "9-10"]
    rpe_colors = ["#A8C686", "#F8C900", "#F27B00", "#AA3D00", "#453E3B"]
    
    fig.add_trace(go.Scatter(
        x=[0, 4], y=[-0.1, -0.1],
        mode="lines", line=dict(color=rpe_colors[0], width=3),
        name="RPE 0-4", showlegend=False
    ))
    rpe_value = rpe_values[0]
    fig.add_annotation(
        x=2, y=-0.3, text=f"RPE<br>{rpe_value}", showarrow=False, font=dict(size=11, color=rpe_colors[0])
    )
    
    fig.add_trace(go.Scatter(
        x=[4, 7], y=[-0.1, -0.1],
        mode="lines", line=dict(color=rpe_colors[1], width=3),
        name="RPE 4-7", showlegend=False
    ))
    rpe_value = rpe_values[1]
    fig.add_annotation(
        x=5.5, y=-0.3, text=f"RPE<br>{rpe_value}", showarrow=False, font=dict(size=11, color=rpe_colors[1])
    )
    
    fig.add_trace(go.Scatter(
        x=[7, 7.5], y=[-0.1, -0.1],
        mode="lines", line=dict(color=rpe_colors[2], width=3),
        name="RPE 7-7.5", showlegend=False
    ))
    rpe_value = rpe_values[2]
    fig.add_annotation(
        x=7.25, y=-0.3, text=f"RPE<br>{rpe_value}", showarrow=False, font=dict(size=11, color=rpe_colors[2])
    )
    
    fig.add_trace(go.Scatter(
        x=[7.5, 9], y=[-0.1, -0.1],
        mode="lines", line=dict(color=rpe_colors[3], width=3),
        name="RPE 7.5-9", showlegend=False
    ))
    rpe_value = rpe_values[3]
    fig.add_annotation(
        x=8.25, y=-0.3, text=f"RPE<br>{rpe_value}", showarrow=False, font=dict(size=11, color=rpe_colors[3])
    )
    
    fig.add_trace(go.Scatter(
        x=[9, 10], y=[-0.1, -0.1],
        mode="lines", line=dict(color=rpe_colors[4], width=3),
        name="RPE 9-10", showlegend=False
    ))
    rpe_value = rpe_values[4]
    fig.add_annotation(
        x=9.5, y=-0.3, text=f"RPE<br>{rpe_value}", showarrow=False, font=dict(size=11, color=rpe_colors[4])
    )
    
    

    # Mise en forme générale
    fig.update_layout(
        autosize=False,  # Désactive l'ajustement automatique de la taille
        width=800,  # Largeur fixe du graphe
        height=200,  # Hauteur fixe du graphe
        margin=dict(t=40, b=0),  # Supprime l'espace réservé au titre et en bas du graphe
        xaxis=dict(visible=False),  # Supprime l'axe des X dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(visible=False),  # Supprime l'axe des X dict(showgrid=False, zeroline=False, showticklabels=False),
        template="simple_white",
        showlegend=False
    )

    return fig


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
    tau = 300 # Constante de temps #300, 450, 600
    
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
    
    
    
def create_pdf_template(df_test, CS_pace, CS_kmh, D_prime_0, CS_graph_path, Durability, Domaines_graph_path) :
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
    elements.append(Spacer(1, 24))  # Ajouter un espace après le titre
    title = Paragraph("BILAN DU TEST DE VITESSE CRITIQUE", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))  # Ajouter un espace après le titre

    
    # =============================================================================
    #     VALEURS DES TESTS
    # =============================================================================
    
    subtitle_1 = Paragraph("Résultats des tests", subtitle2_style)
    elements.append(subtitle_1)

    elements.append(Spacer(1, 6))  # Ajouter un espace après le texte

    # TABLEAU DES VALEURS DE TEST

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

    text = Paragraph("La vitesse critique de l'athlète est de " + str(round(CS_kmh, 2)) + " km/h, soit " + str(CS_pace) + " min/km. Pour rappel, cette intensité permet de délimiter le domaine d'intensité lourd et le domaine d'intensité sévère.", normal_style)
    elements.append(text)
    # text = Paragraph("Pour rappel, cette intensité permet de délimiter le domaine d'intensité lourd et le domaine d'intensité sévère.", normal_style)
    # elements.append(text)
    text = Paragraph("La résèrve anaérobie D' correspondante est de " + str(round(D_prime_0, 1)) + " m.", normal_style)
    elements.append(text)
    text = Paragraph("L'indice de durabilité, calculé à partir de la vitesse limite sur 5 minutes et de la vitesse critique, est de " + str(Durability) + " %.", normal_style)
    elements.append(text)
    if Durability > 90 :
        text = Paragraph("Le profil obtenu est plutôt ENDURANT.", normal_style)
    else :
        text = Paragraph("Le profil obtenu est plutôt RAPIDE.", normal_style)
    elements.append(text)

    # On affiche le graphe d'évolution de l'effort dans la liaison boulonnée en fonction de l'effort extérieur dans le cas où la thermique est prise en compte
    # scale_factor = 0.8  # Réduction de 5%


    # Saut de page
    elements.append(PageBreak())
    elements.append(Spacer(1, 24))  # Ajouter un espace après le titre

    text = Paragraph("La vitesse critique marque la transition entre le domaine d'intensité élevé et le domaine d'intensité sevère. Le diagramme ci-dessous représente les domaines d'intensité de l'athlète basés sur la vitesse critique. Les valeurs associées au premier seuil de lactate (LT1) et au second seuil de lactate (LT2) sont placé à des pourcentages arbitraires de la vitesse critique. Pour le premier seuil, ce pourcentage est calculé à partir de la valeur de la vitesse critique, en se basant sur l'étude de Ben Hunter et al. [2], ajusté en fonction de l'indice de durabilité calculé. Il s'agit d'un point de départ à ajuster avec l'entraînement, à défaut d'avoir recours à des méthodes plus précises (mesure du lactate ou de la ventilation).", normal_style)
    elements.append(text)                 
    
    elements.append(Spacer(1, 12))  # Ajouter un espace après le texte
    
    # Ajustement de la largeur et de la hauteur du graphe 
    Domaines_graph = Image(Domaines_graph_path)
    Domaines_graph.drawHeight = page_width * Domaines_graph.drawHeight / Domaines_graph.drawWidth
    Domaines_graph.drawWidth = page_width

    elements.append(Domaines_graph)
    legend = Paragraph("Figure 2 : Domaines d'intensité de l'athlète", legend_style)
    elements.append(legend)

    
    # Génération du PDF
    doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)

    
    buffer.seek(0)
    return buffer


def estimate_LT1(cs, d_index):
    """
    Estime le premier seuil (LT1) à partir de la vitesse critique (CS) et de l’indice de durabilité (D’index).
    
    Paramètres :
    - cs : float - Vitesse critique (km/h)
    - d_index : float - Indice de durabilité (0 à 1)
    
    Retourne :
    - lt1 : float - Estimation de la vitesse au premier seuil (km/h)
    - lt1ratio : float - Ratio de LT1 par rapport à CS
    """
    # Facteurs de base selon la plage de CS (issus de l'étude)
    if cs <= 12:
        base_factor = 0.806
    elif 12 < cs <= 14:
        base_factor = 0.832
    else:
        base_factor = 0.842
    
    # Ajustement basé sur l'indice de durabilité (k = 0.05, ajustable empiriquement)
    k = 0.1  
    d_ref = 0.85  # Valeur moyenne de D’index
    adjustment = 1 + k * (d_index - d_ref)
    
    # Calcul de LT1 ajusté
    lt1 = cs * base_factor * adjustment
    lt1 = round(lt1, 2)
    lt1ratio = round((lt1/cs)*100, 1)
    return lt1, lt1ratio, base_factor, adjustment  # Arrondi à 2 décimales



import math
from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go

def _fit_powerlaw_xy(x_list, y_list):
    """
    Ajuste log(y)=a+b*log(x). Renvoie A=exp(a), B=b.
    """
    x = [math.log(float(xi)) for xi in x_list]
    y = [math.log(float(yi)) for yi in y_list]
    if len(x_list) == 2:
        (x1, y1), (x2, y2) = (x[0], y[0]), (x[1], y[1])
        if x2 == x1:
            raise ValueError("Deux abscisses identiques pour un fit à 2 points.")
        B = (y2 - y1) / (x2 - x1)
        a = y1 - B * x1
    else:
        n = len(x)
        sx, sy = sum(x), sum(y)
        sxx = sum(xi*xi for xi in x)
        sxy = sum(xi*yi for xi, yi in zip(x, y))
        denom = n*sxx - sx*sx
        if denom == 0:
            raise ValueError("Données dégénérées pour la régression.")
        B = (n*sxy - sx*sy) / denom
        a = (sy - B*sx) / n
    A = math.exp(a)
    return A, B

def powerlaw_vitesse_et_puissance_append_points(
    d: List[float],
    t: List[float],
    p: List[float],
    t_short: float = 300.0,   # 5 minutes
    t_long: float = 1200.0    # 20 minutes
) -> Tuple[List[float], List[float], List[float], go.Figure]:
    """
    - Ajuste v(t)=A*t^B (v en km/h) à partir de d (m) et t (s)
    - Ajuste P(t)=C*t^D (P en W) à partir de p (W) et t (s)
    - Ajoute les points théoriques pour t_short et t_long dans d/t/p
    - Renvoie la figure Plotly de la power law Vitesse (seulement)

    Returns
    -------
    d_out, t_out, p_out, fig
      d_out : d + [d_5', d_20'] (m)
      t_out : t + [300, 1200] (s)
      p_out : p + [P_5', P_20'] (W)
      fig   : figure Plotly (vitesse vs temps)
    """
    # --- validations ---
    if not (len(d) == len(t) == len(p)):
        raise ValueError("d, t et p doivent avoir la même longueur.")
    if len(d) < 2:
        raise ValueError("Fournir au moins 2 points (d,t,p).")
    if any(di <= 0 for di in d) or any(ti <= 0 for ti in t) or any(pi <= 0 for pi in p):
        raise ValueError("Toutes les distances, tous les temps et toutes les puissances doivent être > 0.")

    # --- vitesses observées (km/h) ---
    v_kmh = [3.6 * di / ti for di, ti in zip(d, t)]

    # --- fit power law vitesse: v(t)=A*t^B ---
    A_v, B_v = _fit_powerlaw_xy(t, v_kmh)

    # prédictions vitesses
    v_5_kmh  = A_v * (t_short ** B_v)
    v_20_kmh = A_v * (t_long  ** B_v)

    # distances associées (m) à 5' et 20'
    d_5_m  = v_5_kmh  * (t_short / 3600.0) * 1000.0
    d_20_m = v_20_kmh * (t_long  / 3600.0) * 1000.0

    # --- fit power law puissance: P(t)=C*t^D ---
    C_p, D_p = _fit_powerlaw_xy(t, p)

    # prédictions puissances (W)
    P_5 = C_p * (t_short ** D_p)
    P_20 = C_p * (t_long ** D_p)

    # --- sorties augmentées ---
    d_out = [d_5_m, d_20_m]
    t_out = [t_short, t_long]
    p_out = [P_5, P_20]

    # --- figure (uniquement vitesse) ---
    t_min = 0.8 * min(min(t), t_short)
    t_max = 1.2 * max(max(t), t_long)
    t_range = np.linspace(t_min, t_max, 300)
    v_pred = A_v * (t_range ** B_v)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_range, y=v_pred,
        mode="lines", name="Loi puissance v = A·t^B"
    ))
    fig.add_trace(go.Scatter(
        x=t, y=v_kmh,
        mode="markers", name="Points observés (v)"
    ))
    fig.add_trace(go.Scatter(
        x=[t_short, t_long], y=[v_5_kmh, v_20_kmh],
        mode="markers", marker_symbol="x", marker_size=10,
        name="Prédictions 5′ & 20′ (v)"
    ))
    fig.update_layout(
        title="Power law Vitesse (km/h) en fonction du Temps (s)",
        xaxis_title="Temps (s)",
        yaxis_title="Vitesse (km/h)",
        legend_title=None,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    return d_out, t_out, p_out, fig






# Configuration du titre de la page et du logo
st.set_page_config(page_title="Vitesse Critique", page_icon="Pictures/__LOGO_EAICONE_NOIR.png")



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



st.header("VITESSE CRITIQUE (CRITICAL SPEED)")

# saut de ligne
st.write("\n")
# saut de ligne
st.write("\n")

with st.expander("Définition et hypothèses") :
    st.subheader("Généralités")
    st.write(r"""
La vitesse critique (VC) est un modèle mathématique utilisé pour estimer la frontière entre deux domaines d'intensité d'exercice : l’intensité élevée (où l'homéostasie est maintenue) et l’intensité sévère (où l'homéostasie est rompue et l'épuisement est inévitable). Ce modèle repose sur une relation hyperbolique entre la vitesse et la durée de l’effort, définissant une vitesse seuil soutenable sur une longue période.
""")
    st.image("Pictures/Modele Vitesse Critique.PNG", use_container_width=True)
    st.write(r"""
Ce programme permet de calculer la **vitesse critique (CS)** et la **capacité anaérobie ($D'$)** à partir de tests de course, et de **programmer une séance HIT (> CS)** en déterminant l'évolution de $D'$ pour une séance donnée. La relation qui lie la vitesse au temps sur laquelle est basée le modèle de vitesse critique est rappelée ci-dessous.
""")

    st.latex(r"V(t) = CS + \frac{D'}{t}")
    st.subheader("Définition et utilisation de D'")
    st.write(r"""
Le paramètre $D'$ représente une quantité de travail pouvant être effectuée au-delà de la vitesse critique avant d'atteindre l'épuisement. Plutôt que d’être une simple "réserve anaérobie", $D'$ est mieux décrit comme une capacité de travail finie issue de processus métaboliques non soutenables sur le long terme. Il inclut l’utilisation des phosphagènes (ATP-PCr), de la glycolyse anaérobie et potentiellement de contributions aérobie transitoires.

Lorsque la vitesse excède la vitesse critique, $D'$ est consommé proportionnellement à l’intensité de l’effort. Plus l’athlète court vite, plus $D'$ se vide rapidement. À l’inverse, lorsqu’il ralentit sous la vitesse critique, $D'$ se reconstitue progressivement selon une dynamique exponentielle, dont la vitesse dépend d’une constante de temps $\tau$.

L'utilisation de $D'$ permet de calibrer les séances d'entraînement en haute intensité (HIT) en quantifiant précisément la tolérance à l'effort au-dessus de la vitesse critique. Cela aide à structurer les répétitions et les périodes de récupération. Cependant, il s'agit d'un modèle, et la cinétique de récupération de $D'$ varie d'un individu à l'autre, notamment en fonction de l'entraînement et des caractéristiques physiologiques. Avec l'accumulation des séances, $D'$ peut être ajusté progressivement, permettant ainsi d'affiner la programmation et d'offrir une ligne directrice fiable pour optimiser la performance.
""")
    st.subheader("Pertinence de la Vitesse Critique pour Déterminer le MMSS")
    st.write(r"""
L’article de Jones et al. (2019)​ met en lumière l'importance de la vitesse critique pour identifier le Maximal Metabolic Steady State (MMSS), qui correspond à la plus haute intensité d'exercice où l'homéostasie physiologique peut être maintenue. Contrairement au concept traditionnel de Maximal Lactate Steady State (MLSS), qui repose sur l’accumulation du lactate sanguin, l'étude montre que la vitesse critique est un indicateur plus robuste du MMSS, car elle permet de mieux distinguer :
""")
    st.write("- L’intensité élevée, où la consommation d’oxygène et les niveaux de lactate atteignent un état stable.")
    st.write("- L’intensité sévère, où la demande énergétique excède les capacités oxydatives, entraînant une dérive progressive du métabolisme et l’épuisement.")
with st.expander("Références") :
    st.markdown("""
    **[1] Jones, A. M., Vanhatalo, A., Burnley, M., Morton, R. H., & Poole, D. C. (2010).**  
    Critical power: Implications for determination of VO₂max and exercise tolerance.   
    *Medicine & Science in Sports & Exercise, 42*(10), 1876–1890. 
    [DOI: 10.1249/MSS.0b013e3181d9cf7f](https://journals.lww.com/acsm-msse/fulltext/2010/10000/critical_power__implications_for_determination_of.11.aspx)
    [DOI: 10.1123/ijspp.2024-0101](https://doi.org/10.1123/ijspp.2024-0101)
    """)
    st.markdown("""
    **[2] Hunter, B., Meyler, S., Maunder, E., Cox, T. H., & Muniz-Pumares, D. (2024).**  
    The Relationship Between the Moderate–Heavy Boundary and Critical Speed in Running.  
    *International Journal of Sports Physiology and Performance, 19*(9), 963-972.  
    [DOI: 10.1123/ijspp.2024-0101](https://doi.org/10.1123/ijspp.2024-0101)
    """)
    st.markdown("""
    **[3] Jones, A. M., et al. (2019).**  
    The Maximal Metabolic Steady State: Redefining the Gold Standard.  
    *Physiological Reports, 7*(10), e14098.
    [DOI: 10.14814/phy2.14098](https://physoc.onlinelibrary.wiley.com/doi/10.14814/phy2.14098)​.
    """)




# =============================================================================
# CALCUL DE LA VITESSE CRITIQUE (CS)
# =============================================================================

# saut de ligne
st.write("\n")
# saut de ligne
st.write("\n")

st.subheader("CALCUL DE LA VITESSE CRITIQUE (CS)")

# saut de ligne
st.write("\n")

# methode1, methode2 = st.columns(2)

# with methode1 :
#     methode1 = st.checkbox("Utiliser des données de test")
# with methode2 :
#     methode2 = st.checkbox("Utiliser des données de compétition")

methode = st.radio("", ["Utiliser des données de test", "Utiliser des données de compétition"], horizontal = True, index = 0, label_visibility="collapsed")

if methode == "Utiliser des données de test" :
    st.markdown("### Saisie des données de test")
    
             
    # Crée un état pour stocker l'affichage de l'aide
    #if "show_help" not in st.session_state:
        #st.session_state.show_help = False
    
    # Création de colonnes pour aligner les éléments
    #selec_num_point_col1, selec_num_point_col2, empty_col3 = st.columns([5, 1, 12])  # Ajuster la largeur pour un bon alignement
    
    #with selec_num_point_col1 :
        # Sélection du nombre de tests
    st.markdown("**Nombre de tests à entrer :**") 
    num_points = st.radio("", [2, 3], horizontal = True, index = 0, label_visibility="collapsed")
    
    #with selec_num_point_col2 :
        # Bouton pour afficher/masquer l'aide
        #if st.button("?"):
            #st.session_state.show_help = not st.session_state.show_help
        
    # Affichage du texte explicatif si le bouton est activé
    #if st.session_state.show_help:
    st.info("Pour davantage de précision sur la détermination de la vitesse critique, la littérature conseille de saisir des durées comprises entre 3 et 20 minutes (voir [1]).")
    
    # saut de ligne
    st.write("\n")
    
    
    # Entrée utilisateur
    st.markdown("**Valeurs des tests :**") 
    # st.markdown("### Valeurs des tests : ") 
    # On coche si on veut renseigner les puissances également
    use_power_data = st.checkbox("Ajouter les puissances moyennes associées (si disponibles et mesurées via un pod)")
    
    distances = []
    times = []
    powers = []
    
    for i in range(num_points):
        if use_power_data == False :
            col1, col2 = st.columns(2)
            with col1:
                d = st.text_input(f"Distance {i+1} (m)", placeholder="0")
                
                # Conversion en floatant et en mètre pour pouvoir réaliser les opérations
                if len(d) == 0 :
                    d = 1000
                else :
                    d = float(d)
                    
            with col2:
                t = st.text_input(f"Temps {i+1} (s)", placeholder="0")
        
                if len(t) == 0 :
                    t = 180
                else :
                    t = float(t)
            
            distances.append(d)
            times.append(t)
        else :
            col1, col2, col3 = st.columns(3)
            with col1:
                d = st.text_input(f"Distance {i+1} (m)", placeholder="0")
                
                # Conversion en floatant et en mètre pour pouvoir réaliser les opérations
                if len(d) == 0 :
                    d = 1000
                else :
                    d = float(d)
                    
            with col2:
                t = st.text_input(f"Temps {i+1} (s)", placeholder="0")
        
                if len(t) == 0 :
                    t = 180
                else :
                    t = float(t)
            with col3:
                p = st.text_input(f"Puissance moyenne {i+1} (W)", placeholder="0")
    
                if len(p) == 0 :
                    p = 500
                else :
                    p = float(p)
            
            distances.append(d)
            times.append(t)
            powers.append(p)

else :
    st.markdown("### Saisie des données de course")         
 
    # Sélection du nombre de résultats de course
    st.markdown("**Nombre de rsultats de course à entrer :**") 
    num_points = st.radio("", [2, 3], horizontal = True, index = 0, label_visibility="collapsed")

        
    # saut de ligne
    st.write("\n")
    
    
    # Entrée utilisateur
    st.markdown("**Valeurs des tests :**") 

    # On coche si on veut renseigner les puissances également
    use_power_data = st.checkbox("Ajouter les puissances moyennes associées (si disponibles et mesurées via un pod)")
    
    distances = []
    times = []
    powers = []
    
    for i in range(num_points):
        if use_power_data == False :
            col1, col2 = st.columns(2)
            with col1:
                d = st.text_input(f"Distance {i+1} (m)", placeholder="0")
                
                # Conversion en floatant et en mètre pour pouvoir réaliser les opérations
                if len(d) == 0 :
                    d = 1000
                else :
                    d = float(d)
                    
            with col2:
                t = st.text_input(f"Temps {i+1} (s)", placeholder="0")
        
                if len(t) == 0 :
                    t = 180
                else :
                    t = float(t)
            
            distances.append(d)
            times.append(t)
            if len(distances) > 2 and len(times) > 2 :
                distances, times, powers = powerlaw_vitesse_et_puissance_append_points(distances,times,[2.0, 1.0],t_short = 300.0,t_long = 1200.0)
            
        else :
            col1, col2, col3 = st.columns(3)
            with col1:
                d = st.text_input(f"Distance {i+1} (m)", placeholder="0")
                
                # Conversion en floatant et en mètre pour pouvoir réaliser les opérations
                if len(d) == 0 :
                    d = 1000
                else :
                    d = float(d)
                    
            with col2:
                t = st.text_input(f"Temps {i+1} (s)", placeholder="0")
        
                if len(t) == 0 :
                    t = 180
                else :
                    t = float(t)
            with col3:
                p = st.text_input(f"Puissance moyenne {i+1} (W)", placeholder="0")
    
                if len(p) == 0 :
                    p = 500
                else :
                    p = float(p)
            
            distances.append(d)
            times.append(t)
            powers.append(p)

            if len(distances) > 2 and len(times) > 2 :
                distances, times, powers = powerlaw_vitesse_et_puissance_append_points(distances,times,powers,t_short = 300.0,t_long = 1200.0)



# Vérifier si les variables existent dans session_state
if "CS" not in st.session_state:
    st.session_state.CS = None
    st.session_state.D_prime_0 = None
    st.session_state.fig = None
    

# Calcul de la vitesse critique
# On initialise les valeurs
CS = 3
D_prime_0 = 100
speeds = [0]
CP = 10
W_prime_0 = 100
# On met à jour en calculant avec les données de test
CS, D_prime_0, speeds, CP, W_prime_0 = calculate_critical_speed(distances, times, use_power_data, powers)

# Calcul du point CS_5min
CS_5min = CS + (D_prime_0 / 300)

# Calcul de l'indice de durabilité (en %)
Durability = round((1-np.log(speeds[0]/speeds[-1])/np.log(times[-1]/times[0]))*100,1)

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

        if use_power_data :
            st.session_state.CP = CP
            st.session_state.W_prime_0 = W_prime_0

        # Génération des données pour la courbe
        time_range = np.linspace(20, 2500, 200)  # Étendre jusqu'à 2500 s
        speed_pred = (CS + (D_prime_0 / time_range))*3.6  # Courbe hyperbolique

        # Création du graphique avec Plotly
        fig = go.Figure()
        
        # Ajout des zones de couleur
        # fig.add_trace(go.Scatter(
        #     x=[0, 2000, 2000, 0],
        #     y=[0.8*CS, 0.8*CS, 0.4*CS, 0.4*CS],
        #     fill='toself', fillcolor='rgba(168, 198, 134, 0.3)', #'rgba(168, 198, 134, 0.3)'
        #     line=dict(color='rgba(0,0,0,0)'),
        #     showlegend=False
        # ))
        
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
        
         
        # Ajout de la zone représentant D' sous forme de rectangle
        fig.add_trace(go.Scatter(
            x=[0, 300, 300, 0, 0],
            y=[CS*3.6, CS*3.6, CS_5min*3.6, CS_5min*3.6, CS*3.6],
            fill='toself', fillcolor='rgba(170, 61, 0, 0.2)',# 'rgba(69, 62, 59, 0.3)',
            line=dict(color='rgba(0,0,0,0)'), # dict(color='#A8C686', width=1),
            name=f"Réserve anaérobie (D') = {round(D_prime_0, 2)} m"
        ))
        

        # Points expérimentaux
        fig.add_trace(go.Scatter(
            x=times, y=(np.array(distances) / np.array(times))*3.6,
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
            x=[0, 2500], y=[CS*3.6, CS*3.6],
            mode='lines', line=dict(color='#A8C686', width=2, dash='dash'),
            name=f"Vitesse Critique (CS) = {CS_pace}"
        ))
        
        

        # Mise en forme du graphique
        fig.update_layout(
        autosize=False,  # Désactive l'ajustement automatique de la taille
        width=800,  # Largeur fixe du graphe
        height=400,  # Hauteur fixe du graphe
        plot_bgcolor='white',  # Fond du graphe en blanc pour éviter des problèmes de rendu
        paper_bgcolor='white',  # Fond du "papier" du graphe en blanc aussi
        margin=dict(t=40, b=0),  # Supprime l'espace réservé au titre et en bas du graphe
        # plot_bgcolor="rgba(0,0,0,0)",  # Fond du graphe transparent
        # paper_bgcolor="rgba(0,0,0,0)",  # Fond extérieur transparent
        xaxis=dict(
            title = dict(text = "Temps (s)", font = dict(color = 'black')),
            showline=True,  # Afficher la barre de l'axe X
            linecolor='black',  # Couleur de la barre de l'axe X 
            linewidth=0.5,  # Largeur de la barre de l'axe X
            range=[0, 2500], 
            showgrid=False,
            tickformat='.0f',
            tickfont=dict(color='black')
            ),
        yaxis=dict(
            title = dict(text = "Vitesse (km/h)", font = dict(color = 'black')),
            showline=True,  # Afficher la barre de l'axe Y
            linecolor='black',  # Couleur de la barre de l'axe Y
            linewidth=0.5,  # Largeur de la barre de l'axe Y
            range=[0.5*CS*3.6, max(speed_pred)*0.8], 
            showgrid=False,
            tickformat='.1f',
            tickfont=dict(color='black')
            ),
        legend=dict(
            x=0.95, y=0.95, xanchor='right', yanchor='top',
            bordercolor='#453E3B', borderwidth=0.5
            ),
        template="plotly_white"
        # template=None
        )

        

        # Stocker le graphique dans session_state
        st.session_state.fig = fig
        

    else:
        st.error("❌ Veuillez entrer au moins deux points pour le calcul.")
else :
    st.info("Appuyez sur le bouton pour générer les résultats.")
        

st.write("\n\n")  # Deux lignes vides

st.markdown("### Résultats du test de vitesse critique") 
st.write("\n")  # Une lignes vides

if st.session_state.fig is not None:
    figure = st.session_state.fig
    st.plotly_chart(figure, use_container_width=False)
    
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
    
    figure.write_image(CS_graph_path, scale=4) 


# Affichage des résultats enregistrés
if st.session_state.CS is not None:
    st.write(f"💨 Vitesse Critique estimée : {speed_m_s_to_kmh(st.session_state.CS):.2f} km/h")
    st.write(f"💨 Allure correspondante : {speed_to_pace(st.session_state.CS)}")
    st.write(f"🔋 D' (capacité anaérobie) estimée : {st.session_state.D_prime_0:.2f} m")
    if use_power_data :
        st.write(f"⚡️ Puissance Critique estimée : {st.session_state.CP} W")
        st.write(f"🔋 W' (capacité anaérobie) estimée : {st.session_state.W_prime_0} J")
    st.write("📌 Indice de durabilité estimé : " + str(Durability) + " %")
    
    if Durability > 90 :
        st.write("📌 Profil endurant")
    else :
        st.write("📌 Profil rapide")


    st.write("La vitesse critique marque la transition entre le domaine d'intensité élevé et le domaine d'intensité sevère. Le diagramme ci-dessous représente les domaines d'intensité de l'athlète basés sur la vitesse critique. Les valeurs associées au premier seuil de lactate (LT1) et au second seuil de lactate (LT2) sont placé à des pourcentages arbitraires de la vitesse critique. Pour le premier seuil, ce pourcentage est calculé à partir de la valeur de la vitesse critique, en se basant sur l'étude de Ben Hunter et al. [2], ajusté en fonction de l'indice de durabilité calculé. Il s'agit d'un point de départ à ajuster avec l'entraînement, à défaut d'avoir recours à des méthodes plus précises (mesure du lactate ou de la ventilation).")
    LT2_speed = 0.95*CS
    LT2_pace = speed_to_pace(LT2_speed)
    LT2_pace_without_unit = LT2_pace[:4]
    LT1_speed, LT1_percent, base_factor, adjustment = estimate_LT1(CS*3.6, Durability/100.0)
    # LT1_speed = 0.8*CS
    # st.write("LT1 représente " + str(LT1_percent) + " de la vitesse critique")
    # st.write("base_factor = " + str(base_factor))
    # st.write("adjustment = " + str(adjustment))
    LT1_pace = speed_to_pace(LT1_speed/3.6)
    LT1_pace_without_unit = LT1_pace[:4]
    CS_pace_without_unit = CS_pace[:4]
    pace_values = {
        "LT1 / VT1": LT1_pace_without_unit,
        "LT2": LT2_pace_without_unit,
        "VC": CS_pace_without_unit
    }
    fig_domaines = generate_training_zone_graph(pace_values)
    st.plotly_chart(fig_domaines, use_container_width=False)

    # On affiche la légende du graphe
    st.markdown(
        "<p style='text-align: center; font-size:15px; color:darkgray; font-style:italic;'>"
        "Domaines d'intensité de l'athlète"
        "</p>",
        unsafe_allow_html=True
    )
    st.write("\n\n")  # Deux lignes vides
    
    # Sauvegarder le graphe en tant qu'image
    Domaines_graph_path = "Temp/Domaines_graph.png"
    save_dir = os.path.dirname(Domaines_graph_path)
    
    fig_domaines.write_image(Domaines_graph_path, scale=4) 
    


# =============================================================================
# TELECHARGER LE RAPPORT PDF
# =============================================================================

st.markdown("### Télécharger le rapport pdf") # Partie


# Bouton télécharger
# On fait l'export si le graphe existe
if st.session_state.fig is not None:
    pdf_buffer = create_pdf_template(df_test, CS_pace, CS_kmh, D_prime_0, CS_graph_path, Durability, Domaines_graph_path)


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
    # else : # Il n'y a pas assez de lignes pour tracer un graphe
    #     st.error("Le fichier PDF ne peut pas être exporté.")
# Si le graphe n'existe pas 
else :
    st.warning("Réalisez le calcul de vitesse critique pour pouvoir exporter les résultats")




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
st.write("\n")
st.markdown("**Type de bloc à ajouter**")
choice = st.radio("", ["Bloc simple", "Répétition"], horizontal = True, label_visibility="collapsed")

with st.form("Ajouter un bloc ou une répétition"):
    if choice == "Bloc simple":
        st.markdown("<p style='font-size:14px;'><strong>Durée du bloc</strong></p>", unsafe_allow_html=True)
        duration_bloc_simple_col1, duration_bloc_simple_col2 = st.columns(2)
        with duration_bloc_simple_col1 :
            duration_min = st.number_input("minutes :", min_value=0, step=1, value=5) * 60
        with duration_bloc_simple_col2 :
            duration_sec = st.number_input("secondes :", min_value=0, step=1, value=0)
        duration = duration_min + duration_sec
        st.write("\n")
        percent_CS = st.slider("**Intensité en % de CS**", min_value=50, max_value=150, step=5, value=100)
        submitted = st.form_submit_button("Ajouter ce bloc")

        if submitted:
            st.session_state.session.append((duration, percent_CS))
            st.success(f"Bloc ajouté : {duration//60} min à {percent_CS}% de CS")
            st.rerun()

    elif choice == "Répétition":
        repetitions = st.number_input("**Nombre de répétitions :**", min_value=1, step=1, value=3)
        
        st.write("")
        st.subheader("Bloc 1")
        st.markdown("<p style='font-size:14px;'><strong>Durée du 1er bloc</strong></p>", unsafe_allow_html=True)
        duration_bloc_1_col1, duration_bloc_1_col2 = st.columns(2)
        with duration_bloc_1_col1 :
            duration1_min = st.number_input("minutes :", min_value=0, step=1, value=1) * 60
        with duration_bloc_1_col2 :
            duration1_sec = st.number_input("secondes :", min_value=0, step=1, value=0)
        
        duration1 = duration1_min + duration1_sec
        #st.write("duration1 :", duration1)
        st.write("\n")
        percent_CS1 = st.slider("**Intensité du 1er bloc (% de CS) :**", min_value=50, max_value=150, step=5, value=100)
        
        st.write("")
        st.subheader("Bloc 2")
        st.markdown("<p style='font-size:14px;'><strong>Durée du 2ème bloc</strong></p>", unsafe_allow_html=True)
        
        duration_bloc_2_col1, duration_bloc_2_col2 = st.columns(2)
        with duration_bloc_2_col1 :
            #duration2_min = st.number_input("minutes :", min_value=0, step=1, value=0) * 60
            duration2_min = st.text_input("minutes :", placeholder="0")
        
            # Conversion en floatant et en mètre pour pouvoir réaliser les opérations
            if len(duration2_min) == 0 :
                duration2_min = 0
            else :
                duration2_min = int(duration2_min) * 60
        with duration_bloc_2_col2 :
            #duration2_sec = st.number_input("secondes :", min_value=0, step=1, value=0)
            duration2_sec = st.text_input("secondes :", placeholder="0")
        
            # Conversion en floatant et en mètre pour pouvoir réaliser les opérations
            if len(duration2_sec) == 0 :
                duration2_sec = 0
            else :
                duration2_sec = int(duration2_sec)
                

        duration2 = duration2_min + duration2_sec
        #st.write("duration2 :", duration2)
        #duration2 = st.number_input("minutes :", min_value=1, step=1, value=2) * 60
        st.write("\n")
        percent_CS2 = st.slider("**Intensité du 2ème bloc (% de CS) :**", min_value=50, max_value=150, step=5, value=75)
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
            L_saisie_seance.append(f"🔴 {dur//60} min à {percent_CS}% de CS → {speed_kmh:.2f} km/h ({pace})") # min/km
        else :
            L_saisie_seance.append(f"🟢 {dur//60} min à {percent_CS}% de CS → {speed_kmh:.2f} km/h ({pace})")
    
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















