import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re

# Définir les alias pour les colonnes avec plus de flexibilité
COLUMN_ALIASES = {
    'nom': [
        'nom', 'name', 'joueur', 'player', 'prenom', 'firstname', 
        'nom et prénom', 'participant', 'prénom', 'identifiant'
    ],
    'technique': [
        'technique', 'skill', 'competence', 'capacite', 'tech', 
        'niveau technique', 'niveau', 'note technique', 
        'note ton niveau technique', 'compétence'
    ],
    'physique': [
        'physique', 'physical', 'condition', 'fitness', 'phys', 
        'niveau physique', 'forme', 'endurance', 
        'note physique', 'note ton niveau physique'
    ]
}

def clean_column_name(column):
    """
    Nettoyer et normaliser les noms de colonnes
    """
    return (column.lower()
        .replace(' ', '')
        .replace('é', 'e')
        .replace('è', 'e')
        .replace('ê', 'e')
    )

def extract_numeric_value(value):
    """
    Extraire une valeur numérique d'une chaîne
    Gère les formats comme "4/5", "4 / 5", etc.
    """
    if pd.isna(value):
        return None
    
    # Convertir en chaîne si ce n'est pas déjà le cas
    str_value = str(value).lower().replace(',', '.')
    
    # Chercher un nombre (entier ou decimal)
    match = re.search(r'(\d+(?:\.\d+)?)', str_value)
    
    if match:
        try:
            num = float(match.group(1))
            return num if 1 <= num <= 5 else None
        except (ValueError, TypeError):
            return None
    
    return None

def fuzzy_column_match(data_columns):
    """
    Trouver des correspondances approximatives pour les noms de colonnes
    """
    column_mapping = {}
    unmatched_columns = []

    # Normaliser les noms de colonnes existants
    normalized_data_columns = [clean_column_name(col) for col in data_columns]
    
    for req_col, aliases in COLUMN_ALIASES.items():
        # Rechercher une correspondance
        matched = False
        for alias in aliases:
            normalized_alias = clean_column_name(alias)
            
            # Recherche inclusive
            for idx, norm_col in enumerate(normalized_data_columns):
                if (normalized_alias in norm_col or 
                    norm_col in normalized_alias):
                    orig_col = data_columns[idx]
                    column_mapping[req_col] = orig_col
                    matched = True
                    break
            
            if matched:
                break
        
        if not matched:
            unmatched_columns.append(req_col)
    
    return column_mapping, unmatched_columns

def load_data(file_path):
    """
    Charger des fichiers avec support élargi
    """
    try:
        # Support élargi des fichiers
        if file_path.endswith(('.xls', '.xlsx')):
            try:
                # Premier essai : lecture standard
                data = pd.read_excel(file_path)
            except Exception:
                # Deuxième essai avec openpyxl
                data = pd.read_excel(
                    file_path, 
                    engine='openpyxl', 
                    dtype=str  # Lire tous les types comme des chaînes
                )
        elif file_path.endswith('.csv'):
            # Support des CSV avec différents encodages
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, encoding=encoding)
                    break
                except Exception:
                    continue
            else:
                raise ValueError("Impossible de lire le fichier CSV")
        else:
            raise ValueError("Format de fichier non supporté")

        # Nettoyer les noms de colonnes
        data.columns = data.columns.str.strip()
        
        return data
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier : {e}")

def validate_data(players):
    """
    Valider et nettoyer les données des joueurs
    """
    # Vérifier les colonnes requises
    required_columns = ['nom', 'technique', 'physique']
    if not all(col in players.columns for col in required_columns):
        raise ValueError(f"Le fichier doit contenir les colonnes : {', '.join(required_columns)}")

    # Supprimer les lignes vides
    players = players.dropna()

    # Convertir les colonnes de score
    for col in ['technique', 'physique']:
        players[col] = players[col].apply(extract_numeric_value)

    # Supprimer les lignes avec des valeurs non valides
    players = players.dropna(subset=['technique', 'physique'])

    # Vérifier les limites des scores
    players = players[
        (players['technique'] >= 1) & (players['technique'] <= 5) &
        (players['physique'] >= 1) & (players['physique'] <= 5)
    ]

    return players

def generate_teams(players, num_teams):
    """
    Générer des équipes équilibrées
    """
    players['total'] = players['technique'] + players['physique']
    sorted_players = players.sort_values(by='total', ascending=False)

    # Ajouter un joueur "joker" si nécessaire pour équilibrer les équipes
    if len(sorted_players) % num_teams != 0:
        joker = pd.DataFrame([{"nom": "Joker", "technique": 0, "physique": 0, "total": 0}])
        sorted_players = pd.concat([sorted_players, joker], ignore_index=True)

    # Distribution des joueurs dans les équipes (en serpentin)
    teams = {i: [] for i in range(num_teams)}
    direction = 1  # 1 pour avancer, -1 pour reculer

    for i, player in enumerate(sorted_players.itertuples(index=False)):
        team_idx = i % num_teams if direction == 1 else (num_teams - 1) - (i % num_teams)
        teams[team_idx].append(player)
        if i % num_teams == num_teams - 1:
            direction *= -1  # Changer de direction

    return teams

def export_teams_to_excel(teams, output_file="equipes_frisbee.xlsx"):
    """
    Exporter les équipes dans un fichier Excel
    
    :param teams: Dictionnaire des équipes générées
    :param output_file: Nom du fichier de sortie
    :return: Chemin du fichier exporté
    """
    try:
        # Créer un writer Excel
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            # Calculer les statistiques globales
            global_stats = {
                'Statistique': [
                    'Nombre de joueurs', 
                    'Score total technique', 
                    'Score total physique', 
                    'Score total général'
                ]
            }
            
            # Parcourir chaque équipe
            for team_id, team in teams.items():
                # Convertir l'équipe en DataFrame
                team_df = pd.DataFrame(team)
                
                # Nommer explicitement les colonnes
                team_df.columns = [
                    'Nom', 
                    'Score Technique', 
                    'Score Physique', 
                    'Score Total'
                ][:len(team_df.columns)]
                
                # Écrire l'équipe dans une feuille
                team_df.to_excel(
                    writer, 
                    index=False, 
                    sheet_name=f"Équipe {team_id + 1}"
                )
                
                # Calculer les statistiques de l'équipe
                global_stats[f'Équipe {team_id + 1}'] = [
                    len(team),
                    team_df['Score Technique'].sum() if 'Score Technique' in team_df.columns else 0,
                    team_df['Score Physique'].sum() if 'Score Physique' in team_df.columns else 0,
                    team_df['Score Total'].sum() if 'Score Total' in team_df.columns else 0
                ]
            
            # Créer une feuille de statistiques globales
            stats_df = pd.DataFrame(global_stats)
            stats_df.to_excel(
                writer, 
                index=False, 
                sheet_name="Statistiques Globales"
            )
        
        return output_file
    
    except Exception as e:
        raise ValueError(f"Erreur lors de l'exportation : {e}")

def plot_teams(teams):
    """
    Générer un graphique comparant les scores des équipes
    """
    # Calculer le score total de chaque équipe
    team_totals = {team_id: sum(player.total for player in team) for team_id, team in teams.items()}
    
    # Créer un graphique en barres
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [f"Équipe {team_id + 1}" for team_id in team_totals.keys()], 
        team_totals.values(), 
        color='skyblue', 
        edgecolor='navy'
    )
    ax.set_title("Comparaison des scores totaux des équipes", fontsize=16)
    ax.set_xlabel("Équipe", fontsize=12)
    ax.set_ylabel("Score total (technique + physique)", fontsize=12)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{height:.1f}',
            ha='center', 
            va='bottom'
        )
    
    plt.tight_layout()
    return fig


# Interface principale
def main():
    st.title("Gestion des équipes Frisbee")
    st.write("Importez un fichier Excel ou CSV contenant les informations des joueurs pour démarrer.")
    
    # Importer le fichier
    uploaded_file = st.file_uploader("Choisissez un fichier", type=['xls', 'xlsx', 'csv'])
    
    if uploaded_file:
        # Charger les données
        data = load_and_preview_data(uploaded_file)
        
        if data is not None:
            # Validation des colonnes avec correspondance flexible
            valid, data_renamed, column_mapping = validate_columns(data)
            
            if valid and data_renamed is not None:
                # Nettoyage et validation des données
                clean_data = validate_and_clean_data(data_renamed)
                
                if clean_data is not None:
                    st.success("Les données sont prêtes pour la suite du traitement !")
                    
                    # Options supplémentaires
                    if st.checkbox("Afficher le mapping des colonnes"):
                        st.write("Mapping des colonnes :", column_mapping)
                    
                    # Nombre d'équipes
                    num_teams = st.slider("Nombre d'équipes à créer", min_value=2, max_value=10, value=4)
                    
                    # Générer les équipes
                    if st.button("Générer les équipes"):
                        teams = generate_teams(clean_data, num_teams)
                        
                        # Afficher les équipes
                        for team_id, team in teams.items():
                            st.write(f"### Équipe {team_id + 1}")
                            team_df = pd.DataFrame(team)
                            st.dataframe(team_df[['nom', 'technique', 'physique', 'total']])
                        
                        # Visualisation des scores des équipes
                        st.write("### Comparaison des scores des équipes")
                        fig = plot_teams(teams)
                        st.pyplot(fig)
                        
                        # Option d'export
                        export_teams = st.checkbox("Exporter les équipes")
    if export_teams:
        try:
            # Utiliser la nouvelle fonction d'exportation
            export_file = export_teams_to_excel(teams)
            
            if export_file:
                with open(export_file, 'rb') as file:
                    st.download_button(
                        label="📥 Télécharger les équipes",
                        data=file,
                        file_name=export_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Fichier Excel avec les détails des équipes et leurs statistiques"
                    )
                
                # Afficher un aperçu des statistiques
                stats_df = pd.read_excel(export_file, sheet_name="Statistiques Globales")
                st.subheader("Résumé des Équipes")
                st.dataframe(stats_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erreur lors de la préparation de l'exportation : {e}")

if __name__ == "__main__":
    main()