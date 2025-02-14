import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
import math

# Configuration de base de Streamlit
st.set_page_config(page_title="Gestion des Équipes Frisbee", page_icon=":rugby_football:")

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

def load_and_preview_data(uploaded_file):
    """
    Charger des fichiers avec support élargi
    """
    try:
        # Support élargi des fichiers
        if uploaded_file.name.endswith(('.xls', '.xlsx')):
            try:
                # Premier essai : lecture standard
                data = pd.read_excel(uploaded_file)
            except Exception:
                # Deuxième essai avec openpyxl
                data = pd.read_excel(
                    uploaded_file, 
                    engine='openpyxl', 
                    dtype=str  # Lire tous les types comme des chaînes
                )
        elif uploaded_file.name.endswith('.csv'):
            # Support des CSV avec différents encodages
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    data = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except Exception:
                    continue
            else:
                st.error("Impossible de lire le fichier CSV")
                return None
        else:
            st.error("Format de fichier non supporté")
            return None

        # Nettoyer les noms de colonnes
        data.columns = data.columns.str.strip()
        
        st.info("Fichier chargé avec succès ! Voici un aperçu des premières lignes :")
        st.dataframe(data.head())  # Afficher les 5 premières lignes pour prévisualisation
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

def validate_columns(data):
    """
    Identification et validation des colonnes
    """
    st.write("**Étape 1 : Identification des colonnes**")
    
    # Recherche de correspondance floue
    column_mapping, unmatched_columns = fuzzy_column_match(data.columns)
    
    # Si des colonnes sont manquantes, interaction utilisateur
    if unmatched_columns:
        st.warning(f"Les colonnes suivantes n'ont pas été identifiées automatiquement : {', '.join(unmatched_columns)}")
        
        for missing_col in unmatched_columns:
            # Proposer des options de colonnes existantes
            selected_col = st.selectbox(
                f"Sélectionnez la colonne correspondant à '{missing_col}'", 
                options=list(data.columns),
                key=f"column_select_{missing_col}"
            )
            column_mapping[missing_col] = selected_col
    
    # Renommer les colonnes
    data_renamed = data.copy()
    
    # Vérifier que toutes les colonnes requises sont présentes
    final_columns = ['nom', 'technique', 'physique']
    final_mapping = {}
    
    for req_col in final_columns:
        # Trouver la colonne sélectionnée
        matched_col = column_mapping.get(req_col)
        
        if matched_col:
            if req_col == 'nom':
                # Spécifiquement pour la colonne nom, s'assurer qu'elle n'est pas vide
                data_renamed[req_col] = data_renamed[matched_col].fillna('Joueur')
            
            # Convertir les colonnes technique et physique
            if req_col in ['technique', 'physique']:
                data_renamed[req_col] = data_renamed[matched_col].apply(extract_numeric_value)
            
            final_mapping[req_col] = matched_col
    
    # Vérifier la conversion
    try:
        # Vérifier la conversion des colonnes
        conversion_check = data_renamed[final_columns].notna().all()
        
        # Afficher un avertissement si des valeurs sont manquantes
        if not conversion_check['nom']:
            st.warning("Certains noms n'ont pas pu être extraits correctement.")
        
        if not conversion_check['technique'] or not conversion_check['physique']:
            st.warning("Certaines valeurs de technique ou physique n'ont pas pu être converties.")
    except KeyError:
        st.error("Impossible de convertir toutes les colonnes requises.")
        return False, None, None
    
    st.success("Toutes les colonnes nécessaires sont identifiées !")
    return True, data_renamed, final_mapping

def validate_and_clean_data(data):
    """
    Validation et nettoyage des données
    """
    st.write("**Étape 2 : Validation et nettoyage des données**")
    
    # Convertir les colonnes en numérique
    for col in ['technique', 'physique']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Vérifier les valeurs manquantes ou non valides
    invalid_rows = data[data[['technique', 'physique']].isna().any(axis=1)]
    
    if not invalid_rows.empty:
        st.warning("Certaines lignes ont des valeurs invalides ou manquantes.")
        
        # Option de correction manuelle
        correction_mode = st.checkbox("Activer la correction manuelle des lignes", key="correction_mode")
        
        if correction_mode:
            corrected_data = invalid_rows.copy()
            
            for idx, row in invalid_rows.iterrows():
                st.write(f"\n### Correction pour la ligne {idx}")
                
                for col in ['technique', 'physique']:
                    if pd.isna(row[col]):
                        current_value = st.text_input(
                            f"Valeur pour {col} (Ligne {idx})", 
                            value="", 
                            key=f"correction_{idx}_{col}"
                        )
                        
                        if current_value:
                            try:
                                score = float(current_value)
                                if 1 <= score <= 5:
                                    corrected_data.at[idx, col] = score
                                else:
                                    st.error(f"Le score doit être entre 1 et 5 pour {col}")
                            except ValueError:
                                st.error(f"Valeur invalide pour {col}")
            
            # Mettre à jour les données originales
            data.update(corrected_data)

    # Nettoyer les données : retirer les lignes invalides
    data_cleaned = data.dropna(subset=['technique', 'physique'])

    # Vérifier les plages de valeurs
    data_cleaned = data_cleaned[
        (data_cleaned['technique'] >= 1) & (data_cleaned['technique'] <= 5) &
        (data_cleaned['physique'] >= 1) & (data_cleaned['physique'] <= 5)
    ]

    if data_cleaned.empty:
        st.error("Aucune ligne valide après le nettoyage ! Vérifiez les données dans le fichier.")
        return None

    st.success(f"Données validées et nettoyées ! {len(data_cleaned)} lignes prêtes à l'emploi.")
    st.dataframe(data_cleaned.head())  # Afficher un aperçu des données nettoyées
    return data_cleaned
def suggest_team_configurations(total_players):
    """
    Suggérer des configurations d'équipes optimales
    """
    suggestions = []
    
    # Configurations possibles
    possible_team_sizes = [
        (2, 6, 7),   # 2 équipes
        (3, 4, 5),   # 3 équipes
        (4, 3, 4),   # 4 équipes
        (5, 3, 3),   # 5 équipes
        (6, 2, 3)    # 6 équipes
    ]
    
    for num_teams, min_per_team, max_per_team in possible_team_sizes:
        if (total_players >= num_teams * min_per_team and 
            total_players <= num_teams * max_per_team):
            suggestions.append({
                'num_teams': num_teams,
                'players_per_team': total_players // num_teams,
                'remainder': total_players % num_teams
            })
    
    return suggestions

def display_team_suggestions(total_players):
    """
    Afficher les suggestions de configuration d'équipes
    """
    st.subheader("🏁 Suggestions de Configuration d'Équipes")
    
    suggestions = suggest_team_configurations(total_players)
    
    for suggestion in suggestions:
        st.markdown(f"""
        - **{suggestion['num_teams']} équipes**
          * Joueurs par équipe : {suggestion['players_per_team']} 
          * Joueurs restants : {suggestion['remainder']}
        """)

def plan_tournament(teams, match_duration=20, break_duration=10):
    """
    Planifier un tournoi avec gestion des matchs et pauses
    """
    num_teams = len(teams)
    
    # Calcul du nombre de matchs
    total_matches = math.comb(num_teams, 2)
    
    # Estimation du temps total
    total_match_time = total_matches * match_duration
    total_break_time = (total_matches - 1) * break_duration
    total_tournament_time = total_match_time + total_break_time
    
    st.subheader("📅 Planification du Tournoi")
    
    st.markdown(f"""
    **Détails du Tournoi :**
    - Nombre d'équipes : {num_teams}
    - Nombre total de matchs : {total_matches}
    - Durée de chaque match : {match_duration} minutes
    - Pause entre les matchs : {break_duration} minutes
    
    **Estimation du Temps Total :** {total_tournament_time} minutes (environ {total_tournament_time/60:.1f} heures)
    """)
    
    # Générer un planning potentiel
    st.subheader("Planning Proposé")
    planning = []
    current_time = 0
    
    for match_num in range(total_matches):
        match_start = current_time
        match_end = match_start + match_duration
        
        planning.append({
            'Match': f"Match {match_num + 1}",
            'Début': f"{match_start} min",
            'Fin': f"{match_end} min"
        })
        
        current_time = match_end + break_duration
    
    st.dataframe(planning)

def modify_player_data(data):
    """
    Interface de modification des données des joueurs
    """
    st.subheader("🔧 Modification des Données des Joueurs")
    
    # Créer une copie modifiable des données
    modified_data = data.copy()
    
    # Sélectionner les joueurs à modifier
    selected_players = st.multiselect(
        "Sélectionner les joueurs à modifier", 
        modified_data['nom'].tolist()
    )
    
    if selected_players:
        for player in selected_players:
            st.write(f"### Modification de {player}")
            
            # Colonnes modifiables
            cols = st.columns(3)
            with cols[0]:
                new_technique = st.number_input(
                    f"Niveau Technique de {player}", 
                    min_value=1.0, 
                    max_value=5.0, 
                    value=modified_data.loc[modified_data['nom'] == player, 'technique'].values[0],
                    step=0.5
                )
            
            with cols[1]:
                new_physique = st.number_input(
                    f"Niveau Physique de {player}", 
                    min_value=1.0, 
                    max_value=5.0, 
                    value=modified_data.loc[modified_data['nom'] == player, 'physique'].values[0],
                    step=0.5
                )
            
            # Mettre à jour les données
            modified_data.loc[modified_data['nom'] == player, 'technique'] = new_technique
            modified_data.loc[modified_data['nom'] == player, 'physique'] = new_physique
    
    return modified_data

def generate_teams(players, num_teams):
    """
    Générer des équipes équilibrées
    """
    players['total'] = players['technique'] + players['physique']
    sorted_players = players.sort_values(by='total', ascending=False)

    # Ajouter des joueurs "joker" si nécessaire pour équilibrer les équipes
    while len(sorted_players) % num_teams != 0:
        joker_index = len(sorted_players)
        joker = pd.DataFrame([{
            "nom": f"Joker {joker_index + 1}", 
            "technique": 0, 
            "physique": 0, 
            "total": 0
        }])
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

def export_teams_to_excel(teams):
    """
    Exporter les équipes dans un fichier Excel en mémoire
    
    :param teams: Dictionnaire des équipes générées
    :return: Buffer de fichier Excel
    """
    try:
        # Créer un buffer en mémoire
        output = io.BytesIO()
        
        # Créer un writer Excel
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
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
        
        # Réinitialiser le pointeur du buffer
        output.seek(0)
        return output
    
    except Exception as e:
        st.error(f"Erreur lors de l'exportation : {e}")
        return None

def main():
    st.title("🥏 Gestion des Équipes Frisbee")
    st.write("Importez un fichier Excel ou CSV contenant les informations des joueurs pour démarrer.")
    
    # Importer le fichier
    uploaded_file = st.file_uploader(
        "Choisissez un fichier", 
        type=['xls', 'xlsx', 'csv'],
        help="Fichier Excel ou CSV avec les colonnes : nom, technique, physique"
    )
    
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
                    
                    # Détails supplémentaires
                    with st.expander("Détails du chargement"):
                        st.write("Mapping des colonnes :", column_mapping)
                        st.write(f"Nombre total de joueurs : {len(clean_data)}")
                    
                    # Modification des données
                    st.subheader("🔍 Préparation des Équipes")
                    if st.checkbox("Modifier les données des joueurs"):
                        clean_data = modify_player_data(clean_data)
                    
                    # Suggestions de configuration d'équipes
                    display_team_suggestions(len(clean_data))
                    
                    # Nombre d'équipes
                    col1, col2 = st.columns(2)
                    with col1:
                        num_teams = st.slider(
                            "Nombre d'équipes", 
                            min_value=2, 
                            max_value=min(10, len(clean_data)),
                            value=min(4, len(clean_data) // 2)
                        )
                    
                    with col2:
                        st.write(f"Joueurs par équipe : {len(clean_data) // num_teams}")
                    
                    # Générer les équipes
                    if st.button("🏁 Générer les Équipes"):
                        try:
                            teams = generate_teams(clean_data, num_teams)
                            
                            # Afficher les équipes
                            for team_id, team in teams.items():
                                st.subheader(f"Équipe {team_id + 1}")
                                team_df = pd.DataFrame(team)
                                st.dataframe(
                                    team_df[['nom', 'technique', 'physique', 'total']],
                                    use_container_width=True
                                )
                            
                            # Visualisation des scores des équipes
                            st.subheader("Comparaison des Scores des Équipes")
                            fig = plot_teams(teams)
                            st.pyplot(fig)
                            
                            # Option d'export
                            export_teams = st.checkbox("Exporter les équipes")
                            if export_teams:
                                # Générer le fichier Excel en mémoire
                                output = export_teams_to_excel(teams)
                                
                                if output is not None:
                                    st.download_button(
                                        label="📥 Télécharger les équipes",
                                        data=output,
                                        file_name="equipes_frisbee.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                    
                                    # Afficher les statistiques globales
                                    try:
                                        output.seek(0)
                                        stats_df = pd.read_excel(output, sheet_name="Statistiques Globales")
                                        st.subheader("Résumé des Équipes")
                                        st.dataframe(stats_df, use_container_width=True)
                                    except Exception as stats_error:
                                        st.warning(f"Impossible d'afficher les statistiques : {stats_error}")
                        except Exception as e:
                            st.error(f"Erreur lors de la génération des équipes : {e}")
            else:
                st.error("Impossible de valider les données. Veuillez vérifier votre fichier.")

if __name__ == "__main__":
    main()