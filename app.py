import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# Configuration de base de Streamlit
st.set_page_config(page_title="Gestion des Équipes Frisbee", page_icon=":rugby_football:")

# Définir les alias pour les colonnes
COLUMN_ALIASES = {
    'nom': ['nom', 'name', 'joueur', 'player', 'prenom', 'firstname'],
    'technique': ['technique', 'skill', 'competence', 'capacite', 'tech'],
    'physique': ['physique', 'physical', 'condition', 'fitness', 'phys']
}

def fuzzy_column_match(data_columns):
    """
    Trouver des correspondances approximatives pour les noms de colonnes
    """
    column_mapping = {}
    unmatched_columns = []

    for req_col, aliases in COLUMN_ALIASES.items():
        # Normaliser les noms de colonnes existants
        normalized_data_columns = [col.lower().replace(' ', '') for col in data_columns]
        
        # Rechercher une correspondance
        matched = False
        for alias in aliases:
            normalized_alias = alias.lower().replace(' ', '')
            if normalized_alias in normalized_data_columns:
                # Trouver l'index de la colonne correspondante
                orig_col = data_columns[normalized_data_columns.index(normalized_alias)]
                column_mapping[req_col] = orig_col
                matched = True
                break
        
        if not matched:
            unmatched_columns.append(req_col)
    
    return column_mapping, unmatched_columns

def load_and_preview_data(uploaded_file):
    try:
        # Lire le fichier (support Excel et CSV)
        if uploaded_file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)
        
        st.info("Fichier chargé avec succès ! Voici un aperçu des premières lignes :")
        st.dataframe(data.head())  # Afficher les 5 premières lignes pour prévisualisation
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

def validate_columns(data):
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
    data_renamed = data.rename(columns=column_mapping)
    
    # Vérifier que toutes les colonnes requises sont présentes
    final_columns = ['nom', 'technique', 'physique']
    for col in final_columns:
        if col not in data_renamed.columns:
            st.error(f"La colonne '{col}' est manquante.")
            return False, None, None
    
    st.success("Toutes les colonnes nécessaires sont identifiées !")
    return True, data_renamed, column_mapping

def validate_and_clean_data(data):
    st.write("**Étape 2 : Validation et nettoyage des données**")
    
    # Convertir les colonnes en numérique, forcer les erreurs à NaN
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