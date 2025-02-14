import pandas as pd
import matplotlib.pyplot as plt

# Charger les données depuis un fichier Excel
def load_data(file_path):
    try:
        players = pd.read_excel(file_path)

        # Vérifier que les colonnes essentielles existent
        required_columns = {'nom', 'technique', 'physique'}
        if not required_columns.issubset(players.columns):
            raise ValueError(f"Le fichier doit contenir les colonnes suivantes : {', '.join(required_columns)}")

        return players
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier : {e}")

# Valider les données (vérifications des colonnes et des valeurs)
def validate_data(players):
    if 'nom' not in players.columns or 'technique' not in players.columns or 'physique' not in players.columns:
        raise ValueError("Le fichier doit contenir les colonnes : nom, technique, physique.")

    # Supprimer les lignes vides
    players = players.dropna()

    # Convertir les colonnes "technique" et "physique" en nombres, forcer les erreurs à NaN
    for col in ['technique', 'physique']:
        players[col] = pd.to_numeric(players[col], errors='coerce')

    # Supprimer les lignes avec des valeurs non valides
    players = players.dropna(subset=['technique', 'physique'])

    # Vérifier les limites des scores
    players = players[(players['technique'] >= 1) & (players['technique'] <= 5)]
    players = players[(players['physique'] >= 1) & (players['physique'] <= 5)]

    return players


# Générer des équipes équilibrées
def generate_teams(players, num_teams):
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

# Exporter les équipes dans un fichier Excel
def export_teams_to_excel(teams, output_file="teams.xlsx"):
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for team_id, team in teams.items():
            team_df = pd.DataFrame(team)
            team_df.to_excel(writer, index=False, sheet_name=f"Équipe {team_id + 1}")
    return output_file

# Générer un graphique des équipes
def plot_teams(teams):
    # Calculer le score total de chaque équipe
    team_totals = {team_id: sum(player.total for player in team) for team_id, team in teams.items()}
    
    # Créer un graphique en barres
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(team_totals.keys(), team_totals.values(), color='skyblue')
    ax.set_title("Comparaison des scores totaux des équipes", fontsize=14)
    ax.set_xlabel("Équipe", fontsize=12)
    ax.set_ylabel("Score total (technique + physique)", fontsize=12)
    ax.set_xticks(list(team_totals.keys()))
    ax.set_xticklabels([f"Équipe {team_id + 1}" for team_id in team_totals.keys()])
    
    return fig
