import pandas as pd
import matplotlib.pyplot as plt

# Charger les données depuis un fichier Excel
def load_data(file_path):
    try:
        players = pd.read_excel(file_path)
        return players
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier : {e}")

# Valider les données (niveaux compris entre 1 et 5, pas de valeurs manquantes)
def validate_data(players):
    if 'nom' not in players.columns or 'technique' not in players.columns or 'physique' not in players.columns:
        raise ValueError("Le fichier doit contenir les colonnes : nom, technique, physique.")
    players = players.dropna()  # Supprime les lignes vides
    players = players[(players['technique'] >= 1) & (players['technique'] <= 5)]
    players = players[(players['physique'] >= 1) & (players['physique'] <= 5)]
    return players

# Générer des équipes équilibrées
def generate_teams(players, num_teams):
    players['total'] = players['technique'] + players['physique']
    sorted_players = players.sort_values(by='total', ascending=False)

    # Ajout d'un joker si nécessaire
    if len(sorted_players) % num_teams != 0:
        joker = pd.DataFrame([{"nom": "Joker", "technique": 0, "physique": 0, "total": 0}])
        sorted_players = pd.concat([sorted_players, joker], ignore_index=True)

    teams = {i: [] for i in range(num_teams)}
    direction = 1

    for i, player in enumerate(sorted_players.itertuples()):
        team_idx = i % num_teams if direction == 1 else (num_teams - 1) - (i % num_teams)
        teams[team_idx].append(player)
        if i % num_teams == num_teams - 1:
            direction *= -1

    return teams

def export_teams_to_excel(teams, output_file="teams.xlsx"):
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for team_id, team in teams.items():
            team_df = pd.DataFrame(team)
            team_df.to_excel(writer, index=False, sheet_name=f"Équipe {team_id + 1}")
    return output_file
def plot_teams(teams):
    team_totals = {team_id: sum(player.total for player in team) for team_id, team in teams.items()}
    fig, ax = plt.subplots()
    ax.bar(team_totals.keys(), team_totals.values(), color='skyblue')
    ax.set_title("Comparaison des niveaux des équipes")
    ax.set_xlabel("Équipe")
    ax.set_ylabel("Score total (technique + physique)")
    return fig