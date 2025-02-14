import streamlit as st
from utils import load_data, validate_data, generate_teams
from utils import export_teams_to_excel
from utils import plot_teams

st.title("Création d'équipes pour Ultimate Frisbee 🥏")

# Importer le fichier Excel
uploaded_file = st.file_uploader("Téléchargez un fichier Excel", type=["xlsx"])

if uploaded_file:
    try:
        # Charger et valider les données
        players = load_data(uploaded_file)
        players = validate_data(players)

        st.success("Données importées avec succès !")
        st.write(players)

        # Nombre d'équipes
        num_teams = st.slider("Nombre d'équipes", min_value=2, max_value=10, value=4)

        if st.button("Générer les équipes"):
            teams = generate_teams(players, num_teams)
            for team_id, team in teams.items():
                st.subheader(f"Équipe {team_id + 1}")
                st.write(pd.DataFrame(team))
    except ValueError as e:
        st.error(f"Erreur : {e}")
        if st.button("Exporter les équipes"):
    try:
        output_file = export_teams_to_excel(teams)
        with open(output_file, "rb") as file:
            st.download_button(
                label="Télécharger les équipes (Excel)",
                data=file,
                file_name="teams.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Erreur lors de l'exportation : {e}")
if st.button("Afficher le graphique des équipes"):
    try:
        fig = plot_teams(teams)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique : {e}")