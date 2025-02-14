import streamlit as st
import pandas as pd
from utils import load_data, validate_data, generate_teams, export_teams_to_excel, plot_teams

# Titre principal
st.title("Création d'équipes pour Ultimate Frisbee 🥏")

# Téléchargement du fichier Excel
uploaded_file = st.file_uploader("Téléchargez un fichier Excel contenant les joueurs", type=["xlsx"])

if uploaded_file:
    try:
        # Charger et valider les données
        players = load_data(uploaded_file)
        players = validate_data(players)

        st.success("Données importées avec succès !")
        st.write(players)
    except ValueError as e:
        st.error(f"Erreur lors de l'importation ou validation des données : {e}")

        # Sélectionner le nombre d'équipes
        num_teams = st.slider("Choisissez le nombre d'équipes à générer", min_value=2, max_value=10, value=4)

        # Générer les équipes
        if st.button("Générer les équipes"):
            teams = generate_teams(players, num_teams)
            st.success("Les équipes ont été générées avec succès !")

            # Afficher les équipes
            for team_id, team in teams.items():
                st.subheader(f"Équipe {team_id + 1}")
                team_df = pd.DataFrame(team)
                st.write(team_df)

            # Exporter les équipes dans un fichier Excel
            if st.button("Exporter les équipes en fichier Excel"):
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

            # Afficher un graphique des équipes
            if st.button("Afficher le graphique des équipes"):
                try:
                    fig = plot_teams(teams)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erreur lors de la génération du graphique : {e}")

    except ValueError as e:
        st.error(f"Erreur : {e}")
else:
    st.info("Veuillez importer un fichier Excel pour commencer.")
