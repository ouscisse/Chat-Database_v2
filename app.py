import json
import tempfile
import csv
import streamlit as st
import pandas as pd
import re
from PIL import Image
from dotenv import load_dotenv

# LLM imports
from phi.model.google import Gemini
from phi.model.openai import OpenAIChat
from phi.model.deepseek import DeepSeekChat

from phi.agent.duckdb import DuckDbAgent
from phi.tools.pandas import PandasTools

# For clickable suggestions:
from streamlit_pills import pills  # pip install streamlit-pills

load_dotenv()

# 
im = Image.open("S-satelix.ico")
st.set_page_config(page_title="Satelix", 
                    page_icon=im,
                    layout="wide")

# --- Session state for chat messages ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- Sidebar ---
st.sidebar.image("Satelix1.png", width=250)
password = st.sidebar.text_input("Entrez le mot de passe", type="password")

if password == st.secrets["pwd"]:

    # Sidebar for LLM selection
    st.sidebar.title("⚙️ Configuration LLM")
    llm_choice = st.sidebar.selectbox(
        "Choisissez le modèle LLM:",
        ["Gemini", "OpenAI", "DeepSeek"],
        index=0
    )

    # Depending on the choice, ask for the relevant API key
    if llm_choice == "Gemini":
        gemini_key = st.sidebar.text_input("Saisissez votre clé API Gemini:", type="password")
        if gemini_key:
            st.session_state.gemini_key = gemini_key
            st.sidebar.success("Clé API Gemini sauvegardée !")
        else:
            st.sidebar.warning("Saisir une clé API Gemini pour poursuivre.")
    elif llm_choice == "OpenAI":
        openai_key = st.sidebar.text_input("Saisissez votre clé API OpenAI:", type="password")
        if openai_key:
            st.session_state.openai_key = openai_key
            st.sidebar.success("Clé API OpenAI sauvegardée !")
        else:
            st.sidebar.warning("Saisir une clé API OpenAI pour poursuivre.")
    else:  # DeepSeek
        deepseek_key = st.sidebar.text_input("Saisissez votre clé API DeepSeek:", type="password")
        if deepseek_key:
            st.session_state.deepseek_key = deepseek_key
            st.sidebar.success("Clé API DeepSeek sauvegardée !")
        else:
            st.sidebar.warning("Saisir une clé API DeepSeek pour poursuivre.")

    # Clear chat history button
    def clear_chat():
        st.session_state.chat_messages = []

    st.sidebar.text("Cliquez pour effacer l'historique du Chat !")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat)

    # --- Main App ---
    st.title("📊 Satelix Data Insights ✨")
    uploaded_file = st.file_uploader("Importez un fichier CSV ou Excel", type=["csv", "xlsx"])

    def preprocess_and_save(file):
        """Reads and cleans up the file, then saves to a temp CSV."""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
            else:
                st.error("Format de fichier non pris en compte. Téléchargez un fichier CSV ou Excel.")
                return None, None, None

            # Cleanup strings
            for col in df.select_dtypes(include=['object']):
                df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

            # Try to convert date/numeric
            for col in df.columns:
                if 'date' in col.lower():
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                elif df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except (ValueError, TypeError):
                        pass

            # Save as temp CSV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_path = temp_file.name
                df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

            return temp_path, df.columns.tolist(), df
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {e}")
            return None, None, None

    # Check if user has provided correct LLM API key
    have_key = (
        (llm_choice == "Gemini" and "gemini_key" in st.session_state)
        or (llm_choice == "OpenAI" and "openai_key" in st.session_state)
        or (llm_choice == "DeepSeek" and "deepseek_key" in st.session_state)
    )

    if uploaded_file is not None and have_key:
        temp_path, columns, df = preprocess_and_save(uploaded_file)

        if temp_path and columns and df is not None:
            st.write("Données téléchargées :")
            st.dataframe(df)

            # Build semantic_model for DuckDbAgent
            semantic_model = {
                "tables": [
                    {
                        "name": "uploaded_data",
                        "description": "Contains the uploaded dataset.",
                        "path": temp_path,
                    }
                ]
            }

            # Create model based on user choice
            if llm_choice == "Gemini":
                model = Gemini(model="gemini-1.5-flash", api_key=st.session_state.gemini_key)
            elif llm_choice == "OpenAI":
                model = OpenAIChat(model="gpt-3.5-turbo", api_key=st.session_state.openai_key)
            else:  # DeepSeek
                model = DeepSeekChat(api_key=st.session_state.deepseek_key)

            # Create DuckDbAgent
            duckdb_agent = DuckDbAgent(
                model=model,
                semantic_model=json.dumps(semantic_model),
                markdown=True,
                add_history_to_messages=False,
                followups=False,
                read_tool_call_history=False,
                system_prompt=(
                    "Vous êtes un expert analyste de données. "
                    "Quelle que soit la langue de la question, répondez UNIQUEMENT en français. "
                    "Ne répondez jamais en anglais ou dans une autre langue. "
                    "Fournissez des réponses claires et concises adaptées aux professionnels francophones. "
                    "Ne répondez jamais en anglais. "
                    "Ne fournissez pas de code SQL ou d'explication sur les requêtes."
                )
            )

            # --- Chat Interface ---
            st.subheader("💬 Explorer vos données avec Satelix LLM Insights ?🧐")

            # Display chat history
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # ----------------------------------------------------------
            #  1) Show Example Prompts (Suggestion Pills)
            # ----------------------------------------------------------
            # Example suggestions in FRENCH:
            suggestions_list = [
                "Quel est le nombre de colis traités par dépôt d'origine ?",
                "Nombre de colis traités par dépôt de destination ?",
                "Nombre de colis traités ?",
                "Quel est le délai moyen entre la création du colis et son archivage ?",
                "Top 5 des magasiniers les plus performants ?",
                "Nombre de colis par type de colis ?",
                "Quels sont les terminaux les plus utilisés ?",
                "Nombre de colis avec une date de péremption dépassée ?",
                "Date de la première et de la dernière archive ?"
            ]

            # A small header or instructions above
            st.markdown("**Suggestions de questions (cliquez pour lancer la requête) :**")

            # Use the pills component
            predefined_prompt_selected = pills(
                label="",
                options=suggestions_list,
                index=None,
                clearable=True
            )

            # ----------------------------------------------------------
            #  2) Chat Input
            # ----------------------------------------------------------
            # user_query can come either from typed text or from suggestion
            typed_query = st.chat_input("Tapez votre question ou utilisez une suggestion ci-dessus...")

            # Final user_input determination
            if typed_query and typed_query.strip():
                # If the user typed something, we use that
                user_query = typed_query.strip()
            elif predefined_prompt_selected:
                # Otherwise, if they clicked a pill
                user_query = predefined_prompt_selected
            else:
                user_query = None

            # If there's user input (from either source)
            if user_query:
                # Show it in chat as "user"
                with st.chat_message("user"):
                    st.markdown(user_query)

                # Append "Réponds en français uniquement" for extra assurance
                forced_query = user_query + " Réponds en français uniquement."

                # Call the DuckDbAgent
                try:
                    with st.spinner("Analyse en cours..."):
                        response_obj = duckdb_agent.run(forced_query)

                        if hasattr(response_obj, 'content'):
                            response_content = response_obj.content
                            # remove any code block with SQL
                            response_content = re.sub(r'Voici la requête SQL.*?```sql.*?```', '', response_content, flags=re.DOTALL)
                            response_content = re.sub(r'Here\'s the SQL query.*?```sql.*?```', '', response_content, flags=re.DOTALL)
                            response_content = re.sub(r'```sql.*?```', '', response_content, flags=re.DOTALL)
                            response_content = re.sub(r'```.*?```', '', response_content, flags=re.DOTALL)
                            response_content = re.sub(r'.*?requête SQL.*?\n', '', response_content, flags=re.DOTALL)
                            response_content = re.sub(r'.*?SQL query.*?\n', '', response_content, flags=re.DOTALL)
                            
                        else:
                            response_content = str(response_obj)

                    with st.chat_message("assistant"):
                        st.markdown(response_content)

                    st.session_state.chat_messages.append({"role": "assistant", "content": response_content})

                except Exception as e:
                    error_message = f"Erreur lors de la génération de la réponse: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_message})

    else:
        if not uploaded_file:
            st.info("Veuillez importer un fichier pour démarrer l'analyse.")
        elif not have_key:
            st.warning("Veuillez renseigner la clé API correspondant au LLM choisi.")
