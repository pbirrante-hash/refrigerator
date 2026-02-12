import os
import base64
import requests
import PIL.Image
from io import BytesIO
from dotenv import load_dotenv
import streamlit as st
from crewai import Agent, Task, Crew
import streamlit as st

st.write("Chiavi trovate nei Secrets:", list(st.secrets.keys()))

# 1. Configurazione ambiente
load_dotenv()
#api_key = st.secrets.get("GEMINI_API_KEY")
api_key = st.secrets.get("GEMINI_API_KEY")

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
else:
    st.error("Chiave API non trovata! Configurala nei Secrets di Streamlit.")
    st.stop()

st.set_page_config(page_title="Family Fridge Manager AI", layout="wide")

# --- DATABASE LOGICO: Lo Standard della Famiglia ---
# Questo serve all'agente come termine di paragone
FRIGO_IDEALE = """
- Latte: 2 litri (Soglia minima: 0.5 litri / 25%)
- Uova: 12 unità (Soglia minima: 4 unità)
- Yogurt: 8 vasetti (Soglia minima: 2 unità)
- Frutta (Mele/Banane): 2kg (Soglia minima: 0.5kg)
- Verdura di stagione: 3 tipologie (Soglia minima: 1)
- Formaggio spalmabile: 2 confezioni (Soglia minima: 0.5)
- Carne/Pesce: 2 pasti pronti (Soglia minima: 0)
- Succo di frutta: 1 litro (Soglia minima: 0.3 litri)
"""

# --- VISIONE: Funzione stabile tramite Requests ---
def analizza_quantita(image, api_key):
    # Proviamo l'endpoint v1 standard
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    prompt = "Analizza questa foto del frigo. Elenca i prodotti e le quantità rimanenti in un elenco puntato."
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_str}}
            ]
        }]
    }
    
    response = requests.post(url, json=payload)
    
    # DEBUG: Se dà ancora 404, proviamo l'endpoint v1beta
    if response.status_code == 404:
        url_beta = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key}"
        response = requests.post(url_beta, json=payload)

    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        st.error(f"Errore API {response.status_code}: {response.text}")
        return None

# --- INTERFACCIA STREAMLIT ---
st.title("🛒 Family Fridge Manager AI")
st.info("🎯 Obiettivo: Gestione scorte per una famiglia di 4 persone (2 Adulti, 2 Bambini)")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📸 Inquadra o carica la foto del frigo", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = PIL.Image.open(uploaded_file)
        st.image(img, use_container_width=True, caption="Foto scansionata")

with col2:
    if uploaded_file and st.button("📊 Genera Analisi e Lista Spesa"):
        with st.spinner("🕵️ Scansione visiva in corso..."):
            stato_attuale = analizza_quantita(img, api_key)
            
        if stato_attuale:
            st.subheader("👀 Rilevato nel frigo:")
            st.write(stato_attuale)
            
            with st.spinner("📝 L'agente sta preparando la lista della spesa..."):
                # Definiamo l'agente con il modello Gemini
                config_llm = "gemini/gemini-2.5-flash"

                manager = Agent(
                    role='Inventory Manager Domestico',
                    goal='Identificare cosa manca e cosa sta per finire rispetto allo standard familiare.',
                    backstory='Sei un assistente meticoloso esperto in economia domestica. Il tuo compito è evitare che una famiglia di 4 persone resti senza beni di prima necessità.',
                    llm=config_llm,
                    verbose=False # Cambia in True se vuoi vedere il ragionamento nel terminale
                )

                task_inventory = Task(
                    description=f"""
                    1. Analizza questo STATO ATTUALE: {stato_attuale} 
                    2. Confrontalo con il FRIGO IDEALE di una famiglia di 4 persone: {FRIGO_IDEALE}
                    3. Crea una LISTA DELLA SPESA divisa in:
                       - 🔴 MANCANTI: Prodotti fondamentali assenti o finiti.
                       - 🟡 IN ESAURIMENTO: Prodotti sotto la soglia minima o quasi finiti.
                    4. Aggiungi un consiglio breve per ottimizzare la spesa.
                    """,
                    agent=manager,
                    expected_output="Un report Markdown con lo stato delle scorte e la lista della spesa categorizzata."
                )

                # Creazione ed esecuzione della Crew
                crew = Crew(agents=[manager], tasks=[task_inventory])
                report_spesa = crew.kickoff()

                st.markdown("---")
                st.subheader("📋 Report Spesa Suggerito")
                st.markdown(report_spesa)
                
                # Bottone per scaricare il testo
                st.download_button(
                    label="📩 Scarica Lista Spesa",
                    data=str(report_spesa),
                    file_name="lista_spesa_famiglia.md",
                    mime="text/markdown"
                )
        else:

            st.warning("Non è stato possibile estrarre dati dalla foto. Riprova con un'immagine più chiara.")

