import streamlit as st
from musicagent import run_music_agent
import os

st.set_page_config(page_title="🎧 Music Chatbot", layout="wide")
st.title("🎶 Music Chatbot")

query = st.text_input("Ask me about music or request a song:", placeholder="e.g. Play Bohemian Rhapsody or Who is Freddie Mercury")

if query:
    state = run_music_agent(query)
    tracks = state.get("db_result", [])
    trivia = state.get("trivia")

    if trivia:
        st.subheader("🎸 Music Trivia")
        st.success(trivia)

    if tracks:
        st.subheader("🎵 Found Tracks")

        track_labels = [f"{i+1}. {title} by {artist}" for i, (title, artist, path, lyrics) in enumerate(tracks)]

        selected_index = st.selectbox(
            "Select a track to play:", 
            options=range(len(track_labels)), 
            format_func=lambda i: track_labels[i]
        )

        selected_track = tracks[selected_index]
        title, artist, path, lyrics = selected_track

        st.markdown(f"**Now Playing:** `{title}` by `{artist}`")

        if os.path.exists(path):
            st.audio(path, format="audio/mp3")
        else:
            st.error(f"⚠️ Audio file not found: {path}")

        if lyrics:
            st.markdown("### 🎤 Lyrics")
            st.text(lyrics[:1500])
        else:
            st.info("No lyrics available for this track.")
    else:
        st.warning("No tracks found for this input.")
