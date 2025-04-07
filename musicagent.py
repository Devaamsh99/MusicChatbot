import os
import sqlite3
import re
import streamlit as st  # ✅ for accessing secrets.toml
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from typing import TypedDict, Optional, List, Tuple

# ✅ Use secrets from Streamlit's secure config
llm = AzureChatOpenAI(
    azure_endpoint=st.secrets["AZURE_ENDPOINT"],
    api_key=st.secrets["AZURE_API_KEY"],
    azure_deployment=st.secrets["DEPLOYMENT_NAME"],
    api_version="2024-02-01",
    temperature=0
)

serp_api = SerpAPIWrapper(serpapi_api_key=st.secrets["SERP_API_KEY"])


# --- Shared Utilities ---
def query_database(title=None, artist=None):
    conn = sqlite3.connect("music_library.db")
    cursor = conn.cursor()
    conditions = []
    params = []
    if title:
        conditions.append("title LIKE ?")
        params.append(f"%{title}%")
    if artist:
        conditions.append("artist LIKE ?")
        params.append(f"%{artist}%")
    if conditions:
        query = f"SELECT title, artist, file_path, lyrics FROM tracks WHERE {' OR '.join(conditions)}"
        cursor.execute(query, tuple(params))
        tracks = cursor.fetchall()
    else:
        tracks = []
    conn.close()
    return tracks

# --- Node: ChatDetectAgent ---
def chat_detect_agent(state):
    user_input = state["user_input"]
    detection_prompt = f"""
    Is the following user input asking for music-related trivia (e.g. about a person, history, or music fact)?
    Return only "trivia" or "track".
    Input: "{user_input}"
    """
    response = llm.invoke([{"role": "user", "content": detection_prompt}])
    category = "trivia" if "trivia" in response.content.lower() else "track"
    return {**state, "query_type": category}

# --- Node: DBAgent ---
def db_agent(state):
    user_input = state["user_input"]
    chat_response = llm.invoke([{"role": "user", "content": user_input}])
    extract_prompt = f"""
    Extract the song title and artist from the following response.
    Return in format: "Title: [song name] | Artist: [artist name]".
    Response: '{chat_response.content}'
    """
    track_response = llm.invoke([{"role": "user", "content": extract_prompt}])
    match = re.search(r'Title:\s*(.*?)\s*\|\s*Artist:\s*(.*)', track_response.content)
    title = match.group(1).strip() if match else None
    artist = match.group(2).strip() if match else None
    if not title:
        title_match = re.search(r'Title:\s*(.*)', track_response.content)
        title = title_match.group(1).strip() if title_match else None
    if not artist:
        artist_match = re.search(r'Artist:\s*(.*)', track_response.content)
        artist = artist_match.group(1).strip() if artist_match else None
    tracks = query_database(title, artist)
    return {
        **state,
        "extracted_title": title,
        "extracted_artist": artist,
        "db_result": tracks
    }

# --- Node: WebSearchAgent ---
def web_search_agent(state):
    title = state.get("extracted_title")
    artist = state.get("extracted_artist")
    user_input = state.get("user_input")
    queries = []
    if title:
        queries.append(f"{title} song by {artist if artist else ''}")
    if artist:
        queries.append(f"songs by {artist}")
    if not title and not artist:
        queries.append(user_input)
    new_title, new_artist = None, None
    new_tracks = []
    for q in queries:
        results = serp_api.run(q)
        extraction_prompt = f"""
        Extract a relevant song title and artist from the search results below:
        Format: "Title: [song name] | Artist: [artist name]"
        Results: {results}
        """
        response = llm.invoke([{"role": "user", "content": extraction_prompt}])
        match = re.search(r'Title:\s*(.*?)\s*\|\s*Artist:\s*(.*)', response.content)
        if match:
            new_title = match.group(1).strip()
            new_artist = match.group(2).strip()
            new_tracks = query_database(new_title, new_artist)
            if new_tracks:
                break
    return {
        **state,
        "extracted_title": new_title,
        "extracted_artist": new_artist,
        "db_result": new_tracks
    }

# --- Node: LyricsAgent (DB only) ---
def lyrics_agent(state):
    tracks = state.get("db_result", [])

    # For each track, ensure lyrics are present or set a placeholder
    updated_tracks = []
    for title, artist, path, lyrics in tracks:
        updated_tracks.append((
            title,
            artist,
            path,
            lyrics or "Lyrics not available in database."
        ))

    return {
        **state,
        "db_result": updated_tracks
    }

# --- Node: TriviaAgent ---
def trivia_agent(state):
    user_input = state.get("user_input", "")
    search_results = serp_api.run(user_input)
    trivia_prompt = f"""
    Based on the following search results, answer the user's music-related question or provide a fun fact.
    Be concise, accurate, and conversational.
    Results: {search_results}
    """
    trivia_response = llm.invoke([{"role": "user", "content": trivia_prompt}])
    return {
        **state,
        "trivia": trivia_response.content.strip()
    }

# --- Graph State ---
class GraphState(TypedDict):
    user_input: str
    query_type: Optional[str]
    extracted_title: Optional[str]
    extracted_artist: Optional[str]
    db_result: Optional[List[Tuple[str, str, str, str]]]
    trivia: Optional[str]

# --- Build Graph ---
graph = StateGraph(GraphState)

graph.add_node("DetectType", chat_detect_agent)
graph.add_node("TriviaSearch", trivia_agent)
graph.add_node("DBSearch", db_agent)
graph.add_node("WebSearch", web_search_agent)
graph.add_node("LyricsSearch", lyrics_agent)

# Flow Logic
graph.set_entry_point("DetectType")

graph.add_conditional_edges("DetectType", lambda s: "TriviaSearch" if s["query_type"] == "trivia" else "DBSearch")
graph.add_edge("TriviaSearch", "DBSearch")  # trivia-first flow
graph.add_conditional_edges("DBSearch", lambda s: "WebSearch" if not s["db_result"] else "LyricsSearch")
graph.add_edge("WebSearch", "LyricsSearch")
graph.add_edge("LyricsSearch", END)

# Finalize
graph_app = graph.compile()

# --- Main Execution ---
if __name__ == "__main__":
    user_query = input("Ask a music question: ")
    final_state = graph_app.invoke({"user_input": user_query})

    tracks = final_state.get("db_result", [])
    trivia = final_state.get("trivia")

    if trivia:
        print("\n🎧 Trivia:\n" + trivia)

    if tracks:
        print("\n🎵 Found Tracks:")
        for title, artist, path, lyrics in tracks:
            print(f"- {title} by {artist}")
            print(f"  🎤 Lyrics:\n{lyrics[:500]}...\n")
    else:
        print("\n😢 No tracks found.")

def run_music_agent(user_query: str):
    final_state = graph_app.invoke({"user_input": user_query})
    return final_state

