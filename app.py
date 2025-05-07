import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import itertools
import random
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

# Titel und Beschreibung
st.title('Australian Open Predictor 🎾')
st.write('Vorhersage von ATP-Matches basierend auf historischen Daten.')

# Daten laden
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2018.csv'
    data = pd.read_csv(url)
    return data

data = load_data()

# Datenvorverarbeitung
features = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age', 'winner_ht', 'loser_ht']
data = data.dropna(subset=features)
data['result'] = 1

losers = data.copy()
losers['winner_rank'] = data['loser_rank']
losers['loser_rank'] = data['winner_rank']
losers['winner_age'] = data['loser_age']
losers['loser_age'] = data['winner_age']
losers['winner_ht'] = data['loser_ht']
losers['loser_ht'] = data['winner_ht']
losers['result'] = 0

data = pd.concat([data, losers], ignore_index=True)

# Training und Testdaten
X = data[features]
y = data['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell erstellen
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Genauigkeit anzeigen
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f'Modellgenauigkeit: {accuracy * 100:.2f}%')

# ATP-Spielernamen und Rangliste hinzufügen
players_df = pd.read_csv('https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv', names=['id', 'firstname', 'lastname', 'hand', 'birthdate', 'country'])
rankings_df = pd.read_csv('https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv', names=['date', 'rank', 'player_id', 'points'])

# Spieler basierend auf realen Rankings und Altersverteilung auswählen
n_players = st.slider('Anzahl Spieler', 4, 128, 16)

players = []
for player_id in rankings_df['player_id'].unique()[:n_players]:
    player_row = players_df[players_df['id'] == player_id].iloc[0]
    rank_row = rankings_df[rankings_df['player_id'] == player_id].iloc[0]
    birthdate = player_row['birthdate']
    age = (datetime.now() - datetime.strptime(str(birthdate), '%Y%m%d')).days // 365
    height = random.randint(170, 210)
    players.append({
        'name': f"{player_row['firstname']} {player_row['lastname']}",
        'rank': rank_row['rank'],
        'age': age,
        'height': height
    })

# Turnierbaum erstellen
def plot_bracket(players, round_number):
    G = nx.DiGraph()
    for i, player in enumerate(players):
        G.add_node(player['name'])

    edges = []
    while len(players) > 1:
        next_round = []
        for a, b in zip(players[::2], players[1::2]):
            features = [[a['rank'], b['rank'], a['age'], b['age'], a['height'], b['height']]]
            prediction = model.predict(features)
            winner = a if prediction[0] == 1 else b
            G.add_edge(a['name'], winner['name'])
            G.add_edge(b['name'], winner['name'])
            next_round.append(winner)
        players = next_round
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
    st.pyplot(plt)

# Simulation starten
if st.button('Turnier simulieren'):
    st.subheader(f'🏆 Turnierbaum für {n_players} Spieler')
    plot_bracket(players, 1)

# Benutzeroberfläche für Einzelmatches
st.header('Einzelmatch-Vorhersage')
winner_rank = st.number_input('Ranking Spieler A', min_value=1, max_value=2000)
loser_rank = st.number_input('Ranking Spieler B', min_value=1, max_value=2000)
winner_age = st.slider('Alter Spieler A', 18, 40, 28)
loser_age = st.slider('Alter Spieler B', 18, 40, 28)
winner_ht = st.slider('Größe Spieler A (cm)', 160, 220, 185)
loser_ht = st.slider('Größe Spieler B (cm)', 160, 220, 185)

if st.button('Einzelmatch vorhersagen'):
    prediction = model.predict([[winner_rank, loser_rank, winner_age, loser_age, winner_ht, loser_ht]])
    if prediction[0] == 1:
        st.success('Spieler A wird voraussichtlich gewinnen!')
    else:
        st.error('Spieler B wird voraussichtlich gewinnen!')
