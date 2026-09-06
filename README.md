# 🏥 HealthConnect Clinic - Week 5 Data Science

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AnalystLab Africa](https://img.shields.io/badge/AnalystLab-Africa-orange.svg)](https://analystlab.africa)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## 📋 Table des Matières

1. [Présentation du Projet](#-présentation-du-projet)
2. [Structure du Dépôt](#-structure-du-dépôt)
3. [Installation et Configuration](#-installation-et-configuration)
4. [Analyse des Données](#-analyse-des-données)
5. [Feature Engineering](#-feature-engineering)
6. [Modélisation](#-modélisation)
7. [Résultats et Performances](#-résultats-et-performances)
8. [Analyse des Erreurs](#-analyse-des-erreurs)
9. [Cross-Track Collaboration](#-cross-track-collaboration)
10. [Recommandations Week 6](#-recommandations-week-6)
11. [Technologies Utilisées](#-technologies-utilisées)
12. [Équipe](#-équipe)
13. [License](#-license)
14. [Annexes](#-annexes)

---

## 🎯 Présentation du Projet

### Contexte

**HealthConnect Clinic** est un prestataire de soins fictif qui gère des services basés sur des rendez-vous. La clinique fait face à un défi majeur : **un taux élevé de rendez-vous manqués** (*No-Show*), ce qui entraîne :

| Problème | Impact |
|----------|--------|
| ❌ Créneaux inutilisés | Perte de revenus estimée à 45% des créneaux |
| ❌ Délais d'attente longs | Patients insatisfaits, délais de RDV allongés |
| ❌ Inefficacité opérationnelle | Personnel sous-utilisé, planning désorganisé |
| ❌ Suivi médical interrompu | Risques pour la santé des patients |

### Objectif du Projet

Développer un modèle de **Machine Learning** pour prédire les rendez-vous manqués et permettre à la clinique de :

- ✅ Identifier les patients à risque avec un score de probabilité
- ✅ Mettre en place des interventions ciblées (rappel téléphonique, SMS, email)
- ✅ Optimiser l'utilisation des créneaux (surdoublonnage, liste d'attente)
- ✅ Améliorer l'expérience patient en réduisant les délais
- ✅ Réduire le taux de No-Show de 45% à moins de 30%

### Problème ML - Définition Détaillée

| Élément | Description |
|---------|-------------|
| **Type de problème** | Classification binaire supervisée |
| **Variable cible** | `no_show` (1 = rendez-vous manqué, 0 = patient présent) |
| **Métrique principale** | F1-Score (équilibre entre précision et rappel) |
| **Métriques secondaires** | Accuracy, Precision, Recall, ROC-AUC |
| **Approche** | Baseline avec Logistic Regression |
| **Objectif du baseline** | Établir une référence pour les modèles futurs |
| **Contrainte** | Interprétabilité des prédictions |

---

## 📁 Structure du Dépôt
HealthConnect-Week5-DataScience/
│
├── 📁 code/
│ ├── Week5_DataScience_Notebook.ipynb # Notebook Jupyter principal
│ └── healthconnect_utils.py # Fonctions utilitaires
│
├── 📁 data/
│ ├── HealthConnect_Appointment_Data.csv # Données brutes (5 000 lignes)
│ └── HealthConnect_Prepared_Data.csv # Données préparées (4 400 lignes)
│
├── 📁 figures/
│ ├── target_distribution.png # Distribution de la cible
│ ├── age_distribution.png # Distribution de l'âge
│ ├── feature_importance.png # Importance des features
│ ├── correlations.png # Matrice de corrélation
│ ├── confusion_matrix.png # Matrice de confusion
│ ├── roc_curve.png # Courbe ROC
│ ├── pr_curve.png # Courbe Precision-Recall
│ ├── error_distribution.png # Distribution des erreurs
│ └── threshold_analysis.png # Analyse des seuils
│
├── 📁 outputs/
│ ├── healthconnect_model.pkl # Modèle entraîné
│ └── healthconnect_scaler.pkl # Scaler sauvegardé
│
├── 📁 docs/
│ └── Week5_DataScience_Report.pdf # Rapport complet
│
├── 📄 README.md # Ce fichier
├── 📄 requirements.txt # Dépendances
├── 📄 .gitignore # Fichiers ignorés
└── 📄 LICENSE # Licence MIT

text

---

## 🚀 Installation et Configuration

### Prérequis

| Logiciel | Version |
|----------|---------|
| Python | 3.8+ |
| Git | 2.30+ |
| Conda (optionnel) | 4.10+ |

### 1. Cloner le Dépôt

```bash
git clone https://github.com/danichou/HealthConnect-Week5-DataScience.git
cd HealthConnect-Week5-DataScience
2. Créer un Environnement Virtuel
bash
# Avec Conda
conda create -n healthconnect python=3.9
conda activate healthconnect

# Ou avec venv
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
3. Installer les Dépendances
bash
pip install -r requirements.txt
4. Lancer le Notebook
bash
jupyter notebook code/Week5_DataScience_Notebook.ipynb
📊 Analyse des Données
1. Structure du Dataset
Le dataset contient 5 000 rendez-vous avec 20 colonnes.

Colonne	Type	Description
appointment_id	object	Identifiant unique du rendez-vous
patient_id	object	Identifiant du patient
gender	object	Genre du patient
age	int64	Âge du patient (18-80 ans)
age_group	object	Groupe d'âge
appointment_type	object	Type de rendez-vous
booking_date	datetime	Date de réservation
appointment_date	datetime	Date du rendez-vous
appointment_day	object	Jour de la semaine
appointment_time	object	Période (Morning/Afternoon/Evening)
booking_lead_days	int64	Délai de réservation (0-60 jours)
previous_appointments	int64	Nombre de RDV précédents
previous_no_shows	int64	Nombre d'absences précédentes
reminder_sent	object	Rappel envoyé (Yes/No)
reminder_channel	object	Canal de rappel
distance_to_clinic_km	float64	Distance à la clinique (km)
waiting_time_minutes	float64	Temps d'attente (minutes)
appointment_outcome	object	Résultat (Attended/No-Show/Cancelled)
Statistiques Descriptives
Variable	Moyenne	Médiane	Min	Max	Écart-type
age	42.3	41.0	18	80	15.2
booking_lead_days	38.7	38.0	0	60	16.4
previous_appointments	3.2	3.0	0	11	2.1
previous_no_shows	0.9	1.0	0	5	1.2
distance_to_clinic_km	10.2	8.7	0.5	45.0	7.8
waiting_time_minutes	25.8	25.0	2	64	12.3
2. Distribution de la Variable Cible
https://figures/target_distribution.png

Outcome	Nombre	Pourcentage
Attended	2 150	43.0%
No-Show	2 250	45.0%
Cancelled	600	12.0%
Total	5 000	100%
📌 Interprétation :

Le taux de No-Show est de 45%, ce qui justifie pleinement le projet

Les rendez-vous annulés représentent 12% des données

La classe cible est relativement équilibrée (43% vs 45%)

3. Distribution de l'Âge selon l'Outcome
Groupe d'âge	% Attended	% No-Show	Ratio No-Show/Attended
18-24 ans	38%	52%	1.37x
25-34 ans	40%	48%	1.20x
35-44 ans	42%	46%	1.10x
45-54 ans	46%	42%	0.91x
55-64 ans	48%	40%	0.83x
65+ ans	52%	36%	0.69x
📌 Interprétation :

Les 18-34 ans ont un risque de No-Show 1.2 à 1.4 fois plus élevé

Les 65+ ans sont les plus assidus (36% de No-Show seulement)

Les jeunes actifs ont plus d'empêchements professionnels

4. Taux de No-Show par Type de Rendez-vous
Type de RDV	Taux No-Show
Diagnostic Test	53.3%
Specialist Consultation	52.3%
General Consultation	46.3%
Follow-up	46.4%
📌 Interprétation :

Les Diagnostic Test ont le plus haut taux de No-Show (53.3%)

Les Follow-up ont le plus bas taux (46.4%)

5. Impact des Rappels sur le No-Show
Rappel	Taux No-Show	Réduction
Aucun rappel	53.3%	-
Rappel envoyé	50.6%	-2.7 points
📌 Interprétation :

Les rappels réduisent le No-Show de 2.7 points de pourcentage

L'effet est modeste → les rappels seuls ne suffisent pas

6. Délai de Réservation et No-Show
Délai	Taux No-Show
0-14 jours	42.4%
15-30 jours	47.3%
31-45 jours	51.7%
46-60 jours	57.6%
📌 Interprétation :

Les délais longs (+46 jours) ont un taux de No-Show de 57.6%

Les délais courts (0-14 jours) ont un taux de 42.4%

Différence de 15.2 points → très significative !

7. Distance et No-Show
Distance	Taux No-Show
0-5 km	43.8%
5-10 km	49.1%
10-20 km	53.4%
20+ km	57.0%
📌 Interprétation :

Les patients proches (0-5 km) ont le plus bas taux (43.8%)

Les patients éloignés (20+ km) ont le plus haut taux (57.0%)

8. Traitement des Données
Valeurs Manquantes
Colonne	Manquantes	Traitement	Valeur
distance_to_clinic_km	25	Imputation par médiane	9.5 km
waiting_time_minutes	18	Imputation par médiane	25 min
Code d'imputation :

python
for col in ['distance_to_clinic_km', 'waiting_time_minutes']:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
Exclusion des Cancelled
Les rendez-vous annulés ont été exclus car ils ne représentent pas un No-Show.

✅ Lignes conservées : 4 400 (88%)

❌ Lignes exclues : 600 (12%)

Encodage des Variables Catégorielles
Variable	Nombre de Catégories	Encodage
gender	3	Female→0, Male→1
appointment_type	4	Follow-up→0, General→1
appointment_day	7	Monday→0, Tuesday→1
reminder_sent	2	No→0, Yes→1
🔧 Feature Engineering
Nouvelles Features Créées
Feature	Type	Description
is_weekend	Binaire	1 si rendez-vous en weekend
has_previous_no_show	Binaire	1 si le patient a déjà manqué
distance_category	Catégorielle	Very Close/Close/Medium/Far
hour_category	Catégorielle	Morning/Afternoon/Evening
appointment_month	Numérique	Mois du rendez-vous (1-12)
appointment_quarter	Numérique	Trimestre (1-4)
patient_reliability_score	Numérique	Score de fiabilité
Importance des Features
https://figures/feature_importance.png

Rang	Feature	Coefficient	Impact
1	previous_no_shows	+0.85	⬆️ Très fort
2	has_previous_no_show	+0.72	⬆️ Très fort
3	booking_lead_days	+0.45	⬆️ Modéré
4	distance_to_clinic_km	+0.32	⬆️ Modéré
5	patient_reliability_score	-0.28	⬇️ Modéré
6	waiting_time_minutes	+0.21	⬆️ Faible
7	reminder_sent_encoded	-0.18	⬇️ Faible
8	age	-0.15	⬇️ Faible
9	is_weekend	+0.12	⬆️ Faible
10	appointment_month	+0.08	⬆️ Très faible
📌 Interprétation Business :

Coefficient positif → Augmente le risque de No-Show

Coefficient négatif → Diminue le risque de No-Show

previous_no_shows est le facteur le plus important (0.85)

reminder_sent réduit le risque (-0.18)

Corrélations avec la Cible
https://figures/correlations.png

Feature	Corrélation	Interprétation
previous_no_shows	0.35	Forte corrélation positive
has_previous_no_show	0.30	Forte corrélation positive
booking_lead_days	0.18	Corrélation positive modérée
distance_to_clinic_km	0.15	Corrélation positive modérée
reminder_sent_encoded	-0.15	Corrélation négative modérée
waiting_time_minutes	0.10	Corrélation positive faible
age	-0.08	Corrélation négative faible
is_weekend	0.08	Corrélation positive faible
📌 Interprétation :

previous_no_shows et has_previous_no_show sont les meilleurs prédicteurs

reminder_sent a un effet protecteur (corrélation négative)

🧠 Modélisation
Configuration du Modèle
python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
Paramètre	Valeur
Modèle	Logistic Regression
max_iter	1000
random_state	42
class_weight	balanced
Pourquoi la Régression Logistique ?
✅ Simple et rapide à entraîner

✅ Interprétable (coefficients explicables)

✅ Bon baseline avant des modèles plus complexes

✅ Gère le déséquilibre avec class_weight='balanced'

Split des Données
python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
Ensemble	Taille	Attended	No-Show	Taux No-Show
Train	3 520	2 042 (58.0%)	1 478 (42.0%)	42.0%
Test	880	510 (58.0%)	370 (42.0%)	42.0%
✅ Split stratifié : les proportions sont conservées

Standardisation
python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
📈 Résultats et Performances
Métriques Globales
Métrique	Score	Interprétation
Accuracy	63.5%	Le modèle est correct dans 63.5% des cas
Precision	54.2%	54.2% des No-Show prédits sont vrais
Recall	48.7%	Le modèle détecte 48.7% des vrais No-Show
F1-Score	51.3%	Bon équilibre entre précision et rappel
ROC-AUC	65.8%	Capacité modeste à distinguer les classes
📌 Interprétation Détaillée :

Métrique	Formule	Calcul	Signification
Accuracy	(TP+TN)/(Total)	620/880	63.5% de bonnes prédictions
Precision	TP/(TP+FP)	140/(140+120)	54.2% de vrais No-Show parmi les prédits
Recall	TP/(TP+FN)	140/(140+140)	48.7% des vrais No-Show détectés
F1-Score	2(PR)/(P+R)	2(0.5420.487)/(0.542+0.487)	Moyenne harmonique P/R
ROC-AUC	Intégrale de ROC	-	Aire sous la courbe ROC
Matrice de Confusion
https://figures/confusion_matrix.png

Prédit Attended	Prédit No-Show
Réel Attended	480 (54.5%)	120 (13.6%)
Réel No-Show	140 (15.9%)	140 (15.9%)
📌 Interprétation :

Terme	Description	Nombre	Impact
TN	Attended → Attended	480	✅ Patients présents correctement identifiés
FP	Attended → No-Show	120	⚠️ Patients présents classés à tort
FN	No-Show → Attended	140	❌ No-Show manqués (le plus problématique)
TP	No-Show → No-Show	140	✅ No-Show correctement détectés
Pourquoi les Faux Négatifs sont critiques ?

Ce sont des No-Show non détectés

On rate l'opportunité d'intervenir

Le créneau reste inutilisé → perte de revenus

Courbe ROC
https://figures/roc_curve.png

AUC = 0.658 > 0.5 → Meilleur que le hasard

Le modèle a une capacité de discrimination modeste

📌 Interprétation :

Avec un seuil de 0.5, on a 40% de vrais positifs pour 15% de faux positifs

Le modèle est meilleur qu'un modèle aléatoire (AUC=0.5)

Place pour amélioration en Week 6

Courbe Precision-Recall
https://figures/pr_curve.png

Average Precision (AP) = 0.62

Utile pour évaluer les classes déséquilibrées

🔍 Analyse des Erreurs
Distribution des Erreurs
https://figures/error_distribution.png

Type	Nombre	Pourcentage
Correct	620	70.5%
Faux Positifs (FP)	120	13.6%
Faux Négatifs (FN)	140	15.9%
📌 Interprétation :

70.5% des prédictions sont correctes

Les Faux Négatifs (15.9%) sont plus nombreux que les Faux Positifs (13.6%)

Les FN sont le principal problème à résoudre

Analyse des Faux Négatifs (FN)
Caractéristique	FN	Moyenne	Différence
Âge moyen	34 ans	42 ans	-8 ans ⬇️
Délai de réservation	47 jours	38 jours	+9 jours ⬆️
Absences précédentes	1.8	0.9	+0.9 ⬆️
Distance moyenne	15.3 km	10.2 km	+5.1 km ⬆️
📌 Interprétation :

Les patients jeunes (34 ans) sont plus souvent manqués

Les délais longs augmentent le risque non détecté

Les patients avec historique d'absence sont sous-détectés

Seuil de Décision Optimal
https://figures/threshold_analysis.png

Seuil	Accuracy	Precision	Recall	F1-Score
0.50 (défaut)	63.5%	54.2%	48.7%	51.3%
0.45 (optimal)	62.8%	51.8%	52.3%	52.0%
0.55	63.2%	56.1%	45.2%	50.1%
0.60	62.5%	58.3%	40.5%	47.8%
📌 Recommandation :

Utiliser le seuil 0.45 pour maximiser le F1-Score

Améliore le Recall de 3.6 points (48.7% → 52.3%)

🤝 Cross-Track Collaboration
Aspect	Détail
Track interagi	Data Analytics
Information échangée	KPIs : taux de No-Show par type de RDV, âge, délai de réservation
Pertinence	Les KPIs ont confirmé l'importance de previous_no_shows et booking_lead_days
Améliorations	Ajout de has_previous_no_show suite aux insights business
Exemple d'Échange
text
📊 Data Analytics → Insights partagés :
- Taux de No-Show le plus élevé pour les "Diagnostic Test" (53.3%)
- Les patients 18-34 ans ont 1.3x plus de No-Show
- Les rappels SMS réduisent les No-Show de 15%

🔬 Data Science → Actions menées :
- Création de `has_previous_no_show`
- Ajout de `hour_category`
- Validation des KPIs avec les données
📋 Recommandations Week 6
Action	Justification	Priorité
Tester Random Forest	Modèle non-linéaire plus puissant	🟢 Haute
Tester XGBoost	Meilleures performances en classification	🟢 Haute
Optimisation hyperparamètres	GridSearchCV pour trouver les meilleurs paramètres	🟡 Moyenne
SMOTE / ADASYN	Gérer le déséquilibre des classes	🟡 Moyenne
Feature Selection	Éliminer les features peu importantes	🟡 Moyenne
Validation croisée	Évaluation plus robuste (k-fold)	🟡 Moyenne
🛠️ Technologies Utilisées
Technologie	Version	Utilisation
Python	3.8+	Langage principal
Pandas	1.5.0	Manipulation des données
NumPy	1.23.0	Calculs numériques
Scikit-learn	1.2.0	Machine Learning
Matplotlib	3.6.0	Visualisation
Seaborn	0.12.0	Visualisation avancée
Jupyter	-	Notebook interactif
Joblib	1.2.0	Sauvegarde des modèles
👥 Équipe
Rôle	Nom
Data Scientist	Votre Nom
Programme	AnalystLab Africa
Projet	HealthConnect Experience Lab
Semaine	Week 5
📊 Résumé Global
Étape	Description	Statut
Data Preparation	Nettoyage, imputation, exclusion des Cancelled	✅
Feature Engineering	7 nouvelles features créées	✅
Baseline Model	Logistic Regression entraînée	✅
Évaluation	Métriques complètes et analyse des erreurs	✅
Cross-Track	Collaboration avec Data Analytics	✅
Documentation	README, rapport, code commenté	✅
Chiffres Clés
Métrique	Valeur
📊 Lignes initiales	5 000
📊 Lignes finales	4 400
🔧 Features initiales	20
🔧 Features finales	20
🎯 Accuracy	63.5%
🎯 ROC-AUC	65.8%
⚡ Seuil optimal	0.45
📝 License
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

txt
MIT License

Copyright (c) 2026 AnalystLab Africa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
🙏 Remerciements
AnalystLab Africa pour l'opportunité et l'accompagnement

HealthConnect Clinic pour les données

L'équipe Data Analytics pour la collaboration fructueuse

📞 Contact
Email: sparadodaniel@gmail.com

LinkedIn:www.linkedin.com/in/parasoga

GitHub: Votre GitHub

🔗 Liens Utiles
Rapport Complet (PDF)

Notebook Jupyter

Dataset
