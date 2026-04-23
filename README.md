# Prédiction du risque, de la responsabilité et de la tarification en assurance automobile basée sur l’IA

## Aperçu

Ce projet vise à prédire le niveau de risque d’un accident, évaluer la responsabilité et estimer l’impact sur la tarification de l’assurance à partir de descriptions textuelles non structurées d’accidents automobiles, en utilisant le traitement du langage naturel (NLP) et l’apprentissage automatique.

La solution intègre également un chatbot conversationnel basé sur l’intelligence artificielle, conçu pour assister les experts en sinistres et les analystes d’assurance lors de l’analyse des accidents, de la collecte des données et de la prise de décision.

---

## Objectifs

- Analyser automatiquement les descriptions libres d’accidents  
- Prédire les niveaux de risque d’accident  
- Fournir une évaluation préliminaire de la responsabilité  
- Estimer l’impact sur la prime d’assurance  
- Assister les experts en sinistres grâce à un chatbot conversationnel  
- Réduire le temps de traitement et améliorer la cohérence des évaluations  

---

## Fonctionnalités principales

- Extraction, par NLP, d’informations structurées à partir des récits d’accidents  
- Modèles de machine learning pour la prédiction du risque, de la responsabilité et de la tarification  
- Chatbot conversationnel pour soutenir les experts en sinistres et les enquêteurs  
- Pipeline d’analyse évolutif, reproductible et automatisé  
- Outil d’aide à la décision pour le traitement d’un grand volume de réclamations  

---

## Architecture du projet

data/
 ├── raw_reports/
 ├── processed_data/

models/
 ├── nlp_models/
 ├── pricing_model/

chatbot/
 ├── prompts/
 ├── conversation_logic/

src/
 ├── preprocessing/
 ├── feature_extraction/
 ├── prediction/
 ├── evaluation/

---

## Pile technologique

- Python  
- Traitement du langage naturel (NLP)  
- Apprentissage automatique (Machine Learning)  
- Grands modèles de langage (via Ollama)  
- Sorties structurées au format JSON  
- Intelligence artificielle conversationnelle  

---

## Indicateurs clés de performance (KPI)

- Réduction de 50 % du temps moyen d’analyse des réclamations  
- 85 % de précision analytique validée par des experts du domaine  
- 90 % des utilisateurs jugent l’interface intuitive  
- 80 % des demandes traitées sans intervention humaine  
- Temps de réponse inférieur à 2 secondes pour 95 % des interactions  

---

## Parties prenantes

Yasmine Ameur / Lara Abou-Arraj – Expertes en intelligence artificielle  

Responsables du développement des modèles NLP, de la validation ainsi que de l’évaluation des performances.

---

## Risques et hypothèses

### Hypothèses

- Les descriptions d’accidents contiennent suffisamment d’informations pour permettre une analyse automatisée fiable  
- La validation par des experts permet une amélioration continue du modèle  

### Risques

- Variabilité et qualité inégale des données textuelles  
- Difficulté d’interprétation des scénarios ambigus ou incomplets  

---

## Statut

Ce projet est actuellement en cours de développement et est destiné à des fins académiques et de recherche, avec un fort potentiel d’application dans le domaine de l’assurance et de la gestion des risques.
