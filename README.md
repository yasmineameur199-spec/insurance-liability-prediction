# AI-Based Car Insurance Risk, Liability and Pricing Prediction
## Overview

This project focuses on predicting accident risk, assessing liability, and estimating insurance pricing from unstructured car accident descriptions using Natural Language Processing (NLP) and machine learning techniques.

The solution also integrates a conversational AI chatbot designed to assist claims adjusters and insurance experts during accident analysis, data collection, and decision-making.

## Objectives

Automatically analyze free-text accident descriptions

Predict accident risk levels

Provide preliminary liability assessments

Estimate insurance pricing impact

Assist claims experts through a conversational AI chatbot

Reduce processing time and improve consistency in evaluations

## Key Features

NLP-based extraction of structured information from accident narratives

Machine learning models for risk, liability, and pricing prediction

Conversational chatbot to support claims adjusters and investigators

Scalable and reproducible analysis pipeline

Decision support for high-volume claim processing

## Project Architecture
data/
 ├── raw_reports/
 ├── processed_data/
models/
 ├── nlp_models
 ├── pricing_model/
chatbot/
 ├── prompts/
 ├── conversation_logic/
src/
 ├── preprocessing/
 ├── feature_extraction/
 ├── prediction/
 ├── evaluation/

## Tech Stack

Python

Natural Language Processing (NLP)

Machine Learning

Large Language Models (via Ollama)

JSON-based structured outputs

Conversational AI

## Key Performance Indicators (KPIs)

50% reduction in average claim analysis time

85% analytical accuracy validated by domain experts

90% of users rate the interface as intuitive

80% of requests handled without human intervention

Response time under 2 seconds for 95% of interactions

Stakeholders

Yasmine Ameur/ Lara Abou-Arraj – AI Expert
Responsible for NLP model development, validation, and performance evaluation

## Risks and Assumptions
### Assumptions

Accident descriptions contain sufficient information for reliable automated analysis

Expert validation enables continuous model improvement

### Risks

Variability and inconsistent quality of textual data

Challenges in interpreting ambiguous or incomplete accident scenarios

## Status

This project is under active development and intended for academic and research purposes, with potential applications in insurance and risk management.
