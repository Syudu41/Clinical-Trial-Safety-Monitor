# Clinical Trial Safety Monitoring System

**Status: Enhancement in Progress**

A real-time adverse event processing system for clinical trial safety monitoring, built with cloud-native architecture and machine learning capabilities.

## System Overview

Real-time processing pipeline that monitors FDA adverse events, applies risk assessment algorithms, and generates safety alerts for clinical trial oversight.

## Architecture

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Data Stream   │    │   Processing    │
│                 │    │                 │    │                 │
│ • FDA API       │───▶│ Apache Kafka    │───▶│ AWS Lambda      │
│ • ClinicalTrials│    │ (Docker)        │    │ • Rule Engine   │
│ • 25K Records   │    │ 20 events/min   │    │ • Risk Scoring  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Output      │◀───│    Database     │◀───│   ML Models     │
│                 │    │                 │    │                 │
│ • Risk Alerts   │    │ PostgreSQL RDS  │    │ 86.28% Accuracy │
│ • Classifications│   │ • Events Table  │    │ Gradient Boost  │
│ • Safety Reports│    │ • Alerts Table  │    │ (Training Phase)│
└─────────────────┘    └─────────────────┘    └─────────────────┘




## Technology Stack
Data Processing: Python, Pandas, Scikit-learn
Streaming: Apache Kafka, Docker
Cloud Infrastructure: AWS Lambda, PostgreSQL RDS, S3
APIs: FDA OpenFDA, ClinicalTrials.gov v2.0

## Current Results

Dataset: 25,000 real FDA adverse event records processed
ML Performance: 86.28% accuracy (Gradient Boosting), 96.23% precision
Data Processing: 476MB raw data converted to 19 engineered features
Infrastructure: Deployed Lambda functions with PostgreSQL database
Streaming: Kafka pipeline handling 20 events/minute throughput

## Project Status

Data collection and preprocessing pipeline
Machine learning model training and evaluation
AWS cloud infrastructure deployment
Kafka streaming architecture
Database schema and connections