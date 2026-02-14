"""
Fusion Oncology Suite
=====================

A multi-modal cancer genomics analysis platform fusing gradient-boosted
feature importance (XGBoost) with transformer-based DNA sequence
embeddings (DNABERT-2) to identify high-priority therapeutic targets.

Modules
-------
config      – Central configuration and runtime settings.
data        – Dataset retrieval, caching, and preprocessing.
models      – XGBoost and DNABERT model wrappers plus fusion logic.
analysis    – Instability scoring, pathway enrichment, survival hooks, drug-target mapping.
viz         – Publication-quality plots and HTML report generation.
cli         – Command-line interface powered by Click.
"""

__version__ = "0.1.0"
