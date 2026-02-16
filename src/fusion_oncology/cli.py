"""
Command-line interface for the Fusion Oncology Suite.

Usage examples::

    # Full pipeline with default settings
    fusion-oncology run

    # Quick smoke test (fewer iterations)
    fusion-oncology run --top-k 3 --fuzz-iterations 5

    # Only ingest and cache the dataset
    fusion-oncology ingest

    # Generate report from a previous run
    fusion-oncology report results/fusion_results.csv

    # Run companion diagnostic for a patient
    fusion-oncology companion-dx patient_mutations.json

    # Design CRISPR guide RNAs for top targets
    fusion-oncology crispr results/fusion_results.csv sequences.fasta

    # Simulate tumour response to treatment
    fusion-oncology simulate --drug Osimertinib --efficacy 0.15 --days 365

    # Query clinical evidence for a gene
    fusion-oncology evidence EGFR BRAF KRAS

    # Check resistance mechanisms
    fusion-oncology resistance EGFR ALK
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from fusion_oncology import __version__
from fusion_oncology.config import ProjectConfig
from fusion_oncology.utils.log import setup_logging

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(__version__, prog_name="fusion-oncology")
def main() -> None:
    """Fusion Oncology Suite — multi-modal cancer target discovery.

    Top-level Click group that registers all sub-commands
    (``run``, ``ingest``, ``report``, ``clear-cache``).
    """


# ── run ──────────────────────────────────────────────────────────────────


def _setup_run(
    top_k: int,
    fuzz_iterations: int,
    xgb_trees: int,
    xgb_depth: int,
    output_dir: str,
    log_level: str,
) -> ProjectConfig:
    """Setup logging and create project configuration.

    Parameters
    ----------
    top_k : int
        Number of top-ranked genes to analyse.
    fuzz_iterations : int
        Number of SNP-mutation iterations per gene.
    xgb_trees : int
        Number of XGBoost boosting rounds.
    xgb_depth : int
        Maximum tree depth for XGBoost.
    output_dir : str
        Filesystem path for CSV, figures, and reports.
    log_level : str
        Python logging level name.

    Returns
    -------
    ProjectConfig
        Configured project configuration object.
    """
    setup_logging(log_level)
    return ProjectConfig(
        top_k_genes=top_k,
        fuzz_iterations=fuzz_iterations,
        xgb_n_estimators=xgb_trees,
        xgb_max_depth=xgb_depth,
        output_dir=Path(output_dir),
    )


def _load_data(cfg: ProjectConfig) -> tuple:
    """Load TCGA Pan-Cancer patient data.

    Parameters
    ----------
    cfg : ProjectConfig
        Project configuration object.

    Returns
    -------
    tuple
        Tuple of (X, y) where X is feature matrix and y is labels.
    """
    from fusion_oncology.data.ingestion import DataIngestion

    console.print("[cyan]Ingesting TCGA Pan-Cancer data …[/cyan]")
    ingestor = DataIngestion(cfg)
    X, y = ingestor.get_patient_data()
    msg = f"  Loaded {X.shape[0]} samples x {X.shape[1]} genes, {y.nunique()} cancer types"
    console.print(msg)
    return X, y


def _run_analysis(X, y, cfg: ProjectConfig) -> tuple:
    """Run fusion engine analysis.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target labels.
    cfg : ProjectConfig
        Project configuration object.

    Returns
    -------
    tuple
        Tuple of (results DataFrame, engine object).
    """
    from fusion_oncology.models.fusion import FusionEngine

    console.print("[cyan]Running fusion analysis …[/cyan]")
    engine = FusionEngine(cfg)
    results = engine.run(X, y)
    return results, engine


def _save_results(results, cfg: ProjectConfig) -> None:
    """Save results to CSV file.

    Parameters
    ----------
    results : pd.DataFrame
        Fusion analysis results.
    cfg : ProjectConfig
        Project configuration object.
    """
    csv_path = cfg.output_dir / "fusion_results.csv"
    results.to_csv(csv_path, index=False)
    console.print(f"  Results saved → {csv_path}")


def _make_plots(results, cfg: ProjectConfig):
    """Generate and save plots, return figure dict."""
    from fusion_oncology.viz.plots import (
        fusion_bar,
        importance_vs_instability,
        save_figure,
    )

    console.print("[cyan]Generating figures …[/cyan]")
    fig1, fig2 = fusion_bar(results, cfg), importance_vs_instability(results)
    save_figure(fig1, "fusion_bar", cfg)
    save_figure(fig2, "importance_vs_instability", cfg)
    return {"Fusion Index Ranking": fig1, "Importance vs Instability": fig2}


def _generate_outputs(
    results, engine, cfg: ProjectConfig, skip_plots: bool, skip_report: bool
) -> None:
    """Generate plots and HTML report.

    Parameters
    ----------
    results : pd.DataFrame
        Fusion analysis results.
    engine : FusionEngine
        Fusion engine instance with cv_metrics.
    cfg : ProjectConfig
        Project configuration object.
    skip_plots : bool
        When True, skip matplotlib figure generation.
    skip_report : bool
        When True, skip HTML report assembly.
    """
    from fusion_oncology.viz.report import generate_html_report

    figures = {} if skip_plots else _make_plots(results, cfg)
    if not skip_report:
        console.print("[cyan]Generating HTML report …[/cyan]")
        generate_html_report(results, figures, engine.cv_metrics, cfg)


def _print_summary(results, engine) -> None:
    """Print results table and summary to console.

    Parameters
    ----------
    results : pd.DataFrame
        Fusion analysis results.
    engine : FusionEngine
        Fusion engine instance with summary method.
    """
    _print_results_table(results)
    console.print(engine.summary())
    console.rule("[bold green]DONE[/bold green]")


@main.command()
@click.option("--top-k", default=5, show_default=True, help="Number of top genes to analyse.")
@click.option(
    "--fuzz-iterations",
    default=20,
    show_default=True,
    help="Mutation iterations per gene.",
)
@click.option("--xgb-trees", default=50, show_default=True, help="XGBoost number of estimators.")
@click.option("--xgb-depth", default=4, show_default=True, help="XGBoost max tree depth.")
@click.option("--output-dir", default="results", type=click.Path(), help="Where to write results.")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
@click.option("--skip-plots", is_flag=True, help="Skip figure generation.")
@click.option("--skip-report", is_flag=True, help="Skip HTML report generation.")
def run(
    top_k: int,
    fuzz_iterations: int,
    xgb_trees: int,
    xgb_depth: int,
    output_dir: str,
    log_level: str,
    skip_plots: bool,
    skip_report: bool,
) -> None:
    """Run the full fusion analysis pipeline.

    Orchestrates data ingestion, XGBoost + DNABERT-2 fusion ranking,
    drug-target annotation, figure generation, and HTML reporting.

    Parameters
    ----------
    top_k : int
        Number of top-ranked genes to analyse.
    fuzz_iterations : int
        Number of SNP-mutation iterations per gene for instability
        scoring.
    xgb_trees : int
        Number of XGBoost boosting rounds (``n_estimators``).
    xgb_depth : int
        Maximum tree depth for XGBoost.
    output_dir : str
        Filesystem path for CSV, figures, and reports.
    log_level : str
        Python logging level name.
    skip_plots : bool
        When ``True``, skip matplotlib figure generation.
    skip_report : bool
        When ``True``, skip HTML report assembly.
    """
    cfg = _setup_run(top_k, fuzz_iterations, xgb_trees, xgb_depth, output_dir, log_level)
    console.rule("[bold blue]FUSION ONCOLOGY SUITE[/bold blue]")
    X, y = _load_data(cfg)
    results, engine = _run_analysis(X, y, cfg)
    _save_results(results, cfg)
    _generate_outputs(results, engine, cfg, skip_plots, skip_report)
    _print_summary(results, engine)


# ── ingest ───────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
def ingest(log_level: str) -> None:
    """Download and cache the TCGA dataset (no analysis).

    Parameters
    ----------
    log_level : str
        Python logging level name.
    """
    setup_logging(log_level)
    from fusion_oncology.data.ingestion import DataIngestion

    cfg = ProjectConfig()
    ing = DataIngestion(cfg)
    X, y = ing.get_patient_data()
    console.print(f"[green]Cached {X.shape[0]} samples x {X.shape[1]} genes[/green]")


# ── report ───────────────────────────────────────────────────────────────


def _load_results(csv_path: str, output_dir: str) -> tuple:
    """Load results CSV and create configuration.

    Parameters
    ----------
    csv_path : str
        Path to a previously saved fusion_results.csv.
    output_dir : str
        Directory in which to write the HTML file.

    Returns
    -------
    tuple
        Tuple of (results DataFrame, ProjectConfig).
    """
    import pandas as pd

    cfg = ProjectConfig(output_dir=Path(output_dir))
    results = pd.read_csv(csv_path)
    return results, cfg


def _generate_figures(results, cfg: ProjectConfig) -> dict:
    """Generate figures dictionary for report.

    Parameters
    ----------
    results : pd.DataFrame
        Fusion results DataFrame.
    cfg : ProjectConfig
        Project configuration object.

    Returns
    -------
    dict
        Dictionary mapping figure names to figure objects.
    """
    from fusion_oncology.viz.plots import fusion_bar, importance_vs_instability

    return {
        "Fusion Index Ranking": fusion_bar(results, cfg),
        "Importance vs Instability": importance_vs_instability(results),
    }


def _write_report(results, figures: dict, cfg: ProjectConfig) -> None:
    """Generate and print HTML report path.

    Parameters
    ----------
    results : pd.DataFrame
        Fusion results DataFrame.
    figures : dict
        Dictionary mapping figure names to figure objects.
    cfg : ProjectConfig
        Project configuration object.
    """
    from fusion_oncology.viz.report import generate_html_report

    path = generate_html_report(results, figures, config=cfg)
    console.print(f"[green]Report → {path}[/green]")


@main.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--output-dir", default="results", type=click.Path())
def report(csv_path: str, output_dir: str) -> None:
    """Regenerate the HTML report from a saved CSV.

    Parameters
    ----------
    csv_path : str
        Path to a previously saved ``fusion_results.csv``.
    output_dir : str
        Directory in which to write the HTML file.
    """
    results, cfg = _load_results(csv_path, output_dir)
    figures = _generate_figures(results, cfg)
    _write_report(results, figures, cfg)


# ── cache ────────────────────────────────────────────────────────────────


@main.command()
def clear_cache() -> None:
    """Delete all locally cached artefacts.

    Removes every file created by :class:`ArtifactCache` under the
    configured cache directory and prints the count of deleted items.
    """
    from fusion_oncology.data.cache import ArtifactCache

    cfg = ProjectConfig()
    n = ArtifactCache(cfg.cache_dir).clear()
    console.print(f"[yellow]Cleared {n} cached files[/yellow]")


# ── evidence ─────────────────────────────────────────────────────────────


def _query_gene_evidence(agg, gene: str) -> dict:
    """Query clinical evidence for a single gene.

    Parameters
    ----------
    agg : ClinicalEvidenceAggregator
        Evidence aggregator instance.
    gene : str
        HGNC gene symbol.

    Returns
    -------
    dict
        Evidence profile dictionary.
    """
    console.rule(f"[bold cyan]{gene}[/bold cyan]")
    return agg.profile(gene)


def _add_evidence_row(table, source, score, details):
    """Add a row to the evidence table."""
    table.add_row(source, f"{score:.3f}", details)


def _build_evidence_table(gene: str, profile: dict) -> Table:
    """Build Rich table from evidence profile.

    Parameters
    ----------
    gene : str
        HGNC gene symbol.
    profile : dict
        Evidence profile dictionary.

    Returns
    -------
    Table
        Rich table with evidence data.
    """
    table = Table(title=f"Clinical Evidence: {gene}")
    table.add_column("Source", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Details")
    ot = profile.get("opentargets", {})
    _add_evidence_row(
        table,
        "OpenTargets",
        ot.get("overall_score", 0),
        f"{len(ot.get('diseases', []))} disease associations",
    )
    civic_items = profile.get("civic", [])
    _add_evidence_row(
        table,
        "CIViC",
        min(len(civic_items) / 10, 1.0),
        f"{len(civic_items)} evidence items",
    )
    trial_items = profile.get("trials", [])
    _add_evidence_row(
        table,
        "ClinicalTrials.gov",
        min(len(trial_items) / 10, 1.0),
        f"{len(trial_items)} trials",
    )
    _add_evidence_row(table, "[bold]Composite[/bold]", profile.get("evidence_score", 0), "")
    return table


def _print_evidence_table(agg, gene: str) -> None:
    """Query and print evidence table for gene.

    Parameters
    ----------
    agg : ClinicalEvidenceAggregator
        Evidence aggregator instance.
    gene : str
        HGNC gene symbol.
    """
    profile = _query_gene_evidence(agg, gene)
    table = _build_evidence_table(gene, profile)
    console.print(table)


@main.command()
@click.argument("genes", nargs=-1, required=True)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
def evidence(genes: tuple[str, ...], log_level: str) -> None:
    """Query clinical evidence databases for one or more genes.

    Queries OpenTargets, CIViC, and ClinicalTrials.gov for each gene
    and prints a summary with composite evidence scores.

    Parameters
    ----------
    genes : tuple[str, ...]
        One or more HGNC gene symbols.
    log_level : str
        Python logging level name.
    """
    setup_logging(log_level)
    from fusion_oncology.analysis.clinical_evidence import ClinicalEvidenceAggregator

    cfg = ProjectConfig()
    agg = ClinicalEvidenceAggregator(cfg)
    for gene in genes:
        _print_evidence_table(agg, gene)


# ── resistance ───────────────────────────────────────────────────────────


def _query_resistance(pred, gene: str):
    """Query resistance mechanisms for a gene.

    Parameters
    ----------
    pred : ResistancePredictor
        Resistance predictor instance.
    gene : str
        HGNC gene symbol.

    Returns
    -------
    pd.DataFrame
        Resistance report DataFrame.
    """
    console.rule(f"[bold red]Resistance: {gene}[/bold red]")
    return pred.full_report([gene])


def _print_resistance_table(report, gene: str) -> None:
    """Print resistance mechanisms table.

    Parameters
    ----------
    report : pd.DataFrame
        Resistance report DataFrame.
    gene : str
        HGNC gene symbol.
    """
    if report.empty:
        console.print(f"  No known resistance mechanisms for {gene}")
        return
    table = Table()
    for col in report.columns:
        table.add_column(col)
    for _, row in report.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


@main.command()
@click.argument("genes", nargs=-1, required=True)
def resistance(genes: tuple[str, ...]) -> None:
    """Show known resistance mechanisms for one or more genes.

    Parameters
    ----------
    genes : tuple[str, ...]
        HGNC gene symbols to check.
    """
    from fusion_oncology.analysis.resistance import ResistancePredictor

    cfg = ProjectConfig()
    pred = ResistancePredictor(cfg)
    for gene in genes:
        report = _query_resistance(pred, gene)
        _print_resistance_table(report, gene)


# ── simulate ─────────────────────────────────────────────────────────────


def _setup_twin(drug, efficacy, days, resistance_rate, output_dir):
    """Create and configure DigitalTwin with regimen."""
    from fusion_oncology.models.digital_twin import (
        DigitalTwin,
        DrugRegimen,
        SimulationConfig,
    )

    cfg = ProjectConfig(output_dir=Path(output_dir))
    sim_cfg = SimulationConfig(simulation_days=days)
    twin = DigitalTwin(sim_config=sim_cfg, project_config=cfg)
    twin.add_regimen(
        DrugRegimen(
            name=drug,
            efficacy=efficacy,
            resistance_rate=resistance_rate,
            duration_days=days,
        )
    )
    return twin, cfg


def _run_simulation(
    drug: str, efficacy: float, days: int, resistance_rate: float, output_dir: str
) -> tuple:
    """Setup and run digital twin simulation.

    Parameters
    ----------
    drug : str
        Drug name for labelling.
    efficacy : float
        Kill rate constant (day⁻¹).
    days : int
        Duration of simulation.
    resistance_rate : float
        Rate of sensitive→resistant cell conversion.
    output_dir : str
        Directory for output CSV.

    Returns
    -------
    tuple
        Tuple of (DataFrame trajectory, summary dict, ProjectConfig).
    """
    twin, cfg = _setup_twin(drug, efficacy, days, resistance_rate, output_dir)
    console.rule("[bold blue]Digital Twin Simulation[/bold blue]")
    df = twin.simulate()
    summary = twin.summary()
    return df, summary, cfg


def _print_simulation_summary(drug: str, days: int, summary: dict) -> None:
    """Print simulation results table.

    Parameters
    ----------
    drug : str
        Drug name.
    days : int
        Simulation duration.
    summary : dict
        Simulation summary dictionary.
    """
    table = Table(title="Simulation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Drug", drug)
    table.add_row("Duration", f"{days} days")
    table.add_row("RECIST", summary["recist"])
    br = summary["best_response"]
    table.add_row("Best Response", f"{br['response_pct']:.1f}% on day {br['day']}")
    table.add_row("Final Tumour", f"{summary['final_tumour']:.2e}")
    console.print(table)


def _save_simulation(df, cfg: ProjectConfig) -> None:
    """Save simulation trajectory to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation trajectory DataFrame.
    cfg : ProjectConfig
        Project configuration object.
    """
    csv_path = cfg.output_dir / "simulation_trajectory.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]Trajectory saved → {csv_path}[/green]")


@main.command()
@click.option("--drug", required=True, help="Drug name for simulation.")
@click.option("--efficacy", default=0.15, show_default=True, help="Drug kill rate (day⁻¹).")
@click.option("--days", default=365, show_default=True, help="Simulation duration (days).")
@click.option(
    "--resistance-rate",
    default=0.001,
    show_default=True,
    help="Sensitive→resistant conversion rate.",
)
@click.option("--output-dir", default="results", type=click.Path())
def simulate(
    drug: str,
    efficacy: float,
    days: int,
    resistance_rate: float,
    output_dir: str,
) -> None:
    """Run a digital-twin tumour growth simulation.

    Simulates tumour response to a drug regimen using Gompertzian
    growth kinetics and outputs response trajectory data.

    Parameters
    ----------
    drug : str
        Drug name for labelling.
    efficacy : float
        Kill rate constant (day⁻¹).
    days : int
        Duration of simulation.
    resistance_rate : float
        Rate of sensitive→resistant cell conversion.
    output_dir : str
        Directory for output CSV.
    """
    df, summary, cfg = _run_simulation(drug, efficacy, days, resistance_rate, output_dir)
    _print_simulation_summary(drug, days, summary)
    _save_simulation(df, cfg)


# ── companion-dx ─────────────────────────────────────────────────────────


def _load_mutations(mutations_file: str, cancer_type: str, patient_id: str):
    """Load mutations and create patient profile.

    Parameters
    ----------
    mutations_file : str
        Path to JSON file with mutation list.
    cancer_type : str
        Cancer type label.
    patient_id : str
        Patient identifier string.

    Returns
    -------
    PatientProfile
        Patient profile object with mutations.
    """
    import json
    from fusion_oncology.models.companion_dx import PatientProfile

    with open(mutations_file) as f:
        mutations = json.load(f)
    return PatientProfile(patient_id=patient_id, mutations=mutations, cancer_type=cancer_type)


def _run_companion_analysis(patient, cfg: ProjectConfig) -> tuple:
    """Run companion diagnostic analysis.

    Parameters
    ----------
    patient : PatientProfile
        Patient profile object.
    cfg : ProjectConfig
        Project configuration object.

    Returns
    -------
    tuple
        Tuple of (results, report_text).
    """
    from fusion_oncology.models.companion_dx import CompanionDiagnostic

    dx = CompanionDiagnostic(cfg)
    results = dx.analyse(patient)
    report_text = dx.generate_report(results)
    return results, report_text


def _save_companion_report(report_text: str, patient_id: str, cfg: ProjectConfig) -> None:
    """Save and print companion diagnostic report.

    Parameters
    ----------
    report_text : str
        Generated report text.
    patient_id : str
        Patient identifier string.
    cfg : ProjectConfig
        Project configuration object.
    """
    console.print(report_text)
    report_path = cfg.output_dir / f"companion_dx_{patient_id}.txt"
    report_path.write_text(report_text)
    console.print(f"\n[green]Report saved → {report_path}[/green]")


@main.command(name="companion-dx")
@click.argument("mutations_file", type=click.Path(exists=True))
@click.option("--cancer-type", default="unknown", help="Cancer type label.")
@click.option("--patient-id", default="PATIENT_001", help="Patient identifier.")
@click.option("--output-dir", default="results", type=click.Path())
def companion_dx(
    mutations_file: str,
    cancer_type: str,
    patient_id: str,
    output_dir: str,
) -> None:
    """Run companion diagnostic analysis for a patient.

    Reads a JSON file of somatic mutations and produces a personalised
    treatment recommendation report.

    Parameters
    ----------
    mutations_file : str
        Path to a JSON file with a list of mutation dicts
        (each having ``gene`` and ``variant`` keys).
    cancer_type : str
        Cancer type label (e.g. ``LUAD``, ``BRCA``).
    patient_id : str
        Patient identifier string.
    output_dir : str
        Directory for report output.
    """
    cfg = ProjectConfig(output_dir=Path(output_dir))
    patient = _load_mutations(mutations_file, cancer_type, patient_id)
    results, report_text = _run_companion_analysis(patient, cfg)
    _save_companion_report(report_text, patient_id, cfg)


# ── helpers ──────────────────────────────────────────────────────────────


def _print_results_table(df: "pd.DataFrame") -> None:  # noqa: F821
    """Render *df* as a Rich table in the console.

    Parameters
    ----------
    df : pd.DataFrame
        Fusion results DataFrame whose columns become table headers.
    """
    table = Table(title="Fusion Oncology — Ranked Targets", show_lines=True)
    for col in df.columns:
        table.add_column(col, justify="right" if df[col].dtype in ("float64", "int64") else "left")
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


if __name__ == "__main__":
    main()
