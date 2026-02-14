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


@main.command()
@click.option(
    "--top-k", default=5, show_default=True, help="Number of top genes to analyse."
)
@click.option(
    "--fuzz-iterations",
    default=20,
    show_default=True,
    help="Mutation iterations per gene.",
)
@click.option(
    "--xgb-trees", default=50, show_default=True, help="XGBoost number of estimators."
)
@click.option(
    "--xgb-depth", default=4, show_default=True, help="XGBoost max tree depth."
)
@click.option(
    "--output-dir", default="results", type=click.Path(), help="Where to write results."
)
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
    setup_logging(log_level)

    cfg = ProjectConfig(
        top_k_genes=top_k,
        fuzz_iterations=fuzz_iterations,
        xgb_n_estimators=xgb_trees,
        xgb_max_depth=xgb_depth,
        output_dir=Path(output_dir),
    )

    # Lazy imports to keep CLI startup fast
    from fusion_oncology.data.ingestion import DataIngestion
    from fusion_oncology.models.fusion import FusionEngine
    from fusion_oncology.viz.plots import (
        fusion_bar,
        importance_vs_instability,
        save_figure,
    )
    from fusion_oncology.viz.report import generate_html_report

    console.rule("[bold blue]FUSION ONCOLOGY SUITE[/bold blue]")

    # 1 — Data
    console.print("[cyan]Ingesting TCGA Pan-Cancer data …[/cyan]")
    ingestor = DataIngestion(cfg)
    X, y = ingestor.get_patient_data()
    console.print(
        f"  Loaded {X.shape[0]} samples × {X.shape[1]} genes, {y.nunique()} cancer types"
    )

    # 2 — Fusion analysis (includes drug-target, resistance, SL, network)
    console.print("[cyan]Running fusion analysis …[/cyan]")
    engine = FusionEngine(cfg)
    results = engine.run(X, y)

    # 3 — Save CSV
    csv_path = cfg.output_dir / "fusion_results.csv"
    results.to_csv(csv_path, index=False)
    console.print(f"  Results saved → {csv_path}")

    # 4 — Figures
    figures: dict = {}
    if not skip_plots:
        console.print("[cyan]Generating figures …[/cyan]")
        fig1 = fusion_bar(results, cfg)
        save_figure(fig1, "fusion_bar", cfg)
        fig2 = importance_vs_instability(results)
        save_figure(fig2, "importance_vs_instability", cfg)
        figures["Fusion Index Ranking"] = fusion_bar(results, cfg)
        figures["Importance vs Instability"] = importance_vs_instability(results)

    # 5 — HTML report
    if not skip_report:
        console.print("[cyan]Generating HTML report …[/cyan]")
        generate_html_report(results, figures, engine.cv_metrics, cfg)

    # 6 — Rich table to console
    _print_results_table(results)
    console.print(engine.summary())
    console.rule("[bold green]DONE[/bold green]")


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
    console.print(f"[green]Cached {X.shape[0]} samples × {X.shape[1]} genes[/green]")


# ── report ───────────────────────────────────────────────────────────────


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
    import pandas as pd

    from fusion_oncology.viz.plots import fusion_bar, importance_vs_instability
    from fusion_oncology.viz.report import generate_html_report

    cfg = ProjectConfig(output_dir=Path(output_dir))
    results = pd.read_csv(csv_path)
    figures = {
        "Fusion Index Ranking": fusion_bar(results, cfg),
        "Importance vs Instability": importance_vs_instability(results),
    }
    path = generate_html_report(results, figures, config=cfg)
    console.print(f"[green]Report → {path}[/green]")


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
        console.rule(f"[bold cyan]{gene}[/bold cyan]")
        profile = agg.profile(gene)
        table = Table(title=f"Clinical Evidence: {gene}")
        table.add_column("Source", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Details")

        ot = profile.get("opentargets", {})
        table.add_row(
            "OpenTargets",
            f"{ot.get('overall_score', 0):.3f}",
            f"{len(ot.get('diseases', []))} disease associations",
        )
        civic_items = profile.get("civic", [])
        table.add_row(
            "CIViC",
            f"{min(len(civic_items) / 10, 1.0):.3f}",
            f"{len(civic_items)} evidence items",
        )
        trial_items = profile.get("trials", [])
        table.add_row(
            "ClinicalTrials.gov",
            f"{min(len(trial_items) / 10, 1.0):.3f}",
            f"{len(trial_items)} trials",
        )
        table.add_row(
            "[bold]Composite[/bold]",
            f"[bold]{profile.get('evidence_score', 0):.3f}[/bold]",
            "",
        )
        console.print(table)


# ── resistance ───────────────────────────────────────────────────────────


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
        report = pred.full_report([gene])
        console.rule(f"[bold red]Resistance: {gene}[/bold red]")
        if report.empty:
            console.print(f"  No known resistance mechanisms for {gene}")
        else:
            table = Table()
            for col in report.columns:
                table.add_column(col)
            for _, row in report.iterrows():
                table.add_row(*[str(v) for v in row])
            console.print(table)


# ── simulate ─────────────────────────────────────────────────────────────


@main.command()
@click.option("--drug", required=True, help="Drug name for simulation.")
@click.option(
    "--efficacy", default=0.15, show_default=True, help="Drug kill rate (day⁻¹)."
)
@click.option(
    "--days", default=365, show_default=True, help="Simulation duration (days)."
)
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

    console.rule("[bold blue]Digital Twin Simulation[/bold blue]")
    df = twin.simulate()
    summary = twin.summary()

    table = Table(title="Simulation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Drug", drug)
    table.add_row("Duration", f"{days} days")
    table.add_row("RECIST", summary["recist"])
    table.add_row(
        "Best Response",
        f"{summary['best_response']['response_pct']:.1f}% on day "
        f"{summary['best_response']['day']}",
    )
    table.add_row("Final Tumour", f"{summary['final_tumour']:.2e}")
    console.print(table)

    csv_path = cfg.output_dir / "simulation_trajectory.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]Trajectory saved → {csv_path}[/green]")


# ── companion-dx ─────────────────────────────────────────────────────────


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
    import json

    from fusion_oncology.models.companion_dx import CompanionDiagnostic, PatientProfile

    cfg = ProjectConfig(output_dir=Path(output_dir))

    with open(mutations_file) as f:
        mutations = json.load(f)

    patient = PatientProfile(
        patient_id=patient_id,
        mutations=mutations,
        cancer_type=cancer_type,
    )

    dx = CompanionDiagnostic(cfg)
    results = dx.analyse(patient)
    report_text = dx.generate_report(results)

    console.print(report_text)

    report_path = cfg.output_dir / f"companion_dx_{patient_id}.txt"
    report_path.write_text(report_text)
    console.print(f"\n[green]Report saved → {report_path}[/green]")


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
        table.add_column(
            col, justify="right" if df[col].dtype in ("float64", "int64") else "left"
        )
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


if __name__ == "__main__":
    main()
