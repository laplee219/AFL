"""
AFL Predictor — Command Line Interface

Usage:
    python main.py collect              Collect historical match data
    python main.py features             Build feature matrix from collected data
    python main.py train [--optuna]      Train prediction models
    python main.py predict [--round N]   Predict match outcomes
    python main.py bet [--round N]       Find value bet opportunities
    python main.py ingest --round N      Ingest completed round results
    python main.py monitor              Show model health status
    python main.py report [--round N]    Generate LLM analysis report
    python main.py status               Show system status
    python main.py performance          Show betting performance
    python main.py pipeline [--round N]  Run full pipeline end-to-end
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import settings

# Force UTF-8 output on Windows to avoid GBK encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console(force_terminal=True)


@click.group()
@click.version_option(version="0.1.0", prog_name="AFL Predictor")
def cli():
    """AFL Match Prediction & Value Betting System"""
    pass


@cli.command()
@click.option("--start-year", default=None, type=int, help="Start year for data collection")
@click.option("--end-year", default=None, type=int, help="End year for data collection")
def collect(start_year, end_year):
    """Collect historical match data from Squiggle API."""
    from src.pipeline.feedback_loop import Pipeline

    console.print("[bold blue]Collecting AFL match data...[/bold blue]")
    pipeline = Pipeline()
    matches = pipeline.collect_data(start_year, end_year)

    if not matches.empty:
        console.print(f"[green]✓ Collected {len(matches)} matches[/green]")

        # Show summary table
        table = Table(title="Data Summary by Year")
        table.add_column("Year", style="cyan")
        table.add_column("Matches", style="green")
        table.add_column("Completed", style="yellow")

        for year, group in matches.groupby("year"):
            completed = group["is_complete"].sum() if "is_complete" in group.columns else len(group)
            table.add_row(str(year), str(len(group)), str(completed))

        console.print(table)
    else:
        console.print("[red]✗ No data collected[/red]")


@cli.command()
def features():
    """Build feature matrix from collected match data."""
    from src.pipeline.feedback_loop import Pipeline

    console.print("[bold blue]Building feature matrix...[/bold blue]")
    pipeline = Pipeline()
    fm = pipeline.build_features()

    if isinstance(fm, tuple):
        fm = fm[0]

    if fm is not None and not fm.empty:
        n_features = len([c for c in fm.columns if c not in {
            "match_id", "year", "round", "date", "home_team", "away_team",
            "venue", "target_margin", "target_home_win"
        }])
        console.print(f"[green]✓ Feature matrix: {len(fm)} matches × {n_features} features[/green]")
    else:
        console.print("[red]✗ Failed to build features. Run 'collect' first.[/red]")


@cli.command()
@click.option("--optuna", is_flag=True, help="Use Optuna for hyperparameter optimization")
def train(optuna):
    """Train prediction models."""
    from src.pipeline.feedback_loop import Pipeline

    console.print("[bold blue]Training models...[/bold blue]")
    if optuna:
        console.print("[yellow]Using Optuna optimization (this may take a while)...[/yellow]")

    pipeline = Pipeline()
    model = pipeline.train_model(use_optuna=optuna)

    if model:
        console.print(f"[green]✓ Model trained: {model.version}[/green]")
        console.print(f"  Features: {len(model.feature_names)}")
    else:
        console.print("[red]✗ Training failed. Run 'collect' and 'features' first.[/red]")


@cli.command()
@click.option("--round", "round_num", default=None, type=int, help="Round number to predict")
@click.option("--year", default=None, type=int, help="Season year")
@click.option("--model", "model_version", default=None, type=str, help="Model version to use (e.g. v_20260222_130000)")
def predict(round_num, year, model_version):
    """Generate match predictions."""
    from src.pipeline.feedback_loop import Pipeline
    from src.models.predict import format_predictions

    pipeline = Pipeline(model_version=model_version)
    predictions = pipeline.predict(year, round_num)

    if not predictions.empty:
        output = format_predictions(predictions)
        console.print(output)

        # Also show as a table
        table = Table(title="Prediction Details")
        table.add_column("Home", style="cyan")
        table.add_column("Away", style="cyan")
        table.add_column("Winner", style="bold green")
        table.add_column("Margin", justify="right")
        table.add_column("Prob", justify="right")
        table.add_column("Confidence", justify="right")

        for _, row in predictions.iterrows():
            margin = row["ensemble_margin"]
            prob = row["ensemble_prob"]
            winner = row["home_team"] if margin > 0 else row["away_team"]
            win_prob = prob if margin > 0 else (1 - prob)

            table.add_row(
                row["home_team"], row["away_team"],
                winner,
                f"{abs(margin):.0f} pts",
                f"{win_prob:.0%}",
                f"{row['confidence']:.0%}",
            )

        console.print(table)
    else:
        console.print("[yellow]No predictions available. Ensure data is collected and model is trained.[/yellow]")


@cli.command()
@click.option("--round", "round_num", default=None, type=int, help="Round number")
@click.option("--year", default=None, type=int, help="Season year")
@click.option("--model", "model_version", default=None, type=str, help="Model version to use")
def bet(round_num, year, model_version):
    """Find value bet opportunities."""
    from src.pipeline.feedback_loop import Pipeline
    from src.betting.value import format_value_bets

    pipeline = Pipeline(model_version=model_version)
    value_bets = pipeline.find_bets(year, round_num)

    if not value_bets.empty:
        console.print(format_value_bets(value_bets))

        table = Table(title="Value Bets")
        table.add_column("Match", style="cyan")
        table.add_column("Bet On", style="bold green")
        table.add_column("Odds", justify="right")
        table.add_column("Model", justify="right")
        table.add_column("Book", justify="right")
        table.add_column("Edge", justify="right", style="green")
        table.add_column("EV", justify="right", style="bold green")
        table.add_column("Kelly", justify="right")

        for _, row in value_bets.iterrows():
            table.add_row(
                f"{row['home_team']} v {row['away_team']}",
                row["bet_on"],
                f"${row['decimal_odds']:.2f}",
                f"{row['model_prob']:.0%}",
                f"{row['bookmaker_prob']:.0%}",
                f"+{row['edge']*100:.1f}%",
                f"+{row['expected_value']*100:.1f}%",
                f"{row['kelly_fraction']*100:.1f}%",
            )

        console.print(table)
    else:
        console.print("[yellow]No value bets found. This could mean odds API is not configured or no edge exists.[/yellow]")


@cli.command()
@click.option("--round", "round_num", required=True, type=int, help="Round number to ingest")
@click.option("--year", default=None, type=int, help="Season year")
def ingest(round_num, year):
    """Ingest completed round results and update the system."""
    from src.pipeline.feedback_loop import Pipeline

    console.print(f"[bold blue]Ingesting results for Round {round_num}...[/bold blue]")
    pipeline = Pipeline()
    pipeline.ingest_results(year, round_num)
    console.print(f"[green]✓ Results ingested for R{round_num}[/green]")


@cli.command()
@click.option("--year", default=None, type=int, help="Season year")
@click.option("--round", "round_num", default=None, type=int, help="Current round")
def monitor(year, round_num):
    """Show model health and monitoring status."""
    from src.pipeline.monitor import ModelMonitor

    mon = ModelMonitor()
    year = year or settings.data.current_season
    round_num = round_num or 1

    status = mon.format_status(year, round_num)
    console.print(status)


@cli.command()
@click.option("--round", "round_num", default=None, type=int, help="Round number")
@click.option("--year", default=None, type=int, help="Season year")
def report(round_num, year):
    """Generate LLM analysis report for a round."""
    from src.pipeline.feedback_loop import Pipeline
    from src.llm.reporter import generate_round_report

    pipeline = Pipeline()
    predictions = pipeline.predict(year, round_num)

    if predictions.empty:
        console.print("[yellow]No predictions available for report generation.[/yellow]")
        return

    value_bets = pipeline.find_bets(year, round_num)

    console.print("[bold blue]Generating LLM analysis report...[/bold blue]")
    report_text, report_path = generate_round_report(
        predictions, value_bets,
        round_num=round_num,
        year=year or settings.data.current_season,
    )

    console.print(Panel(report_text, title="AFL Round Report", border_style="blue"))
    console.print(f"[green]Report saved to: {report_path}[/green]")


@cli.command()
def status():
    """Show system status."""
    from src.pipeline.feedback_loop import Pipeline

    pipeline = Pipeline()
    s = pipeline.get_status()

    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("Match Data", "✓ Available" if s["has_match_data"] else "✗ Not found")
    table.add_row("Feature Matrix", "✓ Available" if s["has_feature_matrix"] else "✗ Not found")
    table.add_row("Elo Ratings", "✓ Available" if s["has_elo"] else "✗ Not found")
    table.add_row("Saved Models", f"{s['n_saved_models']} model(s)")
    table.add_row("Latest Model", s.get("latest_model", "None") or "None")
    table.add_row("Bankroll", f"${s['bankroll']:.2f}")
    table.add_row(
        "Stop Loss",
        "[red]TRIGGERED[/red]" if s["stop_loss_triggered"] else "[green]OK[/green]",
    )

    console.print(table)


@cli.command()
def models():
    """List all saved model versions."""
    from config.settings import MODELS_DIR

    if not MODELS_DIR.exists():
        console.print("[yellow]No models directory found. Run 'train' first.[/yellow]")
        return

    versions = sorted(
        [d.name for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("v_")],
        reverse=True,
    )

    if not versions:
        console.print("[yellow]No saved models found.[/yellow]")
        return

    table = Table(title="Saved Models")
    table.add_column("#", style="dim")
    table.add_column("Version", style="cyan")
    table.add_column("Latest", style="green")

    for i, v in enumerate(versions, 1):
        is_latest = "(latest)" if i == 1 else ""
        table.add_row(str(i), v, is_latest)

    console.print(table)
    console.print(f"\nUse [cyan]--model {versions[0]}[/cyan] with predict/bet/pipeline to select a version.")


@cli.command()
def performance():
    """Show betting performance metrics."""
    from src.betting.tracker import BetTracker

    tracker = BetTracker()
    console.print(tracker.format_performance())


@cli.command()
@click.option("--round", "round_num", default=None, type=int, help="Round to predict")
@click.option("--year", default=None, type=int, help="Season year")
@click.option("--optuna", is_flag=True, help="Use Optuna optimization")
@click.option("--model", "model_version", default=None, type=str, help="Model version to use for prediction")
def pipeline(round_num, year, optuna, model_version):
    """Run the full prediction pipeline end-to-end."""
    from src.pipeline.feedback_loop import Pipeline

    console.print(Panel(
        "[bold]AFL PREDICTOR — FULL PIPELINE[/bold]",
        border_style="blue",
    ))

    pipe = Pipeline(model_version=model_version)
    result = pipe.run_full_pipeline(year, round_num, use_optuna=optuna)

    if result:
        console.print(f"\n[green]✓ Pipeline complete. Model: {result.get('model_version')}[/green]")
    else:
        console.print("[red]✗ Pipeline failed[/red]")


@cli.command()
@click.option("--round", "round_num", required=True, type=int, help="Round number")
@click.option("--year", default=None, type=int, help="Season year")
@click.option("--mode", type=click.Choice(["warmstart", "retrain"]), default="warmstart",
              help="Update mode: warmstart (quick) or retrain (full)")
def update(round_num, year, mode):
    """Update model with new data (warm-start or full retrain)."""
    from src.pipeline.feedback_loop import Pipeline

    pipeline = Pipeline()

    console.print(f"[bold blue]Updating model ({mode})...[/bold blue]")

    if mode == "retrain":
        pipeline.train_model()
    else:
        # Warm-start
        try:
            pipeline.model = AFLModel.load_latest()
        except FileNotFoundError:
            console.print("[red]No existing model to warm-start from. Use --mode retrain.[/red]")
            return

        from src.preprocessing.dataset import build_train_test_split
        fm = pipeline._load_feature_matrix()
        if fm.empty:
            console.print("[red]No feature matrix. Run 'features' first.[/red]")
            return

        data = build_train_test_split(fm)
        pipeline.model.warm_start_update(data)
        pipeline.model.save()

    console.print(f"[green]✓ Model updated ({mode})[/green]")


@cli.command()
@click.option("--season", required=True, type=int, help="Season to backtest")
def backtest(season):
    """Backtest model predictions against a historical season."""
    from src.pipeline.feedback_loop import Pipeline
    from src.models.evaluate import format_evaluation

    console.print(f"[bold blue]Backtesting season {season}...[/bold blue]")

    pipeline = Pipeline()
    fm = pipeline._load_feature_matrix()

    if fm.empty:
        console.print("[red]No feature data. Run 'collect' and 'features' first.[/red]")
        return

    # Get completed rounds for the season
    season_data = fm[(fm["year"] == season) & fm["target_margin"].notna()]
    if season_data.empty:
        console.print(f"[red]No completed matches for season {season}[/red]")
        return

    # Train on data before this season, predict this season
    from src.preprocessing.dataset import build_train_test_split

    data = build_train_test_split(fm, test_year=season, val_year=season - 1)
    model = AFLModel()
    model.train(data)

    # Predict each round
    from src.models.predict import Predictor
    from src.models.evaluate import evaluate_predictions

    predictor = Predictor(model)
    all_preds = []

    rounds = sorted(season_data["round"].unique())
    for rnd in rounds:
        preds = predictor.predict_round(fm, season, int(rnd))
        if not preds.empty:
            all_preds.append(preds)

    if all_preds:
        predictions = pd.concat(all_preds, ignore_index=True)

        # Evaluate
        actuals = season_data[["match_id", "target_margin", "target_home_win"]].rename(
            columns={"target_margin": "margin", "target_home_win": "home_win"}
        )
        metrics = evaluate_predictions(predictions, actuals)
        console.print(format_evaluation(metrics))
    else:
        console.print("[yellow]No predictions generated for backtest[/yellow]")


if __name__ == "__main__":
    import pandas as pd
    from src.models.train import AFLModel
    cli()
