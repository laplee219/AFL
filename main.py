"""
AFL Predictor — Command Line Interface

Usage:
    python main.py collect              Collect historical match data
    python main.py features             Build feature matrix from collected data
    python main.py train [--optuna]      Train prediction models
    python main.py predict [--round N]   Predict match outcomes
    python main.py bet [--round N]       Find value bet opportunities
    python main.py analysis [--round N]  Show margin distribution, spread coverage & CLV
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
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import settings
from src.models.train import AFLModel

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
                str(row["home_team"]), str(row["away_team"]),
                str(winner),
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
    import pandas as pd
    from src.pipeline.feedback_loop import Pipeline
    from src.betting.value import find_value_bets, format_value_bets, format_odds_comparison

    pipeline = Pipeline(model_version=model_version)

    # Fetch odds regardless of whether predictions exist
    odds = pipeline.odds_collector.get_best_odds()

    # Fetch predictions (may be empty if collect/features haven't been run)
    predictions = pipeline.predict(year, round_num)

    # Save opening odds snapshot for later CLV comparison
    if not odds.empty:
        try:
            rnd = round_num or (int(predictions["round"].iloc[0]) if not predictions.empty else None)
            yr = year or settings.data.current_season
            if rnd is not None:
                pipeline.odds_collector.save_odds_snapshot(yr, rnd, "opening", odds)
        except Exception:
            pass  # non-critical

    if predictions.empty and odds.empty:
        console.print("[yellow]No predictions or odds available. Run 'collect' then 'features', and check ODDS_API_KEY.[/yellow]")
        return

    # Compute value bets only when both are available
    value_bets = find_value_bets(predictions, odds, spread_sigma=getattr(pipeline.model, "margin_sigma", 30.0),
                                  min_edge=settings.betting.min_edge) if (not predictions.empty and not odds.empty) else pd.DataFrame()

    if not value_bets.empty:
        console.print(format_value_bets(value_bets))
    else:
        # No value bets — show full odds vs model comparison (or raw odds if no predictions)
        console.print(format_odds_comparison(predictions, odds))


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


@cli.command()
@click.option("--round", "round_num", default=None, type=int, help="Round number to analyse")
@click.option("--year", default=None, type=int, help="Season year")
@click.option("--model", "model_version", default=None, type=str, help="Model version to use")
def analysis(round_num, year, model_version):
    """Show margin distribution, spread coverage profile, and closing line value for a round."""
    import pandas as pd
    from src.pipeline.feedback_loop import Pipeline
    from src.betting.analysis import format_distribution_report

    console.print("[bold blue]Loading predictions and odds...[/bold blue]")
    pipeline = Pipeline(model_version=model_version)

    predictions = pipeline.predict(year, round_num)
    if predictions.empty:
        console.print("[yellow]No predictions available. Run 'collect' then 'features' first.[/yellow]")
        return

    yr = year or settings.data.current_season
    rnd = round_num or (int(predictions["round"].iloc[0]) if "round" in predictions else None)
    sigma = getattr(pipeline.model, "margin_sigma", 34.0) if pipeline.model else 34.0

    # Try to load historical closing odds for completed rounds
    closing_odds = pd.DataFrame()
    opening_odds = pd.DataFrame()
    if rnd is not None:
        closing_odds = pipeline.odds_collector.load_odds_snapshot(yr, rnd, "closing")
        opening_odds = pipeline.odds_collector.load_odds_snapshot(yr, rnd, "opening")

    # Prefer closing odds (most accurate CLV), then live odds, then opening
    odds = pipeline.odds_collector.get_best_odds()

    # Save opening snapshot if live odds are available and no opening exists yet
    if not odds.empty and opening_odds.empty and rnd is not None:
        try:
            pipeline.odds_collector.save_odds_snapshot(yr, rnd, "opening", odds)
        except Exception:
            pass

    # For CLV analysis, use closing odds if available (post-match review)
    display_odds = closing_odds if not closing_odds.empty else odds
    odds_label = "closing" if not closing_odds.empty else "live"

    report = format_distribution_report(predictions, display_odds, sigma=sigma)
    console.print(report)

    if not closing_odds.empty:
        console.print(f"[dim]Using saved closing odds for CLV analysis[/dim]")
    elif not opening_odds.empty and not odds.empty:
        console.print(f"[dim]Opening odds on file — run 'ingest --round {rnd}' to capture closing odds[/dim]")


if __name__ == "__main__":
    cli()
