"""
AFL Predictor — Streamlit Web Dashboard

Run with: streamlit run app/streamlit_app.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import settings

st.set_page_config(
    page_title="AFL Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("🏈 AFL Predictor")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate",
    ["Dashboard", "Predictions", "Value Bets", "Model Health", "Betting Performance", "Reports", "Settings"],
)

year = st.sidebar.number_input("Season", min_value=2018, max_value=2030, value=settings.data.current_season)
round_num = st.sidebar.number_input("Round", min_value=1, max_value=28, value=1)


# ── Helper Functions ─────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_predictions(year: int, round_num: int):
    """Load or generate predictions."""
    try:
        from src.pipeline.feedback_loop import Pipeline
        pipeline = Pipeline()
        return pipeline.predict(year, round_num)
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_value_bets(year: int, round_num: int):
    """Load value bet recommendations."""
    try:
        from src.pipeline.feedback_loop import Pipeline
        pipeline = Pipeline()
        return pipeline.find_bets(year, round_num)
    except Exception as e:
        return pd.DataFrame()


def get_system_status():
    """Get current system status."""
    try:
        from src.pipeline.feedback_loop import Pipeline
        pipeline = Pipeline()
        return pipeline.get_status()
    except Exception as e:
        return {"error": str(e)}


def get_performance():
    """Get betting performance."""
    try:
        from src.betting.tracker import BetTracker
        tracker = BetTracker()
        return tracker.get_performance()
    except Exception:
        return {}


def get_monitoring_data(year: int):
    """Get monitoring metrics."""
    try:
        from src.pipeline.monitor import ModelMonitor
        monitor = ModelMonitor()
        return monitor.get_performance_trend(year)
    except Exception:
        return pd.DataFrame()


# ── Pages ────────────────────────────────────────────────────────────

if page == "Dashboard":
    st.title("AFL Predictor Dashboard")

    # Status cards
    status = get_system_status()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model", status.get("latest_model", "None") or "Not trained")
    with col2:
        st.metric("Saved Models", status.get("n_saved_models", 0))
    with col3:
        st.metric("Bankroll", f"${status.get('bankroll', 0):.2f}")
    with col4:
        data_status = "✓ Ready" if status.get("has_match_data") else "✗ No Data"
        st.metric("Data", data_status)

    st.markdown("---")

    # Quick predictions for current round
    st.subheader(f"Round {round_num} Predictions")
    predictions = load_predictions(year, round_num)

    if not predictions.empty:
        for _, row in predictions.iterrows():
            margin = row.get("ensemble_margin", 0)
            prob = row.get("ensemble_prob", 0.5)
            conf = row.get("confidence", 0)
            winner = row["home_team"] if margin > 0 else row["away_team"]
            win_prob = prob if margin > 0 else (1 - prob)

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{row['home_team']}** vs **{row['away_team']}**")
            with col2:
                st.write(f"🏆 {winner} by {abs(margin):.0f}")
            with col3:
                # Color code confidence
                color = "🟢" if conf > 0.4 else ("🟡" if conf > 0.2 else "🔴")
                st.write(f"{color} {win_prob:.0%}")
    else:
        st.info("No predictions available. Run the pipeline first: `python main.py pipeline`")

    # Quick actions
    st.markdown("---")
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Collect Data"):
            with st.spinner("Collecting data from Squiggle API..."):
                from src.pipeline.feedback_loop import Pipeline
                pipeline = Pipeline()
                matches = pipeline.collect_data()
                if not matches.empty:
                    st.success(f"Collected {len(matches)} matches")
                else:
                    st.error("No data collected")

    with col2:
        if st.button("🧠 Train Model"):
            with st.spinner("Training models..."):
                from src.pipeline.feedback_loop import Pipeline
                pipeline = Pipeline()
                model = pipeline.train_model()
                if model:
                    st.success(f"Model trained: {model.version}")
                else:
                    st.error("Training failed")

    with col3:
        if st.button("📊 Run Full Pipeline"):
            with st.spinner("Running full pipeline..."):
                from src.pipeline.feedback_loop import Pipeline
                pipeline = Pipeline()
                result = pipeline.run_full_pipeline(year, round_num)
                if result:
                    st.success("Pipeline complete!")
                    st.rerun()
                else:
                    st.error("Pipeline failed")


elif page == "Predictions":
    st.title(f"Match Predictions — {year} Round {round_num}")

    predictions = load_predictions(year, round_num)

    if not predictions.empty:
        # Predictions table
        display_cols = ["home_team", "away_team", "ensemble_margin", "ensemble_prob", "confidence"]
        available_cols = [c for c in display_cols if c in predictions.columns]
        st.dataframe(
            predictions[available_cols].rename(columns={
                "ensemble_margin": "Pred. Margin",
                "ensemble_prob": "Home Win %",
                "confidence": "Confidence",
            }),
            use_container_width=True,
        )

        # Confidence chart
        st.subheader("Prediction Confidence")
        if "confidence" in predictions.columns:
            fig = px.bar(
                predictions,
                x=predictions.apply(
                    lambda r: f"{r['home_team']} v {r['away_team']}", axis=1
                ),
                y="confidence",
                color="confidence",
                color_continuous_scale="RdYlGn",
                labels={"x": "Match", "confidence": "Confidence"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Margin chart
        st.subheader("Predicted Margins")
        if "ensemble_margin" in predictions.columns:
            fig = px.bar(
                predictions,
                x=predictions.apply(
                    lambda r: f"{r['home_team']} v {r['away_team']}", axis=1
                ),
                y="ensemble_margin",
                color="ensemble_margin",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                labels={"ensemble_margin": "Home Margin"},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Individual model comparison
        st.subheader("Model Comparison")
        model_cols = [c for c in ["xgb_prob", "lgb_prob", "lr_prob", "ensemble_prob"]
                      if c in predictions.columns]
        if model_cols:
            comparison_data = predictions[["home_team", "away_team"] + model_cols].copy()
            comparison_data["match"] = comparison_data.apply(
                lambda r: f"{r['home_team']} v {r['away_team']}", axis=1
            )
            st.dataframe(comparison_data[["match"] + model_cols], use_container_width=True)
    else:
        st.warning("No predictions available for this round.")


elif page == "Value Bets":
    st.title(f"Value Bets — {year} Round {round_num}")

    value_bets = load_value_bets(year, round_num)

    if not value_bets.empty:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Value Bets Found", len(value_bets))
        with col2:
            st.metric("Avg EV", f"+{value_bets['expected_value'].mean()*100:.1f}%")
        with col3:
            st.metric("Avg Edge", f"+{value_bets['edge'].mean()*100:.1f}%")

        # Value bets table
        st.subheader("Recommendations")
        display_df = value_bets[[
            "home_team", "away_team", "bet_on", "decimal_odds",
            "model_prob", "bookmaker_prob", "edge", "expected_value", "kelly_fraction",
        ]].copy()

        display_df.columns = [
            "Home", "Away", "Bet On", "Odds",
            "Model %", "Book %", "Edge", "EV", "Kelly %",
        ]

        # Format percentages
        for col in ["Model %", "Book %"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0%}")
        for col in ["Edge", "EV"]:
            display_df[col] = display_df[col].apply(lambda x: f"+{x*100:.1f}%")
        display_df["Kelly %"] = display_df["Kelly %"].apply(lambda x: f"{x*100:.1f}%")
        display_df["Odds"] = display_df["Odds"].apply(lambda x: f"${x:.2f}")

        st.dataframe(display_df, use_container_width=True)

        # Edge visualization
        st.subheader("Edge vs Bookmaker")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Model Probability",
            x=value_bets["bet_on"],
            y=value_bets["model_prob"],
            marker_color="green",
        ))
        fig.add_trace(go.Bar(
            name="Bookmaker Implied",
            x=value_bets["bet_on"],
            y=value_bets["bookmaker_prob"],
            marker_color="red",
        ))
        fig.update_layout(barmode="group", yaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)

        # Kelly sizing
        st.subheader("Bet Sizing (Quarter-Kelly)")
        perf = get_performance()
        bankroll = perf.get("current_bankroll", settings.betting.initial_bankroll)

        sizing_data = value_bets[["bet_on", "kelly_fraction", "expected_value"]].copy()
        sizing_data["recommended_stake"] = (sizing_data["kelly_fraction"] * bankroll).round(2)
        sizing_data.columns = ["Bet On", "Kelly %", "EV", "Stake ($)"]
        sizing_data["Kelly %"] = sizing_data["Kelly %"].apply(lambda x: f"{x*100:.1f}%")
        sizing_data["EV"] = sizing_data["EV"].apply(lambda x: f"+{x*100:.1f}%")
        st.dataframe(sizing_data, use_container_width=True)
    else:
        st.info("No value bets found for this round. This could mean:\n"
                "- Odds API is not configured\n"
                "- No sufficient edge exists\n"
                "- Model is not trained yet")


elif page == "Model Health":
    st.title("Model Health Monitor")

    monitoring = get_monitoring_data(year)

    if not monitoring.empty:
        # Key metrics
        latest = monitoring.iloc[-1] if not monitoring.empty else {}

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            acc = latest.get("accuracy", 0) if isinstance(latest, (dict, pd.Series)) else 0
            st.metric("Latest Accuracy", f"{acc:.0%}")
        with col2:
            ll = latest.get("log_loss", 0) if isinstance(latest, (dict, pd.Series)) else 0
            st.metric("Latest Log Loss", f"{ll:.4f}")
        with col3:
            brier = latest.get("brier_score", 0) if isinstance(latest, (dict, pd.Series)) else 0
            st.metric("Brier Score", f"{brier:.4f}")
        with col4:
            mae = latest.get("margin_mae", 0) if isinstance(latest, (dict, pd.Series)) else 0
            st.metric("Margin MAE", f"{mae:.1f} pts")

        # Accuracy trend
        st.subheader("Accuracy Trend")
        fig = px.line(
            monitoring, x="round", y="accuracy",
            markers=True,
            labels={"round": "Round", "accuracy": "Accuracy"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                      annotation_text="Random baseline")
        fig.add_hline(y=settings.model.accuracy_alert_threshold, line_dash="dot",
                      line_color="orange", annotation_text="Alert threshold")
        st.plotly_chart(fig, use_container_width=True)

        # Log loss trend
        if "log_loss" in monitoring.columns:
            st.subheader("Log Loss Trend")
            fig = px.line(
                monitoring, x="round", y="log_loss",
                markers=True,
                labels={"round": "Round", "log_loss": "Log Loss"},
            )
            fig.add_hline(y=0.693, line_dash="dash", line_color="red",
                          annotation_text="Coin flip baseline")
            st.plotly_chart(fig, use_container_width=True)

        # Retrain triggers
        if "retrain_triggered" in monitoring.columns:
            retrains = monitoring[monitoring["retrain_triggered"] == 1]
            if not retrains.empty:
                st.warning(f"Retrains triggered in rounds: {retrains['round'].tolist()}")
    else:
        st.info("No monitoring data available. Predictions need to be evaluated first.")

    # Retrain check
    st.subheader("Retrain Check")
    if st.button("Check if retraining needed"):
        from src.pipeline.monitor import ModelMonitor
        monitor = ModelMonitor()
        check = monitor.check_retrain_needed(year, round_num)
        if check["should_retrain"]:
            st.warning(f"⚠️ Retraining recommended: {check['reason']}")
        else:
            st.success("✅ Model is performing within acceptable bounds")
        st.json(check["diagnostics"])


elif page == "Betting Performance":
    st.title("Betting Performance")

    perf = get_performance()

    if perf.get("n_bets", 0) > 0:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Bankroll", f"${perf['current_bankroll']:.2f}",
                       delta=f"{perf['bankroll_growth']:+.1%}")
        with col2:
            st.metric("Total Bets", perf["n_bets"])
        with col3:
            st.metric("Win Rate", f"{perf['win_rate']:.0%}")
        with col4:
            st.metric("ROI", f"{perf['roi']:+.1%}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total P/L", f"${perf['total_profit']:+.2f}")
        with col2:
            st.metric("Yield", f"{perf['yield_pct']:+.1f}%")
        with col3:
            st.metric("Avg Odds", f"${perf['avg_odds']:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{perf['max_drawdown']:.1%}")

        # Bet history
        st.subheader("Bet History")
        try:
            from src.betting.tracker import BetTracker
            tracker = BetTracker()
            history = tracker.get_bet_history(100)
            if not history.empty:
                st.dataframe(history, use_container_width=True)
        except Exception:
            pass

        # Bankroll chart
        st.subheader("Bankroll Over Time")
        try:
            from src.utils.helpers import df_from_db
            bets = df_from_db(
                "SELECT * FROM bets WHERE bankroll_after IS NOT NULL ORDER BY id"
            )
            if not bets.empty:
                fig = px.line(
                    bets, x=bets.index, y="bankroll_after",
                    labels={"x": "Bet #", "bankroll_after": "Bankroll ($)"},
                )
                fig.add_hline(
                    y=settings.betting.initial_bankroll,
                    line_dash="dash", line_color="gray",
                    annotation_text="Starting bankroll"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
    else:
        st.info("No bets placed yet. Use the CLI to find and record bets.")


elif page == "Settings":
    st.title("Settings")

    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Retrain every:** {settings.model.retrain_every_n_rounds} rounds")
        st.write(f"**Max depth:** {settings.model.max_depth}")
        st.write(f"**Estimators:** {settings.model.n_estimators}")
        st.write(f"**Learning rate:** {settings.model.full_train_lr}")
    with col2:
        st.write(f"**Elo K-factor:** {settings.model.elo_k_factor}")
        st.write(f"**Elo home advantage:** {settings.model.elo_home_advantage}")
        st.write(f"**Elo season regression:** {settings.model.elo_season_regression}")
        st.write(f"**Early stopping:** {settings.model.early_stopping_rounds} rounds")

    st.subheader("Betting Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Initial bankroll:** ${settings.betting.initial_bankroll:.2f}")
        st.write(f"**Kelly fraction:** {settings.betting.kelly_fraction}")
        st.write(f"**Max bet:** {settings.betting.max_bet_fraction:.0%} of bankroll")
    with col2:
        st.write(f"**Min EV threshold:** {settings.betting.min_ev_threshold:.0%}")
        st.write(f"**Stop loss:** {settings.betting.stop_loss_fraction:.0%} of initial")

    st.subheader("Data Configuration")
    st.write(f"**Data start year:** {settings.data.data_start_year}")
    st.write(f"**Current season:** {settings.data.current_season}")
    st.write(f"**Sample weight decay:** {settings.data.sample_weight_decay}")

    st.subheader("LLM Configuration")
    st.write(f"**Provider:** {settings.llm.llm_provider}")
    st.write(f"**Model:** {settings.llm.llm_model}")
    api_configured = bool(settings.llm.anthropic_api_key or settings.llm.openai_api_key)
    st.write(f"**API Key:** {'✓ Configured' if api_configured else '✗ Not set'}")

    st.subheader("Odds API")
    odds_configured = bool(settings.odds.odds_api_key)
    st.write(f"**API Key:** {'✓ Configured' if odds_configured else '✗ Not set'}")


elif page == "Reports":
    st.title("📄 LLM Reports")

    from config.settings import REPORTS_DIR

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_files = sorted(REPORTS_DIR.glob("*.md"), reverse=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Reports are saved to `{REPORTS_DIR}`")
    with col2:
        if st.button("🤖 Generate Report"):
            with st.spinner("Generating LLM report..."):
                try:
                    from src.pipeline.feedback_loop import Pipeline
                    from src.llm.reporter import generate_round_report
                    pipeline = Pipeline()
                    predictions = pipeline.predict(year, round_num)
                    value_bets = pipeline.find_bets(year, round_num)
                    if predictions.empty:
                        st.warning("No predictions available — run the pipeline first.")
                    else:
                        _, saved = generate_round_report(
                            predictions, value_bets,
                            round_num=int(round_num), year=int(year),
                        )
                        st.success(f"Report saved: {saved.name}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error generating report: {e}")

    st.markdown("---")

    if not report_files:
        st.info("No reports yet. Generate one using the button above or `python main.py report --round N`.")
    else:
        # File selector
        file_names = [f.name for f in report_files]
        selected_name = st.selectbox("Select report", file_names)
        selected_file = REPORTS_DIR / selected_name

        # Download button
        report_content = selected_file.read_text(encoding="utf-8")
        col1, col2 = st.columns([5, 1])
        with col2:
            st.download_button(
                "⬇ Download",
                data=report_content,
                file_name=selected_name,
                mime="text/markdown",
            )
        with col1:
            # Parse metadata from filename: type_year_Rround_timestamp.md
            parts = selected_name.replace(".md", "").split("_")
            st.caption(f"File: `{selected_name}` — {selected_file.stat().st_size} bytes")

        st.markdown(report_content)
