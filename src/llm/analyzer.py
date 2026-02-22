"""
LLM Analyzer Module

Uses LLM (Claude/GPT) to analyze qualitative factors that ML models can't capture:
- Injury reports and team news
- Contextual factors (rivalry games, weather, crowd expectations)
- Narrative analysis and sentiment
"""

from typing import Optional

from config.settings import settings
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def _get_llm_client():
    """Get the configured LLM client."""
    provider = settings.llm.llm_provider

    if provider == "anthropic":
        try:
            import anthropic
            kwargs = {"api_key": settings.llm.anthropic_api_key}
            if settings.llm.anthropic_base_url:
                kwargs["base_url"] = settings.llm.anthropic_base_url
            return anthropic.Anthropic(**kwargs), "anthropic"
        except ImportError:
            logger.warning("anthropic package not installed")
    
    if provider == "openai" or True:  # Fallback to OpenAI
        try:
            import openai
            return openai.OpenAI(api_key=settings.llm.openai_api_key), "openai"
        except ImportError:
            logger.warning("openai package not installed")
    
    return None, None


def _call_llm(prompt: str, system_prompt: str = "") -> str:
    """Make an LLM API call."""
    client, provider = _get_llm_client()
    
    if client is None:
        return "[LLM unavailable — no API key configured or package not installed]"

    try:
        if provider == "anthropic":
            response = client.messages.create(
                model=settings.llm.llm_model,
                max_tokens=settings.llm.llm_max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=settings.llm.llm_model,
                max_tokens=settings.llm.llm_max_tokens,
                messages=messages,
            )
            return response.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return f"[LLM error: {e}]"


SYSTEM_PROMPT = """You are an expert AFL (Australian Football League) analyst. 
You provide data-driven analysis of matches, combining statistical evidence 
with qualitative factors like injuries, team momentum, tactical matchups, 
and venue conditions. You communicate clearly and concisely, always noting 
your confidence level and key uncertainties."""


def analyze_match(
    home_team: str,
    away_team: str,
    model_prediction: dict,
    team_stats: Optional[dict] = None,
    injury_news: Optional[str] = None,
    venue: str = "",
    additional_context: str = "",
) -> str:
    """
    Generate LLM analysis for a specific match.

    Args:
        home_team: Home team name
        away_team: Away team name
        model_prediction: Dict with ensemble_margin, ensemble_prob, confidence
        team_stats: Dict with team performance stats
        injury_news: Text about injuries/team changes
        venue: Venue name
        additional_context: Any extra context

    Returns:
        Natural language analysis string
    """
    # Build the analysis prompt
    prompt_parts = [
        f"Analyze the upcoming AFL match: {home_team} (home) vs {away_team} (away)",
        f"Venue: {venue}" if venue else "",
        "",
        "== MODEL PREDICTION ==",
        f"Predicted winner: {'Home' if model_prediction.get('ensemble_margin', 0) > 0 else 'Away'}",
        f"Predicted margin: {abs(model_prediction.get('ensemble_margin', 0)):.0f} points",
        f"Home win probability: {model_prediction.get('ensemble_prob', 0.5):.1%}",
        f"Model confidence: {model_prediction.get('confidence', 0):.1%}",
    ]

    if team_stats:
        prompt_parts.extend([
            "",
            "== TEAM STATISTICS ==",
        ])
        for key, value in team_stats.items():
            prompt_parts.append(f"{key}: {value}")

    if injury_news:
        prompt_parts.extend([
            "",
            "== INJURY / TEAM NEWS ==",
            injury_news,
        ])

    if additional_context:
        prompt_parts.extend([
            "",
            "== ADDITIONAL CONTEXT ==",
            additional_context,
        ])

    prompt_parts.extend([
        "",
        "Please provide:",
        "1. A concise match preview (2-3 paragraphs)",
        "2. Key factors that could influence the result",
        "3. Any concerns about the model prediction (e.g., factors it may not capture)",
        "4. Your overall assessment and confidence level",
    ])

    prompt = "\n".join(prompt_parts)
    return _call_llm(prompt, SYSTEM_PROMPT)


def analyze_value_bet(
    bet_info: dict,
    match_context: str = "",
) -> str:
    """
    Generate LLM analysis for a value bet opportunity.

    Args:
        bet_info: Dict with bet_on, model_prob, bookmaker_prob, expected_value, etc.
        match_context: Additional match context

    Returns:
        Analysis and recommendation string
    """
    prompt = f"""Analyze this AFL value bet opportunity:

Match: {bet_info.get('home_team', '')} vs {bet_info.get('away_team', '')}
Bet on: {bet_info.get('bet_on', '')}
Bookmaker odds: ${bet_info.get('decimal_odds', 0):.2f}
Bookmaker implied probability: {bet_info.get('bookmaker_prob', 0):.1%}
Model probability: {bet_info.get('model_prob', 0):.1%}
Edge: {bet_info.get('edge', 0):.1%}
Expected Value: +{bet_info.get('expected_value', 0):.1%}
Predicted margin: {bet_info.get('predicted_margin', 0):+.0f} points

{f"Additional context: {match_context}" if match_context else ""}

Please provide:
1. Brief assessment of whether this value bet is justified
2. Key risks and factors that could invalidate the edge
3. Recommended action (bet, pass, or reduce stake) with reasoning
Keep your response concise (under 200 words)."""

    return _call_llm(prompt, SYSTEM_PROMPT)


def analyze_injuries(injury_text: str) -> dict:
    """
    Parse injury reports into structured impact assessments.

    Returns dict with player impact ratings.
    """
    if not injury_text:
        return {"players": [], "overall_impact": "none"}

    prompt = f"""Parse this AFL injury/team news and assess the impact on match outcomes.

NEWS:
{injury_text}

Return a JSON-formatted assessment with:
- "players": list of affected players with "name", "team", "status" (out/doubt/test), and "impact" (high/medium/low)
- "overall_impact": overall impact on team's chances ("significant", "moderate", "minor", "none")
- "summary": one-sentence summary

Be concise and output valid JSON only."""

    response = _call_llm(prompt, "You are an AFL data parser. Output only valid JSON.")

    try:
        import json
        return json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return {"players": [], "overall_impact": "unknown", "raw": response}
