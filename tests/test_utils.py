"""Tests for AFL Predictor utils."""

import pytest
from src.utils.constants import (
    normalize_team_name,
    normalize_venue_name,
    get_travel_distance,
    get_team_state,
    is_home_ground,
    TEAMS,
    VENUES,
)


class TestNormalizeTeamName:
    def test_exact_match(self):
        assert normalize_team_name("Collingwood") == "Collingwood"

    def test_alias(self):
        assert normalize_team_name("Kangaroos") == "North Melbourne"
        assert normalize_team_name("Adelaide Crows") == "Adelaide"

    def test_substring_match(self):
        # "bulldogs" is a substring of "Western Bulldogs"
        assert normalize_team_name("bulldogs") == "Western Bulldogs"

    def test_unknown_team(self):
        result = normalize_team_name("Unknown Team 123")
        assert result == "Unknown Team 123"

    def test_all_teams_valid(self):
        for team in TEAMS:
            assert normalize_team_name(team) == team


class TestNormalizeVenueName:
    def test_mcg(self):
        assert normalize_venue_name("MCG") == "MCG"
        assert normalize_venue_name("Melbourne Cricket Ground") == "MCG"

    def test_alias(self):
        assert normalize_venue_name("Docklands") == "Marvel Stadium"
        assert normalize_venue_name("Kardinia Park") == "GMHBA Stadium"


class TestGetTeamState:
    def test_victorian_teams(self):
        assert get_team_state("Collingwood") == "VIC"
        assert get_team_state("Essendon") == "VIC"
        assert get_team_state("Richmond") == "VIC"

    def test_interstate_teams(self):
        assert get_team_state("Sydney") == "NSW"
        assert get_team_state("Brisbane Lions") == "QLD"
        assert get_team_state("West Coast") == "WA"
        assert get_team_state("Adelaide") == "SA"


class TestTravelDistance:
    def test_same_state(self):
        dist = get_travel_distance("VIC", "VIC")
        assert dist == 0

    def test_interstate(self):
        dist = get_travel_distance("VIC", "NSW")
        assert dist > 0
        assert dist > 500

    def test_symmetric(self):
        d1 = get_travel_distance("VIC", "WA")
        d2 = get_travel_distance("WA", "VIC")
        assert d1 == d2


class TestIsHomeGround:
    def test_collingwood_mcg(self):
        assert is_home_ground("Collingwood", "MCG") is True

    def test_away_ground(self):
        assert is_home_ground("Collingwood", "Optus Stadium") is False
