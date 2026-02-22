"""
AFL Constants - Team names, venue mappings, and reference data.
"""

# Official AFL team names and common aliases
TEAMS = {
    "Adelaide": {"full": "Adelaide Crows", "abbrev": "ADE", "city": "Adelaide", "state": "SA"},
    "Brisbane Lions": {"full": "Brisbane Lions", "abbrev": "BRL", "city": "Brisbane", "state": "QLD"},
    "Carlton": {"full": "Carlton Blues", "abbrev": "CAR", "city": "Melbourne", "state": "VIC"},
    "Collingwood": {"full": "Collingwood Magpies", "abbrev": "COL", "city": "Melbourne", "state": "VIC"},
    "Essendon": {"full": "Essendon Bombers", "abbrev": "ESS", "city": "Melbourne", "state": "VIC"},
    "Fremantle": {"full": "Fremantle Dockers", "abbrev": "FRE", "city": "Perth", "state": "WA"},
    "Geelong": {"full": "Geelong Cats", "abbrev": "GEE", "city": "Geelong", "state": "VIC"},
    "Gold Coast": {"full": "Gold Coast Suns", "abbrev": "GCS", "city": "Gold Coast", "state": "QLD"},
    "GWS": {"full": "Greater Western Sydney Giants", "abbrev": "GWS", "city": "Sydney", "state": "NSW"},
    "Hawthorn": {"full": "Hawthorn Hawks", "abbrev": "HAW", "city": "Melbourne", "state": "VIC"},
    "Melbourne": {"full": "Melbourne Demons", "abbrev": "MEL", "city": "Melbourne", "state": "VIC"},
    "North Melbourne": {"full": "North Melbourne Kangaroos", "abbrev": "NME", "city": "Melbourne", "state": "VIC"},
    "Port Adelaide": {"full": "Port Adelaide Power", "abbrev": "PTA", "city": "Adelaide", "state": "SA"},
    "Richmond": {"full": "Richmond Tigers", "abbrev": "RIC", "city": "Melbourne", "state": "VIC"},
    "St Kilda": {"full": "St Kilda Saints", "abbrev": "STK", "city": "Melbourne", "state": "VIC"},
    "Sydney": {"full": "Sydney Swans", "abbrev": "SYD", "city": "Sydney", "state": "NSW"},
    "West Coast": {"full": "West Coast Eagles", "abbrev": "WCE", "city": "Perth", "state": "WA"},
    "Western Bulldogs": {"full": "Western Bulldogs", "abbrev": "WBD", "city": "Melbourne", "state": "VIC"},
}

# Mapping of common alternate names used by various data sources
TEAM_NAME_ALIASES = {
    "Adelaide Crows": "Adelaide",
    "Brisbane": "Brisbane Lions",
    "Brisbane Bears": "Brisbane Lions",
    "Greater Western Sydney": "GWS",
    "GW Sydney": "GWS",
    "Kangaroos": "North Melbourne",
    "Sydney Swans": "Sydney",
    "West Coast Eagles": "West Coast",
    "Footscray": "Western Bulldogs",
}

# Venue information with coordinates for travel distance calculation
VENUES = {
    "MCG": {"city": "Melbourne", "state": "VIC", "lat": -37.8200, "lon": 144.9834, "capacity": 100024},
    "Marvel Stadium": {"city": "Melbourne", "state": "VIC", "lat": -37.8165, "lon": 144.9475, "capacity": 53359},
    "GMHBA Stadium": {"city": "Geelong", "state": "VIC", "lat": -38.1580, "lon": 144.3545, "capacity": 36000},
    "Adelaide Oval": {"city": "Adelaide", "state": "SA", "lat": -34.9156, "lon": 138.5961, "capacity": 53583},
    "Optus Stadium": {"city": "Perth", "state": "WA", "lat": -31.9512, "lon": 115.8892, "capacity": 60000},
    "Gabba": {"city": "Brisbane", "state": "QLD", "lat": -27.4858, "lon": 153.0381, "capacity": 42000},
    "SCG": {"city": "Sydney", "state": "NSW", "lat": -33.8917, "lon": 151.2247, "capacity": 48000},
    "ENGIE Stadium": {"city": "Sydney", "state": "NSW", "lat": -33.8474, "lon": 151.0682, "capacity": 24000},
    "Blundstone Arena": {"city": "Hobart", "state": "TAS", "lat": -42.8822, "lon": 147.3228, "capacity": 20000},
    "UTAS Stadium": {"city": "Launceston", "state": "TAS", "lat": -41.4244, "lon": 147.1383, "capacity": 20000},
    "Manuka Oval": {"city": "Canberra", "state": "ACT", "lat": -35.3178, "lon": 149.1347, "capacity": 13550},
    "TIO Stadium": {"city": "Darwin", "state": "NT", "lat": -12.4115, "lon": 130.8447, "capacity": 12500},
    "People First Stadium": {"city": "Cairns", "state": "QLD", "lat": -16.9270, "lon": 145.7453, "capacity": 13500},
    "Mars Stadium": {"city": "Ballarat", "state": "VIC", "lat": -37.5510, "lon": 143.8428, "capacity": 11000},
    "Norwood Oval": {"city": "Adelaide", "state": "SA", "lat": -34.9219, "lon": 138.6368, "capacity": 22000},
}

# Venue alternate names
VENUE_ALIASES = {
    "Melbourne Cricket Ground": "MCG",
    "Docklands": "Marvel Stadium",
    "Etihad Stadium": "Marvel Stadium",
    "Docklands Stadium": "Marvel Stadium",
    "Kardinia Park": "GMHBA Stadium",
    "Simonds Stadium": "GMHBA Stadium",
    "Skilled Stadium": "GMHBA Stadium",
    "Perth Stadium": "Optus Stadium",
    "Brisbane Cricket Ground": "Gabba",
    "Sydney Cricket Ground": "SCG",
    "Sydney Showground": "ENGIE Stadium",
    "Giants Stadium": "ENGIE Stadium",
    "Showground Stadium": "ENGIE Stadium",
    "Bellerive Oval": "Blundstone Arena",
    "York Park": "UTAS Stadium",
    "Aurora Stadium": "UTAS Stadium",
    "Cazaly's Stadium": "People First Stadium",
    "TIO Traeger Park": "TIO Stadium",
    "Traeger Park": "TIO Stadium",
    "Eureka Stadium": "Mars Stadium",
}

# Team home venues
TEAM_HOME_VENUES = {
    "Adelaide": ["Adelaide Oval"],
    "Brisbane Lions": ["Gabba"],
    "Carlton": ["MCG", "Marvel Stadium"],
    "Collingwood": ["MCG", "Marvel Stadium"],
    "Essendon": ["MCG", "Marvel Stadium"],
    "Fremantle": ["Optus Stadium"],
    "Geelong": ["GMHBA Stadium", "MCG"],
    "Gold Coast": ["People First Stadium"],
    "GWS": ["ENGIE Stadium"],
    "Hawthorn": ["MCG", "UTAS Stadium"],
    "Melbourne": ["MCG"],
    "North Melbourne": ["Marvel Stadium"],
    "Port Adelaide": ["Adelaide Oval"],
    "Richmond": ["MCG"],
    "St Kilda": ["Marvel Stadium"],
    "Sydney": ["SCG"],
    "West Coast": ["Optus Stadium"],
    "Western Bulldogs": ["Marvel Stadium", "Mars Stadium"],
}

# Approximate distances between state capitals (km) for travel factor calculation
STATE_DISTANCES = {
    ("VIC", "VIC"): 0,
    ("VIC", "SA"): 725,
    ("VIC", "NSW"): 878,
    ("VIC", "QLD"): 1765,
    ("VIC", "WA"): 3410,
    ("VIC", "TAS"): 430,
    ("VIC", "ACT"): 660,
    ("VIC", "NT"): 3750,
    ("SA", "SA"): 0,
    ("SA", "NSW"): 1390,
    ("SA", "QLD"): 2010,
    ("SA", "WA"): 2700,
    ("SA", "TAS"): 1150,
    ("SA", "ACT"): 1170,
    ("SA", "NT"): 3025,
    ("NSW", "NSW"): 0,
    ("NSW", "QLD"): 920,
    ("NSW", "WA"): 3935,
    ("NSW", "TAS"): 1040,
    ("NSW", "ACT"): 280,
    ("NSW", "NT"): 3935,
    ("QLD", "QLD"): 0,
    ("QLD", "WA"): 4370,
    ("QLD", "TAS"): 2200,
    ("QLD", "ACT"): 1200,
    ("QLD", "NT"): 3425,
    ("WA", "WA"): 0,
    ("WA", "TAS"): 3850,
    ("WA", "ACT"): 3700,
    ("WA", "NT"): 2650,
    ("TAS", "TAS"): 0,
    ("TAS", "ACT"): 1100,
    ("TAS", "NT"): 4180,
    ("ACT", "ACT"): 0,
    ("ACT", "NT"): 3950,
    ("NT", "NT"): 0,
}


def get_travel_distance(state1: str, state2: str) -> int:
    """Get approximate travel distance in km between two states."""
    key = (state1, state2)
    if key in STATE_DISTANCES:
        return STATE_DISTANCES[key]
    # Try reversed key
    key_rev = (state2, state1)
    return STATE_DISTANCES.get(key_rev, 0)


def normalize_team_name(name: str) -> str:
    """Normalize a team name to the canonical short form."""
    if name is None or not isinstance(name, str):
        return ""
    name = name.strip()
    if not name:
        return ""
    if name in TEAMS:
        return name
    if name in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[name]
    # Fuzzy match: check if name is a substring of any team
    for canonical, info in TEAMS.items():
        if name.lower() in canonical.lower() or name.lower() in info["full"].lower():
            return canonical
    return name  # Return as-is if no match


def normalize_venue_name(name: str) -> str:
    """Normalize a venue name to the canonical form."""
    if name is None or not isinstance(name, str):
        return ""
    name = name.strip()
    if not name:
        return ""
    if name in VENUES:
        return name
    if name in VENUE_ALIASES:
        return VENUE_ALIASES[name]
    return name


def get_team_state(team: str) -> str:
    """Get the state a team is based in."""
    if not team:
        return "VIC"
    team = normalize_team_name(team)
    if team in TEAMS:
        return TEAMS[team]["state"]
    return "VIC"  # Default assumption


def get_venue_state(venue: str) -> str:
    """Get the state a venue is in."""
    if not venue:
        return "VIC"
    venue = normalize_venue_name(venue)
    if venue in VENUES:
        return VENUES[venue]["state"]
    return "VIC"


def is_home_ground(team: str, venue: str) -> bool:
    """Check if a venue is a home ground for a team."""
    if not team or not venue:
        return False
    team = normalize_team_name(team)
    venue = normalize_venue_name(venue)
    return venue in TEAM_HOME_VENUES.get(team, [])
