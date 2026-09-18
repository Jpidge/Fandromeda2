from pathlib import Path

path = Path("gptindex.py")
backup = Path("gptindex.before_team_filter.bak")

if not path.exists():
    raise FileNotFoundError("Run this from the folder containing gptindex.py.")

source = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one matching section, found {count}. "
            "No changes were written."
        )
    source = source.replace(old, new, 1)


replace_once(
    """    players_by_name = {}

    for _, row in players.iterrows():
""",
    """    players_by_name = {}
    players_by_last_name = {}

    for _, row in players.iterrows():
""",
    "Player-match setup",
)

replace_once(
    """            players_by_name.setdefault(key, []).append(row)
""",
    """            players_by_name.setdefault(key, []).append(row)
            players_by_last_name.setdefault(
                key.split()[-1], []
            ).append(row)
""",
    "Surname index",
)

replace_once(
    """        yahoo_team = normalize_team(row["team"])
""",
    """        yahoo_team = normalize_team(row["team"])
        yahoo_position = clean_text(row.get("position", "")).upper()
""",
    "Yahoo position",
)

fuzzy_marker = """        # ----------------------------------------------------
        # Fuzzy matching
        # ----------------------------------------------------
"""

name_matcher = """        # Match Yahoo full names to NFLverse abbreviated names, such as
        # "Justin Jefferson" to "J.Jefferson".
        if matched is None and normalized:
            yahoo_parts = normalized.split()

            if len(yahoo_parts) >= 2:
                surname_candidates = [
                    candidate
                    for candidate in players_by_last_name.get(
                        yahoo_parts[-1], []
                    )
                    if candidate["name_normalized"].split()[0][0]
                    == yahoo_parts[0][0]
                ]

                if yahoo_team:
                    surname_candidates = [
                        candidate
                        for candidate in surname_candidates
                        if normalize_team(candidate.get("team", ""))
                        == yahoo_team
                    ]

                if yahoo_position:
                    surname_candidates = [
                        candidate
                        for candidate in surname_candidates
                        if clean_text(candidate.get("position", "")).upper()
                        == yahoo_position
                    ]

                if len(surname_candidates) == 1:
                    matched = surname_candidates[0]
                    method = "surname_initial_team_position"
                    confidence = 98.0

"""

replace_once(
    fuzzy_marker,
    name_matcher + fuzzy_marker,
    "Name-matching block",
)

replace_once(
    """    roster_rows = []

    roster_proj = roster.merge(
""",
    """    roster_rows = []

    fantasy_teams = sorted(
        manager
        for manager in roster["manager"].dropna().map(clean_text).unique()
        if manager
    )

    team_options = "".join(
        f'<option value="{html_escape(manager)}">'
        f'{html_escape(manager)}</option>'
        for manager in fantasy_teams
    )

    roster_proj = roster.merge(
""",
    "Team-filter options",
)

replace_once(
    """    roster_rows.append(
            f\"\"\"
            <div class="player-card">
""",
    """    roster_rows.append(
            f\"\"\"
            <div class="player-card" data-fantasy-team="{html_escape(row.get('manager', ''))}">
""",
    "Roster card attribute",
)

replace_once(
    """table {{
""",
    """ .section-heading {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}}

.team-filter {{
    color: var(--muted);
    font-size: 13px;
}}

select {{
    margin-left: 6px;
    padding: 7px 9px;
    color: var(--text);
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 8px;
}}

table {{
""",
    "Team-filter styling",
)

replace_once(
    """<section>
    <h2>Your Roster</h2>

    {''.join(roster_rows)}
</section>
""",
    """<section>
    <div class="section-heading">
        <h2>Your Roster</h2>

        <label class="team-filter" for="fantasy-team-filter">
            Fantasy team
            <select id="fantasy-team-filter">
                <option value="">All teams</option>
                {team_options}
            </select>
        </label>
    </div>

    {''.join(roster_rows)}
</section>
""",
    "Roster selector",
)

replace_once(
    """</main>
</body>
""",
    """</main>

<script>
const teamSelect = document.getElementById("fantasy-team-filter");

teamSelect.addEventListener("change", function () {{
    const selected = this.value;

    document.querySelectorAll(".player-card[data-fantasy-team]").forEach(
        function (card) {{
            card.hidden = selected !== "" &&
                card.dataset.fantasyTeam !== selected;
        }}
    );
}});
</script>

</body>
""",
    "Team-filter behavior",
)

if not backup.exists():
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

path.write_text(source, encoding="utf-8")

print("Updated gptindex.py successfully.")
print(f"Backup saved as: {backup}")