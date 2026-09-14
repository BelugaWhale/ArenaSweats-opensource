-- Replace {region} with one trusted region code before running this query.
-- The result contains one row per daily top-quarter Challenger player-game.
WITH players AS
(
    SELECT
        player_hash AS player_hash,
        game_id AS game_id,
        game_date AS game_date,
        game_ts AS game_ts,
        arraySort(arrayConcat([player_hash], teammate_hashes)) AS team_hashes,
        placing AS placing,
        pregame_rating AS pregame_rating,
        pregame_league_tier AS pregame_league_tier,
        rating_change AS rating_change,
        length(repeated_teammate_hashes) AS repeated_teammates
    FROM {region}_player_matchhistory
),
teams AS
(
    SELECT
        game_id AS game_id,
        team_hashes AS team_hashes,
        count() AS team_player_count,
        countIf(pregame_league_tier IN ('Grandmaster', 'Challenger')) AS gm_plus_count
    FROM players
    GROUP BY game_id, team_hashes
),
game_dates AS
(
    SELECT DISTINCT
        game_date AS game_date
    FROM players
),
rank_snapshots AS
(
    SELECT
        rank_date AS rank_date
    FROM {region}_player_rankhistory
    GROUP BY rank_date
),
game_snapshots AS
(
    SELECT
        game.game_date AS game_date,
        argMax(snapshot.rank_date, snapshot.rank_date) AS snapshot_date
    FROM game_dates AS game
    CROSS JOIN rank_snapshots AS snapshot
    WHERE snapshot.rank_date < game.game_date
    GROUP BY game.game_date
),
top_quarter_members AS
(
    SELECT
        snapshot.game_date AS game_date,
        rank.player_hash AS player_hash,
        toUInt8(1) AS is_top_quarter_challenger
    FROM game_snapshots AS snapshot
    INNER JOIN {region}_player_rankhistory AS rank ON rank.rank_date = snapshot.snapshot_date
    WHERE rank.player_rank <= if(
        '{region}' IN ('oce', 'jp', 'me', 'ru'),
        13,
        if('{region}' IN ('lan', 'las', 'tr', 'tw'), 25, 50)
    )
    GROUP BY snapshot.game_date, rank.player_hash
)
SELECT
    '{region}' AS region,
    player.player_hash AS player_hash,
    player.game_id AS game_id,
    player.game_date AS game_date,
    player.game_ts AS game_ts,
    player.placing AS placing,
    player.pregame_rating AS pregame_rating,
    player.rating_change AS rating_change,
    player.repeated_teammates AS repeated_teammates,
    team.gm_plus_count - toUInt64(player.pregame_league_tier IN ('Grandmaster', 'Challenger')) AS other_gm_plus_teammates,
    member.is_top_quarter_challenger AS is_top_quarter_challenger,
    team.team_player_count AS team_player_count
FROM players AS player
INNER JOIN teams AS team ON team.game_id = player.game_id AND team.team_hashes = player.team_hashes
INNER JOIN game_snapshots AS snapshot ON snapshot.game_date = player.game_date
INNER JOIN top_quarter_members AS member ON member.game_date = player.game_date AND member.player_hash = player.player_hash
FORMAT CSVWithNames
