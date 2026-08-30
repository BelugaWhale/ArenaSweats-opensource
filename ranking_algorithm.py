import math
from collections import defaultdict
from openskill.models import ThurstoneMostellerFull

IS_3X6 = True
SIGMA_FLOOR = 2.5

# Global configuration for the team-gap modifier (historically named "penalty").
# PENALTY_MIN_MULTIPLIER: lower bound on the multiplier applied to the
#                         higher player's mu/sigma delta once fully reduced.
# GAP_TRIGGER: relative mu gap threshold where scaling starts.
# GAP_SATURATION: relative mu gap threshold where scaling saturates at
#                 PENALTY_MIN_MULTIPLIER.
PENALTY_MIN_MULTIPLIER = 0.05
GAP_TRIGGER = 0.10
GAP_SATURATION = 0.55
FRESH_GAP_TRIGGER = 0.15
FRESH_GAP_SATURATION = 0.65

# Unbalanced lobby configuration.
# A team is considered "unbalanced" if its mu sum is above the lobby's
# median team mu (any positive gap). The check is only performed for teams
# that meet the format-specific GM+ threshold. For such teams we temporarily
# reduce their mu based on the fractional gap before calling model.rate. In
# 3v3, the first 20% uses the 57% slope and any excess uses a 25% slope. The
# fractional gap is additionally scaled by
# (team_mu_min / team_mu_max) ** UNBALANCED_PAIR_RATIO_ALPHA in both 2v2 and 3v3,
# so teams with greater mu spread receive less grace.
# After rating updates we apply the resulting delta mu/sigma on top of the
# original (unreduced) mu/sigma.
# Each eligible team's complete mu/sigma grace budget is measured in an
# isolated OpenSkill rerun, then reallocated by UNBALANCED_GRACE_ALLOCATION_Q.
# This prevents another adjusted team from changing this team's grace.
# Team-gap is applied after that allocation.
UNBALANCED_LOBBY_GRACE_ENABLED = True
UNBALANCED_TEAM_MU_REDUCTION = 0.57 if IS_3X6 else 0.22   # Apply 57% of the effective gap in 3v3, or 22% in 2v2
UNBALANCED_3V3_GRACE_BREAKPOINT = 0.20
UNBALANCED_3V3_GRACE_TAIL_SLOPE = 0.25
UNBALANCED_PAIR_RATIO_ALPHA = 2.5 if IS_3X6 else 3.0
UNBALANCED_GRACE_REPEATED_TEAMMATE_GAP_MIN = 0.33
UNBALANCED_GRACE_ALLOCATION_Q = 1.5
'''
ArenaSweats uses OpenSkill's ThurstoneMostellerFull model for 8-team Arena games.
Each player is represented by:
- mu (μ): current estimated skill
- sigma (σ): uncertainty in that estimate

This module applies production rating updates in three stages:
1) Base OpenSkill rate() update on all teams.
2) Optional unbalanced-lobby grace for teams that meet the GM+ threshold,
   then a q-tilt of that team's natural mu/sigma grace toward lower-mu teammates.
3) Team-gap modifier for high-mu players in GM-scoped teams.

REFERENCES:
- https://openskill.me/
- https://arxiv.org/abs/2401.05451
- https://pypi.org/project/openskill/
- https://github.com/OpenDebates/openskill.py (Note: This is a fork; original is at https://github.com/vivekjoshy/openskill.py)
'''

def calculate_rating(rating):
    """
    Calculate displayed rating as round((mu - 3*sigma) * 75).

    Microsoft research recommends using mu-3*sigma as the "conservative skill estimate" for TrueSkill, and this is commonly applied in similar systems like OpenSkill.
    https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
    """
    base_rating = (rating.mu - 3 * rating.sigma) * 75
    return round(base_rating)

def instantiate_rating_model():
    """
    Creates and returns a ThurstoneMostellerFull model from OpenSkill.

    The ThurstoneMostellerFull class constructor can be customized with several
    parameters that define the behavior of the rating system. These parameters
    are based on the mathematical model of the algorithm.

    Parameters:
        mu (float): The initial mean of a player's skill (μ). This represents the
            assumed skill level of a new player before any matches have been played.
            The default value is 25.0.

        sigma (float): The initial standard deviation of a player's skill (σ).
            This represents the system's uncertainty about the player's initial
            skill. A higher value means the system is less certain. As a player
            plays more games, their sigma will decrease. The default is 25.0 / 3.0.

        beta (float): The "skill variance" that defines the distance in skill points
            that gives a player an 80% chance of winning against another. A smaller
            beta means that a smaller skill difference has a greater impact on the
            win probability, making the system more sensitive to skill gaps.
            The default is 25.0 / 6.0.

        kappa (float): An arbitrary small positive real number that is used to prevent
            the variance of the posterior distribution from becoming too small or
            negative. It can also be thought of as a regularization parameter that
            prevents ratings from changing too drastically. Represented by: κ.
            The default value is 0.0001.

        tau (float): The "dynamic factor" that is added to a player's sigma before
            the match to account for performance variability. A higher value allows
            for more significant rating changes based on a single match performance.
            The default is 25.0 / 300.0.

        gamma (callable): A custom function that returns the amount to be added
            to the winning team's sigma before updating. It must accept parameters
            for rank, num_teams, mu, sigma, team, and player_index.
            Represented by: γ. The default is an internal `_gamma` function.

        limit_sigma (bool): If True, this prevents a player's ordinal rating from
            decreasing, even after a loss. This can be useful for maintaining
            leaderboard stability where ranks should only ever increase or stay
            the same. The default is False.

    Note on Other Parameters:
        - `margin`: This is not a constructor parameter but can be used with the
        model to account for the margin of victory, which can improve accuracy
        in games where score differences matter.
        - `balance`: This is not a constructor parameter. It is a flag that can be
        used to have the rating system adjust its assumptions for players at
        the extreme ends of the skill distribution.

    REFERENCE:
    - https://openskill.me/en/stable/models/openskill.models.weng_lin.thurstone_mosteller_full.html
    """
    # This instantiation creates a model for games with strict rankings (no draws).
    model = ThurstoneMostellerFull(sigma=(25/5.75), beta=(25/6) * (3.75 if IS_3X6 else 4), tau=(25/300) * 1.75)

    return model

def _teammate_penalty_scale_gap_pct(gap_pct: float, trigger=GAP_TRIGGER, saturation=GAP_SATURATION) -> float:
    """
    Compute the multiplier for the high-mu player's mu/sigma delta,
    based on the relative mu gap in [0, 1].
    """
    # Below the trigger we do nothing.
    if gap_pct <= trigger:
        return 1.0

    # At or above saturation, use the minimum multiplier (flat line).
    if gap_pct >= saturation:
        return PENALTY_MIN_MULTIPLIER

    # Linear drop between trigger and saturation.
    progress = (gap_pct - trigger) / (saturation - trigger)
    scale = 1.0 - (1.0 - PENALTY_MIN_MULTIPLIER) * progress

    # Clamp to safety range
    return max(PENALTY_MIN_MULTIPLIER, min(1.0, scale))

def calculate_teammate_gap_modifiers(teams, gm_team_any, team_player_ids, repeated_teammate_ids_by_pid):
    """Calculate each player's worst individual teammate-gap modifier."""
    gap_pct_by_pid = {}
    gap_scale_by_pid = {}
    repeat_team_gap_teammate_by_pid = {}
    repeat_grace_teammate_by_pid = {}
    unbalanced_grace_blocked_by_team = [False] * len(teams)

    for team_index, team in enumerate(teams):
        if not gm_team_any[team_index]:
            continue
        if len(team) < 2:
            raise RuntimeError("Team-gap modifier requires at least 2 players per team.")

        for player_index, player_rating in enumerate(team):
            player_id = team_player_ids[team_index][player_index]
            if player_rating.mu <= 0.0:
                continue
            repeated_teammates = None if repeated_teammate_ids_by_pid is None else repeated_teammate_ids_by_pid[player_id]
            player_gap_pct = 0.0
            player_gap_scale = 1.0
            player_fresh_teammate_scale = 1.0
            player_gap_teammate_id = None
            player_gap_repeated = False
            player_grace_gap_pct = 0.0

            for teammate_index, teammate_rating in enumerate(team):
                if teammate_index == player_index or teammate_rating.mu >= player_rating.mu:
                    continue
                teammate_id = team_player_ids[team_index][teammate_index]
                gap_pct = min(1.0, 1.0 - (teammate_rating.mu / player_rating.mu))
                repeated = repeated_teammates is None or teammate_id in repeated_teammates
                scale = _teammate_penalty_scale_gap_pct(gap_pct) if repeated else _teammate_penalty_scale_gap_pct(
                    gap_pct,
                    FRESH_GAP_TRIGGER,
                    FRESH_GAP_SATURATION,
                )
                if not repeated:
                    player_fresh_teammate_scale = min(player_fresh_teammate_scale, scale)
                if scale < player_gap_scale or (scale == player_gap_scale and gap_pct > player_gap_pct):
                    player_gap_pct = gap_pct
                    player_gap_scale = scale
                    player_gap_teammate_id = teammate_id
                    player_gap_repeated = repeated_teammates is not None and teammate_id in repeated_teammates
                if repeated_teammates is not None and teammate_id in repeated_teammates and gap_pct >= UNBALANCED_GRACE_REPEATED_TEAMMATE_GAP_MIN:
                    unbalanced_grace_blocked_by_team[team_index] = True
                    if gap_pct > player_grace_gap_pct:
                        player_grace_gap_pct = gap_pct
                        repeat_grace_teammate_by_pid[player_id] = teammate_id

            if player_gap_pct > 0.0:
                gap_pct_by_pid[player_id] = player_gap_pct
                gap_scale_by_pid[player_id] = player_gap_scale
            if player_gap_repeated and player_gap_scale < min(
                player_fresh_teammate_scale,
                _teammate_penalty_scale_gap_pct(player_gap_pct, FRESH_GAP_TRIGGER, FRESH_GAP_SATURATION),
            ):
                repeat_team_gap_teammate_by_pid[player_id] = player_gap_teammate_id

    return (
        gap_pct_by_pid,
        gap_scale_by_pid,
        unbalanced_grace_blocked_by_team,
        repeat_team_gap_teammate_by_pid,
        repeat_grace_teammate_by_pid,
    )

def apply_teammate_gap_penalty(model, teams, new_teams, team_player_ids, gap_scale_by_pid):
    """
    Apply precomputed team-gap modifiers to players' rating updates.
    """
    tau = getattr(model, "tau", None)
    if tau is None:
        raise RuntimeError("Rating model must expose 'tau' for prior->posterior sigma scaling.")
    tau = float(tau)
    if tau < 0.0:
        raise ValueError(f"Invalid model.tau={tau}; expected tau >= 0.")

    for team_index, team in enumerate(teams):
        for player_index, old_rating in enumerate(team):
            player_id = team_player_ids[team_index][player_index]
            scale = gap_scale_by_pid.get(player_id, 1.0)
            if scale >= 1.0:
                continue
            new_rating = new_teams[team_index][player_index]

            delta_mu = new_rating.mu - old_rating.mu
            sigma_prior = math.sqrt(old_rating.sigma * old_rating.sigma + tau * tau)
            sigma_delta_from_prior = new_rating.sigma - sigma_prior

            new_teams[team_index][player_index] = model.rating(
                mu=old_rating.mu + delta_mu * scale,
                sigma=sigma_prior + sigma_delta_from_prior * scale
            )

# ----------------------------------------------------------------------
# UNBALANCED LOBBY helpers
# ----------------------------------------------------------------------

def check_for_unbalanced_lobby(model, teams, logger, gm_team_eligible_mask=None):
    """
    Decide whether any team is in an "unbalanced lobby" and, if so, prepare
    the adjusted teams list to feed into model.rate.

    Args:
        model: OpenSkill model instance.
        teams: List of teams, where each team is a list of Rating objects.
        logger: Logger or None.
        gm_team_eligible_mask: Optional list[bool] indicating whether each team
            meets the format-specific GM+ threshold. If provided, only teams
            with True are eligible for the grace.

    Returns:
        teams_for_rate: None if no adjustments are required; otherwise a new
            list of teams whose players may have adjusted mu values for
            unbalanced teams (and copied ratings for others).
        reductions: None if no adjustments are required; otherwise list[float]
            aligned with teams containing per-team temporary reduction pct.
    """
    if not UNBALANCED_LOBBY_GRACE_ENABLED:
        return None, None

    num_teams = len(teams)
    if num_teams == 0:
        return None, None

    # Compute team mu sums and lobby median
    team_mu_sums = []
    for team in teams:
        mu_sum = sum(p.mu for p in team)
        team_mu_sums.append(mu_sum)

    sorted_mu_sums = sorted(team_mu_sums)
    mid = len(sorted_mu_sums) // 2
    if len(sorted_mu_sums) % 2 == 0:
        lobby_median_team_mu = (sorted_mu_sums[mid - 1] + sorted_mu_sums[mid]) / 2.0
    else:
        lobby_median_team_mu = sorted_mu_sums[mid]

    unbalanced_mask = [False] * num_teams
    effective_gap_by_team = [0.0] * num_teams
    any_unbalanced = False

    if lobby_median_team_mu > 0.0:
        for idx, mu_sum in enumerate(team_mu_sums):
            if gm_team_eligible_mask is not None and not gm_team_eligible_mask[idx]:
                continue

            base_gap_pct = (mu_sum - lobby_median_team_mu) / lobby_median_team_mu
            if base_gap_pct > 0.0:
                team_scale = _unbalanced_team_ratio_scale(teams[idx])
                diff_pct = base_gap_pct * team_scale
                if diff_pct > 0.0:
                    effective_gap_by_team[idx] = diff_pct
                    unbalanced_mask[idx] = True
                    any_unbalanced = True
                    if logger is not None:
                        logger.debug(
                            "Unbalanced lobby detected for team index %d: "
                            "mu_sum=%.3f lobby_median=%.3f base_gap_pct=%.3f team_scale=%.3f effective_gap_pct=%.3f",
                            idx, mu_sum, lobby_median_team_mu, base_gap_pct, team_scale, diff_pct
                        )
    else:
        # Log when median is invalid, especially if GM+ teams are present
        if logger is not None and gm_team_eligible_mask is not None and any(gm_team_eligible_mask):
            logger.warning(
                f"Unbalanced lobby check skipped: lobby_median_team_mu={lobby_median_team_mu} "
                f"(team_mu_sums={team_mu_sums})"
            )

    # Fast path: no unbalanced team, skip all adjustment machinery
    if not any_unbalanced:
        return None, None

    # Build adjusted teams for the rate() call.
    teams_for_rate = []
    reductions = [0.0] * num_teams
    for idx, team_ratings in enumerate(teams):
        if unbalanced_mask[idx]:
            adjusted_team = []
            for r in team_ratings:
                # Reduction is scaled by team-vs-lobby gap and internal team balance.
                reduction_pct = _unbalanced_grace_reduction_pct(effective_gap_by_team[idx])
                reductions[idx] = reduction_pct
                adjusted_mu = r.mu * (1.0 - reduction_pct)
                adjusted_team.append(model.rating(mu=adjusted_mu, sigma=r.sigma))
            teams_for_rate.append(adjusted_team)
        else:
            # Clone ratings to keep input to model.rate independent
            teams_for_rate.append([
                model.rating(mu=r.mu, sigma=r.sigma) for r in team_ratings
            ])

    return teams_for_rate, reductions

def _unbalanced_team_ratio_scale(team_ratings, alpha=None):
    """Scale unbalanced-lobby grace down for teams with greater mu spread."""
    current_alpha = UNBALANCED_PAIR_RATIO_ALPHA if alpha is None else alpha
    if current_alpha <= 0.0:
        return 1.0

    mus = [r.mu for r in team_ratings]
    mu_hi = max(mus)
    mu_lo = min(mus)
    if mu_hi <= 0.0 or mu_lo < 0.0:
        raise ValueError(
            f"Invalid mu values for unbalanced team-ratio scaling: mu_hi={mu_hi}, mu_lo={mu_lo}"
        )
    if mu_lo == 0.0:
        return 0.0

    return (mu_lo / mu_hi) ** current_alpha

def _unbalanced_grace_reduction_pct(effective_gap_pct: float) -> float:
    """Convert effective unbalanced-lobby gap into the temporary reduction pct."""
    if not IS_3X6 or effective_gap_pct <= UNBALANCED_3V3_GRACE_BREAKPOINT:
        return effective_gap_pct * UNBALANCED_TEAM_MU_REDUCTION
    return UNBALANCED_3V3_GRACE_BREAKPOINT * UNBALANCED_TEAM_MU_REDUCTION + (effective_gap_pct - UNBALANCED_3V3_GRACE_BREAKPOINT) * UNBALANCED_3V3_GRACE_TAIL_SLOPE

# ----------------------------------------------------------------------
# Main game-processing function
# ----------------------------------------------------------------------

def process_game_ratings(
    model,
    players,
    game_id,
    player_ratings,
    logger,
    gm_set,
    arena_format=None,
    afk_pids=None,
    afk_protected_pids=None,
    repeated_teammate_ids_by_pid=None,
):
    """
    Process a single game's ratings update using OpenSkill ThurstoneMostellerFull with direct team support.

    Args:
        model: ThurstoneMostellerFull model instance
        players: List of (player_id, team_placing) tuples
        game_id: Game identifier for logging
        player_ratings: Dictionary of player_id -> Rating
        logger: Logger instance
        gm_set: Set of player_ids considered GM+ for this game's processing
        afk_pids: Optional set of player_ids identified as AFK for this game. Final
            positive display-rating gains are clamped back to zero for these players.
        afk_protected_pids: Optional set of player_ids whose mu/sigma changes should be zeroed
            only when their final display-rating delta would otherwise be negative.
        repeated_teammate_ids_by_pid: Optional dict[player_id, collection[player_id]].
            Each current teammate uses the repeated curve only when that teammate is
            present in the player's collection. None preserves the strict legacy curve.

    Returns:
        tuple: (success: bool, updated_player_ratings: dict, modifiers: dict[player_id] -> dict)

        Modifiers dictionary contains per-player tracking values:
        - gap_pct: Relative mu gap (1 - mu_low / mu_high) for the high-mu player in a modified team, 0.0 otherwise.
        - gap_scale: Multiplier applied to the high-mu player's delta (0.05-1.0), 1.0 if no modifier.
        - unbalanced_reduction_pct: Temporary mu reduction percentage for unbalanced GM+ teams, 0.0 otherwise.
        - openskill_rating_change: Displayed-rating change before this team's own grace allocation.
        - unbalanced_grace_net: Displayed-rating change allocated to a team with nonzero grace before team-gap.
        - team_gap_net: Displayed-rating change added or removed by team-gap after grace.
        - protection_net: Net points from placement/AFK protection, AFK penalties,
          and protection-debt redistribution.
        - afk_protection_applied: 1 if an AFK-protected teammate was floored to +0, else 0.
        - afk_penalty_applied: 1 if an AFK player's positive gain was floored to +0, else 0.
        - repeat_team_gap_teammate_id: Teammate whose repeat status made the selected team-gap curve stricter, else None.
        - repeat_grace_teammate_id: Teammate whose large repeated gap blocked this player's eligible team grace, else None.
        - repeat_protection_teammate_id: Teammate whose repeat status removed this solo GM+'s third-place protection, else None.
    """
    
    if arena_format is None:
        arena_format = {
            "name": "2x8",
            "team_count": 8,
            "team_size": 2,
            "player_count": 16,
            "placement_count": 8,
            "tophalf_cutoff": 4,
        }

    expected_player_count = int(arena_format["player_count"])
    expected_team_count = int(arena_format["team_count"])
    expected_team_size = int(arena_format["team_size"])
    placement_count = int(arena_format["placement_count"])
    tophalf_cutoff = int(arena_format["tophalf_cutoff"])

    if len(players) != expected_player_count:
        logger.warning(f"Game {game_id} has {len(players)} players, expected {expected_player_count}")
        return False, player_ratings, {}

    # Group players by team placement.
    teams_by_placing = defaultdict(list)
    for player_id, team_placing in players:
        teams_by_placing[team_placing].append(player_id)

    if len(teams_by_placing) != expected_team_count:
        logger.warning(f"Game {game_id} has {len(teams_by_placing)} teams, expected {expected_team_count}")
        return False, player_ratings, {}

    for placing, team_players in teams_by_placing.items():
        if len(team_players) != expected_team_size:
            logger.warning(f"Game {game_id} team placing {placing} has {len(team_players)} players, expected {expected_team_size}")
            return False, player_ratings, {}

    # Prepare teams in order of placing 1 (best) to N (worst)
    teams = []
    gm_team_any = []
    gm_team_unbalanced_eligible = []
    gm_team_counts = []
    team_player_ids = []
    for placing in sorted(teams_by_placing.keys()):
        team_players = teams_by_placing[placing]
        team_ratings = [player_ratings.get(pid, model.rating()) for pid in team_players]
        teams.append(team_ratings)
        team_player_ids.append(team_players)
        if gm_set is not None:
            gm_count = sum(1 for pid in team_players if pid in gm_set)
            gm_team_any.append(gm_count >= 1)
            gm_team_unbalanced_eligible.append(gm_count >= min(2, expected_team_size))
            gm_team_counts.append(gm_count)
        else:
            gm_team_any.append(False)
            gm_team_unbalanced_eligible.append(False)
            gm_team_counts.append(0)

    (
        gap_pct_by_pid,
        gap_scale_by_pid,
        unbalanced_grace_blocked_by_team,
        repeat_team_gap_teammate_by_pid,
        repeat_grace_teammate_by_pid,
    ) = calculate_teammate_gap_modifiers(
        teams,
        gm_team_any,
        team_player_ids,
        repeated_teammate_ids_by_pid,
    )
    gm_team_unbalanced_eligible_before_repeat = list(gm_team_unbalanced_eligible)

    ranks = list(range(len(teams)))

    try:
        rate_input = []
        for team_ratings in teams:
            rate_input.append([
                model.rating(mu=r.mu, sigma=r.sigma) for r in team_ratings
            ])

        adjusted_teams, unbalanced_reductions = check_for_unbalanced_lobby(
            model,
            rate_input,
            logger,
            gm_team_eligible_mask=gm_team_unbalanced_eligible_before_repeat,
        )
        if adjusted_teams is None:
            unbalanced_reductions = [0.0] * len(teams)
        repeat_grace_applied_by_team = [
            unbalanced_grace_blocked_by_team[team_index] and unbalanced_reductions[team_index] > 0.0
            for team_index in range(len(teams))
        ]
        unbalanced_reductions = [
            0.0 if unbalanced_grace_blocked_by_team[team_index] else unbalanced_reductions[team_index]
            for team_index in range(len(teams))
        ]

        ordinary_output = model.rate(rate_input, ranks=ranks)
        ordinary_teams = []
        for team_idx, orig_team in enumerate(teams):
            ordinary_team = []
            for p_idx, orig in enumerate(orig_team):
                final_sigma = orig.sigma + (ordinary_output[team_idx][p_idx].sigma - rate_input[team_idx][p_idx].sigma)
                if final_sigma <= 0.0:
                    raise RuntimeError(f"Game {game_id}: OpenSkill produced non-positive sigma={final_sigma}")
                ordinary_team.append(model.rating(
                    mu=orig.mu + (ordinary_output[team_idx][p_idx].mu - rate_input[team_idx][p_idx].mu),
                    sigma=max(final_sigma, SIGMA_FLOOR),
                ))
            ordinary_teams.append(ordinary_team)

        new_teams = ordinary_teams
        if adjusted_teams is not None:
            new_teams = []
            for team_idx, orig_team in enumerate(teams):
                ordinary_team = ordinary_teams[team_idx]
                if unbalanced_reductions[team_idx] <= 0.0:
                    new_teams.append(ordinary_team)
                    continue

                isolated_input = [
                    [
                        model.rating(mu=rating.mu, sigma=rating.sigma)
                        for rating in (adjusted_teams[current_team_idx] if current_team_idx == team_idx else rate_input[current_team_idx])
                    ]
                    for current_team_idx in range(len(teams))
                ]
                isolated_output = model.rate(isolated_input, ranks=ranks)
                graced_team = []
                for p_idx, orig in enumerate(orig_team):
                    final_sigma = orig.sigma + (isolated_output[team_idx][p_idx].sigma - isolated_input[team_idx][p_idx].sigma)
                    if final_sigma <= 0.0:
                        raise RuntimeError(f"Game {game_id}: OpenSkill produced non-positive isolated sigma={final_sigma}")
                    graced_team.append(model.rating(
                        mu=orig.mu + (isolated_output[team_idx][p_idx].mu - isolated_input[team_idx][p_idx].mu),
                        sigma=max(final_sigma, SIGMA_FLOOR),
                    ))

                current_mu_grace = [graced_team[p_idx].mu - ordinary_team[p_idx].mu for p_idx in range(len(orig_team))]
                current_sigma_grace = [graced_team[p_idx].sigma - ordinary_team[p_idx].sigma for p_idx in range(len(orig_team))]
                team_mu_budget = sum(current_mu_grace)
                team_sigma_budget = sum(current_sigma_grace)
                team_display_budget = 75.0 * (team_mu_budget - 3.0 * team_sigma_budget)
                if team_mu_budget <= 0.0 or team_display_budget <= 0.0:
                    raise RuntimeError(
                        f"Game {game_id}: isolated grace was non-positive for team index {team_idx} "
                        f"(mu={team_mu_budget}, sigma={team_sigma_budget}, display={team_display_budget})"
                    )
                orig_mus = [orig.mu for orig in orig_team]
                low_mu = min(orig_mus)
                if low_mu <= 0.0:
                    raise RuntimeError(f"Game {game_id}: non-positive low_mu={low_mu} during grace allocation")
                mu_tilts = [
                    (low_mu / orig_mus[p_idx]) ** UNBALANCED_GRACE_ALLOCATION_Q
                    for p_idx in range(len(orig_team))
                ]
                mu_weights = [current_mu_grace[p_idx] * mu_tilts[p_idx] for p_idx in range(len(orig_team))]
                sigma_weights = [current_sigma_grace[p_idx] * mu_tilts[p_idx] for p_idx in range(len(orig_team))]
                mu_weight_total = sum(mu_weights)
                sigma_weight_total = sum(sigma_weights)
                if mu_weight_total <= 0.0:
                    raise RuntimeError(f"Game {game_id}: non-positive grace allocation mu_weight_total={mu_weight_total}")
                if team_sigma_budget != 0.0 and (
                    sigma_weight_total * team_sigma_budget <= 0.0
                    or any(delta * team_sigma_budget < 0.0 for delta in current_sigma_grace)
                ):
                    raise RuntimeError(
                        f"Game {game_id}: inconsistent sigma grace allocation "
                        f"(budget={team_sigma_budget}, weight_total={sigma_weight_total}, deltas={current_sigma_grace})"
                    )
                allocated_team = []
                for p_idx in range(len(orig_team)):
                    mu_share = mu_weights[p_idx] / mu_weight_total
                    sigma_share = 0.0 if team_sigma_budget == 0.0 else sigma_weights[p_idx] / sigma_weight_total
                    final_sigma = ordinary_team[p_idx].sigma + team_sigma_budget * sigma_share
                    if final_sigma <= 0.0:
                        raise RuntimeError(f"Game {game_id}: grace allocation produced non-positive sigma={final_sigma}")
                    allocated_team.append(model.rating(
                        mu=ordinary_team[p_idx].mu + team_mu_budget * mu_share,
                        sigma=final_sigma,
                    ))
                new_teams.append(allocated_team)

        openskill_rating_change_by_pid = {}
        unbalanced_grace_net_by_pid = {}
        pre_gap_display_by_pid = {}
        for team_index, orig_team in enumerate(teams):
            for player_index, orig_rating in enumerate(orig_team):
                player_id = team_player_ids[team_index][player_index]
                pre_display = calculate_rating(orig_rating)
                ordinary_display = calculate_rating(ordinary_teams[team_index][player_index])
                pre_gap_display = calculate_rating(new_teams[team_index][player_index])
                if unbalanced_reductions[team_index] > 0.0:
                    openskill_rating_change_by_pid[player_id] = int(ordinary_display - pre_display)
                    unbalanced_grace_net_by_pid[player_id] = int(pre_gap_display - ordinary_display)
                else:
                    openskill_rating_change_by_pid[player_id] = int(pre_gap_display - pre_display)
                    unbalanced_grace_net_by_pid[player_id] = 0
                pre_gap_display_by_pid[player_id] = pre_gap_display

        apply_teammate_gap_penalty(
            model,
            teams,
            new_teams,
            team_player_ids,
            gap_scale_by_pid,
        )

        team_gap_net_by_pid = {}
        for team_index, team in enumerate(new_teams):
            for player_index, rating in enumerate(team):
                player_id = team_player_ids[team_index][player_index]
                team_gap_net_by_pid[player_id] = int(calculate_rating(rating) - pre_gap_display_by_pid[player_id])

        sorted_placings = sorted(teams_by_placing.keys())
        protection_net_by_pid = {}
        afk_protection_applied_by_pid = {}
        afk_penalty_applied_by_pid = {}
        repeat_protection_teammate_by_pid = {}
        donor_entries = []
        debt_mu = 0.0
        debt_sigma = 0.0

        for i, placing in enumerate(sorted_placings):
            team_players = teams_by_placing[placing]
            gm_count = gm_team_counts[i]
            if expected_team_size == 2:
                team_protection_disabled = gm_count == 2
                team_protection_cap = None
            else:
                team_protection_disabled = gm_count >= 2
                team_protection_cap = 2 if gm_count == 1 else tophalf_cutoff
            for team_player_index, pid in enumerate(team_players):
                protection_net_by_pid[pid] = 0
                afk_protection_applied_by_pid[pid] = 0
                afk_penalty_applied_by_pid[pid] = 0
                if team_protection_disabled:
                    continue
                is_gm = pid in gm_set if gm_set is not None else False
                protection_cap = (3 if is_gm else 4) if expected_team_size == 2 else team_protection_cap
                if (
                    expected_team_size == 3
                    and gm_count == 1
                    and is_gm
                    and repeated_teammate_ids_by_pid is not None
                ):
                    repeat_protection_teammate_by_pid[pid] = next(
                        (teammate_pid for teammate_pid in team_players if teammate_pid != pid and teammate_pid in repeated_teammate_ids_by_pid[pid]),
                        None,
                    )
                    if repeat_protection_teammate_by_pid[pid] is None:
                        protection_cap = 3

                if afk_protected_pids and pid in afk_protected_pids:
                    continue
                pre_rating = teams[i][team_player_index]
                post_rating = new_teams[i][team_player_index]
                pre_display = int(calculate_rating(pre_rating))
                post_display = int(calculate_rating(post_rating))
                base_delta = int(round(post_display - pre_display))
                base_delta_mu = post_rating.mu - pre_rating.mu
                base_delta_sigma = post_rating.sigma - pre_rating.sigma

                if placing <= protection_cap and base_delta < 0:
                    new_teams[i][team_player_index] = pre_rating
                    debt_mu += base_delta_mu
                    debt_sigma += base_delta_sigma
                    protection_net_by_pid[pid] += -base_delta
                    continue

                if placing >= tophalf_cutoff + 1 and placing <= placement_count:
                    donor_weight = float(placing - tophalf_cutoff)
                    donor_entries.append((i, team_player_index, pid, donor_weight))

        if abs(debt_mu) > 1e-12 or abs(debt_sigma) > 1e-12:
            weight_total = sum(entry[3] for entry in donor_entries)
            if weight_total <= 0.0:
                raise RuntimeError(
                    f"Game {game_id}: place-protection debt exists (mu={debt_mu}, sigma={debt_sigma}) "
                    f"but no eligible donor placements {tophalf_cutoff + 1}-{placement_count}"
                )

            for i, team_player_index, pid, donor_weight in donor_entries:
                donor_rating_before = new_teams[i][team_player_index]
                donor_share = donor_weight / weight_total
                donor_mu = donor_rating_before.mu + (debt_mu * donor_share)
                donor_sigma = donor_rating_before.sigma + (debt_sigma * donor_share)
                if donor_sigma <= 0.0:
                    raise RuntimeError(
                        f"Game {game_id}: donor sigma became non-positive for pid={pid} "
                        f"(sigma={donor_sigma}, debt_sigma={debt_sigma}, share={donor_share})"
                    )
                donor_sigma = max(donor_sigma, SIGMA_FLOOR)
                donor_rating_after = model.rating(mu=donor_mu, sigma=donor_sigma)
                new_teams[i][team_player_index] = donor_rating_after
                donor_display_before = int(calculate_rating(donor_rating_before))
                donor_display_after = int(calculate_rating(donor_rating_after))
                protection_net_by_pid[pid] -= donor_display_before - donor_display_after

        if afk_pids or afk_protected_pids:
            for i in range(len(teams)):
                team_players = team_player_ids[i]
                for team_player_index, pid in enumerate(team_players):
                    pre_rating = teams[i][team_player_index]
                    post_rating = new_teams[i][team_player_index]
                    final_display_delta = int(round(calculate_rating(post_rating) - calculate_rating(pre_rating)))
                    if afk_protected_pids and pid in afk_protected_pids and final_display_delta < 0:
                        protection_net_by_pid[pid] -= final_display_delta
                        new_teams[i][team_player_index] = pre_rating
                        afk_protection_applied_by_pid[pid] = 1
                        continue
                    if afk_pids and pid in afk_pids and final_display_delta > 0:
                        protection_net_by_pid[pid] -= final_display_delta
                        new_teams[i][team_player_index] = pre_rating
                        afk_penalty_applied_by_pid[pid] = 1

        modifiers = {}
        # Index i corresponds to team position in teams/new_teams/unbalanced_reductions
        # because all three were built by iterating sorted(teams_by_placing.keys()).
        for i, placing in enumerate(sorted_placings):
            team_players = teams_by_placing[placing]
            new_team = new_teams[i]
            for team_player_index, pid in enumerate(team_players):
                player_ratings[pid] = new_team[team_player_index]
                modifiers[pid] = {
                    "gap_pct": gap_pct_by_pid.get(pid, 0.0),
                    "gap_scale": gap_scale_by_pid.get(pid, 1.0),
                    "unbalanced_reduction_pct": unbalanced_reductions[i],
                    "openskill_rating_change": openskill_rating_change_by_pid[pid],
                    "unbalanced_grace_net": unbalanced_grace_net_by_pid[pid],
                    "team_gap_net": team_gap_net_by_pid[pid],
                    "protection_net": protection_net_by_pid.get(pid, 0),
                    "afk_protection_applied": afk_protection_applied_by_pid.get(pid, 0),
                    "afk_penalty_applied": afk_penalty_applied_by_pid.get(pid, 0),
                    "repeat_team_gap_teammate_id": repeat_team_gap_teammate_by_pid.get(pid),
                    "repeat_grace_teammate_id": repeat_grace_teammate_by_pid.get(pid) if repeat_grace_applied_by_team[i] else None,
                    "repeat_protection_teammate_id": repeat_protection_teammate_by_pid.get(pid),
                }

        return True, player_ratings, modifiers

    except Exception as e:
        logger.error(f"Failed to update ratings for game {game_id}: {e}")
        return False, player_ratings, {}
