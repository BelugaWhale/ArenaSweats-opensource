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
GAP_TRIGGER_LOW_MU_RATIO = 0.90
GAP_SATURATION_LOW_MU = 20.0

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
UNBALANCED_LOBBY_GRACE_ENABLED = True
UNBALANCED_TEAM_MU_REDUCTION = 0.57 if IS_3X6 else 0.22   # Apply 57% of the effective gap in 3v3, or 22% in 2v2
UNBALANCED_3V3_GRACE_BREAKPOINT = 0.20
UNBALANCED_3V3_GRACE_TAIL_SLOPE = 0.25
UNBALANCED_PAIR_RATIO_ALPHA = 2.5 if IS_3X6 else 3.0
UNBALANCED_GRACE_REPEATED_TEAMMATE_SCALE_MAX = 0.75
'''
ArenaSweats uses OpenSkill's ThurstoneMostellerFull model for 8-team Arena games.
Each player is represented by:
- mu (μ): current estimated skill
- sigma (σ): uncertainty in that estimate

This module applies production rating updates in three stages:
1) Base OpenSkill rate() update on all teams.
2) Optional unbalanced-lobby grace for teams that meet the GM+ threshold.
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

def _teammate_penalty_scale_gap_pct(gap_pct: float) -> float:
    """
    Compute the multiplier for the high-mu player's mu/sigma delta,
    based on the relative mu gap in [0, 1].
    """
    # Below the trigger we do nothing.
    if gap_pct <= GAP_TRIGGER:
        return 1.0

    # At or above saturation, use the minimum multiplier (flat line).
    if gap_pct >= GAP_SATURATION:
        return PENALTY_MIN_MULTIPLIER

    # Linear drop between trigger and saturation.
    progress = (gap_pct - GAP_TRIGGER) / (GAP_SATURATION - GAP_TRIGGER)
    scale = 1.0 - (1.0 - PENALTY_MIN_MULTIPLIER) * progress

    # Clamp to safety range
    return max(PENALTY_MIN_MULTIPLIER, min(1.0, scale))

def _teammate_penalty_scale(mu_hi: float, mu_lo: float) -> float:
    """
    Compute the team-gap modifier multiplier for the high-mu player's
    mu/sigma delta.

    Behavior:
    - No modifier while mu_lo >= 0.90 * mu_hi.
    - Linear reduction between the trigger and mu_lo == 20.
    - Full reduction at/below mu_lo == 20, capped by PENALTY_MIN_MULTIPLIER.
    """
    trigger_low_mu = mu_hi * GAP_TRIGGER_LOW_MU_RATIO

    # Within the trigger zone we do nothing.
    if mu_lo >= trigger_low_mu:
        return 1.0

    # At or below saturation we apply full reduction.
    if mu_lo <= GAP_SATURATION_LOW_MU:
        return PENALTY_MIN_MULTIPLIER

    # Degenerate range: trigger and saturation overlap; treat as step.
    if trigger_low_mu <= GAP_SATURATION_LOW_MU:
        return PENALTY_MIN_MULTIPLIER

    # Linear drop between trigger and saturation.
    progress = (trigger_low_mu - mu_lo) / (trigger_low_mu - GAP_SATURATION_LOW_MU)
    scale = 1.0 - (1.0 - PENALTY_MIN_MULTIPLIER) * progress

    # Clamp to safety range
    return max(PENALTY_MIN_MULTIPLIER, min(1.0, scale))

def calculate_teammate_gap_modifiers(teams, gm_team_any, team_player_ids, repeated_teammate_ids_by_pid):
    """Calculate each player's worst individual teammate-gap modifier."""
    gap_pct_by_pid = {}
    gap_scale_by_pid = {}
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

            for teammate_index, teammate_rating in enumerate(team):
                if teammate_index == player_index or teammate_rating.mu >= player_rating.mu:
                    continue
                teammate_id = team_player_ids[team_index][teammate_index]
                gap_pct = min(1.0, 1.0 - (teammate_rating.mu / player_rating.mu))
                repeated = repeated_teammates is None or teammate_id in repeated_teammates
                scale = _teammate_penalty_scale_gap_pct(gap_pct) if repeated else max(
                    _teammate_penalty_scale_gap_pct(gap_pct),
                    _teammate_penalty_scale(player_rating.mu, teammate_rating.mu),
                )
                if scale < player_gap_scale or (scale == player_gap_scale and gap_pct > player_gap_pct):
                    player_gap_pct = gap_pct
                    player_gap_scale = scale
                if repeated_teammates is not None and teammate_id in repeated_teammates and scale <= UNBALANCED_GRACE_REPEATED_TEAMMATE_SCALE_MAX:
                    unbalanced_grace_blocked_by_team[team_index] = True

            if player_gap_pct > 0.0:
                gap_pct_by_pid[player_id] = player_gap_pct
                gap_scale_by_pid[player_id] = player_gap_scale

    return gap_pct_by_pid, gap_scale_by_pid, unbalanced_grace_blocked_by_team

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
        - protection_net: Net points from placement protection/debt redistribution.
          Positive means received protection; negative means paid donor debt.
        - afk_protection_applied: 1 if an AFK-protected teammate was floored to +0, else 0.
        - afk_penalty_applied: 1 if an AFK player's positive gain was floored to +0, else 0.
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

    gap_pct_by_pid, gap_scale_by_pid, unbalanced_grace_blocked_by_team = calculate_teammate_gap_modifiers(
        teams,
        gm_team_any,
        team_player_ids,
        repeated_teammate_ids_by_pid,
    )
    gm_team_unbalanced_eligible = [
        eligible and not unbalanced_grace_blocked_by_team[team_index]
        for team_index, eligible in enumerate(gm_team_unbalanced_eligible)
    ]

    ranks = list(range(len(teams)))

    try:
        rate_input = []
        for team_ratings in teams:
            rate_input.append([
                model.rating(mu=r.mu, sigma=r.sigma) for r in team_ratings
            ])

        adjusted_teams, unbalanced_reductions = check_for_unbalanced_lobby(model, rate_input, logger, gm_team_eligible_mask=gm_team_unbalanced_eligible)
        if adjusted_teams is None:
            rate_input_final = rate_input
            unbalanced_reductions = [0.0] * len(teams)
        else:
            rate_input_final = adjusted_teams
        rated_teams = model.rate(rate_input_final, ranks=ranks)

        new_teams = []
        for team_idx in range(len(teams)):
            orig_team = teams[team_idx]
            old_final = rate_input_final[team_idx]
            new_from_rate = rated_teams[team_idx]

            final_team = []
            for p_idx in range(len(orig_team)):
                orig = orig_team[p_idx]
                old_adj = old_final[p_idx]
                new_adj = new_from_rate[p_idx]

                delta_mu = new_adj.mu - old_adj.mu
                delta_sigma = new_adj.sigma - old_adj.sigma

                final_mu = orig.mu + delta_mu
                final_sigma = orig.sigma + delta_sigma
                if final_sigma <= 0.0:
                    raise RuntimeError(f"Game {game_id}: OpenSkill produced non-positive sigma={final_sigma}")
                final_sigma = max(final_sigma, SIGMA_FLOOR)

                final_team.append(model.rating(mu=final_mu, sigma=final_sigma))

            new_teams.append(final_team)

        apply_teammate_gap_penalty(
            model,
            teams,
            new_teams,
            team_player_ids,
            gap_scale_by_pid,
        )

        sorted_placings = sorted(teams_by_placing.keys())
        protection_net_by_pid = {}
        afk_protection_applied_by_pid = {}
        afk_penalty_applied_by_pid = {}
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
                if afk_protected_pids and pid in afk_protected_pids:
                    continue
                pre_rating = teams[i][team_player_index]
                post_rating = new_teams[i][team_player_index]
                pre_display = int(calculate_rating(pre_rating))
                post_display = int(calculate_rating(post_rating))
                base_delta = int(round(post_display - pre_display))
                base_delta_mu = post_rating.mu - pre_rating.mu
                base_delta_sigma = post_rating.sigma - pre_rating.sigma

                is_gm = pid in gm_set if gm_set is not None else False
                protection_cap = (3 if is_gm else 4) if expected_team_size == 2 else team_protection_cap

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
                        new_teams[i][team_player_index] = pre_rating
                        afk_protection_applied_by_pid[pid] = 1
                        continue
                    if afk_pids and pid in afk_pids and final_display_delta > 0:
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
                    "protection_net": protection_net_by_pid.get(pid, 0),
                    "afk_protection_applied": afk_protection_applied_by_pid.get(pid, 0),
                    "afk_penalty_applied": afk_penalty_applied_by_pid.get(pid, 0),
                }

        return True, player_ratings, modifiers

    except Exception as e:
        logger.error(f"Failed to update ratings for game {game_id}: {e}")
        return False, player_ratings, {}
