import math
from collections import defaultdict
from openskill.models import ThurstoneMostellerFull

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
# reduce their mu by UNBALANCED_TEAM_MU_REDUCTION times the fractional gap
# before calling model.rate. The fractional gap is additionally scaled by
# (team_mu_min / team_mu_max) ** UNBALANCED_PAIR_RATIO_ALPHA so internally
# lopsided teams receive less grace.
# After rating updates we apply the resulting delta mu/sigma on top of the
# original (unreduced) mu/sigma.
UNBALANCED_LOBBY_GRACE_ENABLED = True
UNBALANCED_TEAM_MU_REDUCTION = 0.22   # Apply 22% of the effective gap as a temporary mu reduction
UNBALANCED_PAIR_RATIO_ALPHA = 3.0
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
    model = ThurstoneMostellerFull(sigma=(25/5.75), beta=(25/6) * 4, tau=(25/300) * 1.75)

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

def apply_teammate_gap_penalty(model, teams, new_teams, logger, gm_team_any=None, team_player_ids=None, gap_pct_by_pid=None, gap_scale_by_pid=None, recent_teammate_repeat_by_pid=None):
    """
    Apply the team-gap modifier by scaling high-mu players' updates in teams
    with large teammate mu gaps.

    If gm_team_any is provided, modifier is only evaluated for teams where at
    least one teammate is GM+.
    """
    tau = getattr(model, "tau", None)
    if tau is None:
        raise RuntimeError("Rating model must expose 'tau' for prior->posterior sigma scaling.")
    tau = float(tau)
    if tau < 0.0:
        raise ValueError(f"Invalid model.tau={tau}; expected tau >= 0.")

    for i in range(len(teams)):
        if gm_team_any is not None and not gm_team_any[i]:
            continue

        if len(teams[i]) < 2:
            raise RuntimeError("Team-gap modifier requires at least 2 players per team.")

        for player_index in range(len(teams[i])):
            hi_old = teams[i][player_index]
            hi_new = new_teams[i][player_index]
            teammate_mu_avg = sum(teams[i][j].mu for j in range(len(teams[i])) if j != player_index) / (len(teams[i]) - 1)

            mu_hi = hi_old.mu
            if mu_hi <= 0.0 or teammate_mu_avg >= mu_hi:
                continue

            gap_pct = min(1.0, 1.0 - (teammate_mu_avg / mu_hi))
            if gap_pct <= 0.0:
                continue

            hi_pid = None
            if team_player_ids is not None:
                hi_pid = team_player_ids[i][player_index]
            if recent_teammate_repeat_by_pid is None or hi_pid is None or recent_teammate_repeat_by_pid.get(hi_pid, False):
                scale = _teammate_penalty_scale_gap_pct(gap_pct)
            else:
                scale = max(_teammate_penalty_scale_gap_pct(gap_pct), _teammate_penalty_scale(mu_hi, teammate_mu_avg))
            if hi_pid is not None and gap_pct_by_pid is not None:
                gap_pct_by_pid[hi_pid] = gap_pct
            if hi_pid is not None and gap_scale_by_pid is not None:
                gap_scale_by_pid[hi_pid] = scale

            delta_mu = hi_new.mu - hi_old.mu
            sigma_prior = math.sqrt(hi_old.sigma * hi_old.sigma + tau * tau)
            sigma_delta_from_prior = hi_new.sigma - sigma_prior

            new_teams[i][player_index] = model.rating(
                mu=hi_old.mu + delta_mu * scale,
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
                reduction_pct = effective_gap_by_team[idx] * UNBALANCED_TEAM_MU_REDUCTION
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
    """Scale unbalanced-lobby grace down for internally lopsided teams."""
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
    afk_protected_pids=None,
    recent_teammate_repeat_by_pid=None,
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
        afk_protected_pids: Optional set of player_ids whose mu/sigma changes should be zeroed
        recent_teammate_repeat_by_pid: Optional dict[player_id, bool].
            The teammate-gap modifier only scales the higher-mu player's delta, so the
            curve choice is keyed off that specific player's recent-teammate history.
            True uses the existing gap_pct curve. False uses the low-mu trigger curve.

    Returns:
        tuple: (success: bool, updated_player_ratings: dict, modifiers: dict[player_id] -> dict)

        Modifiers dictionary contains per-player tracking values:
        - gap_pct: Relative mu gap (1 - mu_low / mu_high) for the high-mu player in a modified team, 0.0 otherwise.
        - gap_scale: Multiplier applied to the high-mu player's delta (0.05-1.0), 1.0 if no modifier.
        - unbalanced_reduction_pct: Temporary mu reduction percentage for unbalanced GM+ teams, 0.0 otherwise.
        - protection_net: Net points from placement protection/debt redistribution.
          Positive means received protection; negative means paid donor debt.
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

                final_team.append(model.rating(mu=final_mu, sigma=final_sigma))

            new_teams.append(final_team)

        gap_pct_by_pid = {}
        gap_scale_by_pid = {}
        apply_teammate_gap_penalty(
            model,
            teams,
            new_teams,
            logger,
            gm_team_any=gm_team_any,
            team_player_ids=team_player_ids,
            gap_pct_by_pid=gap_pct_by_pid,
            gap_scale_by_pid=gap_scale_by_pid,
            recent_teammate_repeat_by_pid=recent_teammate_repeat_by_pid,
        )

        if afk_protected_pids:
            for i in range(len(teams)):
                team_players = team_player_ids[i]
                for team_player_index, pid in enumerate(team_players):
                    if pid in afk_protected_pids:
                        new_teams[i][team_player_index] = teams[i][team_player_index]

        sorted_placings = sorted(teams_by_placing.keys())
        protection_net_by_pid = {}
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
                donor_rating_after = model.rating(mu=donor_mu, sigma=donor_sigma)
                new_teams[i][team_player_index] = donor_rating_after
                donor_display_before = int(calculate_rating(donor_rating_before))
                donor_display_after = int(calculate_rating(donor_rating_after))
                protection_net_by_pid[pid] -= donor_display_before - donor_display_after

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
                }

        return True, player_ratings, modifiers

    except Exception as e:
        logger.error(f"Failed to update ratings for game {game_id}: {e}")
        return False, player_ratings, {}
