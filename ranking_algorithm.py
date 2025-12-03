import math
from collections import defaultdict
from openskill.models import ThurstoneMostellerFull
from datetime import datetime, timezone

# Constants
RANK_SPLIT = "2025 Split 3"
SPLIT_START_DATE = datetime(2025, 8, 27, tzinfo=timezone.utc)

# Global configuration for teammate gap penalty.
# PENALTY_MIN_MULTIPLIER: lower bound on the multiplier applied to the
#                         higher player's mu/sigma delta once fully penalised.
# GAP_TRIGGER: relative mu gap (0-1) at which the penalty starts to apply.
# GAP_SATURATION: relative mu gap (0-1) at which we are fully at
#                 PENALTY_MIN_MULTIPLIER and remain flat afterwards.
PENALTY_MIN_MULTIPLIER = 0.05
GAP_TRIGGER = 0.10
GAP_SATURATION = 0.50

# Unbalanced lobby configuration.
# A team is considered "unbalanced" if its mu sum is above the lobby's
# median team mu (any positive gap). The check is only performed for teams
# where both players are GM+. For such teams we temporarily reduce their mu by
# UNBALANCED_TEAM_MU_REDUCTION times the fractional gap before calling model.rate.
# After rating updates we apply the resulting delta mu/sigma on top of the
# original (unreduced) mu/sigma.
UNBALANCED_LOBBY_GRACE_ENABLED = True
UNBALANCED_TEAM_MU_REDUCTION = 0.3125   # Apply 31.25% of the gap as a temporary mu reduction

'''
The OpenSkill rating system is an open-source library that provides multiplayer rating algorithms, including the Plackett-Luce model for handling ranked outcomes in multiplayer games.
Like TrueSkill, it represents a player's skill level using a Gaussian distribution, characterized by two key parameters: mu (μ) and sigma (σ).
Mu (μ): The Average Skill
    Mu (μ) represents the system's current estimate of a player's average skill. Think of it as the center of a player's skill range. A higher mu value indicates a higher perceived skill level. When a player wins a game, their mu value increases, and when they lose, it decreases.
Sigma (σ): The Uncertainty
    Sigma (σ) represents the degree of uncertainty the system has about a player's mu value. A high sigma means the system is less confident in its assessment of the player's skill, while a low sigma indicates a higher degree of confidence.
    How it changes: A player's sigma is highest when they are new to the system. As a player participates in more games, providing the system with more data, their sigma value decreases, signifying that the system is becoming more certain about their true skill level. Playing consistently can lead to a lower sigma, while inconsistent performances can result in a higher one.
REFERENCES:
- https://openskill.me/
- https://arxiv.org/abs/2401.05451
- https://pypi.org/project/openskill/
- https://github.com/OpenDebates/openskill.py (Note: This is a fork; original is at https://github.com/vivekjoshy/openskill.py)
'''

def calculate_rating(rating, games_played):
    """Calculate the final rating from mu, sigma and games played"""
    """
    Microsoft research recommends using mu-3*sigma as the "conservative skill estimate" for TrueSkill, and this is commonly applied in similar systems like OpenSkill.
    https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
    """
    base_rating = (rating.mu - 3 * rating.sigma) * 75
    """
    MODIFICATION to the algorithm. (THE RAMP UP)
    Concerns were raised about players at the top of the leaderboard with low games played. Players have also been shown to fluctuate in mu a lot while their sigma is high. A player's sigma is highest when they are new to the system.
    Practically this has lead to a few players having not many games played and a very high skill estimate. To resolve this a scaling factor has been applied. The player's skill estimate ramps up, starting at 50% of the rating at 0 games played, and 100% of the rating at 40 games played. This scales linearly.
    """
    scaling_factor = 0.5 + min(games_played, 40) / 40 * 0.5
    return round(base_rating * scaling_factor)

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
    model = ThurstoneMostellerFull(beta=(25/6) * 4, tau=(25/300))

    return model

def _teammate_penalty_scale(gap_pct: float) -> float:
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

def apply_teammate_gap_penalty(model, teams, new_teams, logger, gm_team_any=None, team_player_ids=None, gap_pct_by_pid=None, gap_scale_by_pid=None):
    """
    Post-process rating updates to dampen gains/losses for the higher-mu player
    in teams with large mu gaps, but only when that player is "high rated".
    """
    for i in range(len(teams)):
        if gm_team_any is not None and not gm_team_any[i]:
            continue

        old_p1, old_p2 = teams[i]
        new_p1, new_p2 = new_teams[i]

        # Identify higher and lower pre-game mu
        if old_p1.mu >= old_p2.mu:
            hi_old, lo_old = old_p1, old_p2
            hi_new, lo_new = new_p1, new_p2
            hi_index = 0
        else:
            hi_old, lo_old = old_p2, old_p1
            hi_new, lo_new = new_p2, new_p1
            hi_index = 1

        mu_hi = hi_old.mu
        mu_lo = lo_old.mu
        if mu_hi <= 0.0:
            continue

        # Relative mu gap (scale-free)
        gap_pct = 1.0 - (mu_lo / mu_hi)
        if gap_pct <= 0.0:
            continue

        if gap_pct < GAP_TRIGGER:
            continue

        gap_pct = min(gap_pct, 1.0)
        scale = _teammate_penalty_scale(gap_pct)
        hi_pid = None
        if team_player_ids is not None:
            ids = team_player_ids[i]
            hi_pid = ids[0] if hi_index == 0 else ids[1]
        if hi_pid is not None and gap_pct_by_pid is not None:
            gap_pct_by_pid[hi_pid] = gap_pct
        if hi_pid is not None and gap_scale_by_pid is not None:
            gap_scale_by_pid[hi_pid] = scale

        # Apply to both mu and sigma deltas for the higher player
        delta_mu = hi_new.mu - hi_old.mu
        delta_sigma = hi_new.sigma - hi_old.sigma

        adj_mu = hi_old.mu + delta_mu * scale
        adj_sigma = hi_old.sigma + delta_sigma * scale

        hi_final = model.rating(mu=adj_mu, sigma=adj_sigma)
        lo_final = lo_new  # unchanged

        # Reassign into new_teams preserving original positional mapping
        if hi_index == 0:
            new_teams[i] = [hi_final, lo_final]
        else:
            new_teams[i] = [lo_final, hi_final]

# ----------------------------------------------------------------------
# UNBALANCED LOBBY helpers
# ----------------------------------------------------------------------

def check_for_unbalanced_lobby(model, teams, logger, gm_team_both_mask=None):
    """
    Decide whether any team is in an "unbalanced lobby" and, if so, prepare
    the adjusted teams list to feed into model.rate.

    Args:
        model: OpenSkill model instance.
        teams: List of teams, where each team is a list of Rating objects.
        logger: Logger or None.
        gm_team_both_mask: Optional list[bool] indicating whether both teammates
            are GM+ for each team (same ordering as teams). If provided, only
            teams with True are eligible for the grace.

    Returns:
        teams_for_rate: None if no adjustments are required; otherwise a new
            list of teams whose players may have adjusted mu values for
            unbalanced teams (and copied ratings for others).
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
    any_unbalanced = False

    if lobby_median_team_mu > 0.0:
        for idx, mu_sum in enumerate(team_mu_sums):
            # Only consider teams where both players are GM (if mask provided)
            if gm_team_both_mask is not None and not gm_team_both_mask[idx]:
                continue

            diff_pct = (mu_sum - lobby_median_team_mu) / lobby_median_team_mu
            if diff_pct > 0.0:
                unbalanced_mask[idx] = True
                any_unbalanced = True
                if logger is not None:
                    logger.debug(
                        "Unbalanced lobby detected for team index %d: "
                        "mu_sum=%.3f lobby_median=%.3f diff_pct=%.3f",
                        idx, mu_sum, lobby_median_team_mu, diff_pct
                    )
    else:
        # Log when median is invalid, especially if GM+ teams are present
        if logger is not None and gm_team_both_mask is not None and any(gm_team_both_mask):
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
                # Reduction is 31.25% of the gap percentage
                mu_sum = team_mu_sums[idx]
                gap_pct = max(0.0, (mu_sum - lobby_median_team_mu) / lobby_median_team_mu)
                reduction_pct = gap_pct * UNBALANCED_TEAM_MU_REDUCTION
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

# ----------------------------------------------------------------------
# Main game-processing function
# ----------------------------------------------------------------------

def process_game_ratings(
    model,
    players,
    game_id,
    player_ratings,
    logger,
    game_date,
    gm_set,
):
    """
    Process a single game's ratings update using OpenSkill ThurstoneMostellerFull with direct team support.
    Assumes exactly 8 teams of 2 players each (16 players total).

    Args:
        model: ThurstoneMostellerFull model instance
        players: List of (player_id, team_placing) tuples
        game_id: Game identifier for logging
        player_ratings: Dictionary of player_id -> Rating
        logger: Logger instance
        game_date: datetime object representing when the game was played

    Returns:
        tuple: (success: bool, updated_player_ratings: dict, modifiers: dict[player_id] -> dict)

        Modifiers dictionary contains per-player tracking values:
        - gap_pct: Relative mu gap (1 - mu_low / mu_high) for the high-mu player in a penalized team, 0.0 otherwise.
        - gap_scale: Multiplier applied to the high-mu player's delta (0.05-1.0), 1.0 if no penalty.
        - sigma_cap_scale: Multiplier applied to sigma for GM+ teams (typically 0.0-1.0).
          Note: This is tracked for all players; value is always 1.0 for non-GM teams.
        - unbalanced_reduction_pct: Temporary mu reduction percentage for unbalanced GM+ teams, 0.0 otherwise.
    """
    
    # Verify exactly 16 players
    if len(players) != 16:
        logger.warning(f"Game {game_id} has {len(players)} players, expected 16")
        return False, player_ratings, {}

    # Group players by team_placing (1-8)
    teams_by_placing = defaultdict(list)
    for player_id, team_placing in players:
        teams_by_placing[team_placing].append(player_id)

    # Verify exactly 8 teams of 2 players each
    if len(teams_by_placing) != 8:
        logger.warning(f"Game {game_id} has {len(teams_by_placing)} teams, expected 8")
        return False, player_ratings, {}

    for placing, team_players in teams_by_placing.items():
        if len(team_players) != 2:
            logger.warning(f"Game {game_id} team placing {placing} has {len(team_players)} players, expected 2")
            return False, player_ratings, {}

    # Prepare teams in order of placing 1 (best) to 8 (worst)
    teams = []
    gm_team_any = []
    gm_team_both = []
    team_player_ids = []
    for placing in sorted(teams_by_placing.keys()):  # 1 to 8
        team_players = teams_by_placing[placing]
        team_ratings = [player_ratings.get(pid, model.rating()) for pid in team_players]
        teams.append(team_ratings)
        team_player_ids.append(team_players)
        if gm_set is not None:
            gm_count = sum(1 for pid in team_players if pid in gm_set)
            gm_team_any.append(gm_count >= 1)
            gm_team_both.append(gm_count == 2)
        else:
            gm_team_any.append(False)
            gm_team_both.append(False)

    ranks = list(range(len(teams)))

    try:
        rate_input = []
        sigma_cap_scale_by_pid = {}
        for idx, team_ratings in enumerate(teams):
            t_pids = team_player_ids[idx]
            if gm_team_any[idx]:
                r0, r1 = team_ratings
                if r0.mu >= r1.mu:
                    s_high = r0.sigma
                    s_low = r1.sigma
                else:
                    s_high = r1.sigma
                    s_low = r0.sigma
                if s_low <= s_high:
                    k = 1.0
                else:
                    current_team_sigma = math.hypot(r0.sigma, r1.sigma)
                    target_team_sigma = math.hypot(s_high, s_high)
                    if current_team_sigma > 0:
                        k = target_team_sigma / current_team_sigma
                    else:
                        k = 1.0
                        logger.warning(
                            f"Game {game_id} team index {idx}: current_team_sigma is 0 "
                            f"(r0.sigma={r0.sigma}, r1.sigma={r1.sigma})"
                        )
                sigma_cap_scale_by_pid[t_pids[0]] = k
                sigma_cap_scale_by_pid[t_pids[1]] = k
                if k != 1.0:
                    rate_input.append([
                        model.rating(mu=r0.mu, sigma=r0.sigma * k),
                        model.rating(mu=r1.mu, sigma=r1.sigma * k),
                    ])
                else:
                    rate_input.append([
                        model.rating(mu=r0.mu, sigma=r0.sigma),
                        model.rating(mu=r1.mu, sigma=r1.sigma),
                    ])
            else:
                sigma_cap_scale_by_pid[t_pids[0]] = 1.0
                sigma_cap_scale_by_pid[t_pids[1]] = 1.0
                rate_input.append([
                    model.rating(mu=r.mu, sigma=r.sigma) for r in team_ratings
                ])

        adjusted_teams, unbalanced_reductions = check_for_unbalanced_lobby(model, rate_input, logger, gm_team_both_mask=gm_team_both)
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

        days_since_split_start = (game_date - SPLIT_START_DATE).days
        gap_pct_by_pid = {}
        gap_scale_by_pid = {}
        if days_since_split_start >= 5:
            apply_teammate_gap_penalty(
                model,
                teams,
                new_teams,
                logger,
                gm_team_any=gm_team_any,
                team_player_ids=team_player_ids,
                gap_pct_by_pid=gap_pct_by_pid,
                gap_scale_by_pid=gap_scale_by_pid,
            )

        sorted_placings = sorted(teams_by_placing.keys())
        modifiers = {}
        # Note: Index i in the loop below corresponds to team position in teams/new_teams/unbalanced_reductions
        # because all three were built by iterating sorted(teams_by_placing.keys()) at line 364
        for i, placing in enumerate(sorted_placings):
            team_players = teams_by_placing[placing]
            new_team = new_teams[i]
            player_ratings[team_players[0]] = new_team[0]
            player_ratings[team_players[1]] = new_team[1]
            for pid in team_players:
                modifiers[pid] = {
                    "gap_pct": gap_pct_by_pid.get(pid, 0.0),
                    "gap_scale": gap_scale_by_pid.get(pid, 1.0),
                    "sigma_cap_scale": sigma_cap_scale_by_pid.get(pid, 1.0),
                    "unbalanced_reduction_pct": unbalanced_reductions[i]  # Always a list at this point (set at line 427)
                }

        return True, player_ratings, modifiers

    except Exception as e:
        logger.error(f"Failed to update ratings for game {game_id}: {e}")
        return False, player_ratings, {}
