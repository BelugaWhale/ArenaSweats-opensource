import math
from collections import defaultdict
from openskill.models import ThurstoneMostellerFull
from datetime import datetime, timezone

# Constants
RANK_SPLIT = "2025 Split 3"
SPLIT_START_DATE = datetime(2025, 8, 27, tzinfo=timezone.utc)

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
    return ThurstoneMostellerFull(beta=(25/6) * 4 , tau=(25/300))

def anti_boost(model, teams, new_teams, logger):
    """
    Prevent boosting in team ratings after model.rate update.

    MODIFICATION to the algorithm. (ANTI-BOOSTING)
    Concerns have been raised about high rated players that are boosted, meaning they have played with a second account where the player is of a significantly higher skill level than the account match-making skill level. Once the account match-making skill level matches the player's skill level, they switch to a different account. The primary account gets the benefit of easier games relative to player skills for all games it plays. This is called boosting.
    Initially convergence was going to be used to resolve this issue, this adds an additional modifier to the updates which aims to close the gap between the two players skill level by altering gains and losses. This would reduce boosting benefit significantly. While initial convergence approach did not result in favourable results, it could still work, but it has several nuanced consequences that have to be considered.
    From the a boosting penalty system was introduced, but now the penalty has been removed and instead its more a prevention system. This significantly reduces the impact of boosting without penalizing players who get caught in the false positives.

    This function works by reducing the mu update for the higher-rated player in a team if there is a significant skill gap (mu) or uncertainty gap (sigma). The process is skipped if the absolute difference in skill is less than MIN_MU_DIFF (30) or if the season is less than five days old, allowing initial ratings to settle.

    Regarding uncertainty gap, uncertainty is highest when player's are new to the system. So a brand new account would have a sigma of 8.33, while an established account, lets say 100 games would have a sigma of less than 3. If these two player's played together, the system would first calculate diff_sigma, the difference in player uncertainty normalized by SIGMA_SPREAD (the typical range of sigma values from a new to a settled player). In this case, the difference would be well over 50% (SIGMA_THRESHOLD) and so for the higher rated player this is a low impact game, meaning the rating change from the game, is reduced by 80% (SIGMA_PENALTY).

    For skill gap, if the lower rated player has a skill level less than 40% (1-MU_THRESHOLD) of the higher rated player, then it is a low impact game fo the higher rated player, meaning a bias of 75% to 95% (BIAS_MIN to BIAS_MAX) is applied, reducing the rating change for the higher rated player by the amount. This scales linearly and maxes when the lower rated player has a skill level of less than 20% of the higher rated player (1-MU_SCALE_END).
    """
    SIGMA_SPREAD = 8.33 - 1.5  # Fixed: 6.83, initial to steady-state sigma
    MIN_MU_DIFF = 30  # Tunable: Minimum absolute mu difference to apply non-zero diff_mu
    
    for i in range(len(teams)):
        old_p1 = teams[i][0]
        old_p2 = teams[i][1]
        new_p1_temp = new_teams[i][0]
        new_p2_temp = new_teams[i][1]
        
        # Identify weaker and stronger
        p1_weaker = old_p1.mu <= old_p2.mu
        if p1_weaker:
            weaker_mu = old_p1.mu
            stronger_mu = old_p2.mu
        else:
            weaker_mu = old_p2.mu
            stronger_mu = old_p1.mu
        
        # Compute diff_mu with ratio-based formula
        abs_mu_diff = abs(weaker_mu - stronger_mu)
        if abs_mu_diff < MIN_MU_DIFF:
            diff_mu = 0.0
        else:
            diff_mu = 1 - abs(weaker_mu) / abs_mu_diff
            diff_mu = min(1.0, max(0.0, diff_mu))  # Normalize to [0,1] for consistent blending
        
        # Compute diff_sigma
        diff_sigma = abs(old_p1.sigma - old_p2.sigma) / SIGMA_SPREAD if SIGMA_SPREAD > 0 else 0.0
        
        # New bias logic: separate thresholds with linear scaling for mu
        MU_THRESHOLD = 0.6
        SIGMA_THRESHOLD = 0.5
        SIGMA_PENALTY = 0.8
        MU_SCALE_END = 0.8
        BIAS_MIN = 0.75
        BIAS_MAX = 0.95
        
        if diff_mu < MU_THRESHOLD and diff_sigma < SIGMA_THRESHOLD:
            continue  # Do nothing for this team
        elif diff_mu < MU_THRESHOLD:  # diff_sigma >= SIGMA_THRESHOLD, apply fixed sigma bias
            bias = SIGMA_PENALTY  # 0.75
        else:  # diff_mu >= MU_THRESHOLD, apply scaled mu bias (ignores sigma)
            fraction = min(1.0, (diff_mu - MU_THRESHOLD) / (MU_SCALE_END - MU_THRESHOLD))
            bias = BIAS_MIN + fraction * (BIAS_MAX - BIAS_MIN)
        
        # Compute deltas and apply bias
        delta_mu_1 = new_p1_temp.mu - old_p1.mu
        delta_mu_2 = new_p2_temp.mu - old_p2.mu
        if p1_weaker:
            # Weaker unchanged
            final_mu_1 = new_p1_temp.mu
            # Adjust stronger
            final_delta_2 = delta_mu_2 * (1 - bias)
            final_mu_2 = old_p2.mu + final_delta_2
        else:
            # Weaker unchanged
            final_mu_2 = new_p2_temp.mu
            # Adjust stronger
            final_delta_1 = delta_mu_1 * (1 - bias)
            final_mu_1 = old_p1.mu + final_delta_1
        
        # Replace with new ratings
        new_teams[i] = [
            model.rating(mu=final_mu_1, sigma=new_p1_temp.sigma),
            model.rating(mu=final_mu_2, sigma=new_p2_temp.sigma)
        ]

def process_game_ratings(model, players, game_id, player_ratings, logger, game_date):
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
        tuple: (success: bool, updated_player_ratings: dict)
    """
    
    # Verify exactly 16 players
    if len(players) != 16:
        logger.warning(f"Game {game_id} has {len(players)} players, expected 16")
        return False, player_ratings
   
    # Group players by team_placing (1-8)
    teams_by_placing = defaultdict(list)
    for player_id, team_placing in players:
        teams_by_placing[team_placing].append(player_id)
   
    # Verify exactly 8 teams of 2 players each
    if len(teams_by_placing) != 8:
        logger.warning(f"Game {game_id} has {len(teams_by_placing)} teams, expected 8")
        return False, player_ratings
   
    for placing, team_players in teams_by_placing.items():
        if len(team_players) != 2:
            logger.warning(f"Game {game_id} team placing {placing} has {len(team_players)} players, expected 2")
            return False, player_ratings
   
    # Prepare teams in order of placing 1 (best) to 8 (worst)
    teams = []
    for placing in sorted(teams_by_placing.keys()):  # 1 to 8
        team_players = teams_by_placing[placing]
        team_ratings = [player_ratings.get(pid, model.rating()) for pid in team_players]
        teams.append(team_ratings)
   
    # Ranks: lower number is better (0 for placing 1, 1 for placing 2, ..., 7 for placing 8)
    ranks = list(range(len(teams)))
   
    # Rate the teams directly
    try:
        new_teams = model.rate(teams, ranks=ranks)
        
        # Skip penalize_boosting if game is within 5 days of split start
        days_since_split_start = (game_date - SPLIT_START_DATE).days
        if days_since_split_start >= 5:
            anti_boost(model, teams, new_teams, logger)
       
        # Update player_ratings
        sorted_placings = sorted(teams_by_placing.keys())
        for i, placing in enumerate(sorted_placings):
            team_players = teams_by_placing[placing]
            new_team = new_teams[i]
            player_ratings[team_players[0]] = new_team[0]
            player_ratings[team_players[1]] = new_team[1]
        
        return True, player_ratings
   
    except Exception as e:
        logger.error(f"Failed to update ratings for game {game_id}: {e}")
        return False, player_ratings