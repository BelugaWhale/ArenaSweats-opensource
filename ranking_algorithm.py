import math
from collections import defaultdict
from openskill.models import ThurstoneMostellerFull

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
    MODIFICATION 1/1 to the algorithm. (THE LOW GAMES CLAMP)
    "A player's sigma is highest when they are new to the system."
    Practically this has lead to a few players having not many games played and a very high skill estimate. To resolve this a scaling factor has been applied. The player's skill estimate is 50% of the rating at 0 games played, and 100% of the rating at 40 games played. This scales linearly.
    """
    scaling_factor = 0.5 + min(games_played, 40) / 40 * 0.5
    return round(base_rating * scaling_factor)

def instantiate_rating_model():
    """
    Creates and returns a PlackettLuce model from OpenSkill.
    The PlackettLuce class constructor can be customized with several
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
    kappa (float): The "dynamic factor" added to the variance before each match
        to account for performance variability. Similar to tau in TrueSkill.
        The default value is 0.0001.
    gamma (callable): Custom function you can pass that must contain parameters 
        for rank, num_teams, mu, sigma, team, player_index, and optionally scores. 
        The function must return a float or int. It represents the amount added to 
        the winning team's sigma before updating. Represented by: γ. The default 
        is the internal _gamma function.
    epsilon (float): Arbitrary small positive real number that is used to prevent the 
        variance of the posterior distribution from becoming too small or negative. It 
        can also be thought of as a regularization parameter. Represented by: κ. The default value is 0.0001.
    margin (float): The margin of victory needed for a win to be considered impressive. 
        This parameter is useful in games where the difference in scores matters, adjusting 
        how much a close win or loss affects the rating update. The default is 0.0.
    balance (bool): Boolean that determines whether to emphasize rating outliers during 
        the update process, potentially affecting how extreme ratings are handled. The default is False.
    
    REFERENCE:
    - https://openskill.me/en/stable/models.html#plackettluce
    """
    """
    CURRENTLY ALL PARAMETERS ARE SET TO DEFAULT
    """
    # This instantiation creates a model for games with strict rankings (no draws).
    return ThurstoneMostellerFull(beta=(25/6) * 4 , tau=(25/300))

def apply_convergence(model, teams, new_teams):
    """
    Apply convergence logic to team ratings after model.rate update.
    This function modifies new_teams in place to implement mu-based convergence
    by unevenly distributing the team mu delta, favoring lower-rated players in positive updates 
    and protecting them in negative ones. For sigma, similarly distribute the team variance delta
    using the same modifiers, but flipped for positive deltas to give more uncertainty reduction 
    (less added variance) to the player with larger mu modifier.
    Dynamic bias quadratic in initial mu diff and beta-normalized.
    """
    CONVERGENCE_STRENGTH = 0.006  # Single tunable param: higher = stronger bias on large gaps
    for i in range(len(teams)):
        old_p1 = teams[i][0]
        old_p2 = teams[i][1]
        new_p1_temp = new_teams[i][0]
        new_p2_temp = new_teams[i][1]
        
        # Compute initial mu difference for dynamic bias
        diff = abs(old_p1.mu - old_p2.mu)
        bias = CONVERGENCE_STRENGTH * (diff / model.beta) ** 2
        bias = min(bias, 0.4)  # Cap for mu bias (<0.5 for stability)
        
        # Compute team-level mu delta from library update
        old_mu_team = old_p1.mu + old_p2.mu
        new_mu_team = new_p1_temp.mu + new_p2_temp.mu
        delta_mu_team = new_mu_team - old_mu_team
        
        # Var deltas
        old_var_1 = old_p1.sigma ** 2
        old_var_2 = old_p2.sigma ** 2
        new_var_team = new_p1_temp.sigma ** 2 + new_p2_temp.sigma ** 2
        delta_var_team = new_var_team - (old_var_1 + old_var_2)
        
        # --- Determine distribution modifiers ---
        if old_p1.mu <= old_p2.mu:
            # p1 weaker, p2 stronger
            if delta_mu_team > 0:
                # reward weaker more
                mod1, mod2 = 0.5 + bias, 0.5 - bias
            elif delta_mu_team < 0:
                # protect weaker, punish stronger
                mod1, mod2 = 0.5 - bias, 0.5 + bias
            else:
                mod1 = mod2 = 0.5
        else:
            # p2 weaker, p1 stronger
            if delta_mu_team > 0:
                mod1, mod2 = 0.5 - bias, 0.5 + bias
            elif delta_mu_team < 0:
                mod1, mod2 = 0.5 + bias, 0.5 - bias
            else:
                mod1 = mod2 = 0.5
        
        # Determine var modifiers: same as mu for reductions (higher mod gets more negative),
        # flipped for increases (higher mod gets less positive)
        var_mod1, var_mod2 = mod1, mod2
        if delta_var_team >= 0:
            var_mod1, var_mod2 = 1 - mod1, 1 - mod2  # Flip to minimize added uncertainty for higher mu mod
        
        # --- Final mus and vars ---
        final_mu_1 = old_p1.mu + delta_mu_team * mod1
        final_mu_2 = old_p2.mu + delta_mu_team * mod2
        final_var_1 = max(old_var_1 + delta_var_team * var_mod1, 0)
        final_var_2 = max(old_var_2 + delta_var_team * var_mod2, 0)
        
        # Sigmas
        final_sigma_1 = math.sqrt(final_var_1)
        final_sigma_2 = math.sqrt(final_var_2)
        
        # Replace with converged ratings
        new_teams[i] = [
            model.rating(mu=final_mu_1, sigma=final_sigma_1),
            model.rating(mu=final_mu_2, sigma=final_sigma_2)
        ]

def process_game_ratings(model, players, game_id, player_ratings, logger):
    """
    Process a single game's ratings update using OpenSkill ThurstoneMostellerFull with direct team support.
    Assumes exactly 8 teams of 2 players each (16 players total).
    
    Args:
        model: ThurstoneMostellerFull model instance
        players: List of (player_id, team_placing) tuples
        game_id: Game identifier for logging
        player_ratings: Dictionary of player_id -> Rating
        logger: Logger instance
    
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
        apply_convergence(model, teams, new_teams)
        
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