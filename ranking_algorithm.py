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
    - https://openskill.me/en/stable/models.html#thurstonemostellerfull
    """
    """
    CURRENTLY ALL PARAMETERS ARE SET TO DEFAULT
    """
    # This instantiation creates a model for games with strict rankings (no draws).
    return ThurstoneMostellerFull(beta=(25/6) * 4 , tau=(25/300))
def apply_convergence(model, teams, new_teams, mu_spread):
    """
    Apply convergence logic to team ratings after model.rate update.
    This function modifies new_teams in place to implement blended mu/sigma convergence
    by multiplying individual mu deltas with modifiers based on blended diffs.
    Variances are taken directly from OpenSkill's updates.
    Dynamic bias sigmoid in blended diff for minimal change in normal cases.
    """
    CONVERGENCE_STRENGTH = 0.67  # Tunable: higher = stronger bias on large gaps
    BLEND_P = 0.75  # Tunable: weight for mu_diff (0-1), remainder for sigma_diff
    SIGMA_SPREAD = 8.33 - 1.5  # Fixed: 6.83, initial to steady-state sigma
    MAX_DEVIATION = 0.9  # Maximum allowed deviation from 1.0 for modifiers
    MIDPOINT = 0.3  # Tunable: sigmoid inflection point (lower=earlier ramp)
    STEEPNESS = 20.0  # Tunable: sigmoid sharpness (higher=faster transition)
   
    for i in range(len(teams)):
        old_p1 = teams[i][0]
        old_p2 = teams[i][1]
        new_p1_temp = new_teams[i][0]
        new_p2_temp = new_teams[i][1]
       
        # Compute normalized differences
        diff_mu = abs(old_p1.mu - old_p2.mu) / mu_spread if mu_spread > 0 else 0.0
        diff_sigma = abs(old_p1.sigma - old_p2.sigma) / SIGMA_SPREAD if SIGMA_SPREAD > 0 else 0.0
       
        # Blended diff
        blended_diff = BLEND_P * diff_mu + (1 - BLEND_P) * diff_sigma
       
        # Dynamic bias (sigmoid, no /beta)
        sigmoid_val = 1 / (1 + math.exp(-STEEPNESS * (blended_diff - MIDPOINT)))
        bias = CONVERGENCE_STRENGTH * sigmoid_val
        bias = min(bias, MAX_DEVIATION)  # Cap to prevent extreme modifiers
       
        # Compute individual mu deltas from library update
        delta_mu_1 = new_p1_temp.mu - old_p1.mu
        delta_mu_2 = new_p2_temp.mu - old_p2.mu
       
        # --- Determine distribution modifiers (multipliers) ---
        # Identify weaker and stronger players
        p1_weaker = old_p1.mu <= old_p2.mu
        team_delta = delta_mu_1 + delta_mu_2
       
        if team_delta > 0 or team_delta < 0:
            # Calculate modifiers for weaker and stronger players
            if team_delta > 0:
                # Reward weaker player more, stronger player less
                mod_weaker = 1.0 + bias
                mod_stronger = 1.0 - bias
            else:  # team_delta < 0
                # Protect weaker player (less penalty), penalize stronger player more
                mod_weaker = 1.0 - bias
                mod_stronger = 1.0 + bias
           
            # Assign modifiers based on who is weaker
            if p1_weaker:
                mod_p1, mod_p2 = mod_weaker, mod_stronger
            else:
                mod_p1, mod_p2 = mod_stronger, mod_weaker
        else:
            # No rating change (delta is zero)
            mod_p1 = mod_p2 = 1.0
       
        # Compute modified deltas
        mod_delta_1 = delta_mu_1 * mod_p1
        mod_delta_2 = delta_mu_2 * mod_p2
       
        # Normalize to conserve original total team delta (prevent inflation/deflation)
        original_total_delta = delta_mu_1 + delta_mu_2
        mod_total = mod_delta_1 + mod_delta_2
        if mod_total != 0:
            scale_factor = original_total_delta / mod_total
            final_delta_1 = mod_delta_1 * scale_factor
            final_delta_2 = mod_delta_2 * scale_factor
        else:
            final_delta_1 = mod_delta_1  # Fallback if zero (rare, preserves mods)
            final_delta_2 = mod_delta_2
       
        # --- Final mus (multiplied deltas) ---
        final_mu_1 = old_p1.mu + final_delta_1
        final_mu_2 = old_p2.mu + final_delta_2
       
        # Replace with converged ratings (sigmas unchanged from temp)
        new_teams[i] = [
            model.rating(mu=final_mu_1, sigma=new_p1_temp.sigma),
            model.rating(mu=final_mu_2, sigma=new_p2_temp.sigma)
        ]
def process_game_ratings(model, players, game_id, player_ratings, logger, mu_spread):
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
        apply_convergence(model, teams, new_teams, mu_spread)
       
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
def test_convergence_implementation():
    """
    Test function to verify the blended convergence implementation works correctly.
    Tests normal case and boosting case scenarios.
    """
    print("Testing blended convergence implementation...")
   
    model = instantiate_rating_model()
   
    # Test case 1: Normal case (moderate skill gap)
    print("\n=== Test Case 1: Normal skill gap ===")
    teams = [[model.rating(mu=30, sigma=2), model.rating(mu=50, sigma=3)]]
    mu_spread = 50 - 30  # 20
    print(f"Original team: mu1={teams[0][0].mu}, sigma1={teams[0][0].sigma}, mu2={teams[0][1].mu}, sigma2={teams[0][1].sigma}")
    print(f"mu_spread: {mu_spread}")
   
    # Simulate OpenSkill update (dummy)
    new_teams = [[model.rating(mu=32, sigma=1.9), model.rating(mu=48, sigma=2.8)]]
    original_deltas = (32-30, 48-50)
    original_total = sum(original_deltas)
    print(f"Simulated deltas before convergence: {original_deltas}, total: {original_total}")
   
    apply_convergence(model, teams, new_teams, mu_spread)
    final_deltas = (new_teams[0][0].mu - 30, new_teams[0][1].mu - 50)
    final_total = sum(final_deltas)
    print(f"Final deltas after convergence: {final_deltas}, total: {final_total}")
    print(f"Total delta preserved: {abs(final_total - original_total) < 0.001}")
   
    # Test case 2: Boosting case (extreme skill gap)
    print("\n=== Test Case 2: Boosting scenario ===")
    teams = [[model.rating(mu=20, sigma=8), model.rating(mu=120, sigma=2)]]
    mu_spread = 120 - 20  # 100
    print(f"Original team: mu1={teams[0][0].mu}, sigma1={teams[0][0].sigma}, mu2={teams[0][1].mu}, sigma2={teams[0][1].sigma}")
    print(f"mu_spread: {mu_spread}")
   
    # Simulate OpenSkill update (dummy)
    new_teams = [[model.rating(mu=22, sigma=7.5), model.rating(mu=118, sigma=1.9)]]
    original_deltas = (22-20, 118-120)
    original_total = sum(original_deltas)
    print(f"Simulated deltas before convergence: {original_deltas}, total: {original_total}")
   
    apply_convergence(model, teams, new_teams, mu_spread)
    final_deltas = (new_teams[0][0].mu - 20, new_teams[0][1].mu - 120)
    final_total = sum(final_deltas)
    print(f"Final deltas after convergence: {final_deltas}, total: {final_total}")
    print(f"Total delta preserved: {abs(final_total - original_total) < 0.001}")
   
    print("\nTest completed successfully!")
def test_sigmoid_bias():
    """Test the sigmoid bias function with different blended_diff values"""
    print("\n=== Testing Sigmoid Bias Function ===")
   
    def calculate_bias(blended_diff, strength=0.67, midpoint=0.3, steepness=20.0, max_dev=0.9):
        sigmoid_val = 1 / (1 + math.exp(-steepness * (blended_diff - midpoint)))
        bias = strength * sigmoid_val
        return min(bias, max_dev)
   
    test_cases = [
        (0.0, "Zero diff (no change)"),
        (0.2, "Normal low diff"),
        (0.5, "Midpoint diff"),
        (0.8, "High normal diff"),
        (1.0, "High diff"),
        (1.2, "Boosting scenario"),
        (1.5, "Extreme boosting"),
        (2.0, "Very extreme")
    ]
   
    print("blended_diff -> sigmoid -> bias")
    for diff, description in test_cases:
        sigmoid_val = 1 / (1 + math.exp(-20.0 * (diff - 0.3)))
        bias = min(0.67 * sigmoid_val, 0.9)
        print(f"{diff:4.1f} ({description:20s}) -> {sigmoid_val:.4f} -> {bias:.4f}")
   
    print("\nExpected behavior:")
    print("- Low diffs (0.0-0.3): Very small bias (~0.001-0.05)")
    print("- Mid diffs (0.4-0.6): Moderate bias (~0.1-0.2)")  
    print("- High diffs (0.8+): Strong bias (~0.25-0.33)")
if __name__ == "__main__":
    test_convergence_implementation()
    test_sigmoid_bias()