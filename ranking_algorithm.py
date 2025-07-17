import trueskill
from collections import defaultdict

'''
The TrueSkill rating system, a sophisticated algorithm developed by Microsoft, assesses player skill in multiplayer games through a Bayesian approach.

At its core, TrueSkill moves beyond a single rating number and represents a player's skill level using a Gaussian distribution, characterized by two key parameters: mu (μ) and sigma (σ).

Mu (μ): The Average Skill
    Mu (μ) represents the system's current estimate of a player's average skill. Think of it as the center of a player's skill range. A higher mu value indicates a higher perceived skill level. When a player wins a game, their mu value increases, and when they lose, it decreases.

Sigma (σ): The Uncertainty

    Sigma (σ) represents the degree of uncertainty the system has about a player's mu value. A high sigma means the system is less confident in its assessment of the player's skill, while a low sigma indicates a higher degree of confidence.

    How it changes: A player's sigma is highest when they are new to the system. As a player participates in more games, providing the system with more data, their sigma value decreases, signifying that the system is becoming more certain about their true skill level. Playing consistently can lead to a lower sigma, while inconsistent performances can result in a higher one.

REFERENCES:
- https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
- https://trueskill.org/
- https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/NIPS2006_0688.pdf
- https://medium.com/aimonks/the-trueskill-algorithm-revolutionizing-player-matchmaking-and-skill-assessment-in-online-gaming-1fadcbdb2eb9
'''


def calculate_rating(trueskill_rating, games_played):
    """Calculate the final rating from TrueSkill mu, sigma and games played"""

    """
    Microsoft research recommends using mu-3*sigma as the "conservative skill estimate"
    https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
    """
    base_rating = (trueskill_rating.mu - 3 * trueskill_rating.sigma) * 100

    """
    MODIFICATION 1/1 to TrueSkill algorithm. (THE LOW GAMES CLAMP)
    "A player's sigma is highest when they are new to the system." 
    Practically this has lead to a few players having not many games played and a very high skill estimate. To resolve this a scaling factor has been applied. The player's skill estimate is 80% of the TrueSkill rating at 0 games played, and 100% of the TrueSkill rating at 30 games played. This scales linearly.
    """
    scaling_factor = 0.8 + min(games_played, 30) / 30 * 0.2
    return round(base_rating * scaling_factor)


def instantiate_trueskill():
    """
    Creates and returns a TrueSkill environment.

    The trueskill.TrueSkill class constructor can be customized with several
    parameters that define the behavior of the rating system. These parameters
    are based on the mathematical model of the TrueSkill algorithm.

    Parameters:
    mu (float): The initial mean of a player's skill (μ). This represents the
        assumed skill level of a new player before any matches have been played.
        The default value is 25.0, as used in Xbox Live. [1, 2]

    sigma (float): The initial standard deviation of a player's skill (σ).
        This represents the system's uncertainty about the player's initial
        skill. A higher value means the system is less certain. As a player
        plays more games, their sigma will decrease. The default is 25.0 / 3.0. [1, 2]

    beta (float): The "skill variance" that defines the distance in skill points
        that gives a player an 80% chance of winning against another. A smaller
        beta means that a smaller skill difference has a greater impact on the
        win probability, making the system more sensitive to skill gaps.
        The default is 25.0 / 6.0. [1, 2]

    tau (float): The "dynamic factor" or "variance of performance" (τ). This value
        is added to a player's sigma before a match to account for performance
        variability. A higher tau means the system assumes that a player's
        performance can vary significantly from their true skill from one game
        to the next. The default value is 25.0 / 300.0. [1, 2]

    draw_probability (float): The probability of a draw occurring in a match.
        This value is used to adjust the rating updates in games that can end
        in a tie. Setting it to 0.0, as done here, assumes that draws are
        impossible, which is suitable for games where there is always a winner
        and a loser (e.g., chess). The default value is 0.10 (10%). [1, 3]

    backend (str or tuple, optional): Specifies the numerical backend to be
        used for calculations. This can be 'scipy', 'mpmath', or a tuple
        containing custom functions for more advanced use cases. The default
        is None, which auto-detects the best available backend. [1]
    
    REFERENCE:
    - https://trueskill.org/
    """
    """
    CURRENTLY ALL PARAMETERS ARE SET TO DEFAULT
    (EXCEPT draw_probability since you can't draw in Arena)
    """
    # This instantiation creates an environment for a game where draws cannot happen.
    return trueskill.TrueSkill(draw_probability=0.0)


def process_game_ratings(env, players, game_id, player_ratings, logger):
    """
    Process a single game's ratings update using TrueSkill.
    
    Args:
        env: TrueSkill environment
        players: List of (player_id, team_placing) tuples
        game_id: Game identifier for logging
        player_ratings: Dictionary of player_id -> Rating
        logger: Logger instance
    
    Returns:
        tuple: (success: bool, updated_player_ratings: dict)
    """
    # Group players by team_placing (1-8)
    teams_by_placing = defaultdict(list)
    for player_id, team_placing in players:
        teams_by_placing[team_placing].append(player_id)
    
    # Arena has 8 teams of 2 players, if we don't have this there is an issue with the data and the game should not be processed. (Less than 100 games are failing across all regions atm)
    # Verify we have exactly 8 teams 
    if len(teams_by_placing) != 8:
        logger.warning(f"Game {game_id} has {len(teams_by_placing)} teams, expected 8")
        return False, player_ratings
    
    # Check that each team has exactly 2 players
    for placing, team_players in teams_by_placing.items():
        if len(team_players) != 2:
            logger.warning(f"Game {game_id} team placing {placing} has {len(team_players)} players, expected 2")
            return False, player_ratings
    
    # Prepare teams and ranks for TrueSkill without modifying player_ratings
    teams = []
    ranks = []
    for placing in sorted(teams_by_placing.keys()):
        team_players = teams_by_placing[placing]
        team_ratings = [player_ratings.get(pid, env.Rating()) for pid in team_players]
        teams.append(team_ratings)
        ranks.append(placing - 1)  # Convert 1-8 placing to 0-7 ranks
    
    # Update ratings using TrueSkill
    try:
        new_teams = env.rate(teams, ranks=ranks)
        
        # Update player_ratings in place (adds new players if needed)
        for i, new_team in enumerate(new_teams):
            placing = ranks[i] + 1  # Convert back to 1-8 placing
            team_players = teams_by_placing[placing]
            for j, pid in enumerate(team_players):
                player_ratings[pid] = new_team[j]
        
        return True, player_ratings
        
    except Exception as e:
        logger.error(f"Failed to update ratings for game {game_id}: {e}")
        return False, player_ratings