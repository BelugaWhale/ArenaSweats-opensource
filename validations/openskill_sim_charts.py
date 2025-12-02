#!/usr/bin/env python3
"""
OpenSkill Simulation Experiments - Charting and Analysis Module

This module contains the experiment logic for analyzing sigma and mu impacts
on team ratings in OpenSkill simulations.
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("Note: matplotlib not available - skipping chart generation")


@dataclass
class AnalysisResult:
    sigma_varied: float
    player_mu_change: float
    player_sigma_change: float
    player_rating_change: float
    teammate_mu_change: float
    teammate_sigma_change: float
    teammate_rating_change: float


def calculate_rating(mu: float, sigma: float) -> float:
    """Calculate final rating using (mu - 3*sigma) * 100 formula"""
    return (mu - 3 * sigma) * 100


def run_sigma_analysis(model, players: List[Dict], teams: List[List[str]], placings: List[int]):
    """Run sigma analysis on a random team and player"""
    # Pick a random team to analyze
    random_team_idx = random.randint(0, len(teams) - 1)
    selected_team = teams[random_team_idx]
    team_placing = placings[random_team_idx]

    # Get the two players from this team
    player1_id, player2_id = selected_team[0], selected_team[1]

    # Find player data
    player1_data = next(p for p in players if p["id"] == player1_id)
    player2_data = next(p for p in players if p["id"] == player2_id)

    # Randomly pick which player's sigma to vary
    if random.choice([True, False]):
        varied_player = player1_data
        fixed_player = player2_data
        varied_player_id = player1_id
        fixed_player_id = player2_id
    else:
        varied_player = player2_data
        fixed_player = player1_data
        varied_player_id = player2_id
        fixed_player_id = player1_id

    print(f"\n🎲 Random Selection Results:")
    print(f"   Team: {varied_player_id} + {fixed_player_id} (placing: {team_placing})")
    print(f"   Varying sigma for: {varied_player_id} ({varied_player.get('name', 'Unknown')})")
    print(f"   Fixed teammate: {fixed_player_id} ({fixed_player.get('name', 'Unknown')})")

    # Get base stats
    base_mu = varied_player.get("mu", 25.0)
    original_sigma = varied_player.get("sigma", 25.0/3.0)
    teammate_mu = fixed_player.get("mu", 25.0)
    teammate_sigma = fixed_player.get("sigma", 25.0/3.0)

    # Range of sigma values to test
    sigma_values = np.linspace(1.5, 8.3, 25)  # 25 points from 1.5 to 8.3

    results = []

    for test_sigma in sigma_values:
        # Create ratings for this test
        varied_rating = model.rating(mu=base_mu, sigma=test_sigma)
        fixed_rating = model.rating(mu=teammate_mu, sigma=teammate_sigma)

        # Build all teams with their ratings and placings
        all_teams = []
        all_placings = []

        for i, team in enumerate(teams):
            if i == random_team_idx:  # Our selected team
                all_teams.append([varied_rating, fixed_rating])
                all_placings.append(placings[i])
            else:  # Opponent teams
                p1_data = next(p for p in players if p["id"] == team[0])
                p2_data = next(p for p in players if p["id"] == team[1])
                p1_mu = p1_data.get("mu", 25.0)
                p1_sigma = p1_data.get("sigma", 25.0/3.0)
                p2_mu = p2_data.get("mu", 25.0)
                p2_sigma = p2_data.get("sigma", 25.0/3.0)
                p1_rating = model.rating(mu=p1_mu, sigma=p1_sigma)
                p2_rating = model.rating(mu=p2_mu, sigma=p2_sigma)
                all_teams.append([p1_rating, p2_rating])
                all_placings.append(placings[i])

        # Sort teams by placing (1=best, 8=worst) and convert to ranks (0=best, 7=worst)
        sorted_indices = sorted(range(len(all_placings)), key=lambda i: all_placings[i])
        sorted_teams = [all_teams[i] for i in sorted_indices]
        ranks = list(range(len(sorted_teams)))

        # Run the rating update
        new_teams = model.rate(sorted_teams, ranks=ranks)

        # Find our team in the results
        our_team_sorted_idx = sorted_indices.index(random_team_idx)
        new_varied = new_teams[our_team_sorted_idx][0]
        new_fixed = new_teams[our_team_sorted_idx][1]

        # Calculate changes
        varied_mu_change = new_varied.mu - base_mu
        varied_sigma_change = new_varied.sigma - test_sigma
        varied_rating_change = calculate_rating(new_varied.mu, new_varied.sigma) - calculate_rating(base_mu, test_sigma)

        fixed_mu_change = new_fixed.mu - teammate_mu
        fixed_sigma_change = new_fixed.sigma - teammate_sigma
        fixed_rating_change = calculate_rating(new_fixed.mu, new_fixed.sigma) - calculate_rating(teammate_mu, teammate_sigma)

        results.append(AnalysisResult(
            sigma_varied=test_sigma,
            player_mu_change=varied_mu_change,
            player_sigma_change=varied_sigma_change,
            player_rating_change=varied_rating_change,
            teammate_mu_change=fixed_mu_change,
            teammate_sigma_change=fixed_sigma_change,
            teammate_rating_change=fixed_rating_change
        ))

    return results, varied_player_id, fixed_player_id, {
        'varied_player': varied_player,
        'fixed_player': fixed_player,
        'team_placing': team_placing,
        'original_sigma': original_sigma
    }


def create_analysis_charts(results: List[AnalysisResult], varied_id: str, fixed_id: str, metadata: dict):
    """Create charts showing sigma impact analysis"""
    if not matplotlib_available:
        return

    sigma_values = [r.sigma_varied for r in results]

    # Set up styling
    plt.style.use('default')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('#F8F9FA')

    # Title with player info
    varied_name = metadata['varied_player'].get('name', 'Unknown')[:20]
    fixed_name = metadata['fixed_player'].get('name', 'Unknown')[:20]
    original_sigma = metadata['original_sigma']
    team_placing = metadata['team_placing']

    fig.suptitle(f'🎮 Sigma Impact Analysis: {varied_id} vs {fixed_id}\n'
                f'🎯 Team placed #{team_placing} | 📊 Varying {varied_id}\'s σ (originally {original_sigma:.2f})',
                fontsize=16, fontweight='bold', color='#2C3E50')

    # Mu changes
    axes[0].plot(sigma_values, [r.player_mu_change for r in results],
                color=colors[0], linewidth=3, marker='o', markersize=6,
                label=f'🎲 {varied_id} (varied σ)', alpha=0.9)
    axes[0].plot(sigma_values, [r.teammate_mu_change for r in results],
                color=colors[1], linewidth=3, marker='D', markersize=6,
                label=f'🤝 {fixed_id} (fixed σ)', alpha=0.9)
    axes[0].set_xlabel('🎯 Player Sigma (varied)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('📈 Mu Change', fontsize=12, fontweight='bold')
    axes[0].set_title('🧠 Skill Level Changes (Mu)', fontsize=14, fontweight='bold', color='#2C3E50')
    axes[0].legend(fontsize=10, loc='best', framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_facecolor('#FAFBFC')

    # Add original sigma marker
    axes[0].axvline(x=original_sigma, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=2)
    axes[0].text(original_sigma, axes[0].get_ylim()[1]*0.9, f'Original σ\n{original_sigma:.2f}',
                ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

    # Add teammate sigma line
    teammate_sigma = metadata['fixed_player'].get('sigma', 25.0/3.0)
    axes[0].axvline(x=teammate_sigma, color='#9B59B6', linestyle='--', alpha=0.6, linewidth=2)
    axes[0].text(teammate_sigma, axes[0].get_ylim()[1]*0.8, f'Teammate σ\n{teammate_sigma:.2f}',
                ha='center', fontsize=9, color='#9B59B6', fontweight='bold')

    # Sigma changes
    axes[1].plot(sigma_values, [r.player_sigma_change for r in results],
                color=colors[2], linewidth=3, marker='s', markersize=6,
                label=f'🎲 {varied_id} (varied σ)', alpha=0.9)
    axes[1].plot(sigma_values, [r.teammate_sigma_change for r in results],
                color=colors[3], linewidth=3, marker='^', markersize=6,
                label=f'🤝 {fixed_id} (fixed σ)', alpha=0.9)
    axes[1].set_xlabel('🎯 Player Sigma (varied)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('📉 Sigma Change', fontsize=12, fontweight='bold')
    axes[1].set_title('🎪 Uncertainty Changes (Sigma)', fontsize=14, fontweight='bold', color='#2C3E50')
    axes[1].legend(fontsize=10, loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_facecolor('#FAFBFC')
    axes[1].axvline(x=original_sigma, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=2)
    axes[1].axvline(x=teammate_sigma, color='#9B59B6', linestyle='--', alpha=0.6, linewidth=2)

    # Rating changes
    axes[2].plot(sigma_values, [r.player_rating_change for r in results],
                color=colors[4], linewidth=4, marker='o', markersize=7,
                label=f'🎲 {varied_id} (varied σ)', alpha=0.9)
    axes[2].plot(sigma_values, [r.teammate_rating_change for r in results],
                color=colors[5], linewidth=4, marker='*', markersize=8,
                label=f'🤝 {fixed_id} (fixed σ)', alpha=0.9)
    axes[2].set_xlabel('🎯 Player Sigma (varied)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('🏆 Rating Change', fontsize=12, fontweight='bold')
    axes[2].set_title('💰 Final Rating Impact\n(μ - 3σ) × 100', fontsize=14, fontweight='bold', color='#2C3E50')
    axes[2].legend(fontsize=10, loc='best', framealpha=0.9)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_facecolor('#FAFBFC')
    axes[2].axvline(x=original_sigma, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=2)
    axes[2].axvline(x=teammate_sigma, color='#9B59B6', linestyle='--', alpha=0.6, linewidth=2)

    plt.tight_layout()
    plt.savefig('sigma_analysis_results.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.show()


def print_summary_table(results: List[AnalysisResult]):
    """Print a summary table of key results"""
    print("\nSigma Impact Analysis Summary")
    print("=" * 80)
    print(f"{'Sigma':<8} {'Player Δμ':<10} {'Player Δσ':<10} {'Player ΔRating':<12} {'Teammate Δμ':<12} {'Teammate Δσ':<12} {'Teammate ΔRating':<14}")
    print("-" * 80)

    # Show every 3rd result to keep table manageable
    for i in range(0, len(results), 3):
        r = results[i]
        print(f"{r.sigma_varied:<8.2f} {r.player_mu_change:<10.4f} {r.player_sigma_change:<10.4f} {r.player_rating_change:<12.4f} "
              f"{r.teammate_mu_change:<12.4f} {r.teammate_sigma_change:<12.4f} {r.teammate_rating_change:<14.4f}")

    # Key insights
    min_sigma_result = results[0]  # sigma = 1.5
    max_sigma_result = results[-1]  # sigma = 8.3

    print(f"\nKey Insights:")
    print(f"When player sigma changes from {min_sigma_result.sigma_varied:.1f} to {max_sigma_result.sigma_varied:.1f}:")
    print(f"  Player mu change varies by: {abs(max_sigma_result.player_mu_change - min_sigma_result.player_mu_change):.4f}")
    print(f"  Teammate mu change varies by: {abs(max_sigma_result.teammate_mu_change - min_sigma_result.teammate_mu_change):.4f}")
    print(f"  Player rating change varies by: {abs(max_sigma_result.player_rating_change - min_sigma_result.player_rating_change):.4f}")
    print(f"  Teammate rating change varies by: {abs(max_sigma_result.teammate_rating_change - min_sigma_result.teammate_rating_change):.4f}")


def run_mu_analysis(model, players: List[Dict], teams: List[List[str]], placings: List[int]):
    """Run mu analysis on a random team and player"""
    # Pick a random team to analyze
    random_team_idx = random.randint(0, len(teams) - 1)
    selected_team = teams[random_team_idx]
    team_placing = placings[random_team_idx]

    # Get the two players from this team
    player1_id, player2_id = selected_team[0], selected_team[1]

    # Find player data
    player1_data = next(p for p in players if p["id"] == player1_id)
    player2_data = next(p for p in players if p["id"] == player2_id)

    # Randomly pick which player's mu to vary
    if random.choice([True, False]):
        varied_player = player1_data
        fixed_player = player2_data
        varied_player_id = player1_id
        fixed_player_id = player2_id
    else:
        varied_player = player2_data
        fixed_player = player1_data
        varied_player_id = player2_id
        fixed_player_id = player1_id

    print(f"\n🎲 Random Selection Results:")
    print(f"   Team: {varied_player_id} + {fixed_player_id} (placing: {team_placing})")
    print(f"   Varying mu for: {varied_player_id} ({varied_player.get('name', 'Unknown')})")
    print(f"   Fixed teammate: {fixed_player_id} ({fixed_player.get('name', 'Unknown')})")

    # Get base stats
    original_mu = varied_player.get("mu", 25.0)
    base_sigma = varied_player.get("sigma", 25.0/3.0)
    teammate_mu = fixed_player.get("mu", 25.0)
    teammate_sigma = fixed_player.get("sigma", 25.0/3.0)

    # Range of mu values to test (vary around the original mu)
    mu_range = max(10.0, original_mu * 0.8)  # Create a reasonable range
    mu_values = np.linspace(original_mu - mu_range, original_mu + mu_range, 25)

    results = []

    for test_mu in mu_values:
        # Create ratings for this test
        varied_rating = model.rating(mu=test_mu, sigma=base_sigma)
        fixed_rating = model.rating(mu=teammate_mu, sigma=teammate_sigma)

        # Build all teams with their ratings and placings
        all_teams = []
        all_placings = []

        for i, team in enumerate(teams):
            if i == random_team_idx:  # Our selected team
                all_teams.append([varied_rating, fixed_rating])
                all_placings.append(placings[i])
            else:  # Opponent teams
                p1_data = next(p for p in players if p["id"] == team[0])
                p2_data = next(p for p in players if p["id"] == team[1])
                p1_mu = p1_data.get("mu", 25.0)
                p1_sigma = p1_data.get("sigma", 25.0/3.0)
                p2_mu = p2_data.get("mu", 25.0)
                p2_sigma = p2_data.get("sigma", 25.0/3.0)
                p1_rating = model.rating(mu=p1_mu, sigma=p1_sigma)
                p2_rating = model.rating(mu=p2_mu, sigma=p2_sigma)
                all_teams.append([p1_rating, p2_rating])
                all_placings.append(placings[i])

        # Sort teams by placing (1=best, 8=worst) and convert to ranks (0=best, 7=worst)
        sorted_indices = sorted(range(len(all_placings)), key=lambda i: all_placings[i])
        sorted_teams = [all_teams[i] for i in sorted_indices]
        ranks = list(range(len(sorted_teams)))

        # Run the rating update
        new_teams = model.rate(sorted_teams, ranks=ranks)

        # Find our team in the results
        our_team_sorted_idx = sorted_indices.index(random_team_idx)
        new_varied = new_teams[our_team_sorted_idx][0]
        new_fixed = new_teams[our_team_sorted_idx][1]

        # Calculate changes
        varied_mu_change = new_varied.mu - test_mu
        varied_sigma_change = new_varied.sigma - base_sigma
        varied_rating_change = calculate_rating(new_varied.mu, new_varied.sigma) - calculate_rating(test_mu, base_sigma)

        fixed_mu_change = new_fixed.mu - teammate_mu
        fixed_sigma_change = new_fixed.sigma - teammate_sigma
        fixed_rating_change = calculate_rating(new_fixed.mu, new_fixed.sigma) - calculate_rating(teammate_mu, teammate_sigma)

        results.append(AnalysisResult(
            sigma_varied=test_mu,  # Reusing the dataclass, but storing mu instead
            player_mu_change=varied_mu_change,
            player_sigma_change=varied_sigma_change,
            player_rating_change=varied_rating_change,
            teammate_mu_change=fixed_mu_change,
            teammate_sigma_change=fixed_sigma_change,
            teammate_rating_change=fixed_rating_change
        ))

    return results, varied_player_id, fixed_player_id, {
        'varied_player': varied_player,
        'fixed_player': fixed_player,
        'team_placing': team_placing,
        'original_mu': original_mu
    }


def create_mu_analysis_charts(results: List[AnalysisResult], varied_id: str, fixed_id: str, metadata: dict):
    """Create charts showing mu impact analysis"""
    if not matplotlib_available:
        return

    mu_values = [r.sigma_varied for r in results]  # sigma_varied field stores mu values

    # Set up styling
    plt.style.use('default')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('#F8F9FA')

    # Title with player info
    varied_name = metadata['varied_player'].get('name', 'Unknown')[:20]
    fixed_name = metadata['fixed_player'].get('name', 'Unknown')[:20]
    original_mu = metadata['original_mu']
    team_placing = metadata['team_placing']

    fig.suptitle(f'🎯 Mu Impact Analysis: {varied_id} vs {fixed_id}\n'
                f'🏆 Team placed #{team_placing} | 📊 Varying {varied_id}\'s μ (originally {original_mu:.2f})',
                fontsize=16, fontweight='bold', color='#2C3E50')

    # Mu changes
    axes[0].plot(mu_values, [r.player_mu_change for r in results],
                color=colors[0], linewidth=3, marker='o', markersize=6,
                label=f'🎯 {varied_id} (varied μ)', alpha=0.9)
    axes[0].plot(mu_values, [r.teammate_mu_change for r in results],
                color=colors[1], linewidth=3, marker='D', markersize=6,
                label=f'🤝 {fixed_id} (fixed μ)', alpha=0.9)
    axes[0].set_xlabel('🎯 Player Mu (varied)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('📈 Mu Change', fontsize=12, fontweight='bold')
    axes[0].set_title('🧠 Skill Level Changes (Mu)', fontsize=14, fontweight='bold', color='#2C3E50')
    axes[0].legend(fontsize=10, loc='best', framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_facecolor('#FAFBFC')

    # Add original mu marker
    axes[0].axvline(x=original_mu, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=2)
    axes[0].text(original_mu, axes[0].get_ylim()[1]*0.9, f'Original μ\n{original_mu:.2f}',
                ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

    # Add teammate mu line
    teammate_mu = metadata['fixed_player'].get('mu', 25.0)
    axes[0].axvline(x=teammate_mu, color='#9B59B6', linestyle='--', alpha=0.6, linewidth=2)
    axes[0].text(teammate_mu, axes[0].get_ylim()[1]*0.8, f'Teammate μ\n{teammate_mu:.2f}',
                ha='center', fontsize=9, color='#9B59B6', fontweight='bold')

    # Sigma changes
    axes[1].plot(mu_values, [r.player_sigma_change for r in results],
                color=colors[2], linewidth=3, marker='s', markersize=6,
                label=f'🎯 {varied_id} (varied μ)', alpha=0.9)
    axes[1].plot(mu_values, [r.teammate_sigma_change for r in results],
                color=colors[3], linewidth=3, marker='^', markersize=6,
                label=f'🤝 {fixed_id} (fixed μ)', alpha=0.9)
    axes[1].set_xlabel('🎯 Player Mu (varied)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('📉 Sigma Change', fontsize=12, fontweight='bold')
    axes[1].set_title('🎪 Uncertainty Changes (Sigma)', fontsize=14, fontweight='bold', color='#2C3E50')
    axes[1].legend(fontsize=10, loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_facecolor('#FAFBFC')
    axes[1].axvline(x=original_mu, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=2)
    axes[1].axvline(x=teammate_mu, color='#9B59B6', linestyle='--', alpha=0.6, linewidth=2)

    # Rating changes
    axes[2].plot(mu_values, [r.player_rating_change for r in results],
                color=colors[4], linewidth=4, marker='o', markersize=7,
                label=f'🎯 {varied_id} (varied μ)', alpha=0.9)
    axes[2].plot(mu_values, [r.teammate_rating_change for r in results],
                color=colors[5], linewidth=4, marker='*', markersize=8,
                label=f'🤝 {fixed_id} (fixed μ)', alpha=0.9)
    axes[2].set_xlabel('🎯 Player Mu (varied)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('🏆 Rating Change', fontsize=12, fontweight='bold')
    axes[2].set_title('💰 Final Rating Impact\n(μ - 3σ) × 100', fontsize=14, fontweight='bold', color='#2C3E50')
    axes[2].legend(fontsize=10, loc='best', framealpha=0.9)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_facecolor('#FAFBFC')
    axes[2].axvline(x=original_mu, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=2)
    axes[2].axvline(x=teammate_mu, color='#9B59B6', linestyle='--', alpha=0.6, linewidth=2)

    plt.tight_layout()
    plt.savefig('mu_analysis_results.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.show()


def print_mu_summary_table(results: List[AnalysisResult]):
    """Print a summary table of mu analysis results"""
    print("\nMu Impact Analysis Summary")
    print("=" * 80)
    print(f"{'Mu':<8} {'Player Δμ':<10} {'Player Δσ':<10} {'Player ΔRating':<12} {'Teammate Δμ':<12} {'Teammate Δσ':<12} {'Teammate ΔRating':<14}")
    print("-" * 80)

    # Show every 3rd result to keep table manageable
    for i in range(0, len(results), 3):
        r = results[i]
        mu_val = r.sigma_varied  # sigma_varied field stores mu values
        print(f"{mu_val:<8.2f} {r.player_mu_change:<10.4f} {r.player_sigma_change:<10.4f} {r.player_rating_change:<12.4f} "
              f"{r.teammate_mu_change:<12.4f} {r.teammate_sigma_change:<12.4f} {r.teammate_rating_change:<14.4f}")

    # Key insights
    min_mu_result = results[0]
    max_mu_result = results[-1]

    print(f"\nKey Insights:")
    print(f"When player mu changes from {min_mu_result.sigma_varied:.1f} to {max_mu_result.sigma_varied:.1f}:")
    print(f"  Player mu change varies by: {abs(max_mu_result.player_mu_change - min_mu_result.player_mu_change):.4f}")
    print(f"  Teammate mu change varies by: {abs(max_mu_result.teammate_mu_change - min_mu_result.teammate_mu_change):.4f}")
    print(f"  Player rating change varies by: {abs(max_mu_result.player_rating_change - min_mu_result.player_rating_change):.4f}")
    print(f"  Teammate rating change varies by: {abs(max_mu_result.teammate_rating_change - min_mu_result.teammate_rating_change):.4f}")


def run_experiment_1(model, players: List[Dict], teams: List[List[str]], placings: List[int]):
    """Run Experiment 1: Sigma Impact Analysis"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: SIGMA IMPACT ANALYSIS")
    print("="*80)
    print("🎮 Running Sigma Impact Analysis...")
    print("📊 Using real data from the simulation")
    print("🎲 Randomly selecting a team and player to analyze\n")

    # Run the analysis
    results, varied_id, fixed_id, metadata = run_sigma_analysis(model, players, teams, placings)

    # Print summary
    print_summary_table(results)

    # Create charts
    print(f"\n🎨 Generating charts...")
    try:
        create_analysis_charts(results, varied_id, fixed_id, metadata)
        print("📈 Charts saved as 'sigma_analysis_results.png'")
    except Exception as e:
        print(f"❌ Error generating charts: {e}")

    print("\n✅ Sigma analysis complete! Run again to get a different random player.")


def run_experiment_2(model, players: List[Dict], teams: List[List[str]], placings: List[int]):
    """Run Experiment 2: Mu Impact Analysis"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: MU IMPACT ANALYSIS (MU STEALING)")
    print("="*80)
    print("🎯 Running Mu Impact Analysis...")
    print("📊 Using real data from the simulation")
    print("🎲 Randomly selecting a team and player to analyze\n")

    # Run the analysis
    results, varied_id, fixed_id, metadata = run_mu_analysis(model, players, teams, placings)

    # Print summary
    print_mu_summary_table(results)

    # Create charts
    print(f"\n🎨 Generating mu analysis charts...")
    try:
        create_mu_analysis_charts(results, varied_id, fixed_id, metadata)
        print("📈 Charts saved as 'mu_analysis_results.png'")
    except Exception as e:
        print(f"❌ Error generating charts: {e}")

    print("\n✅ Mu analysis complete! Run again to get a different random player.")
