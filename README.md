# ArenaSweats Open Source

This repository includes content used for the www.arenasweats.lol website.

**ArenaSweats** is a ranked leaderboard and achievement tracker for LoL Arena gamemode.

The core of ArenaSweats, its ranked leaderboard, is powered by **LOTS** of data. ALL Arena matches of ALL players are tracked **GLOBALLY** (except China) in near real-time.

Once all game data is collected, it goes through a ranking algorithm called **OpenSkill TM**. OpenSkill ThurstoneMosteller model is an industry-leading ranking algorithm with unmatched speed and competitive accuracy. It is a Bayesian ranking algorithm, which is the same as TrueSkill. Such algorithms have been implemented extensively for video game rankings, including by Riot for Summoner's Rift ranked.

## 🎯 ArenaSweats Ranked Principles

Through these **3 principles**, the ArenaSweats ranked algorithm will stay trustworthy and as accurate as possible:

1.  **Use Industry Best System** - Currently OpenSkill TM as the foundation
2.  **Transparent and Open Source** - Every calculation is public and verifiable
3.  **Community-Driven Adjustments** - Any and all adjustments will be decided by the community over on [Discord](https://discord.gg/BvGFJ4WEWg)

This repository's purpose is to bring these principles to life, being the real location where the source code of the LIVE leaderboard ranked algorithm lives.

**This is PROOF of ArenaSweats leaderboard integrity.**

## 🧮 The Current ArenaSweats Ranked Algorithm

### 🎮 The OpenSkill TM Algorithm

ArenaSweats uses **OpenSkill TM**, an industry-leading, battle-tested Bayesian ranking system. Unlike simple win/loss systems, OpenSkill TM is smart about understanding your true skill level.

**Parameters**: Based on player feedback and simulation validation, we currently use:
`return ThurstoneMostellerFull(sigma=(25/5.75), beta=(25/6) * (3.75 in 3v3, 4 in 2v2), tau=(25/300) * 1.75)`

### 📈 Your Skill Profile: Two Numbers That Matter

OpenSkill TM doesn't just track one rating number - it maintains two key pieces of information about every player:

**Your Skill Level (μ "mu")** - This is the system's best guess at your actual skill. Think of it as your "true rating" that goes up when you win and down when you lose.

**Uncertainty (σ "sigma")** - This measures how confident the system is about your skill level. New players start with high uncertainty, but as you play more games, the system becomes more confident in its assessment of your ability.

Sigma has a floor of 2.5. OpenSkill and subsequent ranking modifiers may increase sigma, but the stored post-game sigma never falls below this value.

### ⚙️ Applying OpenSkill TM to Arena

Each Arena match has 8 teams of 2 players (16 total players). Here's what happens behind the scenes:

1.  **Before the match**: The system looks at each player's skill level and uncertainty
2.  **Team strength calculation**: Your duo's combined strength is calculated by adding both players' skill levels together
3.  **Match prediction**: Based on all 8 teams' strengths, the system predicts how likely each team is to finish in each position (1st through 8th)
4.  **After the match**: Rating changes depend on how your actual performance compared to what was expected

### 🎯 Rating Changes: How to Climb the Ladder

**The BEST way to improve your rating is to finish in a better position against stronger opponents.**

Your rating changes are based on:
- **Expected vs. Actual performance**: Beating stronger teams gives more rating than expected, losing to weaker teams hurts more
- **Uncertainty factor**: Players with higher uncertainty see bigger rating swings (this helps new players find their correct rating faster)

## 🛠️ Community-Driven Adjustments


Arena is a complicated mode (8 teams, duos, boosting pressure, bravery, matchmaking limitations) so a ranking model out of the box will not fit this perfectly, Adjustments are needed on-top to keep the leaderboard fair and accurate.

As covered in [Principle #3](#-arenasweats-ranked-principles), ranked adjustments are community-driven and discussed on [Discord](https://discord.gg/BvGFJ4WEWg).

There are currently 3 adjustments in place.

### Team Gap Modifier

This adjustment applies in teams with at least one GM+ player. When a higher-rated player has much lower-rated teammates, that game is treated as less informative for that higher-rated player. In 2v2, the comparison is against the other teammate's μ. In 3v3, the comparison is against the average μ of the other two teammates.

The modifier works by scaling the higher-rated player's post-match μ change and σ change by a multiplier between 1.0 and 0.05.

This scaling is more forgiving when the higher-rated player has not recently played with the same teammate or teammates.

### Unbalanced Lobby Grace

This adjustment only applies to teams with 2 or more GM+ players. If such a team enters a lobby where their team strength is significantly above the typical team in that game, the system temporarily reduces their team strength before the OpenSkill update is calculated. This helps compensate for high-rank matchmaking limits where lobbies can have very low upside and high downside for top teams.

The temporary reduction uses 22% of the effective gap in 2v2. In 3v3, it uses 57% through a 20% effective gap, then continues from that point at a 25% slope. The grace is reduced for GM+ teams with greater μ spread and gets stronger for more similarly-rated GM+ teams. In both 2v2 and 3v3 this internal balance check uses the lower player's μ relative to the higher player's μ, with an exponent of 3 in 2v2 and 2.5 in 3v3.

### Protection

In order to support solo queue without indirectly buffing boosting, two forms of protection are added:

**AFK Protection** - If a player would lose rating and has a teammate with 0 kills, fewer than 3 assists, and less than 3000 damage dealt, that player's rating loss is ignored for that game.

**Place Protection** - This applies only to teams that are not 2 Grandmaster+ players. Grandmaster+ players never lose rating if they place 3rd or above. Players below Grandmaster never lose rating if they place 4th or above.

Protected loss is redistributed to eligible players in 5th-8th place, weighted by placement (8th pays the most, 5th the least).

### 🏆 Your Final Rating

Your displayed rating is calculated as: **round((Skill Level - 3 × Uncertainty) × 75)**

The "conservative estimate" approach (subtracting 3× uncertainty) is a recommended method which means your displayed rating is intentionally lower than your raw skill level - it represents what the system is confident you can achieve consistently.

## 📁 Codebase Highlights

-   **validations/openskill_sim**: Simulator code (`openskill_sim.py`, app/chart tooling, and helpers) used to validate behavior against production data.
-   **ranking_algorithm.py**: **This is the exact code that is used to update ratings for every game played.** The file is commented with detailed information to explain exactly what the code does, and the code itself is available.

