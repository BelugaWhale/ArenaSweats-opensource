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
`return ThurstoneMostellerFull(sigma=(25/5.75), beta=(25/6) * 4, tau=(25/300) * 1.75)`

### 📈 Your Skill Profile: Two Numbers That Matter

OpenSkill TM doesn't just track one rating number - it maintains two key pieces of information about every player:

**Your Skill Level (μ "mu")** - This is the system's best guess at your actual skill. Think of it as your "true rating" that goes up when you win and down when you lose.

**Uncertainty (σ "sigma")** - This measures how confident the system is about your skill level. New players start with high uncertainty, but as you play more games, the system becomes more confident in its assessment of your ability.

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

This adjustment applies in teams with at least one GM+ player. When they have a much lower-rated teammate, that game is treated as less informative for the higher-rated player. Specifically, the modifier starts when the lower teammate is below 90% of the higher teammate's μ, scales up as the gap widens, and reaches its cap once the lower teammate is at 20 μ. The result is lower gains and losses for the higher-rated player in those games.

In practice, the system recognizes it could not learn as much from that result, so uncertainty reduces less than usual or can even go up. That allows future games to carry potentially larger gains or losses.

### Unbalanced Lobby Grace

This adjustment only applies when **both** teammates are GM+. If that duo enters a lobby where their team strength is significantly above the typical team in that game, the system temporarily reduces their team strength before the OpenSkill update is calculated. This helps compensate for high-rank matchmaking limits where lobbies can have very low upside and high downside for top duos.

The grace is reduced for GM+ duos that have a big skill gap between them, and gets stronger for similarly-rated GM+ duos.

### Protection

In order to support solo queue without indirectly buffing boosting, two forms of protection are added:

**AFK Protection** - If a player places 8th and has a teammate with 0 kills, 0 assists, and less than 5000 damage dealt, that player's rating update is ignored for that game.

**Place Protection** - This applies only to teams that are not 2 Grand Master+ players. Grand Master+ players never lose rating if they place 3rd or above. Players below Grand Master never lose rating if they place 4th or above.

Protected loss is redistributed to eligible players in 5th-8th place, weighted by placement (8th pays the most, 5th the least).

### 🏆 Your Final Rating

Your displayed rating is calculated as: **round((Skill Level - 3 × Uncertainty) × 75)**

The "conservative estimate" approach (subtracting 3× uncertainty) is a recommended method which means your displayed rating is intentionally lower than your raw skill level - it represents what the system is confident you can achieve consistently.

## 📁 Codebase Highlights

-   **validations/openskill_sim**: Simulator code (`openskill_sim.py`, app/chart tooling, and helpers) used to validate behavior against production data.
-   **ranking_algorithm.py**: **This is the exact code that is used to update ratings for every game played.** The file is commented with detailed information to explain exactly what the code does, and the code itself is available.

