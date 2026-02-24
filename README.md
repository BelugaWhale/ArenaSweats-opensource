# ArenaSweats Open Source

This repository includes content used for the www.arenasweats.lol website.

**ArenaSweats** is a ranked leaderboard and achievement tracker for LoL Arena gamemode.

The core of ArenaSweats, its ranked leaderboard, is powered by **LOTS** of data. ALL Arena matches of ALL players are tracked **GLOBALLY** (except China) in near real-time.

Once all game data is collected, it goes through a ranking algorithm called **OpenSkill TM**. OpenSkill ThurstoneMosteller model is an industry-leading ranking algorithm with unmatched speed and competitive accuracy. It is a Bayesian ranking algorithm, which is the same as TrueSkill. Such algorithms have been implemented extensively for video game rankings, including by Riot for Summoner's Rift ranked.

## 🎯 ArenaSweats Ranked Principles

Through these **3 principles**, the ArenaSweats ranked algorithm will stay trustworthy and as accurate as possible:

1.  **Use Industry Best System** - Currently OpenSkill TM as the foundation
2.  **Transparent and Open Source** - Every calculation is public and verifiable
3.  **Community-Driven Modifications** - Any and all modifications will be decided by the community over on [Discord](https://discord.gg/BvGFJ4WEWg)

This repository's purpose is to bring these principles to life, being the real location where the source code of the LIVE leaderboard ranked algorithm lives.

**This is PROOF of ArenaSweats leaderboard integrity.**

## 🧮 The Current ArenaSweats Ranked Algorithm

### 🎮 The OpenSkill TM Algorithm

ArenaSweats uses **OpenSkill TM**, an industry-leading, battle-tested Bayesian ranking system. Unlike simple win/loss systems, OpenSkill TM is smart about understanding your true skill level.

**Parameters**: Based on player feedback, we use the following parameters:
`return ThurstoneMostellerFull(sigma=(25/6), beta=(25/6) * 2.5, tau=(25/300) * 3)`

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

### 🎯 Rating Changes: Why Some Games Matter More

Your rating changes are based on:
- **Expected vs. Actual performance**: Beating stronger teams gives more rating than expected, losing to weaker teams hurts more
- **Uncertainty factor**: Players with higher uncertainty see bigger rating swings (this helps new players find their correct rating faster)

### 👥 Team Contribution Distribution

In duo matches, rating changes must be distributed between teammates. ArenaSweats uses OpenSkill TM's default uncertainty-based approach: teammates with higher uncertainty (less confident skill estimates) receive proportionally larger rating changes than teammates with lower uncertainty.

## 🛠️ Community-Driven Modifications

### 🚀 The "New Player" Rating Ramp-up

There's one important modification to pure OpenSkill TM: **the rating ramp-up system**. Because new players start with high uncertainty, they could sometimes achieve artificially high ratings with just a few lucky games. To prevent this:

- Players with 0 games get 50% of their calculated rating
- This scales up linearly to 100% at 40 games played
- This ensures the leaderboard accurately reflects sustained performance

### 🛡️ Anti-Boost

This system aims to prevent boosting by reducing the impact of games with significant skill or certainty disparities. It only takes effect when the season is at least 5 days old.

There are two cases where a game is flagged as "low impact":

1.  **High Uncertainty Gap**: If the difference in uncertainty (σ) between two teammates is greater than 50% of the total sigma range, it is considered a low impact game for the higher rated player, who will experience **80% reduced rating gains and losses**.
2.  **High Skill Gap**: If the weaker player's skill level (μ) is less than 40% of their stronger teammate's skill level, it is a low impact game. The higher rated player will experience **75-95% reduced rating gains and losses**.

### 🏆 Your Final Rating

Your displayed rating is calculated as: **(Skill Level - 3 × Uncertainty) × 100 × Games Scaling Factor**

The "conservative estimate" approach (subtracting 3× uncertainty) is a recommended method which means your displayed rating is intentionally lower than your raw skill level - it represents what the system is confident you can achieve consistently.

## 📁 Codebase Guide

-   **visualizations**: This directory contains a simulator used to tune the parameters of the Anti-Boost system. It also includes a chart comparing the behavior of OpenSkill TM, TrueSkill, and our specific parameter set.
-   **ranking_algorithm.py**: This is the exact code that is used to update ratings for every game played. The file is commented with detailed information to explain exactly what the code does, and the code itself is available.
