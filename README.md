# ArenaSweats Open Source

This repository includes content used for the www.arenasweats.lol website.

**ArenaSweats** is a ranked leaderboard and achievement tracker for LoL Arena gamemode.

The core of ArenaSweats, its ranked leaderboard, is powered by **LOTS** of data. ALL Arena matches of ALL players are tracked **GLOBALLY** (except China) in near real-time.

Once all game data is collected, it goes through a ranking algorithm called **TrueSkill**. TrueSkill is developed by Microsoft and is used in many industry-leading multiplayer games including by Riot for Summoner's Rift ranked. This is an excellent base, but Arena is a complicated game and base TrueSkill may not be sufficient.

## 🎯 ArenaSweats Ranked Principles

Through these **3 principles**, the ArenaSweats ranked algorithm will stay trustworthy and as accurate as possible:

1. **Use Industry Best System** - Currently TrueSkill as the foundation
2. **Transparent and Open Source** - Every calculation is public and verifiable
3. **Community-Driven Modifications** - Any and all modifications will be decided by the community over on [Discord](https://discord.gg/BvGFJ4WEWg)

This repository's purpose is to bring these principles to life, being the real location where the source code of the LIVE leaderboard ranked algorithm lives.

**This is PROOF of ArenaSweats leaderboard integrity.**

## 🧮 The Current ArenaSweats Ranked Algorithm

### 🎮 The TrueSkill Algorithm

ArenaSweats uses **TrueSkill**, Microsoft's battle-tested ranking system that powers Xbox Live and is used by Riot Games for Summoner's Rift ranked. Unlike simple win/loss systems, TrueSkill is smart about understanding your true skill level.

**Configuration**: ArenaSweats uses all of TrueSkill's standard default settings, with one key modification - the draw probability is set to 0% since Arena matches can't end in ties.

### 📈 Your Skill Profile: Two Numbers That Matter

TrueSkill doesn't just track one rating number - it maintains two key pieces of information about every player:

**Your Skill Level (μ "mu")** - This is the system's best guess at your actual skill. Think of it as your "true rating" that goes up when you win and down when you lose.

**Uncertainty (σ "sigma")** - This measures how confident the system is about your skill level. New players start with high uncertainty, but as you play more games, the system becomes more confident in its assessment of your ability.

### ⚙️ Applying TrueSkill to Arena

Each Arena match has 8 teams of 2 players (16 total players). Here's what happens behind the scenes:

1. **Before the match**: The system looks at each player's skill level and uncertainty
2. **Team strength calculation**: Your duo's combined strength is calculated by adding both players' skill levels together
3. **Match prediction**: Based on all 8 teams' strengths, the system predicts how likely each team is to finish in each position (1st through 8th)
4. **After the match**: Rating changes depend on how your actual performance compared to what was expected

### 🎯 Rating Changes: Why Some Games Matter More

Your rating changes are based on:
- **Expected vs. Actual performance**: Beating stronger teams gives more rating than expected, losing to weaker teams hurts more
- **Uncertainty factor**: Players with higher uncertainty see bigger rating swings (this helps new players find their correct rating faster)

### 👥 Team Contribution Distribution

In duo matches, rating changes must be distributed between teammates. ArenaSweats uses TrueSkill's default uncertainty-based approach: teammates with higher uncertainty (less confident skill estimates) receive proportionally larger rating changes than teammates with lower uncertainty.

### 🚀 The "New Player" Rating Ramp-up

There's one important modification to pure TrueSkill: **the rating ramp-up system**. Because new players start with high uncertainty, they could sometimes achieve artificially high ratings with just a few lucky games. To prevent this:

- Players with 0 games get 50% of their calculated rating
- This scales up linearly to 100% at 40 games played
- This ensures the leaderboard accurately reflects sustained performance

### 🏆 Your Final Rating

Your displayed rating is calculated as: **(Skill Level - 3 × Uncertainty) × 100 × Games Scaling Factor**

The "conservative estimate" approach (subtracting 3× uncertainty) is Microsoft Research's recommended method and means your displayed rating is intentionally lower than your raw skill level - it represents what the system is confident you can achieve consistently.