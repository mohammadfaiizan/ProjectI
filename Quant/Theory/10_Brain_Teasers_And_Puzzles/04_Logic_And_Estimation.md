# Logic and Estimation

## Problem 1: Three Hats

**Statement:** Three people see each other's hats but not their own. Hats are either red or blue. At least one red hat. They can see others but not communicate. A bell rings. If you know your hat color, stand up. What happens?

**Solution:** If two people see one red and one blue hat each, they both stand up after second bell (realizing they must have red hat if other saw one red).

**Key insight:** Use elimination reasoning. If I see two blue hats, I know I have red. If I see one red and one blue, I wait. If other person also sees one red and one blue, we both deduce we have red.

## Problem 2: Light Switches

**Statement:** Three switches control three lights in another room. You can only go to the other room once. How do you determine which switch controls which light?

**Solution:** Turn on switch 1, wait 5 minutes, turn it off. Turn on switch 2. Go to room:
- Light on and warm: Switch 1
- Light on and cold: Switch 2  
- Light off and warm: Switch 3

**Key insight:** Use time delay and heat to distinguish.

## Problem 3: Weighing Problem - 12 Balls

**Statement:** 12 balls, one is different weight (heavier or lighter, unknown). Balance scale, 3 weighings. Find the different ball.

**Solution:** 
- Weighing 1: Weigh 4 vs 4
  - If equal: Different in remaining 4
  - If unequal: Different in weighed 8
- Weighing 2: Based on result, narrow to 3-4 candidates
- Weighing 3: Identify the different ball

**Detailed strategy:** Complex but systematic elimination.

## Problem 4: Weighing Problem - 9 Balls

**Statement:** 9 balls, one heavier. 2 weighings. Find heavy ball.

**Solution:**
- Weighing 1: Weigh 3 vs 3
  - If equal: Heavy in remaining 3
  - If unequal: Heavy in heavier group of 3
- Weighing 2: Weigh 1 vs 1 from suspect group
  - Identifies heavy ball

## Problem 5: Fermi Estimation - Piano Tuners

**Statement:** How many piano tuners are there in Chicago?

**Approach:**
1. Population of Chicago: ~3 million
2. Households: ~1 million (assuming 3 per household)
3. Pianos per household: ~1 in 100 → 10,000 pianos
4. Tunings per piano per year: ~1
5. Tunings per tuner per day: ~4
6. Working days per year: ~250
7. Tunings per tuner per year: 4 × 250 = 1,000
8. Number of tuners: 10,000 / 1,000 = 10

**Answer:** Order of magnitude: 10-100 piano tuners.

**Key insight:** Break into estimable components, use reasonable assumptions.

## Problem 6: Fermi Estimation - Gas Stations

**Statement:** How many gas stations in the US?

**Approach:**
1. US population: ~330 million
2. Cars per person: ~0.8 → 264 million cars
3. Fill-ups per car per week: ~1
4. Total fill-ups per week: 264 million
5. Fill-ups per station per day: ~500
6. Fill-ups per station per week: 500 × 7 = 3,500
7. Number of stations: 264M / 3,500 ≈ 75,000

**Answer:** Order of magnitude: 50,000-100,000 gas stations.

## Problem 7: Market Sizing - Smartphones

**Statement:** Estimate annual smartphone sales in the US.

**Approach:**
1. US population: 330 million
2. Smartphone penetration: ~80% → 264 million users
3. Replacement cycle: ~2 years
4. Annual sales: 264M / 2 = 132 million
5. New users: ~5% of population → 16.5 million
6. Total: ~150 million smartphones per year

**Answer:** Order of magnitude: 100-200 million per year.

## Problem 8: Estimation Under Uncertainty

**Statement:** Estimate number of trees in Central Park.

**Approach:**
1. Central Park area: ~840 acres ≈ 3.4 km²
2. Tree density: ~100-200 per acre (varies by area)
3. Average: ~150 per acre
4. Total: 840 × 150 = 126,000

**Answer:** Order of magnitude: 100,000-200,000 trees.

**Uncertainty:** Wide range due to varying density. Report as range or with confidence interval.

## Problem 9: Order-of-Magnitude Reasoning

**Problem 9a:** How many ping-pong balls fit in a school bus?

**Approach:**
1. Bus volume: ~8m × 2.5m × 2.5m = 50 m³ = 50,000,000 cm³
2. Ping-pong ball volume: ~4/3 × π × 2³ ≈ 33 cm³
3. Packing efficiency: ~60% (spheres don't pack perfectly)
4. Number: 50,000,000 × 0.6 / 33 ≈ 900,000

**Answer:** Order of magnitude: 500,000-1,000,000 balls.

**Problem 9b:** How many times does average person's heart beat in a lifetime?

**Approach:**
1. Heart rate: ~70 bpm
2. Lifetime: ~80 years = 80 × 365 × 24 × 60 = 42,048,000 minutes
3. Total beats: 70 × 42,048,000 ≈ 3 billion

**Answer:** ~3 billion heartbeats.

## Problem 10: Logic Puzzle - Knights and Knaves

**Statement:** On an island, knights always tell truth, knaves always lie. You meet two people. A says "B is a knight." B says "We are different types." What are they?

**Solution:** Both are knaves.

**Reasoning:**
- If A is knight: B is knight → B tells truth → "We are different" is false → contradiction
- If A is knave: "B is knight" is false → B is knave → B says "We are different" → true (both knaves) → consistent

## Problem 11: Logic Puzzle - Liar Paradox

**Statement:** "This statement is false." Is it true or false?

**Solution:** Paradox - neither true nor false consistently.

**Resolution:** Self-referential statements create logical issues. In formal logic, such statements may be excluded or handled with type theory.

## Problem 12: Estimation - Internet Traffic

**Statement:** Estimate daily internet traffic in petabytes.

**Approach:**
1. Global population: ~8 billion
2. Internet users: ~60% → 4.8 billion
3. Average data per user per day: ~1-2 GB
4. Total: 4.8B × 1.5 GB = 7.2 billion GB = 7,200 PB

**Answer:** Order of magnitude: 5,000-10,000 petabytes per day.

## Problem 13: Market Sizing - Electric Vehicles

**Statement:** Estimate EV market size in 5 years.

**Approach:**
1. Current EV sales: ~5% of car market
2. Car market: ~15 million per year in US
3. Current EV sales: ~750,000 per year
4. Growth rate: ~30% annually (compound)
5. In 5 years: 750K × 1.3⁵ ≈ 2.8 million
6. Market share: ~15-20%

**Answer:** Order of magnitude: 2-3 million EVs per year (US).

## Problem 14: Estimation - Social Media

**Statement:** Estimate number of photos uploaded to Instagram per day.

**Approach:**
1. Instagram users: ~1 billion active
2. Daily active users: ~500 million
3. Posts per user per day: ~0.1 (average)
4. Photos per post: ~1.5
5. Total: 500M × 0.1 × 1.5 = 75 million

**Answer:** Order of magnitude: 50-100 million photos per day.

## Problem 15: Logic - River Crossing

**Statement:** Wolf, goat, cabbage. Ferry can carry you + one item. Wolf eats goat if alone. Goat eats cabbage if alone. How to get all across?

**Solution:**
1. Take goat across
2. Return alone
3. Take wolf across
4. Bring goat back
5. Take cabbage across
6. Return alone
7. Take goat across

**Key insight:** Need to bring goat back to prevent eating.

## Problem 16: Estimation - Cloud Storage

**Statement:** Estimate total cloud storage capacity worldwide.

**Approach:**
1. Major providers: AWS, Azure, GCP, others
2. Estimated capacity per provider: ~100-500 exabytes each
3. Total: ~1,000-2,000 exabytes
4. In petabytes: 1-2 billion PB

**Answer:** Order of magnitude: 1-2 exabytes total capacity.

## Problem 17: Logic - Truth Tellers

**Statement:** Three people: one always tells truth, one always lies, one random. You can ask 2 yes/no questions. Identify each.

**Solution:** 
- Question 1 to person A: "Is B the random one?"
  - If A is truth-teller: Answer reveals B's type
  - If A is liar: Answer is opposite
  - If A is random: Answer is random
- Question 2: Based on first answer, ask to identify types

**Strategy:** Use questions to eliminate possibilities systematically.

## Problem 18: Estimation - E-commerce

**Statement:** Estimate annual e-commerce sales in the US.

**Approach:**
1. US retail sales: ~$6 trillion
2. E-commerce share: ~15%
3. E-commerce sales: ~$900 billion

**Answer:** Order of magnitude: $800B-$1T annually.

## Problem 19: Fermi - Cell Towers

**Statement:** How many cell towers in the US?

**Approach:**
1. US area: ~10 million km²
2. Coverage needed: ~95% of area
3. Tower range: ~5-10 km radius
4. Coverage per tower: ~π × 7.5² ≈ 175 km²
5. Number: 9.5M / 175 ≈ 54,000

**Answer:** Order of magnitude: 50,000-100,000 towers.

## Problem 20: Estimation - Data Centers

**Statement:** Estimate total electricity consumption of data centers worldwide.

**Approach:**
1. Number of data centers: ~8 million (including small)
2. Average power: ~100 kW per data center
3. Total capacity: 800M kW = 800 GW
4. Utilization: ~50%
5. Consumption: 400 GW × 24 × 365 = 3,504 TWh/year
6. Global electricity: ~25,000 TWh/year
7. Share: ~14%

**Answer:** Order of magnitude: 10-15% of global electricity.

**Key insight:** Data centers are significant energy consumers, growing rapidly.
