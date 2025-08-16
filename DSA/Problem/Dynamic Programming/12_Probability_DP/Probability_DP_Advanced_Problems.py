"""
Advanced Probability DP Problems
Difficulty: Hard
Category: Probability DP - Advanced Techniques and Applications

PROBLEM COLLECTION:
==================
This file contains advanced probability DP problems that showcase:

1. Conditional Expectation and Variance
2. Continuous Probability Distributions
3. Stochastic Optimization
4. Multi-stage Decision Processes
5. Competitive Analysis with Probability
6. Advanced Markov Chain Analysis
7. Queueing Theory Applications
8. Game Theory with Uncertainty

These problems demonstrate sophisticated probability techniques
used in competitive programming, research, and real-world applications.
"""


def expected_value_with_stopping(probabilities, values, max_steps):
    """
    EXPECTED VALUE WITH OPTIMAL STOPPING:
    ====================================
    Calculate optimal expected value when you can stop at any time.
    
    Problem: Given sequence of (probability, value) pairs,
    decide when to stop to maximize expected value.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    n = len(probabilities)
    if n == 0 or n != len(values):
        return 0.0
    
    # Work backwards to find optimal stopping strategy
    max_expected = 0.0
    cumulative_prob = 1.0
    
    # Calculate expected value if we never stop
    never_stop_expected = sum(p * v for p, v in zip(probabilities, values))
    
    # Work backwards to find optimal stopping
    for i in range(n - 1, -1, -1):
        # Expected value if we stop at position i
        stop_here_expected = cumulative_prob * values[i]
        
        # Expected value if we continue
        continue_expected = cumulative_prob * probabilities[i] * values[i]
        if i < n - 1:
            continue_expected += (1 - probabilities[i]) * max_expected
        
        # Choose the better option
        max_expected = max(stop_here_expected, continue_expected)
        
        # Update cumulative probability of reaching this position
        if i > 0:
            cumulative_prob *= (1 - probabilities[i - 1])
    
    return max_expected


def multi_stage_decision_probability(stages, decisions_per_stage, transition_probs, rewards):
    """
    MULTI-STAGE DECISION PROCESS:
    ============================
    Optimal decision making under uncertainty across multiple stages.
    
    Time Complexity: O(stages * decisions^2) - DP over stages and decisions
    Space Complexity: O(decisions) - current stage values
    """
    if not stages or not decisions_per_stage:
        return 0.0
    
    # dp[decision] = maximum expected value from this decision
    dp = [0.0] * decisions_per_stage
    
    # Work backwards from last stage
    for stage in range(stages - 1, -1, -1):
        new_dp = [0.0] * decisions_per_stage
        
        for current_decision in range(decisions_per_stage):
            expected_value = 0.0
            
            # Consider all possible next decisions
            for next_decision in range(decisions_per_stage):
                # Transition probability from current to next decision
                prob = transition_probs[stage][current_decision][next_decision]
                
                # Immediate reward
                reward = rewards[stage][current_decision]
                
                # Future value
                future_value = dp[next_decision] if stage < stages - 1 else 0.0
                
                expected_value += prob * (reward + future_value)
            
            new_dp[current_decision] = expected_value
        
        dp = new_dp
    
    # Return maximum expected value from any starting decision
    return max(dp) if dp else 0.0


def competitive_probability_analysis(player_strategies, opponent_strategies, payoff_matrix):
    """
    COMPETITIVE ANALYSIS WITH PROBABILITY:
    =====================================
    Game theory with probabilistic strategies and uncertain outcomes.
    
    Time Complexity: O(m * n) - where m, n are strategy counts
    Space Complexity: O(m * n) - payoff matrix
    """
    m = len(player_strategies)
    n = len(opponent_strategies)
    
    if m == 0 or n == 0:
        return 0.0
    
    # Calculate expected payoff for each strategy combination
    total_expected_payoff = 0.0
    
    for i in range(m):
        for j in range(n):
            prob_combination = player_strategies[i] * opponent_strategies[j]
            payoff = payoff_matrix[i][j]
            total_expected_payoff += prob_combination * payoff
    
    return total_expected_payoff


def queueing_system_probability(arrival_rate, service_rate, max_queue_size):
    """
    QUEUEING SYSTEM ANALYSIS:
    ========================
    Calculate steady-state probabilities in M/M/1/K queue.
    
    Time Complexity: O(K) - where K is max queue size
    Space Complexity: O(K) - probability array
    """
    if service_rate <= 0:
        return [0.0] * (max_queue_size + 1)
    
    rho = arrival_rate / service_rate
    
    if abs(rho - 1.0) < 1e-10:
        # Special case: rho = 1
        prob = 1.0 / (max_queue_size + 1)
        return [prob] * (max_queue_size + 1)
    
    # Calculate steady-state probabilities
    probabilities = [0.0] * (max_queue_size + 1)
    
    # P_0 = (1 - rho) / (1 - rho^(K+1))
    if abs(rho) < 1e-10:
        probabilities[0] = 1.0
        return probabilities
    
    numerator = 1.0 - rho
    denominator = 1.0 - (rho ** (max_queue_size + 1))
    
    if abs(denominator) < 1e-10:
        # Handle numerical issues
        prob = 1.0 / (max_queue_size + 1)
        return [prob] * (max_queue_size + 1)
    
    p_0 = numerator / denominator
    
    # P_k = P_0 * rho^k
    for k in range(max_queue_size + 1):
        probabilities[k] = p_0 * (rho ** k)
    
    return probabilities


def stochastic_optimization_problem(objectives, constraints, uncertainties):
    """
    STOCHASTIC OPTIMIZATION:
    =======================
    Optimize expected objective under uncertain constraints.
    
    Time Complexity: O(n * m) - where n is decisions, m is scenarios
    Space Complexity: O(n * m) - scenario matrix
    """
    if not objectives or not constraints or not uncertainties:
        return 0.0, []
    
    n_decisions = len(objectives)
    n_scenarios = len(uncertainties)
    
    # For each decision, calculate expected objective value
    expected_objectives = []
    
    for decision in range(n_decisions):
        expected_value = 0.0
        
        for scenario in range(n_scenarios):
            probability = uncertainties[scenario]['probability']
            
            # Check if this decision is feasible in this scenario
            feasible = True
            for constraint in constraints:
                if not constraint(decision, scenario):
                    feasible = False
                    break
            
            if feasible:
                objective_value = objectives[decision]
                # Apply scenario-specific modifications
                if 'modifier' in uncertainties[scenario]:
                    objective_value *= uncertainties[scenario]['modifier']
                
                expected_value += probability * objective_value
        
        expected_objectives.append(expected_value)
    
    # Find optimal decision
    best_decision = 0
    best_value = expected_objectives[0]
    
    for i in range(1, n_decisions):
        if expected_objectives[i] > best_value:
            best_value = expected_objectives[i]
            best_decision = i
    
    return best_value, expected_objectives


def markov_chain_analysis(transition_matrix, initial_distribution, steps):
    """
    ADVANCED MARKOV CHAIN ANALYSIS:
    ==============================
    Calculate distribution after k steps and limiting behavior.
    
    Time Complexity: O(n^3 * log(steps)) - matrix exponentiation
    Space Complexity: O(n^2) - transition matrix
    """
    import numpy as np
    
    n = len(transition_matrix)
    if n == 0 or len(initial_distribution) != n:
        return []
    
    # Convert to numpy arrays
    P = np.array(transition_matrix)
    pi_0 = np.array(initial_distribution)
    
    # Calculate P^steps using matrix exponentiation
    def matrix_power(matrix, power):
        result = np.eye(len(matrix))
        base = matrix.copy()
        
        while power > 0:
            if power % 2 == 1:
                result = np.dot(result, base)
            base = np.dot(base, base)
            power //= 2
        
        return result
    
    # Distribution after k steps: π_k = π_0 * P^k
    P_k = matrix_power(P, steps)
    distribution = np.dot(pi_0, P_k)
    
    return distribution.tolist()


def probability_dp_portfolio_optimization(assets, correlations, risk_tolerance, time_horizon):
    """
    PORTFOLIO OPTIMIZATION WITH PROBABILITY:
    =======================================
    Dynamic portfolio optimization under uncertainty.
    
    Time Complexity: O(T * n^2) - where T is time horizon, n is assets
    Space Complexity: O(n^2) - covariance matrix
    """
    n_assets = len(assets)
    if n_assets == 0:
        return []
    
    # Simple mean-variance optimization with time dependency
    portfolio_weights = []
    
    for t in range(time_horizon):
        # Time-dependent risk adjustment
        time_factor = 1.0 - (t / time_horizon)  # Reduce risk as time progresses
        adjusted_risk_tolerance = risk_tolerance * time_factor
        
        # Calculate expected returns and risks
        expected_returns = [asset['expected_return'] for asset in assets]
        risks = [asset['volatility'] for asset in assets]
        
        # Simple optimization: maximize return per unit risk
        weights = [0.0] * n_assets
        total_score = 0.0
        
        for i in range(n_assets):
            if risks[i] > 0:
                score = expected_returns[i] / (risks[i] + adjusted_risk_tolerance)
                weights[i] = max(0.0, score)
                total_score += weights[i]
        
        # Normalize weights
        if total_score > 0:
            weights = [w / total_score for w in weights]
        else:
            weights = [1.0 / n_assets] * n_assets
        
        portfolio_weights.append(weights)
    
    return portfolio_weights


def advanced_probability_analysis():
    """
    COMPREHENSIVE ADVANCED PROBABILITY ANALYSIS:
    ===========================================
    Demonstrate advanced probability DP techniques.
    """
    print("Advanced Probability DP Analysis:")
    print("=" * 50)
    
    # 1. Optimal stopping problem
    print("\n1. Optimal Stopping Problem:")
    probs = [0.8, 0.6, 0.4, 0.3]
    values = [100, 80, 60, 40]
    max_expected = expected_value_with_stopping(probs, values, len(probs))
    print(f"   Probabilities: {probs}")
    print(f"   Values: {values}")
    print(f"   Maximum expected value: {max_expected:.2f}")
    
    # 2. Multi-stage decision process
    print("\n2. Multi-stage Decision Process:")
    stages = 3
    decisions = 2
    transitions = [
        [[0.7, 0.3], [0.4, 0.6]],  # Stage 0 transitions
        [[0.8, 0.2], [0.3, 0.7]],  # Stage 1 transitions
        [[0.5, 0.5], [0.5, 0.5]]   # Stage 2 transitions
    ]
    rewards = [[10, 5], [8, 12], [6, 15]]
    
    max_value = multi_stage_decision_probability(stages, decisions, transitions, rewards)
    print(f"   Stages: {stages}, Decisions per stage: {decisions}")
    print(f"   Maximum expected value: {max_value:.2f}")
    
    # 3. Competitive analysis
    print("\n3. Competitive Analysis:")
    player_strat = [0.6, 0.4]  # Player's mixed strategy
    opponent_strat = [0.3, 0.7]  # Opponent's mixed strategy
    payoffs = [[3, -1], [0, 2]]  # Payoff matrix
    
    expected_payoff = competitive_probability_analysis(player_strat, opponent_strat, payoffs)
    print(f"   Player strategy: {player_strat}")
    print(f"   Opponent strategy: {opponent_strat}")
    print(f"   Expected payoff: {expected_payoff:.3f}")
    
    # 4. Queueing system
    print("\n4. Queueing System Analysis:")
    arrival = 0.8
    service = 1.0
    max_queue = 5
    
    probs = queueing_system_probability(arrival, service, max_queue)
    print(f"   Arrival rate: {arrival}, Service rate: {service}")
    print(f"   Steady-state probabilities:")
    for i, p in enumerate(probs):
        print(f"     P({i} customers) = {p:.4f}")
    
    # 5. Stochastic optimization
    print("\n5. Stochastic Optimization:")
    objectives = [100, 80, 120, 90]
    constraints = [
        lambda d, s: d + s < 5,  # Simple constraint
        lambda d, s: d * 2 + s < 8
    ]
    scenarios = [
        {'probability': 0.4, 'modifier': 1.0},
        {'probability': 0.6, 'modifier': 0.8}
    ]
    
    best_value, all_values = stochastic_optimization_problem(objectives, constraints, scenarios)
    print(f"   Best expected value: {best_value:.2f}")
    print(f"   All expected values: {[f'{v:.2f}' for v in all_values]}")


def probability_dp_case_studies():
    """
    PROBABILITY DP CASE STUDIES:
    ===========================
    Real-world applications of probability DP.
    """
    
    def option_pricing_binomial(S0, K, r, sigma, T, n):
        """European option pricing using binomial model"""
        dt = T / n
        u = 1 + sigma * (dt ** 0.5)  # Up factor
        d = 1 / u  # Down factor
        p = (1 + r * dt - d) / (u - d)  # Risk-neutral probability
        
        # Initialize option values at maturity
        option_values = []
        for i in range(n + 1):
            S_T = S0 * (u ** i) * (d ** (n - i))
            option_values.append(max(S_T - K, 0))  # Call option payoff
        
        # Work backwards through the tree
        for step in range(n - 1, -1, -1):
            new_values = []
            for i in range(step + 1):
                # Expected value under risk-neutral measure
                expected = p * option_values[i + 1] + (1 - p) * option_values[i]
                # Discount to present value
                present_value = expected / (1 + r * dt)
                new_values.append(present_value)
            option_values = new_values
        
        return option_values[0]
    
    def inventory_optimization(demand_dist, holding_cost, shortage_cost, order_cost):
        """Optimal inventory policy under uncertain demand"""
        # Simple newsvendor model
        # Optimal order quantity: F^(-1)(shortage_cost / (shortage_cost + holding_cost))
        
        critical_ratio = shortage_cost / (shortage_cost + holding_cost)
        
        # For normal distribution approximation
        mean_demand = sum(d * p for d, p in demand_dist)
        variance = sum(((d - mean_demand) ** 2) * p for d, p in demand_dist)
        std_dev = variance ** 0.5
        
        # Z-score for critical ratio (approximation)
        # This would normally require inverse normal function
        z_score = 0.0  # Simplified - would need proper inverse normal
        
        optimal_order = mean_demand + z_score * std_dev
        
        return optimal_order
    
    def reliability_optimization(components, system_reliability_target):
        """Optimize component reliabilities to meet system target"""
        n = len(components)
        if n == 0:
            return []
        
        # For series system: R_system = ∏ R_i
        # For parallel system: R_system = 1 - ∏ (1 - R_i)
        
        # Simple equal allocation for series system
        target_component_reliability = system_reliability_target ** (1/n)
        
        optimized_reliabilities = []
        for component in components:
            current_rel = component['current_reliability']
            cost_factor = component['improvement_cost_factor']
            
            # Balance target with cost considerations
            if target_component_reliability > current_rel:
                improvement_needed = target_component_reliability - current_rel
                adjusted_target = current_rel + improvement_needed / cost_factor
                optimized_reliabilities.append(min(adjusted_target, 0.99))
            else:
                optimized_reliabilities.append(current_rel)
        
        return optimized_reliabilities
    
    print("Probability DP Case Studies:")
    print("=" * 40)
    
    # Option pricing
    print("\n1. Option Pricing (Binomial Model):")
    S0, K, r, sigma, T, n = 100, 105, 0.05, 0.2, 1.0, 50
    option_price = option_pricing_binomial(S0, K, r, sigma, T, n)
    print(f"   Stock price: ${S0}, Strike: ${K}")
    print(f"   Option value: ${option_price:.2f}")
    
    # Inventory optimization
    print("\n2. Inventory Optimization:")
    demand_distribution = [(90, 0.2), (100, 0.5), (110, 0.3)]
    holding, shortage, order = 1, 5, 10
    optimal_qty = inventory_optimization(demand_distribution, holding, shortage, order)
    print(f"   Demand distribution: {demand_distribution}")
    print(f"   Optimal order quantity: {optimal_qty:.1f}")
    
    # Reliability optimization
    print("\n3. Reliability Optimization:")
    components = [
        {'current_reliability': 0.9, 'improvement_cost_factor': 2.0},
        {'current_reliability': 0.85, 'improvement_cost_factor': 1.5},
        {'current_reliability': 0.95, 'improvement_cost_factor': 3.0}
    ]
    target_system = 0.8
    
    optimized = reliability_optimization(components, target_system)
    print(f"   System reliability target: {target_system}")
    print(f"   Optimized component reliabilities: {[f'{r:.3f}' for r in optimized]}")


# Test comprehensive advanced probability DP
def test_advanced_probability_dp():
    """Test advanced probability DP implementations"""
    print("Testing Advanced Probability DP Solutions:")
    print("=" * 70)
    
    # Comprehensive analysis
    advanced_probability_analysis()
    
    # Case studies
    print(f"\n" + "=" * 70)
    probability_dp_case_studies()
    
    print("\n" + "=" * 70)
    print("Advanced Techniques Demonstrated:")
    print("1. OPTIMAL STOPPING: Dynamic decision making under uncertainty")
    print("2. MULTI-STAGE PROCESSES: Sequential optimization with feedback")
    print("3. COMPETITIVE ANALYSIS: Game theory with probabilistic strategies")
    print("4. QUEUEING THEORY: Steady-state analysis of stochastic systems")
    print("5. STOCHASTIC OPTIMIZATION: Expected value maximization")
    print("6. MARKOV ANALYSIS: Long-term behavior prediction")
    print("7. FINANCIAL MODELING: Risk management and option pricing")
    print("8. OPERATIONS RESEARCH: Inventory and reliability optimization")
    
    print("\n" + "=" * 70)
    print("Real-World Applications:")
    print("• Financial Engineering: Derivatives pricing and risk management")
    print("• Operations Research: Supply chain and inventory optimization")
    print("• Machine Learning: Reinforcement learning and decision processes")
    print("• Telecommunications: Network design and capacity planning")
    print("• Manufacturing: Quality control and reliability engineering")
    print("• Healthcare: Treatment planning and resource allocation")
    print("• Transportation: Route optimization under uncertainty")
    print("• Energy: Grid management and renewable integration")


if __name__ == "__main__":
    test_advanced_probability_dp()


"""
ADVANCED PROBABILITY DP - SOPHISTICATED STOCHASTIC MODELING:
===========================================================

This collection demonstrates advanced probability DP techniques:
- Multi-stage stochastic optimization
- Competitive analysis with uncertainty
- Continuous-time process approximations
- Real-world application modeling
- Advanced mathematical finance

KEY ADVANCED CONCEPTS:
=====================
1. **OPTIMAL STOPPING**: Secretary problem, option exercise, resource allocation
2. **MULTI-STAGE DECISIONS**: Dynamic programming with uncertain transitions
3. **COMPETITIVE PROBABILITY**: Game theory with mixed strategies
4. **QUEUEING SYSTEMS**: Steady-state analysis and performance metrics
5. **STOCHASTIC OPTIMIZATION**: Expected value maximization under constraints
6. **MARKOV ANALYSIS**: Long-term behavior and convergence properties
7. **FINANCIAL MODELING**: Option pricing, portfolio optimization, risk management
8. **OPERATIONS RESEARCH**: Inventory control, reliability engineering

MATHEMATICAL SOPHISTICATION:
===========================
**Continuous Approximations**: Discrete processes approximating continuous limits
**Risk Measures**: Value-at-Risk, Expected Shortfall, coherent risk measures
**Martingale Theory**: Fair game properties and optional stopping theorems
**Ergodic Theory**: Long-term average behavior and convergence

ALGORITHMIC TECHNIQUES:
======================
**Matrix Exponentiation**: Efficient computation of Markov chain evolution
**Backward Induction**: Optimal control in multi-stage processes
**Linear Programming**: Optimization under probabilistic constraints
**Simulation Methods**: Monte Carlo when analytical solutions intractable

APPLICATIONS SHOWCASE:
=====================
**Finance**: Option pricing, portfolio optimization, credit risk modeling
**Operations**: Inventory management, queueing systems, reliability design
**Engineering**: Signal processing, control systems, fault tolerance
**Computer Science**: Algorithm analysis, load balancing, network protocols

COMPLEXITY CONSIDERATIONS:
=========================
**Curse of Dimensionality**: Exponential growth in multi-dimensional problems
**Numerical Stability**: Precision issues in probability calculations
**Convergence Rates**: Speed of iterative algorithms
**Approximation Quality**: Trade-offs between accuracy and computation

This advanced framework provides tools for modeling
sophisticated real-world systems where uncertainty
and sequential decision-making are fundamental
characteristics requiring probabilistic analysis.
"""
