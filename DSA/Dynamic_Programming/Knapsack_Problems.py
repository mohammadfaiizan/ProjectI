class KnapsackProblems:
    def __init__(self):
        pass

    # Subset Sum Problem: Check if a subset with a given sum exists
    def subset_sum(self, weights, capacity):
        n = len(weights)
        dp = [[False for _ in range(capacity + 1)] for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = True  # Subset sum of 0 is always possible (empty set)
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = dp[i - 1][w] or dp[i - 1][w - weights[i - 1]]
                else:
                    dp[i][w] = dp[i - 1][w]
        
        return dp[n][capacity]

    # Equal Sum Partition Problem: Partition an array into two subsets with equal sum
    def equal_sum_partition(self, weights):
        total_sum = sum(weights)
        if total_sum % 2 != 0:
            return False
        return self.subset_sum(weights, total_sum // 2)

    # Count Subsets with Given Sum
    def count_subsets_with_given_sum(self, weights, capacity):
        n = len(weights)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = 1  # There is always one subset with sum 0 (empty set)
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = dp[i - 1][w] + dp[i - 1][w - weights[i - 1]]
                else:
                    dp[i][w] = dp[i - 1][w]
        
        return dp[n][capacity]

    # Minimum Subset Sum Difference
    def minimum_subset_sum_difference(self, weights):
        total_sum = sum(weights)
        dp = [[False for _ in range(total_sum + 1)] for _ in range(len(weights) + 1)]
        for i in range(len(weights) + 1):
            dp[i][0] = True
        
        for i in range(1, len(weights) + 1):
            for j in range(1, total_sum + 1):
                if weights[i - 1] <= j:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - weights[i - 1]]
                else:
                    dp[i][j] = dp[i - 1][j]
        
        half_sum = total_sum // 2
        for i in range(half_sum, -1, -1):
            if dp[len(weights)][i]:
                return total_sum - 2 * i

    # Target Sum Problem
    def target_sum(self, nums, target):
        total_sum = sum(nums)
        if total_sum < target or (total_sum - target) % 2 != 0:
            return 0
        return self.count_subsets_with_given_sum(nums, (total_sum - target) // 2)

    # Rod Cutting Problem
    def rod_cutting(self, lengths, prices, rod_length):
        n = len(lengths)
        dp = [0 for _ in range(rod_length + 1)]
        
        for i in range(1, rod_length + 1):
            for j in range(n):
                if lengths[j] <= i:
                    dp[i] = max(dp[i], prices[j] + dp[i - lengths[j]])

        return dp[rod_length]

    # Coin Change Problem - Minimum coins to make a value
    def coin_change_minimum(self, coins, amount):
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0  # 0 amount requires 0 coins
        
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1

    # Coin Change Problem - Count all combinations of coins to make a value
    def coin_change_combinations(self, coins, amount):
        dp = [0] * (amount + 1)
        dp[0] = 1  # There is 1 way to make 0 amount (using no coins)
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        
        return dp[amount]

    # Word Break Problem
    def word_break(self, s, word_dict):
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # Empty string can always be segmented

        for i in range(1, n + 1):
            for word in word_dict:
                if i >= len(word) and dp[i - len(word)] and s[i - len(word):i] == word:
                    dp[i] = True
                    break
        
        return dp[n]


# Example Usage
if __name__ == "__main__":
    knapsack_problems = KnapsackProblems()

    # Subset Sum Problem
    weights = [1, 2, 3, 4, 5]
    print("Subset Sum Problem:", knapsack_problems.subset_sum(weights, 10))  # Check if sum 10 is possible

    # Equal Sum Partition Problem
    print("Equal Sum Partition Problem:", knapsack_problems.equal_sum_partition(weights))

    # Count Subsets with Given Sum
    print("Count Subsets with Given Sum:", knapsack_problems.count_subsets_with_given_sum(weights, 10))

    # Minimum Subset Sum Difference
    print("Minimum Subset Sum Difference:", knapsack_problems.minimum_subset_sum_difference(weights))

    # Target Sum Problem
    print("Target Sum Problem:", knapsack_problems.target_sum([1, 1, 1, 1], 2))

    # Rod Cutting Problem
    lengths = [1, 2, 3, 4, 5]
    prices = [2, 5, 7, 8, 10]
    print("Rod Cutting Problem:", knapsack_problems.rod_cutting(lengths, prices, 5))

    # Coin Change Problem - Minimum coins
    coins = [1, 2, 5]
    amount = 11
    print("Coin Change Problem (Minimum coins):", knapsack_problems.coin_change_minimum(coins, amount))

    # Coin Change Problem - Count combinations
    print("Coin Change Problem (Combinations):", knapsack_problems.coin_change_combinations(coins, amount))

    # Word Break Problem
    s = "leetcode"
    word_dict = ["leet", "code"]
    print("Word Break Problem:", knapsack_problems.word_break(s, word_dict))
