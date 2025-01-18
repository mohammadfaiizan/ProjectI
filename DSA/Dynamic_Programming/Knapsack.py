class Knapsack:
    def __init__(self):
        pass

    # Solution for 0/1 Knapsack Problem - Recursive
    def knapsack_01_recursive(self, weights, values, capacity, n):
        if n == 0 or capacity == 0:
            return 0

        if weights[n - 1] <= capacity:
            return max(
                values[n - 1] + self.knapsack_01_recursive(weights, values, capacity - weights[n - 1], n - 1),
                self.knapsack_01_recursive(weights, values, capacity, n - 1),
            )
        else:
            return self.knapsack_01_recursive(weights, values, capacity, n - 1)

    # Solution for 0/1 Knapsack Problem - Memoization (Top-down DP)
    def knapsack_01_memoization(self, weights, values, capacity):
        n = len(weights)
        dp = [[-1 for _ in range(capacity + 1)] for _ in range(n + 1)]

        def solve(w, c, n):
            if n == 0 or c == 0:
                return 0
            if dp[n][c] != -1:
                return dp[n][c]
            if w[n - 1] <= c:
                dp[n][c] = max(
                    values[n - 1] + solve(w, c - w[n - 1], n - 1), solve(w, c, n - 1)
                )
            else:
                dp[n][c] = solve(w, c, n - 1)
            return dp[n][c]

        return solve(weights, capacity, n)

    # Solution for 0/1 Knapsack Problem - Tabulation (Bottom-up DP)
    def knapsack_01_tabulation(self, weights, values, capacity):
        n = len(weights)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]
                    )
                else:
                    dp[i][w] = dp[i - 1][w]

        return dp[n][capacity]

    # Solution for Unbounded Knapsack - Recursive
    def unbounded_knapsack_recursive(self, weights, values, capacity):
        if capacity == 0:
            return 0

        n = len(weights)

        def solve(c):
            if c == 0:
                return 0
            max_value = 0
            for i in range(n):
                if weights[i] <= c:
                    max_value = max(max_value, values[i] + solve(c - weights[i]))
            return max_value

        return solve(capacity)

    # Solution for Unbounded Knapsack - Tabulation (Bottom-up DP)
    def unbounded_knapsack_tabulation(self, weights, values, capacity):
        n = len(weights)
        dp = [0 for _ in range(capacity + 1)]

        for c in range(1, capacity + 1):
            for i in range(n):
                if weights[i] <= c:
                    dp[c] = max(dp[c], values[i] + dp[c - weights[i]])

        return dp[capacity]
    
    # Solution for Unbounded Knapsack - Tabulation (Bottom-up DP with 2D array)
    def unbounded_knapsack_tabulation_2D(self, weights, values, capacity):
        n = len(weights)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        values[i - 1] + dp[i][w - weights[i - 1]], dp[i - 1][w]
                    )
                else:
                    dp[i][w] = dp[i - 1][w]

        return dp[n][capacity]
    
    # Knapsack with Duplicate Items
    def knapsack_with_duplicates(self, weights, values, capacity):
        n = len(weights)
        dp = [0 for _ in range(capacity + 1)]

        for i in range(n):
            for w in range(weights[i], capacity + 1):
                dp[w] = max(dp[w], values[i] + dp[w - weights[i]])

        return dp[capacity]

    # Fractional Knapsack (Greedy Algorithm)
    def fractional_knapsack(self, weights, values, capacity):
        n = len(weights)
        items = [(values[i] / weights[i], weights[i], values[i]) for i in range(n)]
        items.sort(reverse=True, key=lambda x: x[0])

        total_value = 0
        for ratio, weight, value in items:
            if capacity >= weight:
                total_value += value
                capacity -= weight
            else:
                total_value += ratio * capacity
                break

        return total_value


# Example Usage
if __name__ == "__main__":
    knapsack = Knapsack()

    weights = [1, 2, 3, 4]
    values = [10, 30, 40, 50]
    capacity = 10

    print("0/1 Knapsack - Recursive:", knapsack.knapsack_01_recursive(weights, values, capacity, len(weights)))
    print("0/1 Knapsack - Memoization:", knapsack.knapsack_01_memoization(weights, values, capacity))
    print("0/1 Knapsack - Tabulation:", knapsack.knapsack_01_tabulation(weights, values, capacity))
    print("Unbounded Knapsack - Recursive:", knapsack.unbounded_knapsack_recursive(weights, values, capacity))
    print("Unbounded Knapsack - Tabulation:", knapsack.unbounded_knapsack_tabulation(weights, values, capacity))
    print("Unbounded Knapsack - Tabulation 2D:", knapsack.unbounded_knapsack_tabulation_2D(weights, values, capacity))
    print("Knapsack with Duplicates:", knapsack.knapsack_with_duplicates(weights, values, capacity))
    print("Fractional Knapsack:", knapsack.fractional_knapsack(weights, values, capacity))
