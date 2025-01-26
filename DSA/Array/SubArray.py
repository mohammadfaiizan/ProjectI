class Subarray:
    
    # Topic: Kadane's Algorithm
    # Problem: Maximum Sum Subarray
    # Time Complexity: O(n)
    def max_sum_subarray(self, arr):
        """
        Find the contiguous subarray within a one-dimensional array of numbers
        which has the largest sum.
        """
        max_sum = float('-inf')
        sum = 0
        
        for num in arr:
            sum += num
            if sum > max_sum:
                max_sum = sum
            if sum < 0:
                sum = 0
        return max_sum
    
    # Topic: Sliding Window
    # Problem: Subarray with Given Sum
    # Time Complexity: O(n)
    def subarray_with_given_sum(self, arr, target):
        """
        Find a subarray whose sum is equal to a given number.
        """
        n = len(arr)
        left = 0
        sum = 0
        for right in range(n):
            sum += arr[right]
            while sum > target and left <= right:
                sum -= arr[left]
                left+=1
            
            if sum == target:
                return [left+1, right+1]
        return [-1]
    
    # Topic: Prefix Sum
    # Problem: Find Longest Subarrays with Sum Zero
    # Time Complexity: O(n)
    def LongestSubarrays_Sum_Zero(self, arr):
        map = {}
        curr_sum = 0
        max_len = 0
        for i in range(len(arr)):
            curr_sum += arr[i]
            if curr_sum == 0:
                max_len = i + 1    
            if curr_sum in map:
                curr_len = i - map[curr_sum]
                max_len = max(max_len, curr_len)
            else:
                map[curr_sum] = i
        return max_len
    
    # Topic: Prefix Sum
    # Problem: Count of Subarrays with Sum K
    # Time Complexity: O(n)
    def count_subarrays_with_sum_k(self, arr, target):
        """
        Count the number of subarrays whose sum equals K.
        """
        prefix_sum = {0: 1}
        curr_sum = 0
        count = 0
        for num in arr:
            curr_sum += num
            if curr_sum - target in prefix_sum:
                count += prefix_sum[curr_sum - target]
            prefix_sum[curr_sum] = prefix_sum.get(curr_sum, 0) + 1
        return count
    
    # Topic: Prefix Sum
    # Problem: Maximum Length Subarray with Sum K
    # Time Complexity: O(n)
    def max_len_subarray_with_sum_k(self, arr, target):
        """
        Find the maximum length of a subarray whose sum is equal to a given number.
        """
        prefix_sum = {0: -1}
        curr_sum = 0
        max_len = 0
        for i, num in enumerate(arr):
            curr_sum += num
            if curr_sum - target in prefix_sum:
                max_len = max(max_len, i - prefix_sum[curr_sum - target])
            prefix_sum[curr_sum] = i
        return max_len
    
    # Topic: Sliding Window
    # Problem: Longest Subarray with Sum K
    # Time Complexity: O(n)
    def longest_subarray_with_sum_k(self, arr, target):
        """
        Find the longest subarray whose sum is equal to a given number.
        """
        left = 0
        curr_sum = 0
        max_len = 0
        for right in range(len(arr)):
            curr_sum += arr[right]
            while curr_sum > target and left <= right:
                curr_sum -= arr[left]
                left += 1
            if curr_sum == target:
                max_len = max(max_len, right - left + 1)
        return max_len
    
    # Topic: Sliding Window
    # Problem: Longest Subarray with At Most K Distinct Elements
    # Time Complexity: O(n)
    def longest_subarray_with_k_distinct(self, arr, k):
        """
        Find the length of the longest subarray that contains at most K distinct elements.
        """
        left = 0
        char_count = {}
        max_len = 0
        for right in range(len(arr)):
            char_count[arr[right]] = char_count.get(arr[right], 0) + 1
            while len(char_count) > k:
                char_count[arr[left]] -= 1
                if char_count[arr[left]] == 0:
                    del char_count[arr[left]]
                left += 1
            max_len = max(max_len, right - left + 1)
        return max_len
    
    # Topic: Hash Map
    # Problem: Subarray with Maximum Product
    # Time Complexity: O(n)
    def subarray_with_max_product(self, arr):
        """
        Find a contiguous subarray within a one-dimensional array of numbers which has the largest product.
        """
        max_product = min_product = result = arr[0]
        for num in arr[1:]:
            if num < 0:
                max_product, min_product = min_product, max_product
            max_product = max(num, max_product * num)
            min_product = min(num, min_product * num)
            result = max(result, max_product)
        return result
    
    # Topic: Hash Map
    # Problem: Subarray with Given Sum (Negative Numbers Allowed)
    # Time Complexity: O(n)
    def subarray_with_given_sum_neg(self, arr, target):
        """
        Find the subarray with a given sum, considering negative numbers as well.
        """
        prefix_sum = {0: -1}
        curr_sum = 0
        for i, num in enumerate(arr):
            curr_sum += num
            if curr_sum - target in prefix_sum:
                return arr[prefix_sum[curr_sum - target] + 1: i + 1]
            prefix_sum[curr_sum] = i
        return None
    
    # Topic: Sliding Window
    # Problem: Smallest Subarray with Sum Greater Than K
    # Time Complexity: O(n)
    def smallest_subarray_with_sum_greater_than_k(self, arr, target):
        """
        Find the smallest subarray whose sum is greater than a given number.
        """
        left = 0
        curr_sum = 0
        min_len = float('inf')
        for right in range(len(arr)):
            curr_sum += arr[right]
            while curr_sum > target:
                min_len = min(min_len, right - left + 1)
                curr_sum -= arr[left]
                left += 1
        return min_len if min_len != float('inf') else 0
    
    # Topic: Hash Map
    # Problem: Count of Subarrays with Sum K
    # Time Complexity: O(n)
    def count_subarrays_with_sum_k_v2(self, arr, target):
        """
        Count the number of subarrays whose sum equals K.
        """
        prefix_sum = {0: 1}
        curr_sum = 0
        count = 0
        for num in arr:
            curr_sum += num
            if curr_sum - target in prefix_sum:
                count += prefix_sum[curr_sum - target]
            prefix_sum[curr_sum] = prefix_sum.get(curr_sum, 0) + 1
        return count
    
    # Topic: Two Pointers
    # Problem: Maximum Size Subarray Sum Equals K
    # Time Complexity: O(n)
    def max_size_subarray_sum_k(self, arr, target):
        """
        Find the largest size subarray with a sum equal to a given value.
        """
        left = 0
        curr_sum = 0
        max_size = 0
        for right in range(len(arr)):
            curr_sum += arr[right]
            while curr_sum > target:
                curr_sum -= arr[left]
                left += 1
            if curr_sum == target:
                max_size = max(max_size, right - left + 1)
        return max_size

# Usage Example
subarray = Subarray()

arr = [1, 2, 3, 4, -2, 1, 5, -3]
print(subarray.max_sum_subarray(arr))  # Max Sum Subarray
print(subarray.subarray_with_given_sum(arr, 5))  # Subarray with Given Sum
print(subarray.find_subarrays_with_sum_zero(arr))  # Find All Subarrays with Sum Zero
