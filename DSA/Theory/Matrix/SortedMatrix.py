import heapq

class SortedMatrixClass:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)

    def kthSmallest(self, k):
        '''
        Problem Statement:
        Given an N x N matrix where each row and column is sorted in non-decreasing order, find the kth smallest element in the matrix.

        Approach:
        - We use binary search over the range of matrix values (from the smallest to the largest element).
        - We define a helper function `countLessEqual(mid)` which counts how many elements are less than or equal to `mid` in the matrix.
        - If the count of elements less than or equal to `mid` is less than `k`, we move our search to the right side of the matrix; otherwise, we move it to the left side.
        - The answer will be the `k`th smallest element found at the point where binary search converges.

        Time Complexity: O(n * log(max_value - min_value)), where n is the matrix size and max_value - min_value is the range of values in the matrix.
        Space Complexity: O(1), since we only use a few variables for the binary search and counting operation.

        '''
        def countLessEqual(mid):
            count = 0
            col = self.n - 1  # Start from the last column
            for row in range(self.n):
                while col >= 0 and self.matrix[row][col] > mid:
                    col -= 1
                count += (col + 1)
            return count
        
        low, high = self.matrix[0][0], self.matrix[self.n-1][self.n-1]
        while low < high:
            mid = (low + high) // 2
            if countLessEqual(mid) < k:
                low = mid + 1
            else:
                high = mid
        return low

    def kthLargest(self, k):
        '''
        Problem Statement:
        Given an N x N matrix where each row and column is sorted in non-decreasing order, find the kth largest element in the matrix.

        Approach:
        - We can use the `kthSmallest` function by reversing the logic. To find the kth largest element, we find the (n*n - k + 1)th smallest element.
        
        Time Complexity: O(n * log(max_value - min_value)), same as kthSmallest.
        Space Complexity: O(1).

        '''
        return self.kthSmallest(self.n * self.n - k + 1)

    def median(self):
        '''
        Problem Statement:
        Given an N x N matrix where each row and column is sorted in non-decreasing order, find the median of the matrix.

        Approach:
        - The median is the middle element in a sorted sequence. So we can use the `kthSmallest` function and find the middle element in the matrix.
        - The total number of elements is `n * n`. The median is the (n*n + 1) // 2th smallest element.

        Time Complexity: O(n * log(max_value - min_value)), same as kthSmallest.
        Space Complexity: O(1).

        '''
        total_elements = self.n * self.n
        return self.kthSmallest((total_elements + 1) // 2)

    def countLessEqual(self, x):
        '''
        Problem Statement:
        Given an N x N matrix where each row and column is sorted in non-decreasing order, count the number of elements less than or equal to a given number X.

        Approach:
        - We can use a two-pointer approach to count elements less than or equal to X.
        - Start from the last column of the first row. If the current element is greater than X, move left; otherwise, move down and keep counting.
        
        Time Complexity: O(n), as we traverse each row once.
        Space Complexity: O(1), as we only need a few variables for counting and traversing.

        '''
        count = 0
        col = self.n - 1
        for row in range(self.n):
            while col >= 0 and self.matrix[row][col] > x:
                col -= 1
            count += (col + 1)
        return count

    def searchElement(self, target):
        '''
        Problem Statement:
        Given an N x N matrix where each row and column is sorted in non-decreasing order, search for a given target element in the matrix.

        Approach:
        - Start from the top-right corner of the matrix. If the current element is equal to the target, return True.
        - If the current element is greater than the target, move left. Otherwise, move down.
        - This approach ensures we traverse the matrix in at most O(n) steps.

        Time Complexity: O(n), as we move at most n steps in each direction.
        Space Complexity: O(1), no extra space required other than variables for traversal.

        '''
        row, col = 0, self.n - 1
        while row < self.n and col >= 0:
            if self.matrix[row][col] == target:
                return True
            elif self.matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        return False

    def sortedMatrixToList(self):
        ''''
        Problem Statement:
        Given an N x N matrix where each row and column is sorted in non-decreasing order, convert the matrix into a sorted list.

        Approach:
        - Use a min-heap (priority queue) to efficiently extract the smallest element from the matrix.
        - Add the first element of each row into the heap.
        - Extract the minimum element from the heap, add it to the result, and insert the next element from the same row into the heap.
        
        Time Complexity: O(n^2 log n), as we push and pop elements from the heap n^2 times, and heap operations take O(log n).
        Space Complexity: O(n), for the heap that stores at most n elements at any time.
        '''
        n = len(self.matrix)
        result = []
        
        # Min-heap (priority queue) to store elements and their corresponding row and column indices
        heap = []
        
        # Initialize the heap with the first element of each row
        for row in range(n):
            heapq.heappush(heap, (self.matrix[row][0], row, 0))  # (value, row, col)
        
        # Extract elements one by one from the heap and add the next element from the row
        while heap:
            value, row, col = heapq.heappop(heap)  # Extract the minimum element
            
            # Add the minimum element to the result list
            result.append(value)
            
            # If there is another element in the same row, push it into the heap
            if col + 1 < n:
                heapq.heappush(heap, (self.matrix[row][col + 1], row, col + 1))
        
        return result

    def findPairsWithSum(self, target_sum):
        '''
        Problem Statement:
        Given an N x N matrix where each row and column is sorted in non-decreasing order, find all pairs of elements that sum up to a given target sum.

        Approach:
        - For each row, traverse the elements from right to left and compare them with the current element. If the sum matches the target sum, add the pair to the list.
        
        Time Complexity: O(n), as we are checking the elements in the matrix.
        Space Complexity: O(n), to store the pairs.

        '''
        pairs = []
        for row in range(self.n):
            col = self.n - 1
            while col >= 0:
                current_sum = self.matrix[row][col]
                if current_sum == target_sum:
                    pairs.append((row, col))
                    break
                elif current_sum > target_sum:
                    col -= 1
                else:
                    break
        return pairs

    def smallestRange(self):
        '''
        Problem Statement:
        Given an N x N matrix where each row and column is sorted in non-decreasing order, find the smallest range that includes at least one element from each row.

        Approach:
        - Use a min-heap to efficiently find the smallest range that includes at least one element from each row.
        - Start by pushing the first element from each row into the heap. Keep track of the maximum value seen so far.
        - Extract the smallest element from the heap and push the next element from the same row into the heap. Update the maximum value if needed.
        - The smallest range is the difference between the smallest and largest elements in the heap.
        
        Time Complexity: O(n log n), as we are using a heap to track the minimum and maximum elements from each row.
        Space Complexity: O(n), as we are storing the heap.

        '''
        min_heap = []
        current_max = float('-inf')
        for row in range(self.n):
            heapq.heappush(min_heap, (self.matrix[row][0], row, 0))
            current_max = max(current_max, self.matrix[row][0])
        
        start, end = -1, float('inf')
        
        while True:
            current_min, row, col = heapq.heappop(min_heap)
            
            if end - start > current_max - current_min:
                start, end = current_min, current_max
            
            if col + 1 < self.n:
                next_val = self.matrix[row][col + 1]
                heapq.heappush(min_heap, (next_val, row, col + 1))
                current_max = max(current_max, next_val)
            else:
                break

        return [start, end]

    def rotateMatrix(self):
        '''
        Problem Statement:
        Given an N x N matrix, rotate the matrix 90 degrees clockwise.

        Approach:
        - The rotation can be done in two steps:
        1. Transpose the matrix (swap rows and columns).
        2. Reverse each row of the transposed matrix.
        
        Time Complexity: O(n^2), as we are performing transposition and row reversal.
        Space Complexity: O(1), no extra space required apart from modifying the matrix in place.

        '''
        self.matrix = [list(x) for x in zip(*self.matrix)]
        for row in self.matrix:
            row.reverse()
        return self.matrix

# Main to test the problems
if __name__ == "__main__":
    matrix = [
        [1, 5, 9],
        [10, 11, 13],
        [12, 13, 15]
    ]
    matrix_obj = SortedMatrixClass(matrix)
    
    print(matrix_obj.kthSmallest(8)) # Should print 13
    print(matrix_obj.median()) # Should print 11
    print(matrix_obj.kthLargest(2)) # Should print 13
    print(matrix_obj.sortedMatrixToList()) # Should print the sorted list of all elements
    print(matrix_obj.searchElement(10)) # Should print True
    print(matrix_obj.smallestRange()) # Should print the smallest range
    matrix_obj.rotateMatrix() # Should rotate the matrix
