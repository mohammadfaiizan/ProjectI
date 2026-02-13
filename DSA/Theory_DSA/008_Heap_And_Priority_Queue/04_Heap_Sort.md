# Heap Sort

## Theory

Heap sort is a comparison-based sorting algorithm that uses a binary heap. It consists of two phases:

1. **Build phase**: Transform the array into a max-heap (heapify)
2. **Extract phase**: Repeatedly extract the maximum and place it at the end of the unsorted portion

## Algorithm

1. Build max-heap from the array (bottom-up, starting from last non-leaf)
2. For i from n-1 down to 1:
   - Swap arr[0] with arr[i] (move max to sorted position)
   - Reduce heap size to i
   - Sift down the new root to restore max-heap property

## In-Place Sorting

Heap sort sorts in place. The "sorted" region grows from the end of the array. The heap occupies indices 0 to i-1; indices i to n-1 hold the sorted elements in ascending order.

## Step-by-Step Example

Array: [4, 10, 3, 5, 1]

**Build max-heap:**
```
Initial:     [4, 10, 3, 5, 1]
After heapify: [10, 5, 3, 4, 1]
        10
       /  \
      5    3
     / \
    4   1
```

**Extract phase:**
- Swap 10 and 1: [1, 5, 3, 4 | 10], sift down 1
- Swap 5 and 4: [4, 1, 3 | 5, 10], sift down 4
- Swap 4 and 3: [3, 1 | 4, 5, 10], sift down 3
- Swap 3 and 1: [1 | 3, 4, 5, 10], done

Result: [1, 3, 4, 5, 10]

## Python Implementation

```python
def sift_down_max(arr, n, i):
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        sift_down_max(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        sift_down_max(arr, i, 0)
```

## Time Complexity O(n log n) Proof

- Build heap: O(n) - bottom-up construction
- Extract phase: n-1 extractions, each sift down is O(log (heap_size)). Total: sum from k=1 to n-1 of log k = log((n-1)!) = O(n log n)
- Overall: O(n) + O(n log n) = O(n log n)

## Space Complexity

O(1) - only a few variables, no extra arrays.

## Stability

Heap sort is **not stable**. When we swap the root with the last element, we can change the relative order of equal elements. Example: [2a, 2b, 1] - after building max-heap and swapping, the two 2s may swap order.

## Comparison with Merge Sort

| Property | Heap Sort | Merge Sort |
|----------|-----------|------------|
| Time | O(n log n) | O(n log n) |
| Space | O(1) | O(n) |
| Stable | No | Yes |
| Cache | Poor locality | Better (sequential) |

## Comparison with Quick Sort

| Property | Heap Sort | Quick Sort |
|----------|-----------|------------|
| Average | O(n log n) | O(n log n) |
| Worst | O(n log n) | O(n^2) |
| Space | O(1) | O(log n) recursion |
| Stable | No | No (typical) |
| Practical speed | Slower (constants) | Faster (cache-friendly) |

## When to Use Heap Sort

- When O(1) space is critical and stability is not required
- When worst-case O(n log n) is required (quick sort can degrade)
- Embedded systems with limited memory
- As a teaching example of heap operations
