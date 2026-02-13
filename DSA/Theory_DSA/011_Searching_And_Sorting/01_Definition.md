# Searching and Sorting: Definitions and Fundamentals

## Searching vs Sorting Overview

**Searching** is the process of finding a target element within a data structure. The efficiency depends on whether the data is sorted and the structure used.

**Sorting** is the process of arranging elements in a specific order (ascending or descending). Sorting often enables efficient searching (e.g., binary search requires sorted data).

| Aspect | Searching | Sorting |
|--------|-----------|---------|
| Goal | Locate an element | Arrange elements in order |
| Prerequisite | May require sorted data | None |
| Output | Index or boolean | Reordered array |
| Typical use | Lookup, membership | Preprocessing, ordering |

---

## Stability

A sorting algorithm is **stable** if it preserves the relative order of elements with equal keys. If two elements A and B have the same value and A appears before B in the input, then A appears before B in the output.

**Why stability matters:**
- Multi-key sorting: sort by name, then by age; stable sort preserves name order within same age
- Preserving original order of equal elements for user expectations
- Critical for algorithms that rely on previous sort passes

**Stable sorts:** Merge sort, insertion sort, bubble sort, counting sort, radix sort, tim sort

**Unstable sorts:** Quick sort, heap sort, shell sort, selection sort

---

## In-Place

An algorithm is **in-place** if it uses O(1) extra space (excluding the input array). The sorting happens within the original array.

**In-place sorts:** Quick sort, heap sort, shell sort, insertion sort, bubble sort, selection sort

**Not in-place:** Merge sort (requires O(n) auxiliary space)

---

## Adaptive

An algorithm is **adaptive** if it performs better on partially sorted or nearly sorted input. It takes advantage of existing order.

**Adaptive sorts:** Insertion sort, bubble sort (with early termination), tim sort

**Non-adaptive:** Selection sort, merge sort (always does full work)

---

## Comparison-Based Lower Bound O(n log n)

**Theorem:** Any comparison-based sorting algorithm must perform at least Omega(n log n) comparisons in the worst case.

**Proof sketch (decision tree model):**
1. Each comparison has two outcomes (less than or greater than)
2. For n distinct elements, there are n! possible permutations
3. A correct algorithm must distinguish all n! outcomes
4. A binary decision tree with n! leaves has height at least log2(n!)
5. By Stirling's approximation: log2(n!) = Theta(n log n)
6. Therefore, any comparison-based sort requires Omega(n log n) comparisons

**Implication:** To beat O(n log n), we need non-comparison sorts (counting, radix, bucket) that exploit structure of keys.

---

## Comparison vs Non-Comparison Sorting

| Type | Mechanism | Examples | When to use |
|------|-----------|----------|-------------|
| Comparison-based | Compare pairs of elements | Merge, quick, heap, insertion, bubble, selection | General purpose, arbitrary keys |
| Non-comparison | Use key structure (digits, range) | Counting, radix, bucket | Integer keys, bounded range, uniform distribution |

**Comparison sorts:** Lower bound Omega(n log n). Cannot do better than O(n log n) in worst case.

**Non-comparison sorts:** Can achieve O(n) when keys have structure (e.g., integers in range [0, k]).

---

## Time Complexity Summary Table

### Searching Algorithms

| Algorithm | Best | Average | Worst | Space | Prerequisite |
|-----------|------|---------|-------|-------|--------------|
| Linear search | O(1) | O(n) | O(n) | O(1) | None |
| Sentinel linear search | O(1) | O(n) | O(n) | O(1) | None |
| Binary search (iterative) | O(1) | O(log n) | O(log n) | O(1) | Sorted array |
| Binary search (recursive) | O(1) | O(log n) | O(log n) | O(log n) | Sorted array |
| Jump search | O(1) | O(sqrt(n)) | O(sqrt(n)) | O(1) | Sorted array |
| Interpolation search | O(1) | O(log log n) | O(n) | O(1) | Sorted, uniform |
| Exponential search | O(1) | O(log n) | O(log n) | O(1) | Sorted array |
| Fibonacci search | O(1) | O(log n) | O(log n) | O(1) | Sorted array |

### Sorting Algorithms

| Algorithm | Best | Average | Worst | Space | Stable | In-Place | Adaptive |
|-----------|------|---------|-------|-------|--------|----------|----------|
| Selection sort | O(n^2) | O(n^2) | O(n^2) | O(1) | No | Yes | No |
| Bubble sort | O(n) | O(n^2) | O(n^2) | O(1) | Yes | Yes | Yes |
| Insertion sort | O(n) | O(n^2) | O(n^2) | O(1) | Yes | Yes | Yes |
| Merge sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No | No |
| Quick sort | O(n log n) | O(n log n) | O(n^2) | O(log n) | No | Yes | No |
| Heap sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No | Yes | No |
| Shell sort | O(n log n) | O(n^1.3) | O(n^2) | O(1) | No | Yes | No |
| Counting sort | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes | No | No |
| Radix sort | O(d*n) | O(d*n) | O(d*n) | O(n+k) | Yes | No | No |
| Bucket sort | O(n+k) | O(n+k) | O(n^2) | O(n) | Yes | No | No |
| Tim sort | O(n) | O(n log n) | O(n log n) | O(n) | Yes | No | Yes |
| Introsort | O(n log n) | O(n log n) | O(n log n) | O(log n) | No | Yes | No |

Where: n = number of elements, k = range of keys, d = number of digits
