# NumPy Practical Problems Interview Questions

## Q1: How do you find all indices where a condition is true?

**A1:** Use np.where with a single argument to get indices where the condition is True. For a 1D array: indices = np.where(arr > threshold)[0]. For multidimensional arrays, np.where returns a tuple of arrays (one per dimension). You can also use np.argwhere which returns indices as rows of an array. Alternatively, np.nonzero returns the same result as np.where for finding non-zero elements or applying conditions. The returned indices can be used for further indexing or to modify specific elements efficiently.

## Q2: How do you compute a moving average?

**A2:** Use np.convolve with a uniform kernel for simple moving average: result = np.convolve(arr, np.ones(window_size)/window_size, mode='valid'). For more control, use np.cumsum and array slicing: cumsum = np.cumsum(arr), moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size. The cumsum approach is efficient for large arrays as it avoids repeated summations. For 2D arrays, use np.convolve2d or apply the 1D method along the desired axis. The mode parameter controls edge handling: 'valid' returns only complete windows, 'same' pads to maintain size, 'full' includes partial windows.

## Q3: How do you normalize an array to the range [0, 1]?

**A3:** Use min-max normalization: normalized = (arr - arr.min()) / (arr.max() - arr.min()). This scales values linearly so the minimum becomes 0 and maximum becomes 1. For handling division by zero when min equals max, add a check: if arr.max() != arr.min(): normalized = (arr - arr.min()) / (arr.max() - arr.min()). Alternatively, use np.clip to ensure values stay in [0, 1] range. For normalizing along a specific axis in multidimensional arrays, specify axis parameter: (arr - arr.min(axis=1, keepdims=True)) / (arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True)).

## Q4: How do you find the top-k elements in an array?

**A4:** Use np.argpartition for efficient partial sorting: top_k_indices = np.argpartition(arr, -k)[-k:], then sort these indices: top_k_indices = top_k_indices[np.argsort(arr[top_k_indices])[::-1]]. Alternatively, use np.argsort and take the last k indices: top_k_indices = np.argsort(arr)[-k:][::-1]. For just the values: top_k_values = np.partition(arr, -k)[-k:][::-1]. argpartition is O(n) for finding k elements, faster than full sorting when k is small. For largest elements, use negative k with argpartition; for smallest, use positive k.

## Q5: How do you compute pairwise distances between points?

**A5:** For Euclidean distance between points stored as rows: distances = np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2).sum(axis=2)). This uses broadcasting to compute all pairwise differences, squares them, sums along the feature dimension, and takes square root. For efficiency with large datasets, use scipy.spatial.distance.cdist or compute using matrix operations: D = np.sqrt(np.sum((points[:, None, :] - points[None, :, :])**2, axis=-1)). The broadcasting approach creates a (n, n, d) intermediate array, so for very large n, consider chunked computation or specialized libraries.

## Q6: How do you create a one-hot encoding?

**A6:** Use np.eye to create an identity matrix and index it: one_hot = np.eye(num_classes)[labels]. This creates a (len(labels), num_classes) array where each row has 1 at the position corresponding to the label. Alternatively, initialize zeros and use advanced indexing: one_hot = np.zeros((len(labels), num_classes)), one_hot[np.arange(len(labels)), labels] = 1. The np.eye approach is more concise, while the zeros approach gives more control. Both are vectorized and efficient. For labels outside [0, num_classes-1], add validation or use np.clip to ensure valid indices.

## Q7: How do you compute cosine similarity?

**A7:** Normalize vectors to unit length, then compute dot product: norm_a = np.linalg.norm(a), norm_b = np.linalg.norm(b), similarity = np.dot(a, b) / (norm_a * norm_b). For multiple pairs, use broadcasting: norms = np.linalg.norm(vectors, axis=1, keepdims=True), normalized = vectors / norms, similarities = np.dot(normalized, normalized.T). Alternatively, use scipy.spatial.distance.cosine and subtract from 1, or implement directly: similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)). Cosine similarity measures angle between vectors, ranging from -1 (opposite) to 1 (same direction).

## Q8: How do you find duplicate rows in a 2D array?

**A8:** Use np.unique with return_index and return_inverse: unique_rows, indices, inverse = np.unique(arr, axis=0, return_index=True, return_inverse=True), then find duplicates using np.bincount or by checking which inverse values appear multiple times. Alternatively, convert rows to tuples and use np.unique, or use a more direct approach: sorted_indices = np.lexsort(arr.T), duplicates = np.where(np.all(arr[sorted_indices[1:]] == arr[sorted_indices[:-1]], axis=1))[0]. The lexsort approach sorts rows and compares adjacent ones. For large arrays, consider using pandas DataFrame.duplicated() or converting to a structured array for efficient comparison.

## Q9: How do you replace NaN values with column means?

**A9:** Compute column means ignoring NaN: col_means = np.nanmean(arr, axis=0). Find NaN positions: nan_mask = np.isnan(arr). Replace NaNs: arr[nan_mask] = np.take(col_means, np.where(nan_mask)[1]). Alternatively, use np.where: arr = np.where(np.isnan(arr), np.nanmean(arr, axis=0), arr). The np.where approach uses broadcasting to fill all NaN positions in each column with the corresponding mean. For in-place modification: nan_mask = np.isnan(arr), arr[nan_mask] = np.take(col_means, np.where(nan_mask)[1]). This preserves the original array structure while handling missing values.

## Q10: How do you create a sliding window view?

**A10:** Use np.lib.stride_tricks.sliding_window_view for NumPy 1.20+: windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape). This creates a view (not a copy) showing all possible windows. For older NumPy, use stride_tricks manually: shape = (arr.shape[0] - window_size + 1, window_size) + arr.shape[1:], strides = (arr.strides[0],) + arr.strides, windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides). The stride_tricks approach creates a zero-copy view by manipulating memory layout. For 2D windows on 2D arrays, adjust shape and strides accordingly. Be careful with memory safety when using stride_tricks directly.

## Q11: How do you compute element-wise percentage change?

**A11:** Calculate percentage change from previous element: pct_change = np.diff(arr) / arr[:-1] * 100. This computes (current - previous) / previous * 100. For percentage change from first element: pct_change = (arr - arr[0]) / arr[0] * 100. To prepend NaN or zero for the first element: pct_change = np.concatenate([[0], np.diff(arr) / arr[:-1] * 100]) or use np.insert. For 2D arrays along an axis: pct_change = np.diff(arr, axis=0) / arr[:-1] * 100. Handle division by zero cases where previous values are zero, either by filtering or using np.where to set those to NaN or zero.

## Q12: How do you create a diagonal block matrix?

**A12:** Initialize a larger matrix and place blocks along the diagonal: n_blocks = len(blocks), block_size = blocks[0].shape[0], total_size = n_blocks * block_size, result = np.zeros((total_size, total_size)). Then place each block: for i, block in enumerate(blocks): start = i * block_size, result[start:start+block_size, start:start+block_size] = block. Alternatively, use scipy.linalg.block_diag for convenience. For efficient construction, use np.block with a list of lists where off-diagonals are None or zeros: blocks_list = [[blocks[i] if i==j else None for j in range(n_blocks)] for i in range(n_blocks)], result = np.block(blocks_list). This creates a block diagonal matrix efficiently.

## Q13: How do you compute the mode of an array?

**A13:** Use np.unique with return_counts: values, counts = np.unique(arr, return_counts=True), mode_value = values[np.argmax(counts)]. This finds the most frequent value. For handling multiple modes (ties), find all values with maximum count: mode_values = values[counts == counts.max()]. For continuous data, you might need binning first using np.histogram or np.digitize. Alternatively, use scipy.stats.mode which handles edge cases and returns mode, count, and other statistics. The np.unique approach is efficient and works well for discrete data, while scipy.stats.mode provides more robust handling of edge cases.

## Q14: How do you shuffle rows of a 2D array independently?

**A14:** Generate random indices for each row and use advanced indexing: n_rows, n_cols = arr.shape, shuffled = arr[np.arange(n_rows)[:, None], np.random.permutation(n_cols)]. This creates a permutation for each row and applies it. Alternatively, use a loop with np.random.shuffle, but this is slower. For a vectorized approach using broadcasting: indices = np.random.rand(n_rows, n_cols).argsort(axis=1), shuffled = arr[np.arange(n_rows)[:, None], indices]. The argsort approach generates random permutations by sorting random values. Both methods shuffle each row independently while keeping rows intact.

## Q15: How do you find the most frequent value?

**A15:** Use np.unique with return_counts: values, counts = np.unique(arr, return_counts=True), most_frequent = values[np.argmax(counts)]. This returns the value that appears most often. For handling multiple values with the same maximum frequency: max_count = counts.max(), most_frequent = values[counts == max_count]. If you need the count as well: most_frequent_value = values[np.argmax(counts)], frequency = counts.max(). For large arrays, this approach is efficient. For continuous data, consider binning first using np.histogram to convert to discrete bins before finding the mode.

## Q16: How do you compute the running maximum?

**A16:** Use np.maximum.accumulate: running_max = np.maximum.accumulate(arr). This applies the maximum function cumulatively, keeping track of the maximum value seen so far. For a 2D array along an axis: running_max = np.maximum.accumulate(arr, axis=1). The accumulate method of ufuncs applies the function cumulatively, which is perfect for running statistics. Alternatively, use a loop with np.maximum, but accumulate is vectorized and faster. For running minimum, use np.minimum.accumulate. This is useful for tracking peaks, computing cumulative maximums in time series, or maintaining running statistics efficiently.

## Q17: How do you create a Vandermonde matrix?

**A17:** Use np.vander: vandermonde = np.vander(x, N=None, increasing=False). This creates a matrix where each column is x raised to increasing powers. With N=None, it uses len(x) columns. With increasing=True, powers go 0, 1, 2, ...; with increasing=False (default), powers go n-1, n-2, ..., 0. Alternatively, create manually using broadcasting: powers = np.arange(n)[:, None] if increasing else np.arange(n-1, -1, -1)[:, None], vandermonde = x ** powers.T. The Vandermonde matrix is useful for polynomial interpolation, regression, and solving systems involving polynomial bases. np.vander is optimized and handles edge cases efficiently.

## Q18: How do you compute weighted average along an axis?

**A18:** Multiply array by weights, sum along axis, then divide by sum of weights: weighted_avg = np.average(arr, axis=axis, weights=weights). Alternatively, implement manually: weighted_sum = np.sum(arr * weights, axis=axis), weight_sum = np.sum(weights, axis=axis), weighted_avg = weighted_sum / weight_sum. For broadcasting weights across dimensions: weighted_avg = np.sum(arr * weights[:, None], axis=0) / np.sum(weights) if weights is 1D and arr is 2D. np.average handles broadcasting automatically and is the recommended approach. Ensure weights have compatible shape for broadcasting, or reshape them appropriately using np.newaxis or reshape.

## Q19: How do you efficiently compute outer product of multiple vectors?

**A19:** For two vectors, use np.outer: result = np.outer(a, b). For multiple vectors, use reduce with np.multiply.outer or compute iteratively: result = vectors[0][:, None], for v in vectors[1:]: result = np.multiply.outer(result, v).reshape(result.shape + (len(v),)). Alternatively, use np.einsum: result = np.einsum('i,j,k->ijk', v1, v2, v3) for three vectors. For many vectors, the iterative approach with reshape maintains efficiency. The outer product creates a tensor where each element is the product of corresponding elements from input vectors. np.einsum provides a concise notation for complex tensor operations and can be optimized by NumPy.

## Q20: How do you apply different functions to different columns?

**A20:** Apply functions column-wise using a loop or list comprehension: results = np.array([func(arr[:, i]) for i, func in enumerate(functions)]). For in-place modification: for i, func in enumerate(functions): arr[:, i] = func(arr[:, i]). Alternatively, use np.apply_along_axis with a wrapper function that selects the appropriate function based on column index: def apply_func_1d(arr_1d, funcs): return np.array([funcs[i](arr_1d[i]) for i in range(len(arr_1d))]), result = np.apply_along_axis(lambda col: apply_func_1d(col, functions), axis=0, arr=arr). For better performance with many columns, consider vectorizing where possible or using specialized functions that operate on the entire array with column-specific parameters.

---
