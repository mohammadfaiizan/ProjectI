# NumPy Operations And Functions Interview Questions

## Q1: What is vectorization and why is it fast?

**A1:** Vectorization is performing operations on entire arrays at once rather than iterating over elements in Python loops. NumPy implements vectorized operations in optimized C/Fortran code that processes multiple elements simultaneously using SIMD (Single Instruction Multiple Data) CPU instructions. This eliminates Python interpreter overhead, reduces function call overhead, and enables better cache locality by processing contiguous memory blocks. Vectorized operations can be 10-100x faster than Python loops because they leverage low-level optimizations, parallel processing capabilities, and avoid the overhead of Python's dynamic typing and object system. The key is that operations are applied element-wise to entire arrays without explicit loops.

## Q2: What are ufuncs in NumPy?

**A2:** Universal functions (ufuncs) are NumPy functions that operate element-wise on arrays, supporting broadcasting and type coercion. Examples include np.add, np.multiply, np.sin, np.sqrt, and all mathematical operations. Ufuncs are implemented in C for performance and automatically handle broadcasting, allowing operations between arrays of different shapes. They can operate on scalars, arrays, or combinations, and support various methods like reduce, accumulate, outer, and at. Ufuncs are the foundation of NumPy's vectorization capabilities, enabling fast mathematical operations across entire arrays without Python loops. They also support multiple output arrays and can be extended with custom ufuncs.

## Q3: What is the difference between np.add and the + operator?

**A3:** For basic element-wise addition, np.add and the + operator are functionally equivalent and both use the same underlying ufunc. However, np.add offers additional functionality: it accepts an optional out parameter to specify where results are stored, supports multiple output arrays, and can be used with ufunc methods like reduce and accumulate. The + operator is more concise and Pythonic for simple operations, while np.add provides more control for advanced use cases like in-place operations or custom output arrays. Both are vectorized and equally fast, but np.add gives you more flexibility when you need fine-grained control over the operation.

## Q4: What are ufunc methods: reduce, accumulate, outer, and at?

**A4:** Ufunc methods extend ufunc functionality: reduce applies the ufunc along an axis, reducing dimensions (e.g., np.add.reduce sums elements). accumulate applies the ufunc cumulatively, returning intermediate results (e.g., np.add.accumulate gives cumulative sums). outer applies the ufunc to all pairs of elements from two arrays, creating an outer product-like result. The at method performs unbuffered in-place operations at specified indices, useful when you need to handle repeated indices correctly (e.g., np.add.at(arr, indices, values) adds values at indices, handling duplicates properly). These methods provide powerful ways to apply ufuncs beyond simple element-wise operations, enabling efficient reductions, cumulative operations, and advanced indexing scenarios.

## Q5: What is broadcasting and what are its rules?

**A5:** Broadcasting allows NumPy to perform operations on arrays of different shapes by automatically expanding smaller arrays to match larger ones. The rules are: align dimensions from the right, compare sizes dimension by dimension, and allow broadcasting if sizes are equal, one is 1, or one dimension is missing. Arrays are compatible for broadcasting if, for each dimension, the sizes are equal or one is 1. The result shape has the maximum size in each dimension. For example, a (3, 1) array can broadcast with a (1, 4) array to produce a (3, 4) result. Broadcasting eliminates the need for explicit loops or tiling, making code concise and efficient while enabling operations between arrays of different shapes.

## Q6: Can you provide examples of broadcasting shape compatibility?

**A6:** Compatible shapes include: (5, 3) with (3,) broadcasts to (5, 3) by repeating the 1D array; (5, 1) with (1, 3) broadcasts to (5, 3) by repeating both; (4, 1, 3) with (1, 5, 3) broadcasts to (4, 5, 3). Incompatible shapes include: (5, 3) with (4,) fails because 3 ≠ 4; (5, 3) with (5,) fails because trailing dimensions don't align properly. The key is that dimensions align from the right, missing dimensions are treated as 1, and dimensions of size 1 can stretch to match. Broadcasting enables operations like adding a row vector to every row of a matrix or multiplying arrays with different but compatible shapes without explicit loops.

## Q7: How does the axis parameter work in aggregation functions?

**A7:** The axis parameter specifies which dimension to reduce or along which to perform the aggregation. For a 2D array with shape (m, n), axis=0 means operate along rows (first dimension), collapsing it and producing shape (n,). axis=1 means operate along columns (second dimension), producing shape (m,). axis=None means flatten the array and operate on all elements, producing a scalar. For higher dimensions, axis numbers correspond to dimension indices. When you specify an axis, the function reduces that dimension, keeping all other dimensions intact. Understanding axis is crucial for correctly applying aggregations like sum, mean, max, or std to get the desired output shape.

## Q8: What is the purpose of the keepdims parameter?

**A8:** The keepdims parameter preserves the reduced dimension as a size-1 dimension in the result, rather than removing it entirely. When keepdims=True, an aggregation along axis=0 on a (3, 4) array produces shape (1, 4) instead of (4,). This is useful for maintaining broadcasting compatibility: keeping dimensions allows the result to broadcast correctly with other arrays that have matching dimensions. For example, subtracting the mean along an axis while keeping dimensions enables proper broadcasting for normalization operations. Without keepdims, you'd need to manually reshape or use np.newaxis to restore dimensions for subsequent operations.

## Q9: What is the difference between np.sum and ndarray.sum?

**A9:** Functionally, np.sum(arr) and arr.sum() produce identical results for most cases. However, np.sum is a standalone function that can accept any array-like object, while arr.sum() is a method bound to the array instance. The method form is slightly more concise, while the function form is more consistent with NumPy's functional programming style. Both support the same parameters (axis, dtype, keepdims, etc.). The choice is largely stylistic, though some prefer np.sum for consistency when mixing with other NumPy functions, while others prefer the method form for its object-oriented feel. Performance is identical as both use the same underlying implementation.

## Q10: How do you use np.where for conditional operations?

**A10:** np.where(condition, x, y) returns elements from x where condition is True, and elements from y where condition is False. It's a vectorized version of if-else that operates on entire arrays. With two arguments, np.where(condition) returns indices where condition is True, useful for finding positions meeting criteria. The three-argument form enables element-wise conditional selection: np.where(arr > 0, arr, 0) replaces negative values with 0. It's more efficient than Python loops and supports broadcasting, allowing conditional operations across arrays of compatible shapes. np.where is essential for vectorized conditional logic in NumPy.

## Q11: What is the difference between sort and argsort?

**A11:** sort modifies an array in-place (or returns a sorted copy) by rearranging elements into sorted order. argsort returns an array of indices that would sort the original array, without modifying it. For example, if arr = [3, 1, 2], np.sort(arr) returns [1, 2, 3], while np.argsort(arr) returns [1, 2, 0], indicating that element at index 1 is smallest, index 2 is next, and index 0 is largest. argsort is useful when you need to sort one array based on another or when you need the original indices after sorting. Both support the axis parameter for sorting along specific dimensions in multidimensional arrays.

## Q12: What is the difference between stable and unstable sorting?

**A12:** Stable sorting preserves the relative order of elements that compare equal, while unstable sorting may rearrange equal elements arbitrarily. NumPy's default sort is unstable (quicksort), meaning equal elements might appear in different orders after sorting. Stable sorting (mergesort or heapsort) guarantees that if two elements are equal, their original relative order is maintained. Stability matters when sorting by multiple criteria: first sort by secondary key with stable sort, then by primary key. NumPy's sort function accepts a kind parameter ('quicksort', 'mergesort', 'heapsort') where 'mergesort' is stable. Use stable sorting when preserving order of equal elements is important for your algorithm.

## Q13: What is the difference between fancy indexing and basic indexing regarding views vs copies?

**A13:** Basic indexing (using slices, integers, or ellipsis) typically returns views that share memory with the original array. Fancy indexing (using integer or boolean arrays) always returns copies, never views. For example, arr[1:3] is basic indexing and returns a view, while arr[[1, 3]] is fancy indexing and returns a copy. This is because fancy indexing can select non-contiguous elements in arbitrary order, which can't be represented as a simple view with modified strides. Boolean indexing (arr[mask]) is also fancy indexing and creates copies. Understanding this distinction is crucial for memory efficiency and avoiding unintended modifications when you expect a view but get a copy.

## Q14: How do you use boolean indexing in NumPy?

**A14:** Boolean indexing uses boolean arrays to select elements where the condition is True. You create a boolean mask (array of True/False) using comparison operations like arr > 5, then use it to index: arr[mask] returns elements where mask is True. Boolean indexing works for both reading and assignment: arr[arr > 5] = 0 sets all elements greater than 5 to zero. You can combine conditions using & (and), | (or), and ~ (not), with parentheses for precedence. Boolean indexing is vectorized and efficient, enabling conditional selection and modification without loops. It always returns a copy, not a view, because selected elements may be non-contiguous.

## Q15: How do you use np.unique with return_counts?

**A15:** np.unique returns sorted unique elements of an array. With return_counts=True, it also returns the count of each unique value. For example, np.unique(arr, return_counts=True) returns a tuple of (unique_values, counts) where counts[i] is how many times unique_values[i] appears. This is useful for frequency analysis, finding the most common values, or understanding data distribution. You can combine it with return_index=True to also get indices of first occurrences, or return_inverse=True to get an array mapping original elements to unique value indices. return_counts is particularly valuable for categorical data analysis and histogram-like operations.

## Q16: What is the difference between flatten and ravel?

**A16:** Both flatten and ravel convert multidimensional arrays to 1D, but flatten always returns a copy while ravel returns a view when possible and a copy only when necessary. ravel is more memory-efficient as it avoids copying when the array is already contiguous and can be reshaped to 1D by just changing shape and strides. flatten guarantees a copy, which is safer if you want to modify the result without affecting the original, but uses more memory. In practice, ravel is often preferred for its efficiency, while flatten is used when you explicitly need a copy. Both produce identical 1D arrays, but their memory behavior differs.

## Q17: What is the difference between np.concatenate and np.stack?

**A17:** np.concatenate joins arrays along an existing axis, requiring that all arrays have the same shape except in the concatenation dimension. For example, concatenating (3, 4) arrays along axis=0 produces (6, 4), or along axis=1 produces (3, 8). np.stack creates a new axis and stacks arrays along it, requiring all arrays to have identical shapes. Stacking (3, 4) arrays produces (n, 3, 4) where n is the number of arrays. concatenate increases size in an existing dimension, while stack adds a new dimension. Use concatenate to combine arrays side-by-side or top-to-bottom, and use stack when you want to create a new dimension, like combining multiple 2D arrays into a 3D array.

## Q18: How do you use np.searchsorted?

**A18:** np.searchsorted finds insertion indices that would maintain sorted order. Given a sorted array and values to insert, it returns indices where each value should be placed to keep the array sorted. For example, np.searchsorted([1, 3, 5], [2, 4]) returns [1, 2], indicating 2 should go at index 1 and 4 at index 2. It's useful for binning data, finding percentiles, or performing efficient lookups in sorted arrays. The side parameter controls behavior with duplicates: 'left' returns the leftmost insertion point, 'right' returns the rightmost. searchsorted is O(log n) per value, making it efficient for many lookups in sorted arrays.

## Q19: How do you use np.digitize for binning?

**A19:** np.digitize assigns each element of an array to bins defined by bin edges. Given an array of values and bin edges, it returns indices indicating which bin each value belongs to. For example, np.digitize([0.5, 1.5, 2.5], bins=[0, 1, 2, 3]) returns [1, 2, 3], indicating which bin each value falls into. The right parameter controls whether intervals are right-closed (default False, left-closed). Values outside the bin range get index 0 (below) or len(bins) (above). digitize is useful for histogram-like operations, discretizing continuous data, or categorizing values into ranges. It's related to np.histogram but returns bin assignments rather than counts.

## Q20: What are the basics of np.einsum?

**A20:** np.einsum implements Einstein summation convention, providing a concise way to express complex array operations using subscript notation. The syntax is np.einsum('subscripts', arrays) where subscripts describe how dimensions are summed or multiplied. For example, 'ij,jk->ik' represents matrix multiplication, 'ii->i' gets diagonal elements, and 'ij->' sums all elements. Letters represent dimensions, repeated letters in input arrays are summed over (unless in output), and -> separates inputs from output. einsum can express dot products, matrix multiplication, traces, diagonals, and many other operations in a unified notation. It's powerful for complex tensor operations and can be optimized by NumPy for performance, though it may be less readable than explicit functions for simple operations.

---
