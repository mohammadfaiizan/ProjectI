# NumPy Core Concepts Interview Questions

## Q1: What is NumPy and why use it over Python lists?

**A1:** NumPy (Numerical Python) is a fundamental library for numerical computing in Python that provides a high-performance multidimensional array object called ndarray. Unlike Python lists, NumPy arrays are homogeneous (all elements must be the same type), stored in contiguous memory blocks, and optimized for mathematical operations. NumPy offers significant performance advantages through vectorized operations, which leverage optimized C code and SIMD instructions. Python lists are flexible but slow for numerical work because each element is a Python object with overhead, while NumPy arrays store raw data efficiently and perform operations in bulk without Python interpreter overhead.

## Q2: What is an ndarray in NumPy?

**A2:** An ndarray (n-dimensional array) is NumPy's core data structure representing a multidimensional, homogeneous array of fixed-size items. It consists of a pointer to a contiguous block of memory, data type information (dtype), shape tuple describing dimensions, and strides tuple indicating how many bytes to step in each dimension. The ndarray enables efficient element access and mathematical operations through direct memory manipulation. Unlike Python lists, ndarrays have a fixed size at creation and all elements share the same data type, allowing for optimized storage and computation.

## Q3: What are the key differences between ndarray and Python list in terms of memory and speed?

**A3:** NumPy ndarrays store data in a contiguous block of memory with a single data type, while Python lists store pointers to Python objects scattered throughout memory. This contiguous storage enables better cache locality and vectorized operations using SIMD instructions. NumPy operations are implemented in C/Fortran and avoid Python's interpreter overhead, making them 10-100x faster for numerical computations. Python lists have overhead for each element (type checking, reference counting), while NumPy arrays have minimal overhead per element. Memory-wise, a NumPy array of integers uses exactly 4 bytes per integer (for int32), while a Python list uses much more due to object overhead.

## Q4: How does ndarray store data in memory?

**A4:** NumPy ndarrays store data in a contiguous block of memory as a flat buffer of raw bytes. The array object contains a pointer to this memory block, along with metadata including dtype (data type), shape (dimensions), strides (bytes to step in each dimension), and flags (memory layout information). For a 2D array, elements can be stored in row-major (C-order) or column-major (Fortran-order) format. The contiguous storage allows efficient access patterns and enables vectorized operations that can process multiple elements simultaneously using CPU SIMD instructions. This memory layout is fundamentally different from Python lists, which store pointers to separate objects.

## Q5: What are axes in NumPy and how do axis=0, axis=1, and axis=2 differ?

**A5:** Axes in NumPy represent the dimensions along which operations are performed. For a 2D array, axis=0 refers to rows (vertical direction, first dimension) and axis=1 refers to columns (horizontal direction, second dimension). For a 3D array, axis=0 is depth, axis=1 is rows, and axis=2 is columns. When you use np.sum(arr, axis=0) on a 2D array, you sum along rows (collapsing rows), resulting in a 1D array with one value per column. When using axis=1, you sum along columns (collapsing columns), resulting in one value per row. Understanding axes is crucial for aggregation functions, as the axis parameter determines which dimension gets reduced or along which dimension an operation is applied.

## Q6: What are shape, ndim, and size attributes in NumPy?

**A6:** The shape attribute returns a tuple representing the size of each dimension (e.g., (3, 4) for a 3x4 array). The ndim attribute returns the number of dimensions (rank) of the array as an integer. The size attribute returns the total number of elements in the array, which equals the product of all dimensions. For example, an array with shape (2, 3, 4) has ndim=3 and size=24. These attributes help understand array structure: shape tells you the dimensions, ndim tells you how many dimensions exist, and size tells you the total element count. They are fundamental for array manipulation and understanding how operations affect array dimensions.

## Q7: What is dtype and why does it matter?

**A7:** dtype (data type) specifies the type of data stored in a NumPy array, such as int32, float64, bool, or complex128. It matters because it determines memory usage, precision, and performance. Smaller dtypes (int8 vs int64) use less memory, which is crucial for large arrays. Float32 uses half the memory of float64 but has lower precision. The dtype also affects computation speed, as some operations are optimized for specific types. NumPy arrays are homogeneous, meaning all elements share the same dtype, enabling efficient vectorized operations. Choosing the appropriate dtype balances memory efficiency, precision requirements, and computational performance for your specific use case.

## Q8: What are the major dtypes available in NumPy?

**A8:** NumPy provides several categories of dtypes: integer types (int8, int16, int32, int64, uint8, uint16, uint32, uint64), floating-point types (float16, float32, float64), complex types (complex64, complex128), boolean (bool), string types (str, bytes), and object type for arbitrary Python objects. Integer types vary by size and signedness (signed vs unsigned). Float types differ in precision (half, single, double precision). Complex types store pairs of floats. The default integer type is typically int64 on 64-bit systems, and default float is float64. Specialized dtypes like datetime64 and timedelta64 handle temporal data. Structured dtypes allow arrays with multiple named fields of different types.

## Q9: What is type casting and how does astype work with safe casting?

**A9:** Type casting converts an array from one dtype to another. The astype method creates a new array with the specified dtype, copying and converting data if necessary. Safe casting means the conversion doesn't lose information: for example, int32 to int64 is safe (widening), but float64 to int32 may lose precision (unsafe). NumPy's casting rules define which conversions are safe: same-kind casts (int32 to int64) are safe, while cross-kind casts (float to int) may lose information. The astype method always creates a copy, even if the dtype is the same. For safe conversions, you can use the casting parameter with options like 'safe', 'same_kind', 'unsafe', or 'no' to control what conversions are allowed.

## Q10: What is the difference between views and copies in NumPy?

**A10:** A view is a new array object that shares the same data buffer as the original array, so modifying a view modifies the original. A copy is a completely independent array with its own data buffer, so changes don't affect the original. Views are created by slicing, reshaping (when possible), transposing, and changing dtype (in some cases). Copies are created by explicit copy() calls, fancy indexing, and operations that require new memory layout. Views are memory-efficient and fast, while copies use more memory but provide data independence. Understanding when each is created is crucial for avoiding unintended side effects and optimizing memory usage.

## Q11: How can you check if an array is a view or a copy?

**A11:** You can check if an array shares memory with another using np.shares_memory(arr1, arr2), which returns True if they share memory (view relationship). The base attribute of an array points to the original array if it's a view, or is None if it's the original or a copy. You can also use the OWNDATA flag in the flags attribute: arr.flags.owndata is False for views and True for copies/originals. Another method is to modify one array and check if the other changes, though this is a runtime test. The most reliable approach is np.shares_memory(), which directly checks memory sharing at the C level.

## Q12: What is the .base attribute in NumPy arrays?

**A12:** The base attribute of a NumPy array is a reference to the array from which the current array was derived, if it's a view. If an array is a view created by slicing or reshaping, base points to the original array. If the array owns its data (is the original or a copy), base is None. This attribute helps trace the lineage of arrays and understand memory relationships. For example, if you slice an array, the resulting view's base attribute will reference the original array. This is useful for debugging memory issues and understanding when operations create views versus copies.

## Q13: What are strides and how do they work?

**A13:** Strides are a tuple indicating how many bytes to step in memory to move to the next element along each axis. For a 2D array with shape (3, 4) and dtype int32 (4 bytes), strides might be (16, 4), meaning moving one row forward steps 16 bytes (4 elements × 4 bytes) and moving one column forward steps 4 bytes. Strides enable efficient array manipulation without copying data: operations like transpose, reshape, and slicing can often just modify strides rather than rearranging memory. Non-contiguous arrays have non-standard strides, which can impact performance. Understanding strides helps optimize memory access patterns and understand how NumPy achieves zero-copy operations.

## Q14: What is the difference between C-order and Fortran-order?

**A14:** C-order (row-major) stores multidimensional arrays with the rightmost index varying fastest, meaning consecutive elements in memory are in the same row. Fortran-order (column-major) stores arrays with the leftmost index varying fastest, meaning consecutive elements are in the same column. NumPy uses C-order by default. For a 2D array, C-order means row-by-row storage, while Fortran-order means column-by-column storage. The order affects memory access patterns and performance: C-order is faster for row-wise operations, while Fortran-order is faster for column-wise operations. You can specify order when creating arrays or use np.ascontiguousarray() and np.asfortranarray() to convert between orders.

## Q15: What are contiguity flags in NumPy?

**A15:** Contiguity flags indicate whether an array's data is stored contiguously in memory. C_CONTIGUOUS means the array is stored in C-order (row-major) with no gaps between elements. F_CONTIGUOUS means it's stored in Fortran-order (column-major) contiguously. An array can be both, neither, or one of them. Contiguous arrays enable optimal performance for vectorized operations and memory access. Non-contiguous arrays (like transposed views) may have performance penalties. You can check contiguity using arr.flags.c_contiguous or arr.flags.f_contiguous, and force contiguity using np.ascontiguousarray() or np.asfortranarray(), which create copies if necessary.

## Q16: What is itemsize and nbytes in NumPy?

**A16:** itemsize is the size in bytes of a single element in the array, determined by the dtype (e.g., int32 has itemsize=4, float64 has itemsize=8). nbytes is the total memory consumed by the array's data buffer, calculated as size × itemsize. For example, a (10, 10) array of float64 has itemsize=8 and nbytes=800. Note that nbytes doesn't include the overhead of the array object itself, just the raw data buffer. These attributes help estimate memory usage and are useful for optimizing large-scale computations. Understanding itemsize helps choose appropriate dtypes to minimize memory footprint.

## Q17: What are structured dtypes in NumPy?

**A17:** Structured dtypes allow creating arrays with multiple named fields, each with its own dtype, similar to a C struct or database record. You define them using a list of tuples specifying (field_name, dtype) or using dictionaries. For example, you can create an array where each element has a name (string), age (int), and salary (float). Structured arrays enable efficient storage of heterogeneous data while maintaining NumPy's performance benefits. You access fields using dot notation (arr['name']) or by field name indexing. They're useful for representing records, time series with multiple variables, or any data with mixed types that benefits from vectorized operations.

## Q18: When does reshape return a view versus a copy?

**A18:** Reshape returns a view when the new shape is compatible with the array's memory layout and no data rearrangement is needed. This happens when the array is contiguous and the reshape maintains the same memory order. Reshape returns a copy when the array is non-contiguous or when the reshape requires a different memory layout that can't be achieved by just changing the shape and strides. For example, reshaping a C-contiguous array to a different shape that's still C-contiguous typically returns a view. However, if you try to reshape a Fortran-ordered array to C-order layout, it may create a copy. The key factor is whether the underlying memory can be reinterpreted with new dimensions without moving data.

## Q19: How do you use np.shares_memory to check memory sharing?

**A19:** np.shares_memory(arr1, arr2) returns True if the two arrays share any memory, meaning they reference overlapping data buffers. This function checks at the C level whether the memory addresses overlap, providing a reliable way to determine if an array is a view of another. It's more accurate than checking base attributes because it directly examines memory addresses. You can use it to verify that slicing creates views, that copies are independent, or to debug unexpected side effects from array operations. The function returns False only when arrays have completely separate memory buffers, making it the definitive test for memory sharing.

## Q20: What is np.newaxis and how is it used?

**A20:** np.newaxis is an alias for None that adds a new dimension of size 1 to an array. It's used for broadcasting and reshaping operations. For example, arr[:, np.newaxis] converts a 1D array of shape (n,) to shape (n, 1), and arr[np.newaxis, :] converts it to shape (1, n). This is useful for broadcasting: a (3,) array can't broadcast with a (3, 4) array, but a (3, 1) array can. np.newaxis makes array dimensions compatible for operations without copying data, as it only changes the shape and strides. It's commonly used in machine learning for adding batch dimensions or making arrays compatible for matrix operations.

---
