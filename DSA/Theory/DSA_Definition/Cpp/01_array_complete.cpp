/*
 * =============================================================================
 * COMPLETE ARRAY & VECTOR GUIDE - All Methods & Techniques
 * =============================================================================
 * 
 * This file covers:
 * 1. Different ways to initialize arrays
 * 2. All vector operations
 * 3. Array manipulation techniques
 * 4. Common patterns and methods
 * 5. Memory management approaches
 * 
 * =============================================================================
 */

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <iterator>
using namespace std;

void demonstrate_array_initialization() {
    cout << "\n=== ARRAY INITIALIZATION METHODS ===" << endl;
    
    // 1. Static Array Initialization
    cout << "\n--- Static Arrays ---" << endl;
    
    // Method 1: Initialize with values
    int arr1[5] = {1, 2, 3, 4, 5};
    cout << "Method 1 - With values: ";
    for (int i = 0; i < 5; i++) cout << arr1[i] << " ";
    cout << endl;
    
    // Method 2: Partial initialization (rest become 0)
    int arr2[5] = {1, 2};
    cout << "Method 2 - Partial init: ";
    for (int i = 0; i < 5; i++) cout << arr2[i] << " ";
    cout << endl;
    
    // Method 3: All zeros
    int arr3[5] = {0};
    cout << "Method 3 - All zeros: ";
    for (int i = 0; i < 5; i++) cout << arr3[i] << " ";
    cout << endl;
    
    // Method 4: All same value (using fill)
    int arr4[5];
    fill(arr4, arr4 + 5, 7);
    cout << "Method 4 - All 7s: ";
    for (int i = 0; i < 5; i++) cout << arr4[i] << " ";
    cout << endl;
    
    // Method 5: Character array
    char str1[] = "Hello";
    char str2[10] = "World";
    cout << "Method 5 - Char arrays: " << str1 << " " << str2 << endl;
    
    // Method 6: std::array (C++11)
    array<int, 5> arr5 = {10, 20, 30, 40, 50};
    cout << "Method 6 - std::array: ";
    for (int x : arr5) cout << x << " ";
    cout << endl;
}

void demonstrate_vector_operations() {
    cout << "\n=== VECTOR OPERATIONS ===" << endl;
    
    // 1. Vector Initialization Methods
    cout << "\n--- Vector Initialization ---" << endl;
    
    // Method 1: Empty vector
    vector<int> v1;
    cout << "Empty vector size: " << v1.size() << endl;
    
    // Method 2: With size
    vector<int> v2(5);
    cout << "Vector with size 5: ";
    for (int x : v2) cout << x << " ";
    cout << endl;
    
    // Method 3: With size and value
    vector<int> v3(5, 10);
    cout << "Vector with 5 elements of value 10: ";
    for (int x : v3) cout << x << " ";
    cout << endl;
    
    // Method 4: With initializer list
    vector<int> v4 = {1, 2, 3, 4, 5};
    cout << "Vector with initializer list: ";
    for (int x : v4) cout << x << " ";
    cout << endl;
    
    // Method 5: Copy constructor
    vector<int> v5(v4);
    cout << "Copy of v4: ";
    for (int x : v5) cout << x << " ";
    cout << endl;
    
    // Method 6: From array
    int arr[] = {10, 20, 30, 40, 50};
    vector<int> v6(arr, arr + 5);
    cout << "Vector from array: ";
    for (int x : v6) cout << x << " ";
    cout << endl;
    
    // Method 7: From another vector (range)
    vector<int> v7(v4.begin() + 1, v4.end() - 1);
    cout << "Vector from range: ";
    for (int x : v7) cout << x << " ";
    cout << endl;
}

void demonstrate_vector_modification() {
    cout << "\n=== VECTOR MODIFICATION OPERATIONS ===" << endl;
    
    vector<int> v = {1, 2, 3};
    cout << "Initial vector: ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // Adding elements
    cout << "\n--- Adding Elements ---" << endl;
    
    // push_back - add at end
    v.push_back(4);
    cout << "After push_back(4): ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // insert - add at specific position
    v.insert(v.begin() + 1, 99);
    cout << "After insert(1, 99): ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // insert multiple
    v.insert(v.end(), 2, 100);
    cout << "After insert 2x100 at end: ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // emplace_back - construct in place
    v.emplace_back(200);
    cout << "After emplace_back(200): ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // Removing elements
    cout << "\n--- Removing Elements ---" << endl;
    
    // pop_back - remove last
    v.pop_back();
    cout << "After pop_back(): ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // erase - remove at position
    v.erase(v.begin() + 1);
    cout << "After erase(1): ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // erase range
    v.erase(v.end() - 2, v.end());
    cout << "After erase last 2: ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // clear - remove all
    vector<int> temp = v;
    v.clear();
    cout << "After clear(), size: " << v.size() << endl;
    v = temp; // restore
}

void demonstrate_vector_access() {
    cout << "\n=== VECTOR ACCESS METHODS ===" << endl;
    
    vector<int> v = {10, 20, 30, 40, 50};
    cout << "Vector: ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // Different access methods
    cout << "\n--- Access Methods ---" << endl;
    cout << "v[2] = " << v[2] << endl;           // No bounds checking
    cout << "v.at(2) = " << v.at(2) << endl;     // With bounds checking
    cout << "v.front() = " << v.front() << endl; // First element
    cout << "v.back() = " << v.back() << endl;   // Last element
    
    // Using iterators
    cout << "\n--- Iterator Access ---" << endl;
    cout << "*(v.begin()) = " << *(v.begin()) << endl;
    cout << "*(v.end()-1) = " << *(v.end()-1) << endl;
    
    // Using pointers
    cout << "\n--- Pointer Access ---" << endl;
    cout << "v.data()[0] = " << v.data()[0] << endl;
    
    // Range-based for loop variations
    cout << "\n--- Different Loop Methods ---" << endl;
    
    // Method 1: Range-based for
    cout << "Range-based for: ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // Method 2: Traditional for
    cout << "Traditional for: ";
    for (size_t i = 0; i < v.size(); i++) cout << v[i] << " ";
    cout << endl;
    
    // Method 3: Iterator for
    cout << "Iterator for: ";
    for (auto it = v.begin(); it != v.end(); it++) cout << *it << " ";
    cout << endl;
    
    // Method 4: Reverse iterator
    cout << "Reverse iterator: ";
    for (auto it = v.rbegin(); it != v.rend(); it++) cout << *it << " ";
    cout << endl;
}

void demonstrate_vector_algorithms() {
    cout << "\n=== VECTOR ALGORITHMS ===" << endl;
    
    vector<int> v = {5, 2, 8, 1, 9, 3};
    cout << "Original vector: ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // Sorting
    cout << "\n--- Sorting ---" << endl;
    sort(v.begin(), v.end());
    cout << "After sort(): ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // Reverse
    reverse(v.begin(), v.end());
    cout << "After reverse(): ";
    for (int x : v) cout << x << " ";
    cout << endl;
    
    // Searching
    cout << "\n--- Searching ---" << endl;
    sort(v.begin(), v.end()); // Need sorted for binary_search
    
    auto it = find(v.begin(), v.end(), 5);
    cout << "find(5): " << (it != v.end() ? "Found" : "Not found") << endl;
    
    bool found = binary_search(v.begin(), v.end(), 5);
    cout << "binary_search(5): " << (found ? "Found" : "Not found") << endl;
    
    // Min/Max
    cout << "\n--- Min/Max ---" << endl;
    auto min_it = min_element(v.begin(), v.end());
    auto max_it = max_element(v.begin(), v.end());
    cout << "min_element: " << *min_it << endl;
    cout << "max_element: " << *max_it << endl;
    
    // Accumulate (sum)
    int sum = accumulate(v.begin(), v.end(), 0);
    cout << "sum of elements: " << sum << endl;
    
    // Count
    v.push_back(5);
    v.push_back(5);
    int count_5 = count(v.begin(), v.end(), 5);
    cout << "count of 5: " << count_5 << endl;
}

void demonstrate_vector_capacity() {
    cout << "\n=== VECTOR CAPACITY OPERATIONS ===" << endl;
    
    vector<int> v;
    cout << "Initial - Size: " << v.size() << ", Capacity: " << v.capacity() << endl;
    
    // Reserve capacity
    v.reserve(100);
    cout << "After reserve(100) - Size: " << v.size() << ", Capacity: " << v.capacity() << endl;
    
    // Add elements
    for (int i = 0; i < 10; i++) {
        v.push_back(i);
    }
    cout << "After adding 10 elements - Size: " << v.size() << ", Capacity: " << v.capacity() << endl;
    
    // Resize
    v.resize(5);
    cout << "After resize(5) - Size: " << v.size() << ", Capacity: " << v.capacity() << endl;
    
    // Shrink to fit
    v.shrink_to_fit();
    cout << "After shrink_to_fit() - Size: " << v.size() << ", Capacity: " << v.capacity() << endl;
    
    // Check if empty
    cout << "Is empty? " << (v.empty() ? "Yes" : "No") << endl;
    
    // Max size
    cout << "Max possible size: " << v.max_size() << endl;
}

void demonstrate_2d_arrays() {
    cout << "\n=== 2D ARRAYS & VECTORS ===" << endl;
    
    // 2D Static Array
    cout << "\n--- 2D Static Array ---" << endl;
    int arr2d[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    
    cout << "2D Array:" << endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            cout << arr2d[i][j] << " ";
        }
        cout << endl;
    }
    
    // 2D Vector
    cout << "\n--- 2D Vector ---" << endl;
    
    // Method 1: Direct initialization
    vector<vector<int>> v2d = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    
    cout << "2D Vector (Method 1):" << endl;
    for (const auto& row : v2d) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    // Method 2: With size
    vector<vector<int>> v2d2(3, vector<int>(4, 0));
    cout << "2D Vector (Method 2) - 3x4 with zeros:" << endl;
    for (const auto& row : v2d2) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    // Method 3: Dynamic sizing
    vector<vector<int>> v2d3;
    v2d3.resize(2);
    v2d3[0] = {1, 2, 3, 4, 5};
    v2d3[1] = {6, 7, 8};
    
    cout << "2D Vector (Method 3) - Dynamic:" << endl;
    for (const auto& row : v2d3) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

void demonstrate_array_patterns() {
    cout << "\n=== COMMON ARRAY PATTERNS ===" << endl;
    
    vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Pattern 1: Two pointers
    cout << "\n--- Two Pointers Pattern ---" << endl;
    cout << "Finding pairs that sum to 11:" << endl;
    int left = 0, right = v.size() - 1;
    while (left < right) {
        int sum = v[left] + v[right];
        if (sum == 11) {
            cout << "Pair found: " << v[left] << " + " << v[right] << " = 11" << endl;
            left++;
            right--;
        } else if (sum < 11) {
            left++;
        } else {
            right--;
        }
    }
    
    // Pattern 2: Sliding window
    cout << "\n--- Sliding Window Pattern ---" << endl;
    cout << "Max sum of 3 consecutive elements:" << endl;
    int window_size = 3;
    int max_sum = 0;
    int window_sum = 0;
    
    // Calculate first window
    for (int i = 0; i < window_size; i++) {
        window_sum += v[i];
    }
    max_sum = window_sum;
    
    // Slide the window
    for (size_t i = window_size; i < v.size(); i++) {
        window_sum = window_sum - v[i - window_size] + v[i];
        max_sum = max(max_sum, window_sum);
    }
    cout << "Max sum: " << max_sum << endl;
    
    // Pattern 3: Prefix sum
    cout << "\n--- Prefix Sum Pattern ---" << endl;
    vector<int> prefix(v.size());
    prefix[0] = v[0];
    for (size_t i = 1; i < v.size(); i++) {
        prefix[i] = prefix[i-1] + v[i];
    }
    
    cout << "Original: ";
    for (int x : v) cout << x << " ";
    cout << endl;
    cout << "Prefix sum: ";
    for (int x : prefix) cout << x << " ";
    cout << endl;
    
    // Range sum using prefix
    int left_idx = 2, right_idx = 5;
    int range_sum = prefix[right_idx] - (left_idx > 0 ? prefix[left_idx-1] : 0);
    cout << "Sum from index " << left_idx << " to " << right_idx << ": " << range_sum << endl;
}

int main() {
    cout << "=== COMPLETE ARRAY & VECTOR GUIDE ===" << endl;
    
    demonstrate_array_initialization();
    demonstrate_vector_operations();
    demonstrate_vector_modification();
    demonstrate_vector_access();
    demonstrate_vector_algorithms();
    demonstrate_vector_capacity();
    demonstrate_2d_arrays();
    demonstrate_array_patterns();
    
    return 0;
} 