/*
 * CONSTEXPR AND CONSTEVAL - Compile-time Computation in Modern C++
 * 
 * Compilation: g++ -std=c++20 -Wall -Wextra -g -O2 constexpr_and_consteval.cpp -o constexpr_and_consteval
 * For C++17: g++ -std=c++17 -Wall -Wextra -g -O2 constexpr_and_consteval.cpp -o constexpr_and_consteval
 */

#include <iostream>
#include <array>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <chrono>
#include <limits>

#define CONSTEXPR_TEST(expr) \
    do { \
        constexpr auto result = (expr); \
        std::cout << "[CONSTEXPR] " << #expr << " = " << result << std::endl; \
    } while(0)

#define RUNTIME_TEST(expr) \
    do { \
        auto result = (expr); \
        std::cout << "[RUNTIME] " << #expr << " = " << result << std::endl; \
    } while(0)

namespace ConstexprConsteval {

// ============================================================================
// 1. BASIC CONSTEXPR FUNCTIONS
// ============================================================================

// Simple constexpr function
constexpr int square(int x) {
    return x * x;
}

// Constexpr function with conditional logic
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

// Constexpr function with loops (C++14+)
constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

// Constexpr function with multiple statements
constexpr int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// ============================================================================
// 2. CONSTEXPR WITH ARRAYS AND CONTAINERS
// ============================================================================

// Constexpr array operations
constexpr int array_sum(const int* arr, size_t size) {
    int sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum;
}

// Constexpr with std::array
constexpr std::array<int, 5> make_sequence() {
    std::array<int, 5> arr{};
    for (size_t i = 0; i < arr.size(); ++i) {
        arr[i] = static_cast<int>(i * i);
    }
    return arr;
}

// Constexpr array search
constexpr bool contains(const std::array<int, 5>& arr, int value) {
    for (const auto& element : arr) {
        if (element == value) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// 3. CONSTEXPR STRING OPERATIONS
// ============================================================================

// Constexpr string length
constexpr size_t string_length(const char* str) {
    size_t len = 0;
    while (str[len] != '\0') {
        ++len;
    }
    return len;
}

// Constexpr string comparison
constexpr bool string_equal(const char* a, const char* b) {
    while (*a && *b && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

// Constexpr string operations with std::string_view (C++17+)
#if __cplusplus >= 201703L
constexpr bool starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() && 
           str.substr(0, prefix.size()) == prefix;
}

constexpr size_t count_char(std::string_view str, char c) {
    size_t count = 0;
    for (char ch : str) {
        if (ch == c) ++count;
    }
    return count;
}
#endif

// ============================================================================
// 4. CONSTEXPR CLASSES AND CONSTRUCTORS
// ============================================================================

class Point {
private:
    double x_, y_;

public:
    constexpr Point(double x, double y) : x_(x), y_(y) {}
    
    constexpr double x() const { return x_; }
    constexpr double y() const { return y_; }
    
    constexpr double distance_from_origin() const {
        return x_ * x_ + y_ * y_; // Simplified distance squared
    }
    
    constexpr Point operator+(const Point& other) const {
        return Point(x_ + other.x_, y_ + other.y_);
    }
    
    constexpr bool operator==(const Point& other) const {
        return x_ == other.x_ && y_ == other.y_;
    }
};

// Constexpr class with more complex operations
class Matrix2x2 {
private:
    double data_[4];

public:
    constexpr Matrix2x2(double a, double b, double c, double d) 
        : data_{a, b, c, d} {}
    
    constexpr double operator()(int row, int col) const {
        return data_[row * 2 + col];
    }
    
    constexpr double determinant() const {
        return data_[0] * data_[3] - data_[1] * data_[2];
    }
    
    constexpr Matrix2x2 operator*(const Matrix2x2& other) const {
        return Matrix2x2(
            data_[0] * other(0, 0) + data_[1] * other(1, 0),
            data_[0] * other(0, 1) + data_[1] * other(1, 1),
            data_[2] * other(0, 0) + data_[3] * other(1, 0),
            data_[2] * other(0, 1) + data_[3] * other(1, 1)
        );
    }
};

// ============================================================================
// 5. CONSTEVAL FUNCTIONS (C++20)
// ============================================================================

#if __cplusplus >= 202002L
// consteval functions must be evaluated at compile time
consteval int compile_time_only(int x) {
    return x * x + 1;
}

// consteval with more complex logic
consteval bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

// consteval function generating compile-time data
consteval std::array<int, 10> generate_primes() {
    std::array<int, 10> primes{};
    int count = 0;
    int candidate = 2;
    
    while (count < 10) {
        if (is_prime(candidate)) {
            primes[count++] = candidate;
        }
        ++candidate;
    }
    return primes;
}

// Mixing constexpr and consteval
constexpr int flexible_function(int x) {
    return x * 2;
}

consteval int strict_function(int x) {
    return flexible_function(x) + 1; // Can call constexpr from consteval
}
#endif

// ============================================================================
// 6. CONSTEXPR IF (C++17)
// ============================================================================

#if __cplusplus >= 201703L
template<typename T>
constexpr auto process_value(T value) {
    if constexpr (std::is_integral_v<T>) {
        return value * 2;
    } else if constexpr (std::is_floating_point_v<T>) {
        return value / 2.0;
    } else {
        return value; // For other types, return as-is
    }
}

template<typename Container>
constexpr auto get_size_if_possible(const Container& container) {
    if constexpr (std::is_same_v<Container, std::string>) {
        return container.size();
    } else {
        return size_t{0}; // Unknown size
    }
}
#endif

// ============================================================================
// 7. CONSTEXPR ALGORITHMS
// ============================================================================

// Constexpr sorting algorithm
constexpr void bubble_sort(int* arr, size_t size) {
    for (size_t i = 0; i < size - 1; ++i) {
        for (size_t j = 0; j < size - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Constexpr binary search
constexpr bool binary_search(const int* arr, size_t size, int target) {
    size_t left = 0, right = size;
    
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return true;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return false;
}

// ============================================================================
// 8. COMPILE-TIME HASH AND UTILITIES
// ============================================================================

// Compile-time string hashing
constexpr uint32_t fnv1a_hash(const char* str, size_t len) {
    constexpr uint32_t FNV_OFFSET_BASIS = 2166136261U;
    constexpr uint32_t FNV_PRIME = 16777619U;
    
    uint32_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint32_t>(str[i]);
        hash *= FNV_PRIME;
    }
    return hash;
}

constexpr uint32_t operator""_hash(const char* str, size_t len) {
    return fnv1a_hash(str, len);
}

// ============================================================================
// 9. PERFORMANCE COMPARISON
// ============================================================================

// Runtime version for comparison
int factorial_runtime(int n) {
    return (n <= 1) ? 1 : n * factorial_runtime(n - 1);
}

void performance_comparison() {
    std::cout << "\n=== PERFORMANCE COMPARISON ===" << std::endl;
    
    constexpr int n = 10;
    
    // Compile-time computation
    auto start = std::chrono::high_resolution_clock::now();
    constexpr int compile_time_result = factorial(n);
    auto end = std::chrono::high_resolution_clock::now();
    auto compile_time_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    std::cout << "Compile-time factorial(" << n << ") = " << compile_time_result << std::endl;
    std::cout << "Compile-time duration: " << compile_time_duration.count() << " ns (mostly measurement overhead)" << std::endl;
    
    // Runtime computation
    start = std::chrono::high_resolution_clock::now();
    int runtime_result = factorial_runtime(n);
    end = std::chrono::high_resolution_clock::now();
    auto runtime_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    std::cout << "Runtime factorial(" << n << ") = " << runtime_result << std::endl;
    std::cout << "Runtime duration: " << runtime_duration.count() << " ns" << std::endl;
}

// ============================================================================
// DEMONSTRATION FUNCTIONS
// ============================================================================

void demonstrate_basic_constexpr() {
    std::cout << "\n=== BASIC CONSTEXPR FUNCTIONS ===" << std::endl;
    
    CONSTEXPR_TEST(square(5));
    CONSTEXPR_TEST(factorial(5));
    CONSTEXPR_TEST(fibonacci(10));
    CONSTEXPR_TEST(gcd(48, 18));
    
    // These can also be used at runtime
    int runtime_input = 7;
    RUNTIME_TEST(square(runtime_input));
    RUNTIME_TEST(factorial(runtime_input));
}

void demonstrate_constexpr_containers() {
    std::cout << "\n=== CONSTEXPR WITH CONTAINERS ===" << std::endl;
    
    constexpr int arr[] = {1, 2, 3, 4, 5};
    CONSTEXPR_TEST(array_sum(arr, 5));
    
    constexpr auto sequence = make_sequence();
    std::cout << "Generated sequence: ";
    for (const auto& val : sequence) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    CONSTEXPR_TEST(contains(sequence, 4));  // 2^2 = 4
    CONSTEXPR_TEST(contains(sequence, 5));  // Not in sequence
}

void demonstrate_constexpr_strings() {
    std::cout << "\n=== CONSTEXPR STRING OPERATIONS ===" << std::endl;
    
    CONSTEXPR_TEST(string_length("Hello World"));
    CONSTEXPR_TEST(string_equal("hello", "hello"));
    CONSTEXPR_TEST(string_equal("hello", "world"));
    
#if __cplusplus >= 201703L
    CONSTEXPR_TEST(starts_with("Hello World", "Hello"));
    CONSTEXPR_TEST(starts_with("Hello World", "World"));
    CONSTEXPR_TEST(count_char("Hello World", 'l'));
#endif
}

void demonstrate_constexpr_classes() {
    std::cout << "\n=== CONSTEXPR CLASSES ===" << std::endl;
    
    constexpr Point p1(3.0, 4.0);
    constexpr Point p2(1.0, 2.0);
    
    CONSTEXPR_TEST(p1.x());
    CONSTEXPR_TEST(p1.y());
    CONSTEXPR_TEST(p1.distance_from_origin());
    
    constexpr Point sum = p1 + p2;
    std::cout << "Sum point: (" << sum.x() << ", " << sum.y() << ")" << std::endl;
    
    constexpr Matrix2x2 m1(1, 2, 3, 4);
    constexpr Matrix2x2 m2(2, 0, 1, 2);
    
    CONSTEXPR_TEST(m1.determinant());
    
    constexpr Matrix2x2 product = m1 * m2;
    std::cout << "Matrix product (0,0): " << product(0, 0) << std::endl;
}

#if __cplusplus >= 202002L
void demonstrate_consteval() {
    std::cout << "\n=== CONSTEVAL FUNCTIONS (C++20) ===" << std::endl;
    
    // These MUST be evaluated at compile time
    constexpr int result1 = compile_time_only(5);
    std::cout << "compile_time_only(5) = " << result1 << std::endl;
    
    constexpr bool prime_check = is_prime(17);
    std::cout << "is_prime(17) = " << prime_check << std::endl;
    
    constexpr auto primes = generate_primes();
    std::cout << "First 10 primes: ";
    for (const auto& prime : primes) {
        std::cout << prime << " ";
    }
    std::cout << std::endl;
    
    constexpr int strict_result = strict_function(10);
    std::cout << "strict_function(10) = " << strict_result << std::endl;
}
#endif

#if __cplusplus >= 201703L
void demonstrate_constexpr_if() {
    std::cout << "\n=== CONSTEXPR IF (C++17) ===" << std::endl;
    
    CONSTEXPR_TEST(process_value(42));      // integral
    CONSTEXPR_TEST(process_value(3.14));    // floating point
    
    std::string str = "hello";
    std::cout << "String size: " << get_size_if_possible(str) << std::endl;
}
#endif

void demonstrate_constexpr_algorithms() {
    std::cout << "\n=== CONSTEXPR ALGORITHMS ===" << std::endl;
    
    // Compile-time sorting
    constexpr auto sorted_array = []() {
        int arr[] = {5, 2, 8, 1, 9, 3};
        bubble_sort(arr, 6);
        return std::array<int, 6>{arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]};
    }();
    
    std::cout << "Sorted array: ";
    for (const auto& val : sorted_array) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Compile-time binary search
    constexpr int sorted_arr[] = {1, 3, 5, 7, 9, 11, 13, 15};
    CONSTEXPR_TEST(binary_search(sorted_arr, 8, 7));
    CONSTEXPR_TEST(binary_search(sorted_arr, 8, 6));
}

void demonstrate_compile_time_utilities() {
    std::cout << "\n=== COMPILE-TIME UTILITIES ===" << std::endl;
    
    constexpr uint32_t hash1 = "hello"_hash;
    constexpr uint32_t hash2 = "world"_hash;
    constexpr uint32_t hash3 = "hello"_hash;
    
    std::cout << "Hash of 'hello': " << hash1 << std::endl;
    std::cout << "Hash of 'world': " << hash2 << std::endl;
    std::cout << "Hash of 'hello' again: " << hash3 << std::endl;
    std::cout << "Same hash for same string: " << (hash1 == hash3) << std::endl;
}

} // namespace ConstexprConsteval

int main() {
    std::cout << "=== CONSTEXPR AND CONSTEVAL TUTORIAL ===" << std::endl;
    
    try {
        ConstexprConsteval::demonstrate_basic_constexpr();
        ConstexprConsteval::demonstrate_constexpr_containers();
        ConstexprConsteval::demonstrate_constexpr_strings();
        ConstexprConsteval::demonstrate_constexpr_classes();
        
#if __cplusplus >= 202002L
        ConstexprConsteval::demonstrate_consteval();
#else
        std::cout << "\n[INFO] consteval features require C++20" << std::endl;
#endif

#if __cplusplus >= 201703L
        ConstexprConsteval::demonstrate_constexpr_if();
#else
        std::cout << "\n[INFO] constexpr if requires C++17" << std::endl;
#endif
        
        ConstexprConsteval::demonstrate_constexpr_algorithms();
        ConstexprConsteval::demonstrate_compile_time_utilities();
        ConstexprConsteval::performance_comparison();
        
        std::cout << "\n=== TUTORIAL COMPLETED ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 