/*
 * =============================================================================
 * FUNCTIONS - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Function declarations and definitions
 * 2. Function overloading and default parameters
 * 3. Function pointers and references
 * 4. Lambda expressions and closures
 * 5. Function templates and specialization
 * 6. Modern C++ function features (auto, trailing return types)
 * 7. Performance considerations and optimizations
 * 8. Common pitfalls and debugging techniques
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 functions.cpp -o functions
 * Run with debugging: gdb ./functions
 * =============================================================================
 */

#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>
#include <type_traits>
#include <memory>
#include <cassert>

// Debug macro for function call tracing
#define FUNC_TRACE() \
    std::cout << "[TRACE] Entering function: " << __PRETTY_FUNCTION__ << std::endl;

// Performance measurement for functions
#define MEASURE_FUNCTION(func_call, description) \
    { \
        auto start = std::chrono::high_resolution_clock::now(); \
        auto result = func_call; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
        std::cout << "[PERF] " << description << ": " << duration.count() << " μs, Result: " << result << std::endl; \
    }

// =============================================================================
// BASIC FUNCTION DEMONSTRATIONS
// =============================================================================

// Simple function with different parameter types
int add(int a, int b) {
    FUNC_TRACE();
    return a + b;
}

// Function with default parameters
double calculate_area(double length, double width = 1.0, double height = 1.0) {
    FUNC_TRACE();
    std::cout << "Calculating area with dimensions: " << length << "x" << width << "x" << height << std::endl;
    return length * width * height;
}

// Function with reference parameters (pass by reference)
void swap_values(int& a, int& b) {
    FUNC_TRACE();
    std::cout << "Before swap: a=" << a << ", b=" << b << std::endl;
    int temp = a;
    a = b;
    b = temp;
    std::cout << "After swap: a=" << a << ", b=" << b << std::endl;
}

// Function with const reference (efficient for large objects)
void print_vector(const std::vector<int>& vec) {
    FUNC_TRACE();
    std::cout << "Vector contents: ";
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

// Function returning reference (can be used as lvalue)
int& get_element(std::vector<int>& vec, size_t index) {
    FUNC_TRACE();
    if (index >= vec.size()) {
        throw std::out_of_range("Index out of range");
    }
    return vec[index];
}

// =============================================================================
// FUNCTION OVERLOADING
// =============================================================================

// Overloaded functions (same name, different parameters)
void print_info(int value) {
    std::cout << "Integer: " << value << std::endl;
}

void print_info(double value) {
    std::cout << "Double: " << value << std::endl;
}

void print_info(const std::string& value) {
    std::cout << "String: " << value << std::endl;
}

void print_info(const char* value) {
    std::cout << "C-string: " << value << std::endl;
}

// Overloaded with different number of parameters
int multiply(int a, int b) {
    return a * b;
}

int multiply(int a, int b, int c) {
    return a * b * c;
}

// =============================================================================
// FUNCTION POINTERS AND FUNCTION OBJECTS
// =============================================================================

// Function to be used with function pointers
int square(int x) {
    return x * x;
}

int cube(int x) {
    return x * x * x;
}

// Function that takes another function as parameter
int apply_operation(int x, int (*operation)(int)) {
    FUNC_TRACE();
    return operation(x);
}

// Function that takes std::function (more flexible)
int apply_function_object(int x, const std::function<int(int)>& func) {
    FUNC_TRACE();
    return func(x);
}

// =============================================================================
// LAMBDA EXPRESSIONS
// =============================================================================

void demonstrate_lambdas() {
    std::cout << "\n🚀 LAMBDA EXPRESSIONS DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === BASIC LAMBDA ===
    std::cout << "\n📝 BASIC LAMBDAS\n";
    std::cout << std::string(20, '-') << std::endl;
    
    // Simple lambda
    auto simple_lambda = []() {
        std::cout << "Hello from lambda!" << std::endl;
    };
    simple_lambda();
    
    // Lambda with parameters
    auto add_lambda = [](int a, int b) {
        return a + b;
    };
    std::cout << "Lambda add(5, 3): " << add_lambda(5, 3) << std::endl;
    
    // Lambda with return type specification
    auto divide_lambda = [](double a, double b) -> double {
        if (b == 0.0) {
            throw std::invalid_argument("Division by zero");
        }
        return a / b;
    };
    std::cout << "Lambda divide(10.0, 3.0): " << divide_lambda(10.0, 3.0) << std::endl;
    
    // === CAPTURE MODES ===
    std::cout << "\n🎯 CAPTURE MODES\n";
    std::cout << std::string(20, '-') << std::endl;
    
    int x = 10, y = 20;
    
    // Capture by value
    auto capture_by_value = [x, y]() {
        std::cout << "Captured by value: x=" << x << ", y=" << y << std::endl;
        // x++; // This would be an error - captured by value is const by default
    };
    capture_by_value();
    
    // Capture by reference
    auto capture_by_reference = [&x, &y]() {
        std::cout << "Before modification: x=" << x << ", y=" << y << std::endl;
        x += 5;
        y += 10;
        std::cout << "After modification: x=" << x << ", y=" << y << std::endl;
    };
    capture_by_reference();
    
    // Capture all by value
    auto capture_all_by_value = [=]() {
        std::cout << "All by value: x=" << x << ", y=" << y << std::endl;
    };
    capture_all_by_value();
    
    // Capture all by reference
    int z = 30;
    auto capture_all_by_reference = [&]() {
        x *= 2;
        y *= 2;
        z *= 2;
        std::cout << "All by reference modified: x=" << x << ", y=" << y << ", z=" << z << std::endl;
    };
    capture_all_by_reference();
    
    // Mutable lambda (can modify captured-by-value variables)
    int counter = 0;
    auto mutable_lambda = [counter](int increment) mutable {
        counter += increment;
        std::cout << "Mutable lambda counter: " << counter << std::endl;
        return counter;
    };
    mutable_lambda(5);
    mutable_lambda(3);
    std::cout << "Original counter unchanged: " << counter << std::endl;
    
    // === LAMBDA WITH STL ALGORITHMS ===
    std::cout << "\n🔧 LAMBDAS WITH STL ALGORITHMS\n";
    std::cout << std::string(35, '-') << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Find even numbers
    auto even_count = std::count_if(numbers.begin(), numbers.end(), 
                                   [](int n) { return n % 2 == 0; });
    std::cout << "Even numbers count: " << even_count << std::endl;
    
    // Transform with lambda
    std::vector<int> squared_numbers;
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(squared_numbers),
                   [](int n) { return n * n; });
    
    std::cout << "Squared numbers: ";
    for (const auto& num : squared_numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // Sort with custom comparator
    std::vector<std::string> words = {"apple", "pie", "cherry", "banana"};
    std::sort(words.begin(), words.end(), 
              [](const std::string& a, const std::string& b) {
                  return a.length() < b.length(); // Sort by length
              });
    
    std::cout << "Words sorted by length: ";
    for (const auto& word : words) {
        std::cout << word << " ";
    }
    std::cout << std::endl;
}

// =============================================================================
// FUNCTION TEMPLATES
// =============================================================================

// Basic function template
template<typename T>
T find_maximum(T a, T b) {
    FUNC_TRACE();
    return (a > b) ? a : b;
}

// Template with multiple type parameters
template<typename T, typename U>
auto add_different_types(T a, U b) -> decltype(a + b) {
    FUNC_TRACE();
    return a + b;
}

// Template specialization
template<>
const char* find_maximum<const char*>(const char* a, const char* b) {
    FUNC_TRACE();
    std::cout << "Using specialized template for const char*" << std::endl;
    return (std::strcmp(a, b) > 0) ? a : b;
}

// Variadic template
template<typename... Args>
void print_all(Args... args) {
    FUNC_TRACE();
    ((std::cout << args << " "), ...); // C++17 fold expression
    std::cout << std::endl;
}

// Template with SFINAE (Substitution Failure Is Not An Error)
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
safe_divide(T a, T b) {
    FUNC_TRACE();
    if (b == T{}) {
        throw std::invalid_argument("Division by zero");
    }
    return a / b;
}

// =============================================================================
// MODERN C++ FUNCTION FEATURES
// =============================================================================

// Auto return type deduction (C++14)
auto calculate_something(int x, double y) {
    FUNC_TRACE();
    if (x > 0) {
        return x * y; // Returns double
    } else {
        return static_cast<double>(x); // Also returns double
    }
}

// Trailing return type
auto create_vector(size_t size) -> std::vector<int> {
    FUNC_TRACE();
    return std::vector<int>(size, 42);
}

// decltype(auto) for perfect forwarding
template<typename Container>
decltype(auto) get_first_element(Container&& container) {
    FUNC_TRACE();
    return std::forward<Container>(container)[0];
}

// constexpr function (compile-time evaluation)
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

// noexcept specification
int safe_operation(int x) noexcept {
    FUNC_TRACE();
    return x * 2; // Guaranteed not to throw
}

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_basic_functions() {
    std::cout << "\n🔧 BASIC FUNCTIONS DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === SIMPLE FUNCTION CALLS ===
    std::cout << "\n📞 SIMPLE FUNCTION CALLS\n";
    std::cout << std::string(30, '-') << std::endl;
    
    int result = add(5, 3);
    std::cout << "add(5, 3) = " << result << std::endl;
    
    // Default parameters
    std::cout << "\nDefault parameters demonstration:" << std::endl;
    std::cout << "1D: " << calculate_area(5.0) << std::endl;
    std::cout << "2D: " << calculate_area(5.0, 3.0) << std::endl;
    std::cout << "3D: " << calculate_area(5.0, 3.0, 2.0) << std::endl;
    
    // === REFERENCE PARAMETERS ===
    std::cout << "\n🔄 REFERENCE PARAMETERS\n";
    std::cout << std::string(25, '-') << std::endl;
    
    int a = 10, b = 20;
    std::cout << "Before function call: a=" << a << ", b=" << b << std::endl;
    swap_values(a, b);
    std::cout << "After function call: a=" << a << ", b=" << b << std::endl;
    
    // === CONST REFERENCE FOR EFFICIENCY ===
    std::cout << "\n📋 CONST REFERENCE EFFICIENCY\n";
    std::cout << std::string(35, '-') << std::endl;
    
    std::vector<int> large_vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    print_vector(large_vector); // Passed by const reference - no copy
    
    // === RETURNING REFERENCES ===
    std::cout << "\n↩️ RETURNING REFERENCES\n";
    std::cout << std::string(25, '-') << std::endl;
    
    std::vector<int> vec = {10, 20, 30, 40, 50};
    std::cout << "Original vector: ";
    print_vector(vec);
    
    get_element(vec, 2) = 999; // Modify element through reference
    std::cout << "After modifying element 2: ";
    print_vector(vec);
}

void demonstrate_function_overloading() {
    std::cout << "\n🔄 FUNCTION OVERLOADING DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // Same function name, different parameter types
    print_info(42);
    print_info(3.14159);
    print_info(std::string("Hello World"));
    print_info("C-style string");
    
    // Same function name, different parameter count
    std::cout << "multiply(3, 4) = " << multiply(3, 4) << std::endl;
    std::cout << "multiply(3, 4, 5) = " << multiply(3, 4, 5) << std::endl;
}

void demonstrate_function_pointers() {
    std::cout << "\n👉 FUNCTION POINTERS DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === TRADITIONAL FUNCTION POINTERS ===
    std::cout << "\n🔗 TRADITIONAL FUNCTION POINTERS\n";
    std::cout << std::string(35, '-') << std::endl;
    
    // Function pointer declaration and assignment
    int (*operation)(int) = square;
    std::cout << "Using function pointer for square(5): " << apply_operation(5, operation) << std::endl;
    
    operation = cube;
    std::cout << "Using function pointer for cube(3): " << apply_operation(3, operation) << std::endl;
    
    // Array of function pointers
    int (*operations[])(int) = {square, cube};
    std::cout << "Array of function pointers:" << std::endl;
    for (size_t i = 0; i < 2; ++i) {
        std::cout << "  Operation " << i << " on 4: " << operations[i](4) << std::endl;
    }
    
    // === STD::FUNCTION (MORE FLEXIBLE) ===
    std::cout << "\n🎯 STD::FUNCTION\n";
    std::cout << std::string(20, '-') << std::endl;
    
    // std::function can hold lambdas, function pointers, functors
    std::function<int(int)> func_obj;
    
    func_obj = square;
    std::cout << "std::function with square(6): " << apply_function_object(6, func_obj) << std::endl;
    
    func_obj = [](int x) { return x * x * x * x; }; // Lambda for x^4
    std::cout << "std::function with lambda x^4(2): " << apply_function_object(2, func_obj) << std::endl;
}

void demonstrate_templates() {
    std::cout << "\n📋 FUNCTION TEMPLATES DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === BASIC TEMPLATE USAGE ===
    std::cout << "\n🔧 BASIC TEMPLATE USAGE\n";
    std::cout << std::string(30, '-') << std::endl;
    
    std::cout << "max(5, 3): " << find_maximum(5, 3) << std::endl;
    std::cout << "max(5.5, 3.2): " << find_maximum(5.5, 3.2) << std::endl;
    std::cout << "max('a', 'z'): " << find_maximum('a', 'z') << std::endl;
    
    // Template specialization
    const char* str1 = "hello";
    const char* str2 = "world";
    std::cout << "max(\"hello\", \"world\"): " << find_maximum(str1, str2) << std::endl;
    
    // === MULTIPLE TYPE PARAMETERS ===
    std::cout << "\n🎭 MULTIPLE TYPE PARAMETERS\n";
    std::cout << std::string(30, '-') << std::endl;
    
    auto result1 = add_different_types(5, 3.14);
    std::cout << "add_different_types(5, 3.14): " << result1 << " (type: double)" << std::endl;
    
    auto result2 = add_different_types(2.5f, 10);
    std::cout << "add_different_types(2.5f, 10): " << result2 << " (type: float)" << std::endl;
    
    // === VARIADIC TEMPLATES ===
    std::cout << "\n📦 VARIADIC TEMPLATES\n";
    std::cout << std::string(25, '-') << std::endl;
    
    std::cout << "print_all with various types: ";
    print_all(1, 2.5, "hello", 'A', true);
    
    // === SFINAE EXAMPLE ===
    std::cout << "\n🚫 SFINAE EXAMPLE\n";
    std::cout << std::string(20, '-') << std::endl;
    
    std::cout << "safe_divide(10, 3): " << safe_divide(10, 3) << std::endl;
    std::cout << "safe_divide(15.0, 4.0): " << safe_divide(15.0, 4.0) << std::endl;
    // safe_divide("hello", "world"); // Would not compile - not arithmetic type
}

void demonstrate_modern_features() {
    std::cout << "\n🆕 MODERN C++ FEATURES DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === AUTO RETURN TYPE ===
    std::cout << "\n🤖 AUTO RETURN TYPE\n";
    std::cout << std::string(25, '-') << std::endl;
    
    auto result1 = calculate_something(5, 2.5);
    std::cout << "calculate_something(5, 2.5): " << result1 << std::endl;
    
    auto result2 = calculate_something(-3, 1.5);
    std::cout << "calculate_something(-3, 1.5): " << result2 << std::endl;
    
    // === TRAILING RETURN TYPE ===
    std::cout << "\n↩️ TRAILING RETURN TYPE\n";
    std::cout << std::string(30, '-') << std::endl;
    
    auto vec = create_vector(5);
    std::cout << "Created vector size: " << vec.size() << ", first element: " << vec[0] << std::endl;
    
    // === CONSTEXPR FUNCTIONS ===
    std::cout << "\n⚡ CONSTEXPR FUNCTIONS\n";
    std::cout << std::string(25, '-') << std::endl;
    
    constexpr int fact5 = factorial(5); // Computed at compile time
    std::cout << "factorial(5) [compile-time]: " << fact5 << std::endl;
    
    int n = 6;
    int fact6 = factorial(n); // Computed at runtime (n is not constexpr)
    std::cout << "factorial(6) [runtime]: " << fact6 << std::endl;
    
    // === NOEXCEPT SPECIFICATION ===
    std::cout << "\n🛡️ NOEXCEPT SPECIFICATION\n";
    std::cout << std::string(30, '-') << std::endl;
    
    std::cout << "safe_operation(21): " << safe_operation(21) << std::endl;
    std::cout << "Is safe_operation noexcept? " << std::boolalpha 
              << noexcept(safe_operation(10)) << std::endl;
}

void demonstrate_performance_considerations() {
    std::cout << "\n⚡ PERFORMANCE CONSIDERATIONS\n";
    std::cout << std::string(40, '=') << std::endl;
    
    const int ITERATIONS = 1000000;
    
    // === FUNCTION CALL OVERHEAD ===
    std::cout << "\n🏃 FUNCTION CALL OVERHEAD\n";
    std::cout << std::string(30, '-') << std::endl;
    
    // Regular function call
    MEASURE_FUNCTION(
        [&]() {
            int sum = 0;
            for (int i = 0; i < ITERATIONS; ++i) {
                sum += square(i % 100);
            }
            return sum;
        }(), 
        "Regular function calls"
    );
    
    // Inline lambda (likely inlined by compiler)
    auto inline_square = [](int x) { return x * x; };
    MEASURE_FUNCTION(
        [&]() {
            int sum = 0;
            for (int i = 0; i < ITERATIONS; ++i) {
                sum += inline_square(i % 100);
            }
            return sum;
        }(), 
        "Lambda calls (likely inlined)"
    );
    
    // std::function overhead
    std::function<int(int)> func_square = square;
    MEASURE_FUNCTION(
        [&]() {
            int sum = 0;
            for (int i = 0; i < ITERATIONS; ++i) {
                sum += func_square(i % 100);
            }
            return sum;
        }(), 
        "std::function calls (overhead)"
    );
}

// Main function demonstrating all concepts
int main() {
    std::cout << "🚀 C++ FUNCTIONS - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(55, '=') << std::endl;
    
    try {
        // Run all demonstrations
        demonstrate_basic_functions();
        demonstrate_function_overloading();
        demonstrate_function_pointers();
        demonstrate_lambdas();
        demonstrate_templates();
        demonstrate_modern_features();
        demonstrate_performance_considerations();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
        // === DEBUGGING TIPS ===
        std::cout << "\n🐛 DEBUGGING TIPS:\n";
        std::cout << "1. Use FUNC_TRACE() macro to track function calls\n";
        std::cout << "2. Set breakpoints on function entry and exit\n";
        std::cout << "3. Use 'bt' (backtrace) in gdb to see call stack\n";
        std::cout << "4. Use 'info args' to see function parameters\n";
        std::cout << "5. Use 'finish' to run until function returns\n";
        std::cout << "6. Watch parameter changes with 'watch parameter_name'\n";
        
        std::cout << "\n📚 UNDERSTANDING POINTS:\n";
        std::cout << "1. Pass large objects by const reference for efficiency\n";
        std::cout << "2. Use function overloading for different parameter types\n";
        std::cout << "3. Lambdas are often more efficient than std::function\n";
        std::cout << "4. Templates provide compile-time polymorphism\n";
        std::cout << "5. constexpr functions enable compile-time computation\n";
        std::cout << "6. noexcept helps with optimization and exception safety\n";
        std::cout << "7. auto return type deduction simplifies complex types\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
