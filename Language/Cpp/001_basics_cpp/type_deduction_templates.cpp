/*
 * =============================================================================
 * TYPE DEDUCTION AND TEMPLATES - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Template type deduction rules
 * 2. Function template argument deduction
 * 3. Class template argument deduction (CTAD)
 * 4. SFINAE (Substitution Failure Is Not An Error)
 * 5. Template metaprogramming basics
 * 6. Concepts and constraints (C++20 preview)
 * 7. Common pitfalls and debugging techniques
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 type_deduction_templates.cpp -o type_deduction_templates
 * For C++20 features: g++ -std=c++20 -Wall -Wextra -g -O0 type_deduction_templates.cpp -o type_deduction_templates
 * Run with debugging: gdb ./type_deduction_templates
 * =============================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <memory>
#include <functional>
#include <array>
#include <map>
#include <utility>
#include <algorithm>

// Type name demangling for better output
#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>

std::string demangle(const char* name) {
    int status = -4;
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };
    return (status == 0) ? res.get() : name;
}
#else
std::string demangle(const char* name) {
    return name;
}
#endif

// Macro to display type information
#define SHOW_TYPE(expr) \
    std::cout << #expr << " -> Type: " << demangle(typeid(expr).name()) << std::endl;

// Macro to display deduced template parameter
#define SHOW_TEMPLATE_TYPE(T) \
    std::cout << "Template type T = " << demangle(typeid(T).name()) << std::endl;

// =============================================================================
// TEMPLATE TYPE DEDUCTION RULES
// =============================================================================

// Template function for demonstrating type deduction
template<typename T>
void func_by_value(T param) {
    std::cout << "\n--- func_by_value ---" << std::endl;
    SHOW_TEMPLATE_TYPE(T);
    std::cout << "param type: " << demangle(typeid(param).name()) << std::endl;
}

template<typename T>
void func_by_reference(T& param) {
    std::cout << "\n--- func_by_reference ---" << std::endl;
    SHOW_TEMPLATE_TYPE(T);
    std::cout << "param type: " << demangle(typeid(param).name()) << std::endl;
}

template<typename T>
void func_by_const_reference(const T& param) {
    std::cout << "\n--- func_by_const_reference ---" << std::endl;
    SHOW_TEMPLATE_TYPE(T);
    std::cout << "param type: " << demangle(typeid(param).name()) << std::endl;
}

template<typename T>
void func_by_universal_reference(T&& param) {
    std::cout << "\n--- func_by_universal_reference ---" << std::endl;
    SHOW_TEMPLATE_TYPE(T);
    std::cout << "param type: " << demangle(typeid(param).name()) << std::endl;
    std::cout << "Is lvalue reference: " << std::is_lvalue_reference_v<T> << std::endl;
    std::cout << "Is rvalue reference: " << std::is_rvalue_reference_v<T> << std::endl;
}

void demonstrate_basic_type_deduction() {
    std::cout << "\n🔍 BASIC TEMPLATE TYPE DEDUCTION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === SETUP TEST VARIABLES ===
    int x = 42;
    const int cx = x;
    const int& rx = x;
    const char* const ptr = "hello";
    
    std::cout << "\nOriginal types:" << std::endl;
    std::cout << "x: int" << std::endl;
    std::cout << "cx: const int" << std::endl;
    std::cout << "rx: const int&" << std::endl;
    std::cout << "ptr: const char* const" << std::endl;
    
    // === BY VALUE (T param) ===
    std::cout << "\n📝 BY VALUE DEDUCTION" << std::endl;
    std::cout << std::string(25, '-') << std::endl;
    
    func_by_value(x);      // T = int
    func_by_value(cx);     // T = int (const stripped)
    func_by_value(rx);     // T = int (const and reference stripped)
    func_by_value(ptr);    // T = const char* (top-level const stripped)
    
    // === BY REFERENCE (T& param) ===
    std::cout << "\n🔗 BY REFERENCE DEDUCTION" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    func_by_reference(x);   // T = int
    func_by_reference(cx);  // T = const int (const preserved)
    func_by_reference(rx);  // T = const int (reference collapsed)
    // func_by_reference(27); // ERROR: can't bind rvalue to lvalue reference
    
    // === BY CONST REFERENCE (const T& param) ===
    std::cout << "\n🔒 BY CONST REFERENCE DEDUCTION" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    func_by_const_reference(x);     // T = int
    func_by_const_reference(cx);    // T = int (const in param, not in T)
    func_by_const_reference(rx);    // T = int (const in param, not in T)
    func_by_const_reference(27);    // T = int (can bind rvalue to const reference)
    
    // === BY UNIVERSAL REFERENCE (T&& param) ===
    std::cout << "\n🔄 BY UNIVERSAL REFERENCE DEDUCTION" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    func_by_universal_reference(x);         // T = int& (lvalue -> lvalue reference)
    func_by_universal_reference(cx);        // T = const int& (const lvalue -> const lvalue ref)
    func_by_universal_reference(rx);        // T = const int& (reference collapsed)
    func_by_universal_reference(27);        // T = int (rvalue -> no reference in T)
    func_by_universal_reference(std::move(x)); // T = int (rvalue -> no reference in T)
}

// =============================================================================
// ARRAY AND FUNCTION TYPE DEDUCTION
// =============================================================================

template<typename T>
void func_array_by_value(T param) {
    std::cout << "\n--- Array by value ---" << std::endl;
    SHOW_TEMPLATE_TYPE(T);
    std::cout << "sizeof(param): " << sizeof(param) << std::endl;
}

template<typename T>
void func_array_by_reference(T& param) {
    std::cout << "\n--- Array by reference ---" << std::endl;
    SHOW_TEMPLATE_TYPE(T);
    std::cout << "sizeof(param): " << sizeof(param) << std::endl;
}

// Helper to get array size
template<typename T, std::size_t N>
constexpr std::size_t array_size(T(&)[N]) noexcept {
    return N;
}

void func_pointer(int x) {
    std::cout << "Function called with: " << x << std::endl;
}

template<typename T>
void func_with_function_param(T param) {
    std::cout << "\n--- Function parameter ---" << std::endl;
    SHOW_TEMPLATE_TYPE(T);
    param(42);
}

void demonstrate_array_function_deduction() {
    std::cout << "\n📚 ARRAY AND FUNCTION TYPE DEDUCTION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === ARRAY TYPE DEDUCTION ===
    std::cout << "\n📋 ARRAY TYPE DEDUCTION" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    const char name[] = "C++ Template";
    const char* ptrToName = name;
    
    std::cout << "Original array: const char[" << sizeof(name) << "]" << std::endl;
    std::cout << "sizeof(name): " << sizeof(name) << std::endl;
    
    func_array_by_value(name);      // T = const char* (array decays to pointer)
    func_array_by_reference(name);  // T = const char[13] (array type preserved)
    
    // Using array size helper
    std::cout << "Array size using helper: " << array_size(name) << std::endl;
    
    // === FUNCTION TYPE DEDUCTION ===
    std::cout << "\n🔧 FUNCTION TYPE DEDUCTION" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    func_with_function_param(func_pointer);      // T = void(*)(int)
    
    // Lambda function
    auto lambda = [](int x) { std::cout << "Lambda called with: " << x << std::endl; };
    func_with_function_param(lambda);            // T = lambda type
}

// =============================================================================
// TEMPLATE ARGUMENT DEDUCTION GUIDES
// =============================================================================

// Custom container class
template<typename T>
class Container {
    std::vector<T> data;
public:
    Container() = default;
    
    // Constructor from initializer list
    Container(std::initializer_list<T> init) : data(init) {}
    
    // Constructor from iterators
    template<typename Iter>
    Container(Iter first, Iter last) : data(first, last) {}
    
    void print() const {
        std::cout << "Container contents: ";
        for (const auto& item : data) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
    
    size_t size() const { return data.size(); }
};

// Deduction guides (C++17)
template<typename Iter>
Container(Iter, Iter) -> Container<typename std::iterator_traits<Iter>::value_type>;

// Custom pair-like class
template<typename T1, typename T2>
struct MyPair {
    T1 first;
    T2 second;
    
    MyPair(const T1& f, const T2& s) : first(f), second(s) {}
    
    void print() const {
        std::cout << "MyPair: {" << first << ", " << second << "}" << std::endl;
    }
};

// Deduction guide for MyPair
template<typename T1, typename T2>
MyPair(T1, T2) -> MyPair<T1, T2>;

void demonstrate_deduction_guides() {
    std::cout << "\n🏗️ TEMPLATE ARGUMENT DEDUCTION GUIDES\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === STANDARD LIBRARY CTAD ===
    std::cout << "\n📚 STANDARD LIBRARY CTAD" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    // Vector deduction
    std::vector vec1{1, 2, 3, 4, 5};           // std::vector<int>
    std::vector vec2{1.1, 2.2, 3.3};          // std::vector<double>
    
    // Pair deduction
    std::pair pair1{42, "hello"};              // std::pair<int, const char*>
    std::pair pair2{3.14, std::string("pi")};  // std::pair<double, std::string>
    
    // Array deduction
    std::array arr{1, 2, 3, 4, 5};             // std::array<int, 5>
    
    SHOW_TYPE(vec1);
    SHOW_TYPE(pair1);
    SHOW_TYPE(arr);
    
    // === CUSTOM DEDUCTION GUIDES ===
    std::cout << "\n🎨 CUSTOM DEDUCTION GUIDES" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    // Using initializer list constructor
    Container container1{10, 20, 30, 40};      // Container<int>
    
    // Using iterator constructor with deduction guide
    std::vector<double> source{1.1, 2.2, 3.3, 4.4};
    Container container2(source.begin(), source.end()); // Container<double>
    
    SHOW_TYPE(container1);
    SHOW_TYPE(container2);
    
    container1.print();
    container2.print();
    
    // Custom pair with deduction guide
    MyPair my_pair{100, "test"};               // MyPair<int, const char*>
    SHOW_TYPE(my_pair);
    my_pair.print();
}

// =============================================================================
// SFINAE DEMONSTRATIONS
// =============================================================================

// SFINAE example 1: Enable if type has iterator
template<typename T>
auto print_if_iterable(const T& container) 
    -> decltype(container.begin(), container.end(), void()) {
    std::cout << "Iterable container: ";
    for (const auto& item : container) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

// Fallback for non-iterable types
template<typename T>
void print_if_iterable(const T& value, ...) {
    std::cout << "Non-iterable value: " << value << std::endl;
}

// SFINAE example 2: Enable if arithmetic type
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
safe_divide(T a, T b) {
    if (b == T{}) {
        throw std::invalid_argument("Division by zero");
    }
    return a / b;
}

// SFINAE example 3: Type traits
template<typename T>
void describe_type(const T& value) {
    std::cout << "\nType analysis for: " << demangle(typeid(T).name()) << std::endl;
    std::cout << "Value: " << value << std::endl;
    std::cout << "Is integral: " << std::is_integral_v<T> << std::endl;
    std::cout << "Is floating point: " << std::is_floating_point_v<T> << std::endl;
    std::cout << "Is arithmetic: " << std::is_arithmetic_v<T> << std::endl;
    std::cout << "Is pointer: " << std::is_pointer_v<T> << std::endl;
    std::cout << "Is reference: " << std::is_reference_v<T> << std::endl;
    std::cout << "Size: " << sizeof(T) << " bytes" << std::endl;
}

// SFINAE example 4: Has member function
template<typename T>
class has_size_method {
    template<typename U>
    static auto test(int) -> decltype(std::declval<U>().size(), std::true_type{});
    
    template<typename>
    static std::false_type test(...);
    
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename T>
void print_size_if_available(const T& obj) {
    if constexpr (has_size_method<T>::value) {
        std::cout << "Size: " << obj.size() << std::endl;
    } else {
        std::cout << "No size() method available" << std::endl;
    }
}

void demonstrate_sfinae() {
    std::cout << "\n🚫 SFINAE DEMONSTRATIONS\n";
    std::cout << std::string(30, '=') << std::endl;
    
    // === ENABLE IF ITERABLE ===
    std::cout << "\n🔄 ENABLE IF ITERABLE" << std::endl;
    std::cout << std::string(25, '-') << std::endl;
    
    std::vector<int> vec{1, 2, 3, 4, 5};
    std::string str = "hello";
    int number = 42;
    
    print_if_iterable(vec);     // Uses iterable version
    print_if_iterable(str);     // Uses iterable version  
    print_if_iterable(number);  // Uses fallback version
    
    // === ARITHMETIC TYPE CONSTRAINTS ===
    std::cout << "\n🔢 ARITHMETIC TYPE CONSTRAINTS" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    std::cout << "safe_divide(10, 3): " << safe_divide(10, 3) << std::endl;
    std::cout << "safe_divide(15.0, 3.0): " << safe_divide(15.0, 3.0) << std::endl;
    // safe_divide("hello", "world"); // Compilation error - not arithmetic
    
    // === TYPE ANALYSIS ===
    std::cout << "\n🔍 TYPE ANALYSIS" << std::endl;
    std::cout << std::string(20, '-') << std::endl;
    
    describe_type(42);
    describe_type(3.14);
    describe_type(&number);
    
    // === HAS MEMBER DETECTION ===
    std::cout << "\n🔎 HAS MEMBER DETECTION" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    std::cout << "Vector has size(): " << has_size_method<std::vector<int>>::value << std::endl;
    std::cout << "Int has size(): " << has_size_method<int>::value << std::endl;
    
    print_size_if_available(vec);
    print_size_if_available(number);
}

// =============================================================================
// TEMPLATE METAPROGRAMMING BASICS
// =============================================================================

// Compile-time factorial
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N-1>::value;
};

template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// Type list
template<typename... Types>
struct TypeList {};

// Get size of type list
template<typename>
struct TypeListSize;

template<typename... Types>
struct TypeListSize<TypeList<Types...>> {
    static constexpr size_t value = sizeof...(Types);
};

// Check if type is in type list
template<typename T, typename TypeList>
struct Contains;

template<typename T>
struct Contains<T, TypeList<>> : std::false_type {};

template<typename T, typename Head, typename... Tail>
struct Contains<T, TypeList<Head, Tail...>> 
    : std::conditional_t<std::is_same_v<T, Head>, 
                         std::true_type, 
                         Contains<T, TypeList<Tail...>>> {};

// Conditional type selection
template<bool Condition, typename TrueType, typename FalseType>
using conditional_t = typename std::conditional<Condition, TrueType, FalseType>::type;

// Perfect forwarding function template
template<typename Func, typename... Args>
auto call_function(Func&& func, Args&&... args) 
    -> decltype(std::forward<Func>(func)(std::forward<Args>(args)...)) {
    std::cout << "Calling function with " << sizeof...(args) << " arguments" << std::endl;
    return std::forward<Func>(func)(std::forward<Args>(args)...);
}

void demonstrate_template_metaprogramming() {
    std::cout << "\n🧮 TEMPLATE METAPROGRAMMING BASICS\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === COMPILE-TIME CALCULATIONS ===
    std::cout << "\n⚡ COMPILE-TIME CALCULATIONS" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    constexpr int fact5 = Factorial<5>::value;
    constexpr int fact10 = Factorial<10>::value;
    
    std::cout << "Factorial<5>::value = " << fact5 << std::endl;
    std::cout << "Factorial<10>::value = " << fact10 << std::endl;
    
    // === TYPE LIST OPERATIONS ===
    std::cout << "\n📋 TYPE LIST OPERATIONS" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    using MyTypes = TypeList<int, double, std::string, char>;
    constexpr size_t type_count = TypeListSize<MyTypes>::value;
    
    std::cout << "Number of types in list: " << type_count << std::endl;
    std::cout << "Contains int: " << Contains<int, MyTypes>::value << std::endl;
    std::cout << "Contains float: " << Contains<float, MyTypes>::value << std::endl;
    
    // === CONDITIONAL TYPE SELECTION ===
    std::cout << "\n🔀 CONDITIONAL TYPE SELECTION" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    using IntOrDouble = conditional_t<true, int, double>;   // int
    using DoubleOrInt = conditional_t<false, int, double>;  // double
    
    IntOrDouble var1 = 42;
    DoubleOrInt var2 = 3.14;
    
    std::cout << "IntOrDouble type: " << demangle(typeid(IntOrDouble).name()) << std::endl;
    std::cout << "DoubleOrInt type: " << demangle(typeid(DoubleOrInt).name()) << std::endl;
    std::cout << "var1: " << var1 << ", var2: " << var2 << std::endl;
    
    // === PERFECT FORWARDING ===
    std::cout << "\n📤 PERFECT FORWARDING" << std::endl;
    std::cout << std::string(25, '-') << std::endl;
    
    auto add = [](int a, int b) { return a + b; };
    auto multiply = [](int a, int b, int c) { return a * b * c; };
    
    auto result1 = call_function(add, 5, 3);
    auto result2 = call_function(multiply, 2, 3, 4);
    
    std::cout << "Addition result: " << result1 << std::endl;
    std::cout << "Multiplication result: " << result2 << std::endl;
}

// =============================================================================
// VARIADIC TEMPLATES
// =============================================================================

// Print all arguments
template<typename... Args>
void print_all(Args... args) {
    ((std::cout << args << " "), ...); // C++17 fold expression
    std::cout << std::endl;
}

// Sum all arguments
template<typename... Args>
auto sum_all(Args... args) {
    return (args + ...); // C++17 fold expression
}

// Apply function to all arguments
template<typename Func, typename... Args>
void apply_to_all(Func func, Args... args) {
    (func(args), ...); // C++17 fold expression
}

// Recursive variadic template (pre-C++17 style)
template<typename T>
void print_recursive(const T& value) {
    std::cout << value << std::endl;
}

template<typename T, typename... Args>
void print_recursive(const T& first, const Args&... rest) {
    std::cout << first << " ";
    print_recursive(rest...);
}

// Tuple-like structure
template<typename... Types>
class Tuple;

template<>
class Tuple<> {};

template<typename Head, typename... Tail>
class Tuple<Head, Tail...> : private Tuple<Tail...> {
    Head head;
public:
    Tuple(Head h, Tail... tail) : Tuple<Tail...>(tail...), head(h) {}
    
    Head& get_head() { return head; }
    const Head& get_head() const { return head; }
    
    Tuple<Tail...>& get_tail() { return *this; }
    const Tuple<Tail...>& get_tail() const { return *this; }
};

void demonstrate_variadic_templates() {
    std::cout << "\n📦 VARIADIC TEMPLATES\n";
    std::cout << std::string(30, '=') << std::endl;
    
    // === FOLD EXPRESSIONS ===
    std::cout << "\n🔃 FOLD EXPRESSIONS (C++17)" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    std::cout << "print_all: ";
    print_all(1, 2.5, "hello", 'c', true);
    
    auto sum_int = sum_all(1, 2, 3, 4, 5);
    auto sum_double = sum_all(1.1, 2.2, 3.3);
    
    std::cout << "Sum of integers: " << sum_int << std::endl;
    std::cout << "Sum of doubles: " << sum_double << std::endl;
    
    // Apply function to all
    auto square_print = [](auto x) { std::cout << x * x << " "; };
    std::cout << "Squares: ";
    apply_to_all(square_print, 1, 2, 3, 4, 5);
    std::cout << std::endl;
    
    // === RECURSIVE VARIADIC TEMPLATES ===
    std::cout << "\n🔄 RECURSIVE VARIADIC TEMPLATES" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    std::cout << "print_recursive: ";
    print_recursive(42, 3.14, "world", 'x');
    
    // === CUSTOM TUPLE ===
    std::cout << "\n📦 CUSTOM TUPLE" << std::endl;
    std::cout << std::string(20, '-') << std::endl;
    
    Tuple<int, double, std::string> my_tuple(42, 3.14, "hello");
    
    std::cout << "Tuple head: " << my_tuple.get_head() << std::endl;
    std::cout << "Tuple tail head: " << my_tuple.get_tail().get_head() << std::endl;
    std::cout << "Tuple tail tail head: " << my_tuple.get_tail().get_tail().get_head() << std::endl;
}

// =============================================================================
// DEBUGGING AND COMMON PITFALLS
// =============================================================================

// Template debugging helper
template<typename T>
struct TypeDisplayer;

// This will cause compilation error showing the type
// Usage: TypeDisplayer<decltype(some_expression)> td;

template<typename T>
void force_template_instantiation_error() {
    // Uncomment the next line to see the type T in compilation error
    // static_assert(false, "Type debugging");
    std::cout << "Template instantiated with type: " << demangle(typeid(T).name()) << std::endl;
}

// Common pitfall: Template argument deduction with auto
template<typename T>
void dangerous_auto_template(T&& param) {
    auto local_copy = param;  // May not be what you expect!
    auto&& perfect_forward = std::forward<T>(param);  // Better approach
    
    std::cout << "param type: " << demangle(typeid(param).name()) << std::endl;
    std::cout << "local_copy type: " << demangle(typeid(local_copy).name()) << std::endl;
    std::cout << "perfect_forward type: " << demangle(typeid(perfect_forward).name()) << std::endl;
}

void demonstrate_debugging_pitfalls() {
    std::cout << "\n🐛 DEBUGGING AND COMMON PITFALLS\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === TEMPLATE INSTANTIATION DEBUGGING ===
    std::cout << "\n🔍 TEMPLATE INSTANTIATION DEBUGGING" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    force_template_instantiation_error<int>();
    force_template_instantiation_error<std::vector<std::string>>();
    
    // === PERFECT FORWARDING PITFALLS ===
    std::cout << "\n⚠️ PERFECT FORWARDING PITFALLS" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    int x = 42;
    const int cx = 100;
    
    std::cout << "\nWith lvalue int:" << std::endl;
    dangerous_auto_template(x);
    
    std::cout << "\nWith const int:" << std::endl;
    dangerous_auto_template(cx);
    
    std::cout << "\nWith rvalue int:" << std::endl;
    dangerous_auto_template(200);
    
    // === TEMPLATE SPECIALIZATION PITFALLS ===
    std::cout << "\n🎯 TEMPLATE SPECIALIZATION NOTES" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    std::cout << "1. Function templates can't be partially specialized" << std::endl;
    std::cout << "2. Specializations must come after the primary template" << std::endl;
    std::cout << "3. Template argument deduction doesn't work with explicit specializations" << std::endl;
    std::cout << "4. Be careful with template template parameters" << std::endl;
}

// Main function demonstrating all concepts
int main() {
    std::cout << "🚀 C++ TYPE DEDUCTION AND TEMPLATES - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(75, '=') << std::endl;
    
    try {
        // Run all demonstrations
        demonstrate_basic_type_deduction();
        demonstrate_array_function_deduction();
        demonstrate_deduction_guides();
        demonstrate_sfinae();
        demonstrate_template_metaprogramming();
        demonstrate_variadic_templates();
        demonstrate_debugging_pitfalls();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
        // === DEBUGGING TIPS ===
        std::cout << "\n🐛 DEBUGGING TIPS:\n";
        std::cout << "1. Use TypeDisplayer trick to see deduced types\n";
        std::cout << "2. Use static_assert with std::is_same_v for type checking\n";
        std::cout << "3. Enable verbose template instantiation: -ftemplate-backtrace-limit=0\n";
        std::cout << "4. Use typename and template keywords where needed\n";
        std::cout << "5. Be explicit about template arguments when debugging\n";
        std::cout << "6. Use concepts (C++20) for better error messages\n";
        
        std::cout << "\n📚 UNDERSTANDING POINTS:\n";
        std::cout << "1. Template argument deduction follows specific rules\n";
        std::cout << "2. Universal references (T&&) enable perfect forwarding\n";
        std::cout << "3. SFINAE allows conditional template instantiation\n";
        std::cout << "4. Template metaprogramming enables compile-time computation\n";
        std::cout << "5. Variadic templates handle variable number of arguments\n";
        std::cout << "6. CTAD (C++17) reduces template argument verbosity\n";
        std::cout << "7. Always be explicit about const-correctness in templates\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
