/*
 * TYPE TRAITS USAGE - Advanced C++ Type Inspection and Manipulation
 * 
 * Compilation: g++ -std=c++17 -Wall -Wextra -g -O2 type_traits_usage.cpp -o type_traits_usage
 */

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>
#include <utility>
#include <memory>
#include <functional>
#include <typeinfo>
#include <array>
#include <map>
#include <limits>

#define TRAIT_TEST(expr) std::cout << "[TRAIT] " << #expr << " = " << (expr) << std::endl
#define TYPE_INFO(T) std::cout << "[TYPE] " << #T << " = " << typeid(T).name() << std::endl

namespace TypeTraitsUsage {

// ============================================================================
// 1. STANDARD LIBRARY TYPE TRAITS
// ============================================================================

void demonstrate_primary_type_categories() {
    std::cout << "\n=== PRIMARY TYPE CATEGORIES ===" << std::endl;
    
    // Fundamental types
    TRAIT_TEST(std::is_void_v<void>);
    TRAIT_TEST(std::is_null_pointer_v<std::nullptr_t>);
    TRAIT_TEST(std::is_integral_v<int>);
    TRAIT_TEST(std::is_floating_point_v<double>);
    TRAIT_TEST(std::is_array_v<int[5]>);
    TRAIT_TEST(std::is_class_v<std::string>);
    TRAIT_TEST(std::is_function_v<int(int)>);
    TRAIT_TEST(std::is_pointer_v<int*>);
    TRAIT_TEST(std::is_lvalue_reference_v<int&>);
    TRAIT_TEST(std::is_rvalue_reference_v<int&&>);
}

void demonstrate_composite_type_categories() {
    std::cout << "\n=== COMPOSITE TYPE CATEGORIES ===" << std::endl;
    
    TRAIT_TEST(std::is_fundamental_v<int>);
    TRAIT_TEST(std::is_arithmetic_v<double>);
    TRAIT_TEST(std::is_scalar_v<int*>);
    TRAIT_TEST(std::is_object_v<std::string>);
    TRAIT_TEST(std::is_compound_v<std::vector<int>>);
    TRAIT_TEST(std::is_reference_v<int&>);
}

void demonstrate_type_properties() {
    std::cout << "\n=== TYPE PROPERTIES ===" << std::endl;
    
    TRAIT_TEST(std::is_const_v<const int>);
    TRAIT_TEST(std::is_volatile_v<volatile int>);
    TRAIT_TEST(std::is_trivial_v<int>);
    TRAIT_TEST(std::is_trivially_copyable_v<std::string>);
    TRAIT_TEST(std::is_standard_layout_v<std::string>);
    TRAIT_TEST(std::is_empty_v<std::string>);
    TRAIT_TEST(std::is_polymorphic_v<std::string>);
    TRAIT_TEST(std::is_abstract_v<std::string>);
    TRAIT_TEST(std::is_signed_v<int>);
    TRAIT_TEST(std::is_unsigned_v<unsigned int>);
}

void demonstrate_supported_operations() {
    std::cout << "\n=== SUPPORTED OPERATIONS ===" << std::endl;
    
    TRAIT_TEST(std::is_constructible_v<std::string, const char*>);
    TRAIT_TEST(std::is_trivially_constructible_v<int>);
    TRAIT_TEST(std::is_nothrow_constructible_v<std::string>);
    TRAIT_TEST(std::is_default_constructible_v<std::vector<int>>);
    TRAIT_TEST(std::is_copy_constructible_v<std::string>);
    TRAIT_TEST(std::is_move_constructible_v<std::unique_ptr<int>>);
    TRAIT_TEST(std::is_assignable_v<std::string&, const std::string&>);
    TRAIT_TEST(std::is_copy_assignable_v<std::string>);
    TRAIT_TEST(std::is_move_assignable_v<std::string>);
    TRAIT_TEST(std::is_destructible_v<std::string>);
    TRAIT_TEST(std::is_trivially_destructible_v<int>);
    TRAIT_TEST(std::is_nothrow_destructible_v<std::string>);
}

// ============================================================================
// 2. TYPE RELATIONSHIPS
// ============================================================================

void demonstrate_type_relationships() {
    std::cout << "\n=== TYPE RELATIONSHIPS ===" << std::endl;
    
    TRAIT_TEST(std::is_same_v<int, int>);
    TRAIT_TEST(std::is_same_v<int, const int>); // false
    TRAIT_TEST(std::is_convertible_v<int, double>);
    TRAIT_TEST(std::is_convertible_v<std::string, int>); // false
    
    // C++17 concepts-like usage
    if constexpr (std::is_convertible_v<int, double>) {
        std::cout << "int is convertible to double" << std::endl;
    }
}

// ============================================================================
// 3. TYPE MODIFICATIONS
// ============================================================================

void demonstrate_type_modifications() {
    std::cout << "\n=== TYPE MODIFICATIONS ===" << std::endl;
    
    using int_const = std::add_const_t<int>;
    using int_volatile = std::add_volatile_t<int>;
    using int_cv = std::add_cv_t<int>;
    
    TYPE_INFO(int_const);
    TRAIT_TEST(std::is_const_v<int_const>);
    
    using int_ptr = std::add_pointer_t<int>;
    using int_lref = std::add_lvalue_reference_t<int>;
    using int_rref = std::add_rvalue_reference_t<int>;
    
    TYPE_INFO(int_ptr);
    TYPE_INFO(int_lref);
    TYPE_INFO(int_rref);
    
    using const_int_removed = std::remove_const_t<const int>;
    using int_ptr_removed = std::remove_pointer_t<int*>;
    using int_ref_removed = std::remove_reference_t<int&>;
    using int_all_removed = std::remove_cvref_t<const volatile int&>;
    
    TRAIT_TEST(std::is_same_v<const_int_removed, int>);
    TRAIT_TEST(std::is_same_v<int_ptr_removed, int>);
    TRAIT_TEST(std::is_same_v<int_ref_removed, int>);
    TRAIT_TEST(std::is_same_v<int_all_removed, int>);
}

// ============================================================================
// 4. CUSTOM TYPE TRAITS
// ============================================================================

// Custom trait to detect if a type has a specific member
template<typename T, typename = void>
struct has_size_method : std::false_type {};

template<typename T>
struct has_size_method<T, std::void_t<decltype(std::declval<T>().size())>> : std::true_type {};

template<typename T>
constexpr bool has_size_method_v = has_size_method<T>::value;

// Custom trait to detect container-like types
template<typename T, typename = void>
struct is_container : std::false_type {};

template<typename T>
struct is_container<T, std::void_t<
    typename T::value_type,
    typename T::iterator,
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end()),
    decltype(std::declval<T>().size())
>> : std::true_type {};

template<typename T>
constexpr bool is_container_v = is_container<T>::value;

// Custom trait to detect callable types
template<typename T, typename... Args>
struct is_callable {
private:
    template<typename U>
    static auto test(int) -> decltype(std::declval<U>()(std::declval<Args>()...), std::true_type{});
    
    template<typename>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename T, typename... Args>
constexpr bool is_callable_v = is_callable<T, Args...>::value;

// Custom trait to detect smart pointer types
template<typename T>
struct is_smart_pointer : std::false_type {};

template<typename T>
struct is_smart_pointer<std::unique_ptr<T>> : std::true_type {};

template<typename T>
struct is_smart_pointer<std::shared_ptr<T>> : std::true_type {};

template<typename T>
struct is_smart_pointer<std::weak_ptr<T>> : std::true_type {};

template<typename T>
constexpr bool is_smart_pointer_v = is_smart_pointer<T>::value;

// ============================================================================
// 5. TEMPLATE METAPROGRAMMING WITH TYPE TRAITS
// ============================================================================

// Conditional compilation based on type traits
template<typename T>
void process_type(const T& value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Processing integral type: " << value << std::endl;
        std::cout << "  Size: " << sizeof(T) << " bytes" << std::endl;
        std::cout << "  Min: " << std::numeric_limits<T>::min() << std::endl;
        std::cout << "  Max: " << std::numeric_limits<T>::max() << std::endl;
    }
    else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Processing floating-point type: " << value << std::endl;
        std::cout << "  Size: " << sizeof(T) << " bytes" << std::endl;
        std::cout << "  Precision: " << std::numeric_limits<T>::digits10 << " digits" << std::endl;
    }
    else if constexpr (std::is_same_v<T, std::string>) {
        std::cout << "Processing string: \"" << value << "\"" << std::endl;
        std::cout << "  Length: " << value.length() << std::endl;
    }
    else if constexpr (is_container_v<T>) {
        std::cout << "Processing container with " << value.size() << " elements" << std::endl;
    }
    else {
        std::cout << "Processing unknown type: " << typeid(T).name() << std::endl;
    }
}

// SFINAE with type traits
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, T>
safe_divide(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        return (b != 0) ? a / b : T{};
    } else {
        return (b != T{}) ? a / b : std::numeric_limits<T>::quiet_NaN();
    }
}

// Perfect forwarding with type traits
template<typename T>
void universal_print(T&& value) {
    std::cout << "Value: " << value << std::endl;
    
    if constexpr (std::is_lvalue_reference_v<T>) {
        std::cout << "  Passed as lvalue reference" << std::endl;
    }
    else if constexpr (std::is_rvalue_reference_v<T>) {
        std::cout << "  Passed as rvalue reference" << std::endl;
    }
    else {
        std::cout << "  Passed by value" << std::endl;
    }
    
    std::cout << "  Type: " << typeid(T).name() << std::endl;
    std::cout << "  Is const: " << std::is_const_v<std::remove_reference_t<T>> << std::endl;
}

// ============================================================================
// 6. PRACTICAL APPLICATIONS
// ============================================================================

// Generic serialization based on type traits
template<typename T>
std::string serialize(const T& value) {
    if constexpr (std::is_arithmetic_v<T>) {
        return std::to_string(value);
    }
    else if constexpr (std::is_same_v<T, std::string>) {
        return "\"" + value + "\"";
    }
    else if constexpr (is_container_v<T>) {
        std::string result = "[";
        bool first = true;
        for (const auto& item : value) {
            if (!first) result += ", ";
            result += serialize(item);
            first = false;
        }
        result += "]";
        return result;
    }
    else {
        return "non-serializable";
    }
}

// Type-safe factory function
template<typename T, typename... Args>
std::enable_if_t<std::is_constructible_v<T, Args...>, std::unique_ptr<T>>
make_safe(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

// ============================================================================
// 7. ADVANCED TYPE TRAIT PATTERNS
// ============================================================================

// Check if all types satisfy a predicate
template<template<typename> class Predicate, typename... Types>
struct all_of : std::conjunction<Predicate<Types>...> {};

template<template<typename> class Predicate, typename... Types>
constexpr bool all_of_v = all_of<Predicate, Types...>::value;

// Check if any type satisfies a predicate
template<template<typename> class Predicate, typename... Types>
struct any_of : std::disjunction<Predicate<Types>...> {};

template<template<typename> class Predicate, typename... Types>
constexpr bool any_of_v = any_of<Predicate, Types...>::value;

// Count types satisfying a predicate
template<template<typename> class Predicate, typename... Types>
struct count_if {
    static constexpr size_t value = (Predicate<Types>::value + ...);
};

template<template<typename> class Predicate, typename... Types>
constexpr size_t count_if_v = count_if<Predicate, Types...>::value;

// ============================================================================
// DEMONSTRATION FUNCTIONS
// ============================================================================

void demonstrate_custom_traits() {
    std::cout << "\n=== CUSTOM TYPE TRAITS ===" << std::endl;
    
    TRAIT_TEST(has_size_method_v<std::vector<int>>);
    TRAIT_TEST(has_size_method_v<std::string>);
    TRAIT_TEST(has_size_method_v<int>);
    
    TRAIT_TEST(is_container_v<std::vector<int>>);
    TRAIT_TEST(is_container_v<std::string>);
    TRAIT_TEST(is_container_v<int>);
    
    auto lambda = [](int x) { return x * 2; };
    TRAIT_TEST(is_callable_v<decltype(lambda), int>);
    TRAIT_TEST(is_callable_v<int, int>);
    
    TRAIT_TEST(is_smart_pointer_v<std::unique_ptr<int>>);
    TRAIT_TEST(is_smart_pointer_v<std::shared_ptr<int>>);
    TRAIT_TEST(is_smart_pointer_v<int*>);
}

void demonstrate_conditional_compilation() {
    std::cout << "\n=== CONDITIONAL COMPILATION ===" << std::endl;
    
    process_type(42);
    process_type(3.14);
    process_type(std::string("hello"));
    
    std::vector<int> vec = {1, 2, 3};
    process_type(vec);
    
    struct CustomType {};
    process_type(CustomType{});
}

void demonstrate_sfinae_applications() {
    std::cout << "\n=== SFINAE APPLICATIONS ===" << std::endl;
    
    std::cout << "safe_divide(10, 3): " << safe_divide(10, 3) << std::endl;
    std::cout << "safe_divide(10.0, 3.0): " << safe_divide(10.0, 3.0) << std::endl;
    std::cout << "safe_divide(10, 0): " << safe_divide(10, 0) << std::endl;
    
    int x = 42;
    universal_print(x);        // lvalue
    universal_print(42);       // rvalue
    universal_print(std::move(x)); // moved lvalue
}

void demonstrate_practical_applications() {
    std::cout << "\n=== PRACTICAL APPLICATIONS ===" << std::endl;
    
    std::cout << "Serialization:" << std::endl;
    std::cout << "  int: " << serialize(42) << std::endl;
    std::cout << "  string: " << serialize(std::string("hello")) << std::endl;
    
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::cout << "  vector: " << serialize(vec) << std::endl;
    
    // Type-safe factory
    auto str_ptr = make_safe<std::string>("Hello World");
    if (str_ptr) {
        std::cout << "Created string: " << *str_ptr << std::endl;
    }
}

void demonstrate_advanced_patterns() {
    std::cout << "\n=== ADVANCED TYPE TRAIT PATTERNS ===" << std::endl;
    
    std::cout << "All arithmetic: " << all_of_v<std::is_arithmetic, int, double, float> << std::endl;
    std::cout << "All arithmetic: " << all_of_v<std::is_arithmetic, int, double, std::string> << std::endl;
    
    std::cout << "Any arithmetic: " << any_of_v<std::is_arithmetic, int, std::string> << std::endl;
    std::cout << "Any arithmetic: " << any_of_v<std::is_arithmetic, std::string, std::vector<int>> << std::endl;
    
    std::cout << "Count arithmetic: " << count_if_v<std::is_arithmetic, int, double, std::string, float> << std::endl;
}

} // namespace TypeTraitsUsage

int main() {
    std::cout << "=== TYPE TRAITS USAGE TUTORIAL ===" << std::endl;
    
    try {
        TypeTraitsUsage::demonstrate_primary_type_categories();
        TypeTraitsUsage::demonstrate_composite_type_categories();
        TypeTraitsUsage::demonstrate_type_properties();
        TypeTraitsUsage::demonstrate_supported_operations();
        TypeTraitsUsage::demonstrate_type_relationships();
        TypeTraitsUsage::demonstrate_type_modifications();
        TypeTraitsUsage::demonstrate_custom_traits();
        TypeTraitsUsage::demonstrate_conditional_compilation();
        TypeTraitsUsage::demonstrate_sfinae_applications();
        TypeTraitsUsage::demonstrate_practical_applications();
        TypeTraitsUsage::demonstrate_advanced_patterns();
        
        std::cout << "\n=== TUTORIAL COMPLETED ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 