/*
 * =============================================================================
 * AUTO AND DECLTYPE - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. auto keyword and type deduction
 * 2. decltype specifier and usage
 * 3. Type deduction rules and edge cases
 * 4. AAA (Almost Always Auto) idiom
 * 5. decltype(auto) for perfect forwarding
 * 6. CTAD (Class Template Argument Deduction)
 * 7. Common pitfalls and debugging techniques
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 auto_decltype.cpp -o auto_decltype
 * Run with debugging: gdb ./auto_decltype
 * =============================================================================
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <functional>
#include <memory>
#include <utility>
#include <array>

// Type name demangling for better output (compiler-specific)
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

// Macro to print type information
#define PRINT_TYPE(var) \
    std::cout << #var << " -> Type: " << demangle(typeid(var).name()) \
              << ", Value: " << var << std::endl;

// Template to check if types are the same
template<typename T, typename U>
void check_same_type(const std::string& desc) {
    std::cout << desc << ": " << std::boolalpha << std::is_same_v<T, U> << std::endl;
}

// =============================================================================
// AUTO KEYWORD DEMONSTRATIONS
// =============================================================================

void demonstrate_basic_auto() {
    std::cout << "\n🤖 BASIC AUTO KEYWORD DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === BASIC AUTO USAGE ===
    std::cout << "\n📝 BASIC AUTO USAGE\n";
    std::cout << std::string(25, '-') << std::endl;
    
    auto a = 42;                    // int
    auto b = 3.14;                  // double
    auto c = 3.14f;                 // float
    auto d = 'x';                   // char
    auto e = true;                  // bool
    auto f = "hello";               // const char*
    auto g = std::string("world");  // std::string
    
    PRINT_TYPE(a);
    PRINT_TYPE(b);
    PRINT_TYPE(c);
    PRINT_TYPE(d);
    PRINT_TYPE(e);
    PRINT_TYPE(f);
    PRINT_TYPE(g);
    
    // === AUTO WITH CONTAINERS ===
    std::cout << "\n📦 AUTO WITH CONTAINERS\n";
    std::cout << std::string(30, '-') << std::endl;
    
    auto vec = std::vector<int>{1, 2, 3, 4, 5};
    auto map = std::map<std::string, int>{{"one", 1}, {"two", 2}};
    auto arr = std::array<double, 3>{1.1, 2.2, 3.3};
    
    PRINT_TYPE(vec);
    std::cout << "map type: " << demangle(typeid(map).name()) << std::endl;
    PRINT_TYPE(arr);
    
    // === AUTO WITH ITERATORS ===
    std::cout << "\n🔄 AUTO WITH ITERATORS\n";
    std::cout << std::string(25, '-') << std::endl;
    
    auto it1 = vec.begin();         // std::vector<int>::iterator
    auto it2 = vec.cbegin();        // std::vector<int>::const_iterator
    auto it3 = map.find("one");     // std::map<std::string, int>::iterator
    
    std::cout << "it1 type: " << demangle(typeid(it1).name()) << std::endl;
    std::cout << "it2 type: " << demangle(typeid(it2).name()) << std::endl;
    std::cout << "it3 type: " << demangle(typeid(it3).name()) << std::endl;
    
    // Using auto in range-based for loop
    std::cout << "Vector elements: ";
    for (auto element : vec) {       // auto deduces to int
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Map elements: ";
    for (const auto& pair : map) {   // auto deduces to std::pair<const std::string, int>
        std::cout << "{" << pair.first << ", " << pair.second << "} ";
    }
    std::cout << std::endl;
}

void demonstrate_auto_deduction_rules() {
    std::cout << "\n📏 AUTO DEDUCTION RULES\n";
    std::cout << std::string(30, '=') << std::endl;
    
    // === AUTO STRIPS REFERENCES AND CV-QUALIFIERS ===
    std::cout << "\n🚫 AUTO STRIPS REFERENCES AND CV-QUALIFIERS\n";
    std::cout << std::string(50, '-') << std::endl;
    
    int x = 42;
    const int cx = x;
    const int& rx = x;
    int* px = &x;
    const int* pcx = &cx;
    
    auto a1 = x;        // int (not int&)
    auto a2 = cx;       // int (not const int)
    auto a3 = rx;       // int (not const int&)
    auto a4 = px;       // int*
    auto a5 = pcx;      // const int*
    
    std::cout << "Original types:" << std::endl;
    std::cout << "x: int, cx: const int, rx: const int&" << std::endl;
    std::cout << "px: int*, pcx: const int*" << std::endl;
    
    std::cout << "\nAuto deduced types:" << std::endl;
    PRINT_TYPE(a1);
    PRINT_TYPE(a2);
    PRINT_TYPE(a3);
    PRINT_TYPE(a4);
    PRINT_TYPE(a5);
    
    // === PRESERVING REFERENCES WITH AUTO& ===
    std::cout << "\n🔗 PRESERVING REFERENCES WITH AUTO&\n";
    std::cout << std::string(40, '-') << std::endl;
    
    auto& ra1 = x;      // int&
    auto& ra2 = cx;     // const int& (const preserved with reference)
    auto& ra3 = rx;     // const int&
    
    PRINT_TYPE(ra1);
    std::cout << "ra2 type: " << demangle(typeid(ra2).name()) << std::endl;
    std::cout << "ra3 type: " << demangle(typeid(ra3).name()) << std::endl;
    
    // === CONST AUTO ===
    std::cout << "\n🔒 CONST AUTO\n";
    std::cout << std::string(15, '-') << std::endl;
    
    const auto ca1 = x;   // const int
    const auto& ca2 = x;  // const int&
    
    std::cout << "ca1 type: " << demangle(typeid(ca1).name()) << std::endl;
    std::cout << "ca2 type: " << demangle(typeid(ca2).name()) << std::endl;
    
    // === AUTO WITH POINTERS ===
    std::cout << "\n👉 AUTO WITH POINTERS\n";
    std::cout << std::string(25, '-') << std::endl;
    
    auto pa1 = &x;        // int*
    auto* pa2 = &x;       // int* (equivalent to above)
    const auto* pa3 = &x; // const int*
    auto* const pa4 = &x; // int* const
    
    PRINT_TYPE(pa1);
    PRINT_TYPE(pa2);
    std::cout << "pa3 type: " << demangle(typeid(pa3).name()) << std::endl;
    std::cout << "pa4 type: " << demangle(typeid(pa4).name()) << std::endl;
    
    // === AUTO WITH ARRAYS ===
    std::cout << "\n📚 AUTO WITH ARRAYS\n";
    std::cout << std::string(20, '-') << std::endl;
    
    int array[5] = {1, 2, 3, 4, 5};
    auto aa1 = array;     // int* (array decays to pointer)
    auto& aa2 = array;    // int (&)[5] (reference preserves array type)
    
    std::cout << "aa1 type: " << demangle(typeid(aa1).name()) << std::endl;
    std::cout << "aa2 type: " << demangle(typeid(aa2).name()) << std::endl;
    std::cout << "sizeof(array): " << sizeof(array) << std::endl;
    std::cout << "sizeof(aa1): " << sizeof(aa1) << std::endl;
    std::cout << "sizeof(aa2): " << sizeof(aa2) << std::endl;
}

void demonstrate_auto_with_functions() {
    std::cout << "\n🔧 AUTO WITH FUNCTIONS\n";
    std::cout << std::string(30, '=') << std::endl;
    
    // === AUTO WITH FUNCTION RETURN TYPES ===
    std::cout << "\n↩️ AUTO WITH FUNCTION RETURN TYPES\n";
    std::cout << std::string(40, '-') << std::endl;
    
    // Lambda with auto return type
    auto lambda1 = [](int x) -> auto { return x * 2; };
    auto lambda2 = [](auto x) { return x * x; }; // Generic lambda
    
    auto result1 = lambda1(5);
    auto result2 = lambda2(3.14);
    auto result3 = lambda2(std::string("test"));
    
    PRINT_TYPE(lambda1);
    PRINT_TYPE(result1);
    PRINT_TYPE(result2);
    PRINT_TYPE(result3);
    
    // === AUTO WITH FUNCTION POINTERS ===
    std::cout << "\n👉 AUTO WITH FUNCTION POINTERS\n";
    std::cout << std::string(35, '-') << std::endl;
    
    auto func_ptr = [](int a, int b) { return a + b; };
    std::function<int(int, int)> std_func = func_ptr;
    
    auto fp1 = &std::abs;         // Function pointer
    auto fp2 = std::sin;          // Function pointer (& optional for functions)
    
    std::cout << "func_ptr type: " << demangle(typeid(func_ptr).name()) << std::endl;
    std::cout << "std_func type: " << demangle(typeid(std_func).name()) << std::endl;
    std::cout << "fp1 type: " << demangle(typeid(fp1).name()) << std::endl;
    std::cout << "fp2 type: " << demangle(typeid(fp2).name()) << std::endl;
    
    // === AUTO WITH MEMBER FUNCTION POINTERS ===
    std::cout << "\n🎯 AUTO WITH MEMBER FUNCTION POINTERS\n";
    std::cout << std::string(45, '-') << std::endl;
    
    auto mem_func = &std::string::size;
    auto mem_var = &std::pair<int, double>::first;
    
    std::cout << "mem_func type: " << demangle(typeid(mem_func).name()) << std::endl;
    std::cout << "mem_var type: " << demangle(typeid(mem_var).name()) << std::endl;
    
    std::string str = "hello";
    std::cout << "String size using member function pointer: " << (str.*mem_func)() << std::endl;
}

// =============================================================================
// DECLTYPE DEMONSTRATIONS
// =============================================================================

void demonstrate_basic_decltype() {
    std::cout << "\n🔍 BASIC DECLTYPE DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === DECLTYPE WITH VARIABLES ===
    std::cout << "\n📝 DECLTYPE WITH VARIABLES\n";
    std::cout << std::string(30, '-') << std::endl;
    
    int x = 42;
    const int cx = 42;
    int& rx = x;
    const int& crx = cx;
    
    decltype(x) dx = 10;        // int
    decltype(cx) dcx = 20;      // const int
    decltype(rx) drx = x;       // int& (requires initialization)
    decltype(crx) dcrx = cx;    // const int& (requires initialization)
    
    std::cout << "decltype(x): " << demangle(typeid(decltype(x)).name()) << std::endl;
    std::cout << "decltype(cx): " << demangle(typeid(decltype(cx)).name()) << std::endl;
    std::cout << "decltype(rx): " << demangle(typeid(decltype(rx)).name()) << std::endl;
    std::cout << "decltype(crx): " << demangle(typeid(decltype(crx)).name()) << std::endl;
    
    // === DECLTYPE WITH EXPRESSIONS ===
    std::cout << "\n🧮 DECLTYPE WITH EXPRESSIONS\n";
    std::cout << std::string(35, '-') << std::endl;
    
    int a = 5, b = 10;
    decltype(a + b) sum = a + b;        // int
    decltype(a * 1.5) product = a * 1.5; // double
    
    PRINT_TYPE(sum);
    PRINT_TYPE(product);
    
    // === DECLTYPE WITH FUNCTION CALLS ===
    std::cout << "\n📞 DECLTYPE WITH FUNCTION CALLS\n";
    std::cout << std::string(35, '-') << std::endl;
    
    auto get_int = []() -> int { return 42; };
    auto get_ref = [&x]() -> int& { return x; };
    
    decltype(get_int()) di = 100;       // int
    decltype(get_ref()) dr = x;         // int& (function returns reference)
    
    std::cout << "decltype(get_int()): " << demangle(typeid(decltype(get_int())).name()) << std::endl;
    std::cout << "decltype(get_ref()): " << demangle(typeid(decltype(get_ref())).name()) << std::endl;
    
    // === DECLTYPE VS AUTO ===
    std::cout << "\n⚖️ DECLTYPE VS AUTO\n";
    std::cout << std::string(25, '-') << std::endl;
    
    const int& cref = x;
    
    auto auto_var = cref;           // int (strips const and reference)
    decltype(cref) decltype_var = x; // const int& (preserves exact type)
    
    std::cout << "auto from const int&: " << demangle(typeid(auto_var).name()) << std::endl;
    std::cout << "decltype from const int&: " << demangle(typeid(decltype_var).name()) << std::endl;
    
    check_same_type<decltype(auto_var), int>("auto_var is int");
    check_same_type<decltype(decltype_var), const int&>("decltype_var is const int&");
}

void demonstrate_decltype_special_cases() {
    std::cout << "\n🎯 DECLTYPE SPECIAL CASES\n";
    std::cout << std::string(30, '=') << std::endl;
    
    // === DECLTYPE WITH PARENTHESES ===
    std::cout << "\n🔲 DECLTYPE WITH PARENTHESES\n";
    std::cout << std::string(35, '-') << std::endl;
    
    int x = 42;
    
    decltype(x) dx1 = 10;        // int (variable name)
    decltype((x)) dx2 = x;       // int& (parenthesized expression is lvalue)
    
    std::cout << "decltype(x): " << demangle(typeid(decltype(x)).name()) << std::endl;
    std::cout << "decltype((x)): " << demangle(typeid(decltype((x))).name()) << std::endl;
    
    // === DECLTYPE WITH ARRAY INDEXING ===
    std::cout << "\n📚 DECLTYPE WITH ARRAY INDEXING\n";
    std::cout << std::string(40, '-') << std::endl;
    
    int arr[5] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {10, 20, 30};
    
    decltype(arr[0]) arr_elem = arr[0];    // int& (array indexing returns lvalue)
    decltype(vec[0]) vec_elem = vec[0];    // int& (vector indexing returns lvalue)
    
    std::cout << "decltype(arr[0]): " << demangle(typeid(decltype(arr[0])).name()) << std::endl;
    std::cout << "decltype(vec[0]): " << demangle(typeid(decltype(vec[0])).name()) << std::endl;
    
    // === DECLTYPE WITH MEMBER ACCESS ===
    std::cout << "\n🎯 DECLTYPE WITH MEMBER ACCESS\n";
    std::cout << std::string(35, '-') << std::endl;
    
    struct Point { int x, y; };
    Point p{10, 20};
    Point* pp = &p;
    
    decltype(p.x) member1 = 5;      // int& (member access on lvalue)
    decltype(pp->x) member2 = 6;    // int& (member access through pointer)
    decltype(Point{}.x) member3 = 7; // int (member access on rvalue)
    
    std::cout << "decltype(p.x): " << demangle(typeid(decltype(p.x)).name()) << std::endl;
    std::cout << "decltype(pp->x): " << demangle(typeid(decltype(pp->x)).name()) << std::endl;
    std::cout << "decltype(Point{}.x): " << demangle(typeid(decltype(Point{}.x)).name()) << std::endl;
}

// =============================================================================
// DECLTYPE(AUTO) DEMONSTRATIONS
// =============================================================================

template<typename Container>
auto get_element_auto(Container&& c, std::size_t index) -> auto {
    return std::forward<Container>(c)[index];
}

template<typename Container>
decltype(auto) get_element_decltype_auto(Container&& c, std::size_t index) {
    return std::forward<Container>(c)[index];
}

void demonstrate_decltype_auto() {
    std::cout << "\n🔄 DECLTYPE(AUTO) DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === DECLTYPE(AUTO) FOR PERFECT FORWARDING ===
    std::cout << "\n🎯 DECLTYPE(AUTO) FOR PERFECT FORWARDING\n";
    std::cout << std::string(50, '-') << std::endl;
    
    std::vector<int> vec = {1, 2, 3, 4, 5};
    const std::vector<int> cvec = {10, 20, 30, 40, 50};
    
    // Using auto (always copies or strips references)
    auto elem1 = get_element_auto(vec, 0);          // int (copy)
    auto elem2 = get_element_auto(cvec, 0);         // int (copy)
    
    // Using decltype(auto) (preserves exact return type)
    decltype(auto) elem3 = get_element_decltype_auto(vec, 0);   // int& (reference)
    decltype(auto) elem4 = get_element_decltype_auto(cvec, 0);  // const int& (const ref)
    
    std::cout << "get_element_auto from vector: " << demangle(typeid(elem1).name()) << std::endl;
    std::cout << "get_element_auto from const vector: " << demangle(typeid(elem2).name()) << std::endl;
    std::cout << "get_element_decltype_auto from vector: " << demangle(typeid(elem3).name()) << std::endl;
    std::cout << "get_element_decltype_auto from const vector: " << demangle(typeid(elem4).name()) << std::endl;
    
    // Demonstrate the difference
    elem3 = 999;  // This modifies the original vector element
    // elem4 = 888;  // This would be a compile error (const)
    
    std::cout << "Modified vec[0]: " << vec[0] << std::endl;
    
    // === DECLTYPE(AUTO) PITFALL ===
    std::cout << "\n⚠️ DECLTYPE(AUTO) PITFALL\n";
    std::cout << std::string(30, '-') << std::endl;
    
    auto get_value = []() -> decltype(auto) {
        int x = 42;
        return x;        // Returns int (correct)
        // return (x);   // Would return int& - DANGEROUS! (dangling reference)
    };
    
    auto safe_value = get_value();
    std::cout << "Safe value: " << safe_value << std::endl;
    std::cout << "Type: " << demangle(typeid(safe_value).name()) << std::endl;
}

// =============================================================================
// CLASS TEMPLATE ARGUMENT DEDUCTION (CTAD)
// =============================================================================

void demonstrate_ctad() {
    std::cout << "\n🏗️ CLASS TEMPLATE ARGUMENT DEDUCTION (CTAD)\n";
    std::cout << std::string(50, '=') << std::endl;
    
    // === BASIC CTAD ===
    std::cout << "\n📝 BASIC CTAD (C++17)\n";
    std::cout << std::string(25, '-') << std::endl;
    
    // Before C++17 (explicit template arguments)
    std::vector<int> old_style_vec{1, 2, 3, 4, 5};
    std::pair<int, std::string> old_style_pair{42, "hello"};
    
    // C++17 CTAD (template arguments deduced)
    std::vector new_style_vec{1, 2, 3, 4, 5};        // std::vector<int>
    std::pair new_style_pair{42, "hello"};           // std::pair<int, const char*>
    std::array new_style_array{1.1, 2.2, 3.3};      // std::array<double, 3>
    
    std::cout << "new_style_vec type: " << demangle(typeid(new_style_vec).name()) << std::endl;
    std::cout << "new_style_pair type: " << demangle(typeid(new_style_pair).name()) << std::endl;
    std::cout << "new_style_array type: " << demangle(typeid(new_style_array).name()) << std::endl;
    
    // === CTAD WITH CUSTOM CLASSES ===
    std::cout << "\n🎨 CTAD WITH CUSTOM CLASSES\n";
    std::cout << std::string(35, '-') << std::endl;
    
    // Custom class with deduction guide
    template<typename T>
    class MyContainer {
    public:
        MyContainer(std::initializer_list<T> list) : data(list) {}
        void print() const {
            for (const auto& item : data) {
                std::cout << item << " ";
            }
            std::cout << std::endl;
        }
    private:
        std::vector<T> data;
    };
    
    // CTAD automatically deduces T
    MyContainer container{1, 2, 3, 4, 5};    // MyContainer<int>
    MyContainer str_container{"a", "b", "c"}; // MyContainer<const char*>
    
    std::cout << "container type: " << demangle(typeid(container).name()) << std::endl;
    std::cout << "str_container type: " << demangle(typeid(str_container).name()) << std::endl;
    
    container.print();
    str_container.print();
}

// =============================================================================
// ADVANCED TYPE DEDUCTION SCENARIOS
// =============================================================================

void demonstrate_advanced_scenarios() {
    std::cout << "\n🚀 ADVANCED TYPE DEDUCTION SCENARIOS\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === AUTO WITH TRAILING RETURN TYPE ===
    std::cout << "\n↩️ AUTO WITH TRAILING RETURN TYPE\n";
    std::cout << std::string(40, '-') << std::endl;
    
    auto multiply = [](auto a, auto b) -> decltype(a * b) {
        return a * b;
    };
    
    auto result1 = multiply(5, 10);         // int
    auto result2 = multiply(3.14, 2);       // double
    auto result3 = multiply(2.5f, 4.0);     // double
    
    PRINT_TYPE(result1);
    PRINT_TYPE(result2);
    PRINT_TYPE(result3);
    
    // === AUTO WITH STRUCTURED BINDINGS (C++17) ===
    std::cout << "\n📦 AUTO WITH STRUCTURED BINDINGS (C++17)\n";
    std::cout << std::string(45, '-') << std::endl;
    
    std::map<std::string, int> score_map = {{"Alice", 95}, {"Bob", 87}};
    
    // Traditional way
    for (const auto& pair : score_map) {
        std::cout << "Traditional: " << pair.first << " = " << pair.second << std::endl;
    }
    
    // Structured bindings
    for (const auto& [name, score] : score_map) {
        std::cout << "Structured binding: " << name << " = " << score << std::endl;
    }
    
    // With std::tuple
    auto tuple_data = std::make_tuple(42, 3.14, "hello");
    auto [num, pi, str] = tuple_data;
    
    PRINT_TYPE(num);
    PRINT_TYPE(pi);
    PRINT_TYPE(str);
    
    // === AUTO WITH LAMBDAS AND CLOSURES ===
    std::cout << "\n🔧 AUTO WITH LAMBDAS AND CLOSURES\n";
    std::cout << std::string(40, '-') << std::endl;
    
    int capture_value = 100;
    
    auto lambda_by_value = [capture_value](int x) {
        return capture_value + x;
    };
    
    auto lambda_by_ref = [&capture_value](int x) {
        capture_value += x;
        return capture_value;
    };
    
    auto generic_lambda = [](auto x, auto y) {
        return x + y;
    };
    
    std::cout << "lambda_by_value type: " << demangle(typeid(lambda_by_value).name()) << std::endl;
    std::cout << "lambda_by_ref type: " << demangle(typeid(lambda_by_ref).name()) << std::endl;
    std::cout << "generic_lambda type: " << demangle(typeid(generic_lambda).name()) << std::endl;
    
    std::cout << "lambda_by_value(5): " << lambda_by_value(5) << std::endl;
    std::cout << "lambda_by_ref(10): " << lambda_by_ref(10) << std::endl;
    std::cout << "capture_value after lambda_by_ref: " << capture_value << std::endl;
    
    auto generic_result1 = generic_lambda(1, 2);
    auto generic_result2 = generic_lambda(1.5, 2.7);
    auto generic_result3 = generic_lambda(std::string("Hello "), std::string("World"));
    
    PRINT_TYPE(generic_result1);
    PRINT_TYPE(generic_result2);
    PRINT_TYPE(generic_result3);
}

// =============================================================================
// COMMON PITFALLS AND DEBUGGING
// =============================================================================

void demonstrate_common_pitfalls() {
    std::cout << "\n⚠️ COMMON PITFALLS AND DEBUGGING\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === PITFALL 1: AUTO WITH INITIALIZER LISTS ===
    std::cout << "\n🚫 PITFALL 1: AUTO WITH INITIALIZER LISTS\n";
    std::cout << std::string(45, '-') << std::endl;
    
    auto list1 = {1, 2, 3};           // std::initializer_list<int>
    // auto list2 = {1, 2.0, 3};      // ERROR: mixed types
    auto list3{42};                   // int (single element)
    auto list4 = {42};                // std::initializer_list<int>
    
    std::cout << "auto list1 = {1, 2, 3}: " << demangle(typeid(list1).name()) << std::endl;
    std::cout << "auto list3{42}: " << demangle(typeid(list3).name()) << std::endl;
    std::cout << "auto list4 = {42}: " << demangle(typeid(list4).name()) << std::endl;
    
    // === PITFALL 2: AUTO WITH PROXY OBJECTS ===
    std::cout << "\n🎭 PITFALL 2: AUTO WITH PROXY OBJECTS\n";
    std::cout << std::string(40, '-') << std::endl;
    
    std::vector<bool> bool_vec = {true, false, true};
    
    auto bool_elem1 = bool_vec[0];     // std::vector<bool>::reference (proxy)
    bool bool_elem2 = bool_vec[0];     // bool (actual value)
    
    std::cout << "auto from vector<bool>: " << demangle(typeid(bool_elem1).name()) << std::endl;
    std::cout << "bool from vector<bool>: " << demangle(typeid(bool_elem2).name()) << std::endl;
    
    // === PITFALL 3: DANGLING REFERENCES ===
    std::cout << "\n💀 PITFALL 3: DANGLING REFERENCES\n";
    std::cout << std::string(35, '-') << std::endl;
    
    auto get_temp_ref = []() -> const std::string& {
        // return std::string("temporary");  // DANGEROUS: returns reference to temporary
        static std::string safe_string = "safe";
        return safe_string;  // SAFE: returns reference to static
    };
    
    const auto& safe_ref = get_temp_ref();
    std::cout << "Safe reference: " << safe_ref << std::endl;
    
    // === DEBUGGING TECHNIQUES ===
    std::cout << "\n🔍 DEBUGGING TECHNIQUES\n";
    std::cout << std::string(25, '-') << std::endl;
    
    // Technique 1: Force compilation error to see type
    auto mystery_var = std::make_unique<int>(42);
    // mystery_var = nullptr;  // Uncomment to see error message with type
    
    std::cout << "mystery_var type: " << demangle(typeid(mystery_var).name()) << std::endl;
    
    // Technique 2: Use type traits for compile-time checks
    static_assert(std::is_same_v<decltype(mystery_var), std::unique_ptr<int>>, 
                  "mystery_var should be std::unique_ptr<int>");
    
    std::cout << "Type assertion passed!" << std::endl;
}

// Main function demonstrating all concepts
int main() {
    std::cout << "🚀 C++ AUTO AND DECLTYPE - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(65, '=') << std::endl;
    
    try {
        // Run all demonstrations
        demonstrate_basic_auto();
        demonstrate_auto_deduction_rules();
        demonstrate_auto_with_functions();
        demonstrate_basic_decltype();
        demonstrate_decltype_special_cases();
        demonstrate_decltype_auto();
        demonstrate_ctad();
        demonstrate_advanced_scenarios();
        demonstrate_common_pitfalls();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
        // === DEBUGGING TIPS ===
        std::cout << "\n🐛 DEBUGGING TIPS:\n";
        std::cout << "1. Use typeid().name() to inspect deduced types\n";
        std::cout << "2. Force compilation errors to see type information\n";
        std::cout << "3. Use static_assert with std::is_same_v for type checking\n";
        std::cout << "4. Be careful with auto and initializer lists\n";
        std::cout << "5. Watch out for proxy objects and reference types\n";
        std::cout << "6. Use decltype(auto) for perfect forwarding\n";
        
        std::cout << "\n📚 UNDERSTANDING POINTS:\n";
        std::cout << "1. auto strips references and cv-qualifiers by default\n";
        std::cout << "2. decltype preserves the exact type of expressions\n";
        std::cout << "3. decltype(auto) enables perfect return type forwarding\n";
        std::cout << "4. CTAD (C++17) reduces template argument verbosity\n";
        std::cout << "5. Parentheses in decltype change the result type\n";
        std::cout << "6. Use auto& or const auto& to preserve references\n";
        std::cout << "7. Be cautious with auto and proxy objects\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
