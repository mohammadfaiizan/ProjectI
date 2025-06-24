/*
 * C++ LATEST AND FUTURE - CONCEPTS AND CONSTRAINTS
 * 
 * Comprehensive guide to C++20 concepts including concept definitions,
 * constraints, requires expressions, and practical applications.
 * 
 * Compilation: g++ -std=c++20 -Wall -Wextra concepts_constraints.cpp -o concepts_demo
 */

#include <iostream>
#include <vector>
#include <string>
#include <concepts>
#include <type_traits>
#include <iterator>
#include <algorithm>
#include <memory>

// =============================================================================
// BASIC CONCEPT DEFINITIONS
// =============================================================================

// Simple type concepts
template<typename T>
concept Integral = std::is_integral_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template<typename T>
concept Arithmetic = Integral<T> || FloatingPoint<T>;

template<typename T>
concept Pointer = std::is_pointer_v<T>;

// Demonstrate basic concepts
void demonstrate_basic_concepts() {
    std::cout << "\n=== Basic Concept Demonstrations ===" << std::endl;
    
    auto test_concept = [](auto value, const std::string& type_name) {
        using T = decltype(value);
        std::cout << type_name << ": ";
        std::cout << "Integral=" << Integral<T> << ", ";
        std::cout << "FloatingPoint=" << FloatingPoint<T> << ", ";
        std::cout << "Arithmetic=" << Arithmetic<T> << ", ";
        std::cout << "Pointer=" << Pointer<T> << std::endl;
    };
    
    test_concept(42, "int");
    test_concept(3.14, "double");
    test_concept("hello", "const char*");
    test_concept(std::string{}, "std::string");
}

// =============================================================================
// ADVANCED CONCEPT DEFINITIONS
// =============================================================================

// Container concepts
template<typename T>
concept Container = requires(T t) {
    typename T::value_type;
    typename T::iterator;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.size() } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept Iterable = requires(T t) {
    { t.begin() } -> std::input_or_output_iterator;
    { t.end() } -> std::input_or_output_iterator;
};

template<typename T>
concept Printable = requires(T t) {
    std::cout << t;
};

// Function object concepts
template<typename F, typename... Args>
concept Callable = requires(F f, Args... args) {
    f(args...);
};

template<typename F, typename T>
concept Predicate = requires(F f, T t) {
    { f(t) } -> std::convertible_to<bool>;
};

// Mathematical concepts
template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
};

template<typename T>
concept Multipliable = requires(T a, T b) {
    { a * b } -> std::same_as<T>;
};

template<typename T>
concept Numeric = Arithmetic<T> && Addable<T> && Multipliable<T>;

// =============================================================================
// REQUIRES EXPRESSIONS AND CLAUSES
// =============================================================================

// Complex requires expressions
template<typename T>
concept Serializable = requires(T t) {
    // Simple requirement
    { t.serialize() } -> std::convertible_to<std::string>;
    
    // Type requirement
    typename T::SerializationFormat;
    
    // Compound requirement
    { T::deserialize(std::string{}) } -> std::same_as<T>;
    
    // Nested requirement
    requires std::is_default_constructible_v<T>;
};

template<typename T>
concept Comparable = requires(T a, T b) {
    { a == b } -> std::convertible_to<bool>;
    { a != b } -> std::convertible_to<bool>;
    { a < b } -> std::convertible_to<bool>;
    { a <= b } -> std::convertible_to<bool>;
    { a > b } -> std::convertible_to<bool>;
    { a >= b } -> std::convertible_to<bool>;
};

template<typename T>
concept Sortable = Comparable<T> && requires(std::vector<T> v) {
    std::sort(v.begin(), v.end());
};

// =============================================================================
// CONCEPT-CONSTRAINED FUNCTIONS
// =============================================================================

// Function templates with concept constraints
template<Numeric T>
T add_numbers(T a, T b) {
    std::cout << "Adding numeric values: " << a << " + " << b << " = ";
    T result = a + b;
    std::cout << result << std::endl;
    return result;
}

template<Container C>
void print_container(const C& container) {
    std::cout << "Container contents: ";
    for (const auto& item : container) {
        if constexpr (Printable<typename C::value_type>) {
            std::cout << item << " ";
        } else {
            std::cout << "[item] ";
        }
    }
    std::cout << "(size: " << container.size() << ")" << std::endl;
}

template<Iterable I, Predicate<typename I::value_type> P>
auto count_if_concept(const I& iterable, P predicate) {
    std::cout << "Counting elements matching predicate..." << std::endl;
    return std::count_if(iterable.begin(), iterable.end(), predicate);
}

// Alternative syntax with requires clause
template<typename T>
requires Sortable<T>
void sort_and_print(std::vector<T> values) {
    std::cout << "Before sorting: ";
    for (const auto& val : values) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::sort(values.begin(), values.end());
    
    std::cout << "After sorting: ";
    for (const auto& val : values) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// =============================================================================
// CONCEPT COMPOSITION AND REFINEMENT
// =============================================================================

// Concept composition with logical operators
template<typename T>
concept SignedIntegral = Integral<T> && std::is_signed_v<T>;

template<typename T>
concept UnsignedIntegral = Integral<T> && std::is_unsigned_v<T>;

template<typename T>
concept NumericContainer = Container<T> && Numeric<typename T::value_type>;

// Concept refinement hierarchy
template<typename T>
concept Shape = requires(T t) {
    { t.area() } -> std::convertible_to<double>;
    { t.perimeter() } -> std::convertible_to<double>;
};

template<typename T>
concept ColoredShape = Shape<T> && requires(T t) {
    { t.get_color() } -> std::convertible_to<std::string>;
    t.set_color(std::string{});
};

template<typename T>
concept DrawableShape = ColoredShape<T> && requires(T t) {
    t.draw();
    { t.is_visible() } -> std::convertible_to<bool>;
};

// =============================================================================
// PRACTICAL EXAMPLES WITH CONCEPTS
// =============================================================================

// Example classes that satisfy concepts
class Circle {
private:
    double radius_;
    std::string color_ = "red";
    bool visible_ = true;
    
public:
    Circle(double r) : radius_(r) {}
    
    double area() const { return 3.14159 * radius_ * radius_; }
    double perimeter() const { return 2 * 3.14159 * radius_; }
    
    std::string get_color() const { return color_; }
    void set_color(const std::string& color) { color_ = color; }
    
    void draw() const {
        std::cout << "Drawing circle with radius " << radius_ 
                  << " and color " << color_ << std::endl;
    }
    
    bool is_visible() const { return visible_; }
};

class Rectangle {
private:
    double width_, height_;
    std::string color_ = "blue";
    bool visible_ = true;
    
public:
    Rectangle(double w, double h) : width_(w), height_(h) {}
    
    double area() const { return width_ * height_; }
    double perimeter() const { return 2 * (width_ + height_); }
    
    std::string get_color() const { return color_; }
    void set_color(const std::string& color) { color_ = color; }
    
    void draw() const {
        std::cout << "Drawing rectangle " << width_ << "x" << height_ 
                  << " with color " << color_ << std::endl;
    }
    
    bool is_visible() const { return visible_; }
};

// Generic algorithms using concepts
template<DrawableShape T>
void render_shape(const T& shape) {
    if (shape.is_visible()) {
        std::cout << "Rendering shape (area: " << shape.area() << "): ";
        shape.draw();
    } else {
        std::cout << "Shape is not visible, skipping render" << std::endl;
    }
}

template<Shape T>
double calculate_total_area(const std::vector<T>& shapes) {
    double total = 0.0;
    for (const auto& shape : shapes) {
        total += shape.area();
    }
    std::cout << "Total area of " << shapes.size() << " shapes: " << total << std::endl;
    return total;
}

// =============================================================================
// CONCEPT-BASED FUNCTION OVERLOADING
// =============================================================================

// Different implementations based on concepts
template<SignedIntegral T>
void process_number(T value) {
    std::cout << "Processing signed integer: " << value << std::endl;
    if (value < 0) {
        std::cout << "  Negative value detected" << std::endl;
    }
}

template<UnsignedIntegral T>
void process_number(T value) {
    std::cout << "Processing unsigned integer: " << value << std::endl;
    std::cout << "  Always non-negative" << std::endl;
}

template<FloatingPoint T>
void process_number(T value) {
    std::cout << "Processing floating-point: " << value << std::endl;
    if (value != static_cast<long long>(value)) {
        std::cout << "  Has fractional part" << std::endl;
    }
}

// =============================================================================
// ADVANCED CONCEPT TECHNIQUES
// =============================================================================

// Concept with multiple template parameters
template<typename T, typename U>
concept ConvertibleTo = std::convertible_to<T, U>;

template<typename Container, typename Value>
concept ContainerOf = Container<Container> && 
                     std::same_as<typename Container::value_type, Value>;

// Concept using SFINAE-like techniques
template<typename T>
concept HasToString = requires(T t) {
    { t.to_string() } -> std::convertible_to<std::string>;
} || requires(T t) {
    { std::to_string(t) } -> std::convertible_to<std::string>;
};

// Generic to_string function using concepts
template<HasToString T>
std::string generic_to_string(const T& value) {
    if constexpr (requires { value.to_string(); }) {
        return value.to_string();
    } else {
        return std::to_string(value);
    }
}

// =============================================================================
// CONCEPT SPECIALIZATION AND CUSTOMIZATION
// =============================================================================

// Customization point for different types
template<typename T>
concept Streamable = requires(T t) {
    std::cout << t;
};

template<typename T>
void debug_print(const T& value) requires Streamable<T> {
    std::cout << "Debug: " << value << std::endl;
}

template<typename T>
void debug_print(const T& value) requires (!Streamable<T>) {
    std::cout << "Debug: [non-streamable object of type " 
              << typeid(T).name() << "]" << std::endl;
}

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_concept_constraints() {
    std::cout << "\n=== Concept-Constrained Functions ===" << std::endl;
    
    // Numeric operations
    add_numbers(10, 20);
    add_numbers(3.14, 2.86);
    
    // Container operations
    std::vector<int> numbers{1, 2, 3, 4, 5};
    std::vector<std::string> words{"hello", "world", "concepts"};
    
    print_container(numbers);
    print_container(words);
    
    // Predicate operations
    auto even_count = count_if_concept(numbers, [](int n) { return n % 2 == 0; });
    std::cout << "Even numbers count: " << even_count << std::endl;
    
    // Sorting
    std::vector<int> unsorted{5, 2, 8, 1, 9};
    sort_and_print(unsorted);
}

void demonstrate_shape_concepts() {
    std::cout << "\n=== Shape Concept Hierarchy ===" << std::endl;
    
    Circle circle(5.0);
    Rectangle rect(10.0, 8.0);
    
    // Test concept satisfaction
    static_assert(Shape<Circle>);
    static_assert(ColoredShape<Circle>);
    static_assert(DrawableShape<Circle>);
    
    std::cout << "Concept checks passed for Circle" << std::endl;
    
    // Use concept-constrained functions
    render_shape(circle);
    render_shape(rect);
    
    std::vector<Circle> circles{Circle(1), Circle(2), Circle(3)};
    calculate_total_area(circles);
}

void demonstrate_concept_overloading() {
    std::cout << "\n=== Concept-Based Overloading ===" << std::endl;
    
    process_number(42);        // signed int
    process_number(-17);       // signed int
    process_number(42u);       // unsigned int
    process_number(3.14);      // double
    process_number(2.0f);      // float
}

void demonstrate_advanced_concepts() {
    std::cout << "\n=== Advanced Concept Techniques ===" << std::endl;
    
    // HasToString concept
    std::cout << "String conversion: " << generic_to_string(42) << std::endl;
    std::cout << "String conversion: " << generic_to_string(3.14) << std::endl;
    
    // Debug printing with concept specialization
    debug_print(123);
    debug_print("hello");
    
    struct NonStreamable {};
    NonStreamable obj;
    debug_print(obj);
}

void demonstrate_concept_benefits() {
    std::cout << "\n=== Concept System Benefits ===" << std::endl;
    
    std::cout << "1. Better Error Messages:" << std::endl;
    std::cout << "   • Clear constraint violations" << std::endl;
    std::cout << "   • Precise template instantiation errors" << std::endl;
    
    std::cout << "\n2. Self-Documenting Code:" << std::endl;
    std::cout << "   • Function signatures express requirements" << std::endl;
    std::cout << "   • Concepts serve as documentation" << std::endl;
    
    std::cout << "\n3. Improved Overload Resolution:" << std::endl;
    std::cout << "   • More precise function selection" << std::endl;
    std::cout << "   • Eliminates SFINAE complexity" << std::endl;
    
    std::cout << "\n4. Template Specialization:" << std::endl;
    std::cout << "   • Cleaner than enable_if patterns" << std::endl;
    std::cout << "   • More maintainable code" << std::endl;
}

int main() {
    std::cout << "C++20 CONCEPTS AND CONSTRAINTS\n";
    std::cout << "==============================\n";
    
    try {
        demonstrate_basic_concepts();
        demonstrate_concept_constraints();
        demonstrate_shape_concepts();
        demonstrate_concept_overloading();
        demonstrate_advanced_concepts();
        demonstrate_concept_benefits();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nKey Concept Features:" << std::endl;
    std::cout << "• Type constraints with readable syntax" << std::endl;
    std::cout << "• Improved template error messages" << std::endl;
    std::cout << "• Function overloading based on concepts" << std::endl;
    std::cout << "• Concept composition and refinement" << std::endl;
    std::cout << "• Requires expressions for complex constraints" << std::endl;
    std::cout << "• Better template specialization control" << std::endl;
    
    return 0;
}

/*
CONCEPT SYNTAX SUMMARY:

1. Basic Concept Definition:
template<typename T>
concept MyConstraint = <constraint-expression>;

2. Requires Expression:
template<typename T>
concept MyConstraint = requires(T t) {
    // Simple requirement
    t.method();
    
    // Type requirement
    typename T::type;
    
    // Compound requirement
    { t.function() } -> std::same_as<int>;
    
    // Nested requirement
    requires std::is_integral_v<T>;
};

3. Function Constraints:
// Concept as template parameter
template<MyConstraint T>
void function(T t);

// Requires clause
template<typename T>
requires MyConstraint<T>
void function(T t);

// Abbreviated function template
void function(MyConstraint auto t);

4. Concept Composition:
template<typename T>
concept Combined = Concept1<T> && Concept2<T>;

template<typename T>
concept Either = Concept1<T> || Concept2<T>;
*/ 