/*
 * =============================================================================
 * POLYMORPHISM AND VTABLES - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Runtime polymorphism with virtual functions
 * 2. Virtual function tables (vtables) and vpointers
 * 3. Pure virtual functions and abstract classes
 * 4. Virtual destructors and polymorphic destruction
 * 5. Performance implications of virtual functions
 * 6. Function overriding vs overloading vs hiding
 * 7. Best practices and debugging techniques
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 polymorphism_and_vtables.cpp -o polymorphism_and_vtables
 * =============================================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <typeinfo>
#include <chrono>

#define TRACE(msg) std::cout << "[TRACE] " << msg << std::endl;

// =============================================================================
// BASIC POLYMORPHISM HIERARCHY
// =============================================================================

class Shape {
protected:
    std::string name;
    double x, y;  // Position
    
public:
    Shape(const std::string& n, double x_pos = 0, double y_pos = 0) 
        : name(n), x(x_pos), y(y_pos) {
        TRACE("Shape '" + name + "' created at (" + std::to_string(x) + ", " + std::to_string(y) + ")");
    }
    
    virtual ~Shape() {
        TRACE("Shape '" + name + "' destroyed");
    }
    
    // Pure virtual functions (abstract interface)
    virtual double area() const = 0;
    virtual double perimeter() const = 0;
    virtual void draw() const = 0;
    
    // Virtual function with default implementation
    virtual void move(double dx, double dy) {
        x += dx;
        y += dy;
        std::cout << name << " moved to (" << x << ", " << y << ")" << std::endl;
    }
    
    // Virtual function for type identification
    virtual std::string get_type() const { return "Shape"; }
    
    // Non-virtual functions
    const std::string& get_name() const { return name; }
    double get_x() const { return x; }
    double get_y() const { return y; }
    
    // Virtual function to demonstrate vtable
    virtual void print_info() const {
        std::cout << "Shape: " << name << " at (" << x << ", " << y << ")" << std::endl;
    }
};

class Rectangle : public Shape {
protected:
    double width, height;
    
public:
    Rectangle(const std::string& n, double w, double h, double x_pos = 0, double y_pos = 0) 
        : Shape(n, x_pos, y_pos), width(w), height(h) {
        TRACE("Rectangle '" + name + "' created: " + std::to_string(w) + "x" + std::to_string(h));
    }
    
    ~Rectangle() override {
        TRACE("Rectangle '" + name + "' destroyed");
    }
    
    // Implement pure virtual functions
    double area() const override {
        return width * height;
    }
    
    double perimeter() const override {
        return 2 * (width + height);
    }
    
    void draw() const override {
        std::cout << "Drawing rectangle '" << name << "' (" << width << "x" << height << ")" << std::endl;
        for (int i = 0; i < static_cast<int>(height); ++i) {
            for (int j = 0; j < static_cast<int>(width); ++j) {
                std::cout << "* ";
            }
            std::cout << std::endl;
        }
    }
    
    // Override virtual functions
    std::string get_type() const override { return "Rectangle"; }
    
    void print_info() const override {
        Shape::print_info();  // Call base implementation
        std::cout << "  Dimensions: " << width << "x" << height << std::endl;
        std::cout << "  Area: " << area() << ", Perimeter: " << perimeter() << std::endl;
    }
    
    // Rectangle-specific methods
    double get_width() const { return width; }
    double get_height() const { return height; }
    
    void resize(double new_width, double new_height) {
        width = new_width;
        height = new_height;
        std::cout << name << " resized to " << width << "x" << height << std::endl;
    }
};

class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(const std::string& n, double r, double x_pos = 0, double y_pos = 0) 
        : Shape(n, x_pos, y_pos), radius(r) {
        TRACE("Circle '" + name + "' created with radius " + std::to_string(r));
    }
    
    ~Circle() override {
        TRACE("Circle '" + name + "' destroyed");
    }
    
    // Implement pure virtual functions
    double area() const override {
        return 3.14159 * radius * radius;
    }
    
    double perimeter() const override {
        return 2 * 3.14159 * radius;
    }
    
    void draw() const override {
        std::cout << "Drawing circle '" << name << "' with radius " << radius << std::endl;
        int size = static_cast<int>(2 * radius);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                double dx = j - radius;
                double dy = i - radius;
                if (dx*dx + dy*dy <= radius*radius) {
                    std::cout << "* ";
                } else {
                    std::cout << "  ";
                }
            }
            std::cout << std::endl;
        }
    }
    
    // Override virtual functions
    std::string get_type() const override { return "Circle"; }
    
    void print_info() const override {
        Shape::print_info();
        std::cout << "  Radius: " << radius << std::endl;
        std::cout << "  Area: " << area() << ", Circumference: " << perimeter() << std::endl;
    }
    
    // Circle-specific methods
    double get_radius() const { return radius; }
    
    void set_radius(double new_radius) {
        radius = new_radius;
        std::cout << name << " radius changed to " << radius << std::endl;
    }
};

class Square : public Rectangle {
public:
    Square(const std::string& n, double side, double x_pos = 0, double y_pos = 0) 
        : Rectangle(n, side, side, x_pos, y_pos) {
        TRACE("Square '" + name + "' created with side " + std::to_string(side));
    }
    
    ~Square() override {
        TRACE("Square '" + name + "' destroyed");
    }
    
    // Override to maintain square property
    void resize(double new_side) {
        Rectangle::resize(new_side, new_side);
    }
    
    // Hide the base class resize method
    void resize(double, double) = delete;
    
    std::string get_type() const override { return "Square"; }
    
    void print_info() const override {
        Shape::print_info();
        std::cout << "  Side length: " << width << std::endl;
        std::cout << "  Area: " << area() << ", Perimeter: " << perimeter() << std::endl;
    }
    
    double get_side() const { return width; }
};

// =============================================================================
// VTABLE DEMONSTRATION CLASSES
// =============================================================================

class Base {
public:
    Base() { TRACE("Base constructor"); }
    virtual ~Base() { TRACE("Base destructor"); }
    
    virtual void func1() const { std::cout << "Base::func1()" << std::endl; }
    virtual void func2() const { std::cout << "Base::func2()" << std::endl; }
    virtual void func3() const { std::cout << "Base::func3()" << std::endl; }
    
    void non_virtual_func() const { std::cout << "Base::non_virtual_func()" << std::endl; }
    
    // Function to show vtable pointer
    void show_vtable_info() const {
        std::cout << "Object address: " << this << std::endl;
        std::cout << "Type: " << typeid(*this).name() << std::endl;
        // vtable pointer is typically at the beginning of the object
        void** vtable_ptr = *reinterpret_cast<void*** const>(this);
        std::cout << "VTable address: " << vtable_ptr << std::endl;
    }
};

class Derived1 : public Base {
public:
    Derived1() { TRACE("Derived1 constructor"); }
    ~Derived1() override { TRACE("Derived1 destructor"); }
    
    void func1() const override { std::cout << "Derived1::func1()" << std::endl; }
    void func2() const override { std::cout << "Derived1::func2()" << std::endl; }
    // func3 not overridden - uses Base::func3
    
    // New virtual function
    virtual void derived1_specific() const { std::cout << "Derived1::derived1_specific()" << std::endl; }
};

class Derived2 : public Base {
public:
    Derived2() { TRACE("Derived2 constructor"); }
    ~Derived2() override { TRACE("Derived2 destructor"); }
    
    void func1() const override { std::cout << "Derived2::func1()" << std::endl; }
    // func2 not overridden - uses Base::func2
    void func3() const override { std::cout << "Derived2::func3()" << std::endl; }
    
    // New virtual function
    virtual void derived2_specific() const { std::cout << "Derived2::derived2_specific()" << std::endl; }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_basic_polymorphism() {
    std::cout << "\n🎭 BASIC POLYMORPHISM DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n📋 Creating shapes polymorphically...\n";
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Rectangle>("Rect1", 4, 3));
    shapes.push_back(std::make_unique<Circle>("Circle1", 2.5));
    shapes.push_back(std::make_unique<Square>("Square1", 3));
    shapes.push_back(std::make_unique<Rectangle>("Rect2", 5, 2, 10, 5));
    
    std::cout << "\n🔍 Polymorphic operations:\n";
    for (const auto& shape : shapes) {
        std::cout << "\n--- " << shape->get_type() << " ---" << std::endl;
        shape->print_info();
        shape->draw();
        std::cout << "Area: " << shape->area() << std::endl;
        std::cout << "Perimeter: " << shape->perimeter() << std::endl;
        shape->move(1, 1);
    }
    
    std::cout << "\n📊 Shape statistics:\n";
    double total_area = 0;
    for (const auto& shape : shapes) {
        total_area += shape->area();
    }
    std::cout << "Total area of all shapes: " << total_area << std::endl;
}

void demonstrate_vtable_behavior() {
    std::cout << "\n🔧 VTABLE BEHAVIOR DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    std::cout << "\n📋 Creating objects with different vtables...\n";
    Base base_obj;
    Derived1 derived1_obj;
    Derived2 derived2_obj;
    
    std::cout << "\n🔍 VTable information:\n";
    std::cout << "Base object:\n";
    base_obj.show_vtable_info();
    
    std::cout << "\nDerived1 object:\n";
    derived1_obj.show_vtable_info();
    
    std::cout << "\nDerived2 object:\n";
    derived2_obj.show_vtable_info();
    
    std::cout << "\n🔍 Virtual function calls through base pointers:\n";
    std::vector<std::unique_ptr<Base>> objects;
    objects.push_back(std::make_unique<Base>());
    objects.push_back(std::make_unique<Derived1>());
    objects.push_back(std::make_unique<Derived2>());
    
    for (size_t i = 0; i < objects.size(); ++i) {
        std::cout << "\nObject " << i << " (" << typeid(*objects[i]).name() << "):\n";
        objects[i]->func1();  // Virtual dispatch
        objects[i]->func2();  // Virtual dispatch
        objects[i]->func3();  // Virtual dispatch
        objects[i]->non_virtual_func();  // Direct call (no virtual dispatch)
    }
}

void demonstrate_virtual_destruction() {
    std::cout << "\n💀 VIRTUAL DESTRUCTION DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n📋 Proper polymorphic destruction:\n";
    {
        std::vector<std::unique_ptr<Shape>> shapes;
        shapes.push_back(std::make_unique<Rectangle>("DestructRect", 2, 3));
        shapes.push_back(std::make_unique<Circle>("DestructCircle", 1.5));
        shapes.push_back(std::make_unique<Square>("DestructSquare", 2));
        
        std::cout << "Shapes created, going out of scope...\n";
    }
    std::cout << "All shapes properly destroyed with virtual destructors\n";
    
    std::cout << "\n⚠️ What happens without virtual destructors:\n";
    std::cout << "Without virtual destructors:\n";
    std::cout << "1. Only base class destructor would be called\n";
    std::cout << "2. Derived class resources might leak\n";
    std::cout << "3. Undefined behavior in polymorphic scenarios\n";
}

void demonstrate_performance_implications() {
    std::cout << "\n⚡ PERFORMANCE IMPLICATIONS\n";
    std::cout << std::string(35, '=') << std::endl;
    
    const int NUM_CALLS = 1000000;
    
    // Direct function calls
    Rectangle direct_rect("DirectRect", 1, 1);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_CALLS; ++i) {
        volatile double area = direct_rect.area();  // Direct call
        (void)area;  // Prevent optimization
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto direct_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Virtual function calls
    std::unique_ptr<Shape> virtual_rect = std::make_unique<Rectangle>("VirtualRect", 1, 1);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_CALLS; ++i) {
        volatile double area = virtual_rect->area();  // Virtual call
        (void)area;  // Prevent optimization
    }
    end = std::chrono::high_resolution_clock::now();
    auto virtual_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "📊 Performance comparison (" << NUM_CALLS << " calls):\n";
    std::cout << "  Direct calls:  " << direct_duration.count() << " μs\n";
    std::cout << "  Virtual calls: " << virtual_duration.count() << " μs\n";
    std::cout << "  Overhead:      " << (virtual_duration.count() - direct_duration.count()) << " μs\n";
    
    double overhead_per_call = static_cast<double>(virtual_duration.count() - direct_duration.count()) / NUM_CALLS;
    std::cout << "  Per-call overhead: ~" << (overhead_per_call * 1000) << " ns\n";
    
    std::cout << "\n💡 Performance notes:\n";
    std::cout << "1. Virtual function overhead is typically 1-3 ns per call\n";
    std::cout << "2. Modern CPUs can predict virtual calls effectively\n";
    std::cout << "3. The flexibility often outweighs the small overhead\n";
    std::cout << "4. Avoid virtual functions in tight inner loops if performance critical\n";
}

void demonstrate_function_hiding_vs_overriding() {
    std::cout << "\n🎯 FUNCTION HIDING VS OVERRIDING\n";
    std::cout << std::string(40, '=') << std::endl;
    
    class BaseDemo {
    public:
        virtual void virtual_func(int x) const {
            std::cout << "BaseDemo::virtual_func(int): " << x << std::endl;
        }
        
        void non_virtual_func(int x) const {
            std::cout << "BaseDemo::non_virtual_func(int): " << x << std::endl;
        }
        
        virtual void overloaded_func(int x) const {
            std::cout << "BaseDemo::overloaded_func(int): " << x << std::endl;
        }
        
        virtual void overloaded_func(double x) const {
            std::cout << "BaseDemo::overloaded_func(double): " << x << std::endl;
        }
    };
    
    class DerivedDemo : public BaseDemo {
    public:
        // Proper override
        void virtual_func(int x) const override {
            std::cout << "DerivedDemo::virtual_func(int): " << x << std::endl;
        }
        
        // Function hiding (not overriding due to different signature)
        void non_virtual_func(double x) const {
            std::cout << "DerivedDemo::non_virtual_func(double): " << x << std::endl;
        }
        
        // This hides ALL base class overloaded_func versions
        void overloaded_func(int x) const override {
            std::cout << "DerivedDemo::overloaded_func(int): " << x << std::endl;
        }
        
        // To make base versions available, use 'using' declaration
        using BaseDemo::overloaded_func;
    };
    
    std::cout << "\n📋 Testing function overriding vs hiding:\n";
    
    DerivedDemo derived;
    BaseDemo* base_ptr = &derived;
    
    std::cout << "\nVirtual function (proper override):\n";
    base_ptr->virtual_func(42);  // Calls DerivedDemo version
    derived.virtual_func(42);    // Calls DerivedDemo version
    
    std::cout << "\nNon-virtual function (no polymorphism):\n";
    base_ptr->non_virtual_func(42);  // Calls BaseDemo version
    derived.non_virtual_func(42.5);  // Calls DerivedDemo version (different signature)
    
    std::cout << "\nOverloaded functions:\n";
    derived.overloaded_func(42);     // DerivedDemo version (int)
    derived.overloaded_func(42.5);   // BaseDemo version (double, via 'using')
}

void demonstrate_abstract_classes() {
    std::cout << "\n🎭 ABSTRACT CLASSES AND INTERFACES\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n📋 Cannot instantiate abstract classes:\n";
    std::cout << "// Shape abstract_shape;  // ERROR: Cannot instantiate\n";
    
    std::cout << "\n📋 Must implement all pure virtual functions:\n";
    Rectangle concrete_rect("ConcreteRect", 3, 4);
    concrete_rect.print_info();
    
    std::cout << "\n🔍 Abstract class characteristics:\n";
    std::cout << "1. Contains at least one pure virtual function (= 0)\n";
    std::cout << "2. Cannot be instantiated directly\n";
    std::cout << "3. Can have constructors and destructors\n";
    std::cout << "4. Can have non-virtual and virtual functions\n";
    std::cout << "5. Can have data members\n";
    std::cout << "6. Derived classes must implement all pure virtual functions\n";
}

void demonstrate_best_practices() {
    std::cout << "\n📚 BEST PRACTICES AND GUIDELINES\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n✅ POLYMORPHISM BEST PRACTICES:\n";
    std::cout << "1. Always use virtual destructors in base classes\n";
    std::cout << "2. Use 'override' specifier for clarity and safety\n";
    std::cout << "3. Prefer pure virtual functions for interfaces\n";
    std::cout << "4. Keep virtual function signatures consistent\n";
    std::cout << "5. Consider performance implications in hot paths\n";
    std::cout << "6. Use 'final' to prevent further overriding when appropriate\n";
    
    std::cout << "\n⚠️ COMMON PITFALLS:\n";
    std::cout << "1. Forgetting virtual destructors\n";
    std::cout << "2. Function hiding instead of overriding\n";
    std::cout << "3. Calling virtual functions in constructors/destructors\n";
    std::cout << "4. Object slicing when copying polymorphic objects\n";
    std::cout << "5. Not understanding vtable overhead\n";
    std::cout << "6. Overusing virtual functions unnecessarily\n";
    
    std::cout << "\n🔍 DEBUGGING TIPS:\n";
    std::cout << "1. Use 'typeid' to check actual object types\n";
    std::cout << "2. Set breakpoints in virtual functions to trace calls\n";
    std::cout << "3. Examine vtable contents with debugger\n";
    std::cout << "4. Check for proper destructor chaining\n";
    std::cout << "5. Use compiler warnings to catch override issues\n";
    std::cout << "6. Profile virtual function call overhead if needed\n";
}

int main() {
    std::cout << "🚀 C++ POLYMORPHISM AND VTABLES - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(70, '=') << std::endl;
    
    try {
        demonstrate_basic_polymorphism();
        demonstrate_vtable_behavior();
        demonstrate_virtual_destruction();
        demonstrate_performance_implications();
        demonstrate_function_hiding_vs_overriding();
        demonstrate_abstract_classes();
        demonstrate_best_practices();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
