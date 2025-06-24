/*
 * CRTP PATTERN - Curiously Recurring Template Pattern
 * 
 * Compilation: g++ -std=c++17 -Wall -Wextra -g -O2 CRTP_pattern.cpp -o CRTP_pattern
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <type_traits>
#include <chrono>
#include <typeinfo>

#define CRTP_INFO(msg) std::cout << "[CRTP INFO] " << msg << std::endl

namespace CRTPPattern {

// ============================================================================
// 1. BASIC CRTP PATTERN
// ============================================================================

// Base CRTP class
template<typename Derived>
class Printable {
public:
    void print() const {
        static_cast<const Derived*>(this)->print_impl();
    }
    
    void print_type() const {
        std::cout << "Type: " << typeid(Derived).name() << std::endl;
    }

protected:
    // Protected constructor to prevent direct instantiation
    Printable() = default;
    friend Derived;
};

// Derived classes
class Document : public Printable<Document> {
private:
    std::string content_;

public:
    Document(const std::string& content) : content_(content) {}
    
    void print_impl() const {
        std::cout << "Document: " << content_ << std::endl;
    }
    
    const std::string& get_content() const { return content_; }
};

class Image : public Printable<Image> {
private:
    std::string filename_;
    int width_, height_;

public:
    Image(const std::string& filename, int width, int height) 
        : filename_(filename), width_(width), height_(height) {}
    
    void print_impl() const {
        std::cout << "Image: " << filename_ << " (" << width_ << "x" << height_ << ")" << std::endl;
    }
};

// ============================================================================
// 2. CRTP FOR STATIC POLYMORPHISM
// ============================================================================

template<typename Derived>
class Shape {
public:
    double area() const {
        return static_cast<const Derived*>(this)->area_impl();
    }
    
    double perimeter() const {
        return static_cast<const Derived*>(this)->perimeter_impl();
    }
    
    void draw() const {
        static_cast<const Derived*>(this)->draw_impl();
    }
    
    std::string name() const {
        return static_cast<const Derived*>(this)->name_impl();
    }

protected:
    Shape() = default;
    friend Derived;
};

class Circle : public Shape<Circle> {
private:
    double radius_;

public:
    explicit Circle(double radius) : radius_(radius) {}
    
    double area_impl() const {
        return 3.14159 * radius_ * radius_;
    }
    
    double perimeter_impl() const {
        return 2 * 3.14159 * radius_;
    }
    
    void draw_impl() const {
        std::cout << "Drawing circle with radius " << radius_ << std::endl;
    }
    
    std::string name_impl() const {
        return "Circle";
    }
    
    double get_radius() const { return radius_; }
};

class Rectangle : public Shape<Rectangle> {
private:
    double width_, height_;

public:
    Rectangle(double width, double height) : width_(width), height_(height) {}
    
    double area_impl() const {
        return width_ * height_;
    }
    
    double perimeter_impl() const {
        return 2 * (width_ + height_);
    }
    
    void draw_impl() const {
        std::cout << "Drawing rectangle " << width_ << "x" << height_ << std::endl;
    }
    
    std::string name_impl() const {
        return "Rectangle";
    }
    
    double get_width() const { return width_; }
    double get_height() const { return height_; }
};

// ============================================================================
// 3. CRTP FOR MIXINS
// ============================================================================

// Equality mixin
template<typename Derived>
class Equality {
public:
    bool operator!=(const Derived& other) const {
        return !static_cast<const Derived*>(this)->operator==(other);
    }

protected:
    Equality() = default;
    friend Derived;
};

// Comparison mixin
template<typename Derived>
class Comparable : public Equality<Derived> {
public:
    bool operator>(const Derived& other) const {
        return other < static_cast<const Derived&>(*this);
    }
    
    bool operator<=(const Derived& other) const {
        return !(static_cast<const Derived&>(*this) > other);
    }
    
    bool operator>=(const Derived& other) const {
        return !(static_cast<const Derived&>(*this) < other);
    }

protected:
    Comparable() = default;
    friend Derived;
};

// Arithmetic mixin
template<typename Derived>
class Arithmetic {
public:
    Derived& operator+=(const Derived& other) {
        auto& self = static_cast<Derived&>(*this);
        self = self + other;
        return self;
    }
    
    Derived& operator-=(const Derived& other) {
        auto& self = static_cast<Derived&>(*this);
        self = self - other;
        return self;
    }
    
    Derived& operator*=(const Derived& other) {
        auto& self = static_cast<Derived&>(*this);
        self = self * other;
        return self;
    }

protected:
    Arithmetic() = default;
    friend Derived;
};

// Example class using mixins
class Point : public Comparable<Point>, public Arithmetic<Point> {
private:
    double x_, y_;

public:
    Point(double x, double y) : x_(x), y_(y) {}
    
    // Required for Equality mixin
    bool operator==(const Point& other) const {
        return x_ == other.x_ && y_ == other.y_;
    }
    
    // Required for Comparable mixin
    bool operator<(const Point& other) const {
        return (x_ < other.x_) || (x_ == other.x_ && y_ < other.y_);
    }
    
    // Required for Arithmetic mixin
    Point operator+(const Point& other) const {
        return Point(x_ + other.x_, y_ + other.y_);
    }
    
    Point operator-(const Point& other) const {
        return Point(x_ - other.x_, y_ - other.y_);
    }
    
    Point operator*(const Point& other) const {
        return Point(x_ * other.x_, y_ * other.y_);
    }
    
    double x() const { return x_; }
    double y() const { return y_; }
    
    void print() const {
        std::cout << "Point(" << x_ << ", " << y_ << ")" << std::endl;
    }
};

// ============================================================================
// 4. CRTP FOR SINGLETON PATTERN
// ============================================================================

template<typename Derived>
class Singleton {
public:
    static Derived& instance() {
        static Derived instance_;
        return instance_;
    }
    
    // Delete copy constructor and assignment
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    
    // Delete move constructor and assignment
    Singleton(Singleton&&) = delete;
    Singleton& operator=(Singleton&&) = delete;

protected:
    Singleton() = default;
    virtual ~Singleton() = default;
    friend Derived;
};

class Logger : public Singleton<Logger> {
private:
    std::vector<std::string> logs_;

public:
    void log(const std::string& message) {
        logs_.push_back(message);
        std::cout << "[LOG] " << message << std::endl;
    }
    
    void print_logs() const {
        std::cout << "All logs:" << std::endl;
        for (size_t i = 0; i < logs_.size(); ++i) {
            std::cout << "  " << i + 1 << ". " << logs_[i] << std::endl;
        }
    }
    
    size_t log_count() const { return logs_.size(); }

private:
    friend class Singleton<Logger>;
    Logger() = default;
};

// ============================================================================
// 5. CRTP FOR FLUENT INTERFACE
// ============================================================================

template<typename Derived>
class FluentBuilder {
public:
    Derived& set_name(const std::string& name) {
        name_ = name;
        return static_cast<Derived&>(*this);
    }
    
    Derived& set_id(int id) {
        id_ = id;
        return static_cast<Derived&>(*this);
    }

protected:
    std::string name_;
    int id_ = 0;
    
    FluentBuilder() = default;
    friend Derived;
};

class PersonBuilder : public FluentBuilder<PersonBuilder> {
private:
    int age_ = 0;
    std::string email_;

public:
    PersonBuilder& set_age(int age) {
        age_ = age;
        return *this;
    }
    
    PersonBuilder& set_email(const std::string& email) {
        email_ = email;
        return *this;
    }
    
    void build() const {
        std::cout << "Person built:" << std::endl;
        std::cout << "  Name: " << name_ << std::endl;
        std::cout << "  ID: " << id_ << std::endl;
        std::cout << "  Age: " << age_ << std::endl;
        std::cout << "  Email: " << email_ << std::endl;
    }
};

// ============================================================================
// DEMONSTRATION FUNCTIONS
// ============================================================================

void demonstrate_basic_crtp() {
    std::cout << "\n=== BASIC CRTP PATTERN ===" << std::endl;
    
    Document doc("Hello World Document");
    Image img("photo.jpg", 1920, 1080);
    
    doc.print();
    doc.print_type();
    
    img.print();
    img.print_type();
}

void demonstrate_static_polymorphism() {
    std::cout << "\n=== STATIC POLYMORPHISM ===" << std::endl;
    
    Circle circle(5.0);
    Rectangle rect(4.0, 6.0);
    
    std::cout << circle.name() << " - Area: " << circle.area() 
              << ", Perimeter: " << circle.perimeter() << std::endl;
    circle.draw();
    
    std::cout << rect.name() << " - Area: " << rect.area() 
              << ", Perimeter: " << rect.perimeter() << std::endl;
    rect.draw();
}

void demonstrate_mixins() {
    std::cout << "\n=== CRTP MIXINS ===" << std::endl;
    
    Point p1(1.0, 2.0);
    Point p2(3.0, 4.0);
    Point p3(1.0, 2.0);
    
    std::cout << "p1: "; p1.print();
    std::cout << "p2: "; p2.print();
    std::cout << "p3: "; p3.print();
    
    std::cout << "p1 == p2: " << (p1 == p2) << std::endl;
    std::cout << "p1 == p3: " << (p1 == p3) << std::endl;
    std::cout << "p1 != p2: " << (p1 != p2) << std::endl;
    std::cout << "p1 < p2: " << (p1 < p2) << std::endl;
    std::cout << "p1 > p2: " << (p1 > p2) << std::endl;
    
    Point sum = p1 + p2;
    std::cout << "p1 + p2: "; sum.print();
    
    p1 += p2;
    std::cout << "p1 after += p2: "; p1.print();
}

void demonstrate_singleton() {
    std::cout << "\n=== CRTP SINGLETON ===" << std::endl;
    
    Logger& logger1 = Logger::instance();
    Logger& logger2 = Logger::instance();
    
    std::cout << "Same instance: " << (&logger1 == &logger2) << std::endl;
    
    logger1.log("First message");
    logger2.log("Second message");
    logger1.log("Third message");
    
    logger1.print_logs();
    std::cout << "Total logs: " << logger2.log_count() << std::endl;
}

void demonstrate_fluent_interface() {
    std::cout << "\n=== CRTP FLUENT INTERFACE ===" << std::endl;
    
    PersonBuilder builder;
    builder.set_name("John Doe")
           .set_id(12345)
           .set_age(30)
           .set_email("john.doe@example.com")
           .build();
}

} // namespace CRTPPattern

int main() {
    std::cout << "=== CRTP PATTERN TUTORIAL ===" << std::endl;
    
    try {
        CRTPPattern::demonstrate_basic_crtp();
        CRTPPattern::demonstrate_static_polymorphism();
        CRTPPattern::demonstrate_mixins();
        CRTPPattern::demonstrate_singleton();
        CRTPPattern::demonstrate_fluent_interface();
        
        std::cout << "\n=== TUTORIAL COMPLETED ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 