/*
 * =============================================================================
 * CLASS VS STRUCT - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Differences between class and struct in C++
 * 2. Access modifiers (public, private, protected)
 * 3. When to use class vs struct
 * 4. Data encapsulation and information hiding
 * 5. Best practices and design patterns
 * 6. Performance considerations
 * 7. Common pitfalls and debugging techniques
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 class_vs_struct.cpp -o class_vs_struct
 * Run with debugging: gdb ./class_vs_struct
 * =============================================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <iomanip>
#include <chrono>
#include <cassert>

// Debug macro for tracing
#define TRACE(msg) \
    std::cout << "[TRACE] " << __FUNCTION__ << ":" << __LINE__ << " - " << msg << std::endl;

// Performance measurement macro
#define MEASURE_TIME(code_block, description) \
    { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
        std::cout << "[PERF] " << description << ": " << duration.count() << " μs" << std::endl; \
    }

// =============================================================================
// BASIC STRUCT DEMONSTRATIONS
// =============================================================================

// Simple struct - all members public by default
struct Point2D {
    double x, y;  // Public by default
    
    // Constructor
    Point2D(double x = 0.0, double y = 0.0) : x(x), y(y) {
        TRACE("Point2D constructor called");
    }
    
    // Member function
    double distance_from_origin() const {
        return std::sqrt(x * x + y * y);
    }
    
    // Operator overloading
    Point2D operator+(const Point2D& other) const {
        return Point2D(x + other.x, y + other.y);
    }
    
    void print() const {
        std::cout << "Point2D(" << x << ", " << y << ")" << std::endl;
    }
};

// Struct with mixed access modifiers
struct Rectangle {
public:
    Rectangle(double w, double h) : width(w), height(h) {
        TRACE("Rectangle constructor called");
        validate_dimensions();
    }
    
    double area() const { return width * height; }
    double perimeter() const { return 2 * (width + height); }
    
    // Getters
    double get_width() const { return width; }
    double get_height() const { return height; }
    
    // Setters with validation
    void set_width(double w) {
        if (w > 0) width = w;
        else throw std::invalid_argument("Width must be positive");
    }
    
    void set_height(double h) {
        if (h > 0) height = h;
        else throw std::invalid_argument("Height must be positive");
    }

private:
    double width, height;  // Private members in struct
    
    void validate_dimensions() {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
    }
};

// =============================================================================
// BASIC CLASS DEMONSTRATIONS
// =============================================================================

// Simple class - all members private by default
class BankAccount {
private:  // Explicit private (default for class)
    std::string account_number;
    std::string owner_name;
    double balance;
    static int next_account_id;  // Static member
    
public:
    // Constructor
    BankAccount(const std::string& owner, double initial_balance = 0.0) 
        : account_number(generate_account_number()), 
          owner_name(owner), 
          balance(initial_balance) {
        TRACE("BankAccount constructor called");
        if (initial_balance < 0) {
            throw std::invalid_argument("Initial balance cannot be negative");
        }
    }
    
    // Copy constructor
    BankAccount(const BankAccount& other) 
        : account_number(generate_account_number()),  // New account number
          owner_name(other.owner_name), 
          balance(other.balance) {
        TRACE("BankAccount copy constructor called");
    }
    
    // Destructor
    ~BankAccount() {
        TRACE("BankAccount destructor called");
    }
    
    // Public interface
    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            std::cout << "Deposited $" << amount << ". New balance: $" << balance << std::endl;
        } else {
            throw std::invalid_argument("Deposit amount must be positive");
        }
    }
    
    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            std::cout << "Withdrew $" << amount << ". New balance: $" << balance << std::endl;
            return true;
        } else if (amount > balance) {
            std::cout << "Insufficient funds. Current balance: $" << balance << std::endl;
            return false;
        } else {
            throw std::invalid_argument("Withdrawal amount must be positive");
        }
    }
    
    // Getters (const methods)
    const std::string& get_account_number() const { return account_number; }
    const std::string& get_owner_name() const { return owner_name; }
    double get_balance() const { return balance; }
    
    // Display account information
    void display_info() const {
        std::cout << "Account: " << account_number 
                  << ", Owner: " << owner_name 
                  << ", Balance: $" << std::fixed << std::setprecision(2) << balance << std::endl;
    }
    
private:
    static std::string generate_account_number() {
        return "ACC" + std::to_string(++next_account_id);
    }
};

// Static member definition
int BankAccount::next_account_id = 1000;

// Class with inheritance and access control
class SavingsAccount : public BankAccount {
private:
    double interest_rate;
    
public:
    SavingsAccount(const std::string& owner, double initial_balance = 0.0, double rate = 0.02) 
        : BankAccount(owner, initial_balance), interest_rate(rate) {
        TRACE("SavingsAccount constructor called");
    }
    
    void apply_interest() {
        double interest = get_balance() * interest_rate;
        deposit(interest);
        std::cout << "Applied interest: $" << interest << std::endl;
    }
    
    double get_interest_rate() const { return interest_rate; }
    
    void set_interest_rate(double rate) {
        if (rate >= 0 && rate <= 1.0) {
            interest_rate = rate;
        } else {
            throw std::invalid_argument("Interest rate must be between 0 and 1");
        }
    }
};

// =============================================================================
// ADVANCED ACCESS CONTROL DEMONSTRATIONS
// =============================================================================

class Vehicle {
protected:  // Accessible to derived classes
    std::string make, model;
    int year;
    double mileage;
    
public:
    Vehicle(const std::string& make, const std::string& model, int year) 
        : make(make), model(model), year(year), mileage(0.0) {
        TRACE("Vehicle constructor called");
    }
    
    virtual ~Vehicle() {
        TRACE("Vehicle destructor called");
    }
    
    // Pure virtual function (makes this an abstract class)
    virtual void start_engine() = 0;
    virtual void stop_engine() = 0;
    
    // Virtual function with default implementation
    virtual void display_info() const {
        std::cout << year << " " << make << " " << model 
                  << " (Mileage: " << mileage << " miles)" << std::endl;
    }
    
    // Non-virtual public interface
    void add_mileage(double miles) {
        if (miles > 0) {
            mileage += miles;
        }
    }
    
    // Getters
    const std::string& get_make() const { return make; }
    const std::string& get_model() const { return model; }
    int get_year() const { return year; }
    double get_mileage() const { return mileage; }

protected:
    // Protected helper function
    void log_maintenance(const std::string& action) const {
        std::cout << "[MAINTENANCE] " << action << " for " << make << " " << model << std::endl;
    }
};

class Car : public Vehicle {
private:
    int num_doors;
    bool engine_running;
    
public:
    Car(const std::string& make, const std::string& model, int year, int doors = 4) 
        : Vehicle(make, model, year), num_doors(doors), engine_running(false) {
        TRACE("Car constructor called");
    }
    
    ~Car() override {
        TRACE("Car destructor called");
        if (engine_running) {
            stop_engine();
        }
    }
    
    // Implementation of pure virtual functions
    void start_engine() override {
        if (!engine_running) {
            engine_running = true;
            log_maintenance("Engine started");
            std::cout << "Car engine started!" << std::endl;
        } else {
            std::cout << "Engine is already running!" << std::endl;
        }
    }
    
    void stop_engine() override {
        if (engine_running) {
            engine_running = false;
            log_maintenance("Engine stopped");
            std::cout << "Car engine stopped!" << std::endl;
        } else {
            std::cout << "Engine is already stopped!" << std::endl;
        }
    }
    
    // Override virtual function
    void display_info() const override {
        Vehicle::display_info();  // Call base class implementation
        std::cout << "  Type: Car, Doors: " << num_doors 
                  << ", Engine: " << (engine_running ? "Running" : "Stopped") << std::endl;
    }
    
    // Car-specific methods
    int get_num_doors() const { return num_doors; }
    bool is_engine_running() const { return engine_running; }
};

// =============================================================================
// DESIGN PATTERNS AND BEST PRACTICES
// =============================================================================

// RAII (Resource Acquisition Is Initialization) example
class FileHandler {
private:
    std::string filename;
    bool is_open;
    
public:
    explicit FileHandler(const std::string& fname) : filename(fname), is_open(false) {
        TRACE("FileHandler constructor called");
        open_file();
    }
    
    ~FileHandler() {
        TRACE("FileHandler destructor called");
        close_file();
    }
    
    // Delete copy constructor and assignment operator (RAII pattern)
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
    
    // Move constructor and assignment operator
    FileHandler(FileHandler&& other) noexcept 
        : filename(std::move(other.filename)), is_open(other.is_open) {
        other.is_open = false;
        TRACE("FileHandler move constructor called");
    }
    
    FileHandler& operator=(FileHandler&& other) noexcept {
        if (this != &other) {
            close_file();
            filename = std::move(other.filename);
            is_open = other.is_open;
            other.is_open = false;
        }
        TRACE("FileHandler move assignment called");
        return *this;
    }
    
    void write_data(const std::string& data) {
        if (is_open) {
            std::cout << "Writing to " << filename << ": " << data << std::endl;
        } else {
            throw std::runtime_error("File is not open");
        }
    }
    
    bool is_file_open() const { return is_open; }
    
private:
    void open_file() {
        std::cout << "Opening file: " << filename << std::endl;
        is_open = true;  // Simulate file opening
    }
    
    void close_file() {
        if (is_open) {
            std::cout << "Closing file: " << filename << std::endl;
            is_open = false;
        }
    }
};

// Singleton pattern example
class Logger {
private:
    static std::unique_ptr<Logger> instance;
    static std::once_flag init_flag;
    
    Logger() {
        TRACE("Logger constructor called");
    }
    
public:
    static Logger& get_instance() {
        std::call_once(init_flag, []() {
            instance = std::unique_ptr<Logger>(new Logger());
        });
        return *instance;
    }
    
    // Delete copy and move operations
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;
    
    void log(const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::cout << "[LOG " << std::put_time(std::localtime(&time_t), "%H:%M:%S") 
                  << "] " << message << std::endl;
    }
};

// Static member definitions
std::unique_ptr<Logger> Logger::instance = nullptr;
std::once_flag Logger::init_flag;

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_basic_struct_vs_class() {
    std::cout << "\n🏗️ BASIC STRUCT VS CLASS DEMONSTRATION\n";
    std::cout << std::string(50, '=') << std::endl;
    
    // === STRUCT USAGE ===
    std::cout << "\n📊 STRUCT USAGE\n";
    std::cout << std::string(20, '-') << std::endl;
    
    Point2D p1(3.0, 4.0);
    Point2D p2(1.0, 2.0);
    
    std::cout << "Direct member access (struct): ";
    std::cout << "p1.x = " << p1.x << ", p1.y = " << p1.y << std::endl;
    
    Point2D p3 = p1 + p2;
    std::cout << "Point addition result: ";
    p3.print();
    
    Rectangle rect(5.0, 3.0);
    std::cout << "Rectangle area: " << rect.area() << std::endl;
    std::cout << "Rectangle perimeter: " << rect.perimeter() << std::endl;
    
    // === CLASS USAGE ===
    std::cout << "\n🏛️ CLASS USAGE\n";
    std::cout << std::string(15, '-') << std::endl;
    
    BankAccount account1("John Doe", 1000.0);
    account1.display_info();
    
    account1.deposit(500.0);
    account1.withdraw(200.0);
    account1.withdraw(2000.0);  // Should fail
    
    // Demonstrate encapsulation
    // account1.balance = 10000;  // ERROR: private member
    std::cout << "Current balance (via getter): $" << account1.get_balance() << std::endl;
}

void demonstrate_inheritance_and_access_control() {
    std::cout << "\n👨‍👩‍👧‍👦 INHERITANCE AND ACCESS CONTROL\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === SAVINGS ACCOUNT (INHERITANCE) ===
    std::cout << "\n💰 SAVINGS ACCOUNT DEMONSTRATION\n";
    std::cout << std::string(40, '-') << std::endl;
    
    SavingsAccount savings("Alice Smith", 5000.0, 0.03);
    savings.display_info();
    savings.apply_interest();
    savings.display_info();
    
    // === POLYMORPHISM ===
    std::cout << "\n🔄 POLYMORPHISM DEMONSTRATION\n";
    std::cout << std::string(35, '-') << std::endl;
    
    std::vector<std::unique_ptr<Vehicle>> vehicles;
    vehicles.push_back(std::make_unique<Car>("Toyota", "Camry", 2022, 4));
    vehicles.push_back(std::make_unique<Car>("Honda", "Civic", 2021, 2));
    
    for (auto& vehicle : vehicles) {
        vehicle->display_info();
        vehicle->start_engine();
        vehicle->add_mileage(100.0);
        vehicle->stop_engine();
        std::cout << std::endl;
    }
}

void demonstrate_design_patterns() {
    std::cout << "\n🎨 DESIGN PATTERNS DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === RAII PATTERN ===
    std::cout << "\n🔒 RAII PATTERN\n";
    std::cout << std::string(20, '-') << std::endl;
    
    {
        FileHandler file1("data.txt");
        file1.write_data("Hello, RAII!");
        
        // Move semantics
        FileHandler file2 = std::move(file1);
        file2.write_data("Moved file handler!");
        
        // file1 is now in moved-from state
        // Resources automatically cleaned up when going out of scope
    }
    std::cout << "Files automatically closed when going out of scope" << std::endl;
    
    // === SINGLETON PATTERN ===
    std::cout << "\n🏺 SINGLETON PATTERN\n";
    std::cout << std::string(25, '-') << std::endl;
    
    Logger& logger1 = Logger::get_instance();
    Logger& logger2 = Logger::get_instance();
    
    std::cout << "Same instance? " << std::boolalpha << (&logger1 == &logger2) << std::endl;
    
    logger1.log("First message");
    logger2.log("Second message from same instance");
}

void demonstrate_performance_considerations() {
    std::cout << "\n⚡ PERFORMANCE CONSIDERATIONS\n";
    std::cout << std::string(40, '=') << std::endl;
    
    const int ITERATIONS = 1000000;
    
    // === STRUCT PERFORMANCE ===
    std::cout << "\n📊 STRUCT PERFORMANCE\n";
    std::cout << std::string(25, '-') << std::endl;
    
    MEASURE_TIME({
        std::vector<Point2D> points;
        points.reserve(ITERATIONS);
        for (int i = 0; i < ITERATIONS; ++i) {
            points.emplace_back(i, i);
        }
        
        double sum = 0.0;
        for (const auto& point : points) {
            sum += point.x + point.y;  // Direct access
        }
        std::cout << "Sum: " << sum << std::endl;
    }, "Struct direct access");
    
    // === CLASS PERFORMANCE ===
    std::cout << "\n🏛️ CLASS PERFORMANCE\n";
    std::cout << std::string(25, '-') << std::endl;
    
    MEASURE_TIME({
        std::vector<Rectangle> rectangles;
        rectangles.reserve(ITERATIONS);
        for (int i = 1; i <= ITERATIONS; ++i) {
            rectangles.emplace_back(i, i);
        }
        
        double total_area = 0.0;
        for (const auto& rect : rectangles) {
            total_area += rect.area();  // Method call
        }
        std::cout << "Total area: " << total_area << std::endl;
    }, "Class method access");
    
    // === VIRTUAL FUNCTION OVERHEAD ===
    std::cout << "\n🔄 VIRTUAL FUNCTION OVERHEAD\n";
    std::cout << std::string(35, '-') << std::endl;
    
    std::vector<std::unique_ptr<Vehicle>> vehicles;
    vehicles.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        vehicles.push_back(std::make_unique<Car>("Brand", "Model", 2020));
    }
    
    MEASURE_TIME({
        for (auto& vehicle : vehicles) {
            vehicle->start_engine();
            vehicle->stop_engine();
        }
    }, "Virtual function calls");
}

void demonstrate_common_pitfalls() {
    std::cout << "\n⚠️ COMMON PITFALLS AND BEST PRACTICES\n";
    std::cout << std::string(50, '=') << std::endl;
    
    // === ACCESS MODIFIER PITFALLS ===
    std::cout << "\n🚫 ACCESS MODIFIER PITFALLS\n";
    std::cout << std::string(35, '-') << std::endl;
    
    std::cout << "1. Default access: struct = public, class = private" << std::endl;
    std::cout << "2. Use struct for simple data holders" << std::endl;
    std::cout << "3. Use class for objects with behavior and encapsulation" << std::endl;
    std::cout << "4. Protected members are accessible to derived classes" << std::endl;
    
    // === ENCAPSULATION BEST PRACTICES ===
    std::cout << "\n🔒 ENCAPSULATION BEST PRACTICES\n";
    std::cout << std::string(40, '-') << std::endl;
    
    std::cout << "1. Keep data members private" << std::endl;
    std::cout << "2. Provide public interface through methods" << std::endl;
    std::cout << "3. Use const methods for read-only operations" << std::endl;
    std::cout << "4. Validate input in setters" << std::endl;
    std::cout << "5. Use RAII for resource management" << std::endl;
    
    // === INHERITANCE GUIDELINES ===
    std::cout << "\n👨‍👩‍👧‍👦 INHERITANCE GUIDELINES\n";
    std::cout << std::string(35, '-') << std::endl;
    
    std::cout << "1. Use public inheritance for 'is-a' relationships" << std::endl;
    std::cout << "2. Make destructors virtual in base classes" << std::endl;
    std::cout << "3. Prefer composition over inheritance when possible" << std::endl;
    std::cout << "4. Use abstract base classes to define interfaces" << std::endl;
    std::cout << "5. Follow the Liskov Substitution Principle" << std::endl;
}

// Main function demonstrating all concepts
int main() {
    std::cout << "🚀 C++ CLASS VS STRUCT - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(65, '=') << std::endl;
    
    try {
        // Run all demonstrations
        demonstrate_basic_struct_vs_class();
        demonstrate_inheritance_and_access_control();
        demonstrate_design_patterns();
        demonstrate_performance_considerations();
        demonstrate_common_pitfalls();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
        // === DEBUGGING TIPS ===
        std::cout << "\n🐛 DEBUGGING TIPS:\n";
        std::cout << "1. Use TRACE macro to track object lifecycle\n";
        std::cout << "2. Set breakpoints in constructors and destructors\n";
        std::cout << "3. Use 'info vtbl' in gdb to examine virtual function tables\n";
        std::cout << "4. Check object addresses to verify copy vs move operations\n";
        std::cout << "5. Use static analysis tools to detect access violations\n";
        std::cout << "6. Profile code to measure virtual function overhead\n";
        
        std::cout << "\n📚 KEY TAKEAWAYS:\n";
        std::cout << "1. struct and class are nearly identical (default access differs)\n";
        std::cout << "2. Use struct for simple data, class for complex objects\n";
        std::cout << "3. Encapsulation protects object invariants\n";
        std::cout << "4. Inheritance enables polymorphism and code reuse\n";
        std::cout << "5. Virtual functions enable runtime polymorphism\n";
        std::cout << "6. RAII ensures proper resource management\n";
        std::cout << "7. Design patterns solve common architectural problems\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
