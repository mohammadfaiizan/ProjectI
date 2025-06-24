/*
 * =============================================================================
 * CONSTRUCTOR DESTRUCTOR ORDER - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Constructor and destructor call order
 * 2. Base class vs derived class construction/destruction
 * 3. Member initialization order
 * 4. Static member initialization
 * 5. Virtual destructors and polymorphic destruction
 * 6. Exception safety during construction
 * 7. RAII and resource management
 * 8. Common pitfalls and debugging techniques
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 constructor_destructor_order.cpp -o constructor_destructor_order
 * Run with debugging: gdb ./constructor_destructor_order
 * =============================================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <iomanip>

// Debug macro for tracing construction/destruction
#define TRACE_LIFECYCLE(class_name, action) \
    std::cout << "[" << std::setw(12) << class_name << "] " << action \
              << " (this=" << this << ")" << std::endl;

// Static counter for tracking object creation order
static int global_construction_counter = 0;

// =============================================================================
// BASIC CONSTRUCTION/DESTRUCTION ORDER
// =============================================================================

class Component {
private:
    std::string name;
    int creation_order;
    
public:
    explicit Component(const std::string& n) 
        : name(n), creation_order(++global_construction_counter) {
        TRACE_LIFECYCLE("Component", "Constructor: " + name + " (order: " + std::to_string(creation_order) + ")");
    }
    
    ~Component() {
        TRACE_LIFECYCLE("Component", "Destructor: " + name + " (was order: " + std::to_string(creation_order) + ")");
    }
    
    // Copy constructor
    Component(const Component& other) 
        : name(other.name + "_copy"), creation_order(++global_construction_counter) {
        TRACE_LIFECYCLE("Component", "Copy Constructor: " + name + " (order: " + std::to_string(creation_order) + ")");
    }
    
    // Move constructor
    Component(Component&& other) noexcept 
        : name(std::move(other.name)), creation_order(other.creation_order) {
        other.name = "moved_from";
        TRACE_LIFECYCLE("Component", "Move Constructor: " + name + " (order: " + std::to_string(creation_order) + ")");
    }
    
    const std::string& get_name() const { return name; }
    int get_creation_order() const { return creation_order; }
};

// Class demonstrating member initialization order
class ComplexObject {
private:
    // Members are initialized in declaration order, NOT initializer list order!
    Component first_component;   // Initialized first
    Component second_component;  // Initialized second
    Component third_component;   // Initialized third
    int value;
    std::string description;
    
public:
    // Note: Initialization order follows declaration order, not this list order
    ComplexObject(const std::string& desc) 
        : third_component("third"),      // Actually initialized third (declaration order)
          description(desc),             // Initialized fourth
          first_component("first"),      // Actually initialized first (declaration order)
          value(42),                     // Initialized fifth
          second_component("second") {   // Actually initialized second (declaration order)
        TRACE_LIFECYCLE("ComplexObject", "Constructor: " + description);
    }
    
    ~ComplexObject() {
        TRACE_LIFECYCLE("ComplexObject", "Destructor: " + description);
        // Destructors called in reverse order of construction
    }
    
    void display_info() const {
        std::cout << "ComplexObject '" << description << "' contains:\n";
        std::cout << "  - " << first_component.get_name() << " (order: " << first_component.get_creation_order() << ")\n";
        std::cout << "  - " << second_component.get_name() << " (order: " << second_component.get_creation_order() << ")\n";
        std::cout << "  - " << third_component.get_name() << " (order: " << third_component.get_creation_order() << ")\n";
        std::cout << "  - value: " << value << std::endl;
    }
};

// =============================================================================
// INHERITANCE CONSTRUCTION/DESTRUCTION ORDER
// =============================================================================

class Base {
protected:
    std::string base_name;
    Component base_component;
    
public:
    explicit Base(const std::string& name) 
        : base_name(name), base_component("base_" + name) {
        TRACE_LIFECYCLE("Base", "Constructor: " + base_name);
    }
    
    virtual ~Base() {  // Virtual destructor for proper polymorphic destruction
        TRACE_LIFECYCLE("Base", "Destructor: " + base_name);
    }
    
    virtual void identify() const {
        std::cout << "I am Base: " << base_name << std::endl;
    }
    
    const std::string& get_base_name() const { return base_name; }
};

class Derived : public Base {
private:
    std::string derived_name;
    Component derived_component;
    
public:
    Derived(const std::string& base_n, const std::string& derived_n) 
        : Base(base_n),  // Base constructor called first
          derived_name(derived_n), 
          derived_component("derived_" + derived_n) {  // Derived members initialized after base
        TRACE_LIFECYCLE("Derived", "Constructor: " + derived_name);
    }
    
    ~Derived() override {
        TRACE_LIFECYCLE("Derived", "Destructor: " + derived_name);
        // Derived destructor called first, then base destructor
    }
    
    void identify() const override {
        std::cout << "I am Derived: " << derived_name << " (base: " << base_name << ")" << std::endl;
    }
    
    const std::string& get_derived_name() const { return derived_name; }
};

// Multiple inheritance example
class Mixin1 {
public:
    Mixin1() { TRACE_LIFECYCLE("Mixin1", "Constructor"); }
    virtual ~Mixin1() { TRACE_LIFECYCLE("Mixin1", "Destructor"); }
};

class Mixin2 {
public:
    Mixin2() { TRACE_LIFECYCLE("Mixin2", "Constructor"); }
    virtual ~Mixin2() { TRACE_LIFECYCLE("Mixin2", "Destructor"); }
};

class MultipleInheritance : public Mixin1, public Mixin2 {
private:
    Component component;
    
public:
    MultipleInheritance() : component("multiple") {
        TRACE_LIFECYCLE("MultipleInheritance", "Constructor");
    }
    
    ~MultipleInheritance() {
        TRACE_LIFECYCLE("MultipleInheritance", "Destructor");
    }
};

// =============================================================================
// STATIC MEMBER INITIALIZATION
// =============================================================================

class StaticExample {
private:
    static Component static_component;  // Static member
    Component instance_component;       // Instance member
    static int instance_count;
    
public:
    StaticExample() : instance_component("instance_" + std::to_string(++instance_count)) {
        TRACE_LIFECYCLE("StaticExample", "Constructor (instance #" + std::to_string(instance_count) + ")");
    }
    
    ~StaticExample() {
        TRACE_LIFECYCLE("StaticExample", "Destructor (instance #" + std::to_string(instance_count--) + ")");
    }
    
    static void access_static() {
        std::cout << "Static component: " << static_component.get_name() << std::endl;
    }
};

// Static member definitions (initialized before main())
Component StaticExample::static_component("static_global");
int StaticExample::instance_count = 0;

// Global object (constructed before main())
Component global_component("global");

// =============================================================================
// EXCEPTION SAFETY DURING CONSTRUCTION
// =============================================================================

class RiskyComponent {
private:
    std::string name;
    bool should_throw;
    
public:
    RiskyComponent(const std::string& n, bool throw_exception = false) 
        : name(n), should_throw(throw_exception) {
        TRACE_LIFECYCLE("RiskyComponent", "Constructor: " + name);
        if (should_throw) {
            throw std::runtime_error("Construction failed for " + name);
        }
    }
    
    ~RiskyComponent() {
        TRACE_LIFECYCLE("RiskyComponent", "Destructor: " + name);
    }
    
    const std::string& get_name() const { return name; }
};

class ExceptionSafetyDemo {
private:
    RiskyComponent safe_component;
    RiskyComponent risky_component;
    RiskyComponent never_constructed;  // Won't be constructed if risky_component throws
    
public:
    ExceptionSafetyDemo(bool cause_exception = false) 
        : safe_component("safe", false),
          risky_component("risky", cause_exception),
          never_constructed("never", false) {
        TRACE_LIFECYCLE("ExceptionSafetyDemo", "Constructor completed");
    }
    
    ~ExceptionSafetyDemo() {
        TRACE_LIFECYCLE("ExceptionSafetyDemo", "Destructor");
    }
};

// =============================================================================
// VIRTUAL DESTRUCTOR AND POLYMORPHIC DESTRUCTION
// =============================================================================

class AbstractResource {
protected:
    std::string resource_name;
    
public:
    explicit AbstractResource(const std::string& name) : resource_name(name) {
        TRACE_LIFECYCLE("AbstractResource", "Constructor: " + resource_name);
    }
    
    // Virtual destructor ensures proper cleanup in inheritance hierarchy
    virtual ~AbstractResource() {
        TRACE_LIFECYCLE("AbstractResource", "Destructor: " + resource_name);
    }
    
    virtual void use_resource() = 0;
    virtual void cleanup_resource() = 0;
};

class FileResource : public AbstractResource {
private:
    Component file_handle;
    
public:
    explicit FileResource(const std::string& filename) 
        : AbstractResource(filename), file_handle("file_handle_" + filename) {
        TRACE_LIFECYCLE("FileResource", "Constructor: " + resource_name);
    }
    
    ~FileResource() override {
        TRACE_LIFECYCLE("FileResource", "Destructor: " + resource_name);
        cleanup_resource();
    }
    
    void use_resource() override {
        std::cout << "Using file resource: " << resource_name << std::endl;
    }
    
    void cleanup_resource() override {
        std::cout << "Cleaning up file resource: " << resource_name << std::endl;
    }
};

class NetworkResource : public AbstractResource {
private:
    Component connection;
    
public:
    explicit NetworkResource(const std::string& url) 
        : AbstractResource(url), connection("connection_" + url) {
        TRACE_LIFECYCLE("NetworkResource", "Constructor: " + resource_name);
    }
    
    ~NetworkResource() override {
        TRACE_LIFECYCLE("NetworkResource", "Destructor: " + resource_name);
        cleanup_resource();
    }
    
    void use_resource() override {
        std::cout << "Using network resource: " << resource_name << std::endl;
    }
    
    void cleanup_resource() override {
        std::cout << "Cleaning up network resource: " << resource_name << std::endl;
    }
};

// =============================================================================
// RAII AND RESOURCE MANAGEMENT
// =============================================================================

class RAIIManager {
private:
    std::vector<std::unique_ptr<AbstractResource>> resources;
    Component manager_component;
    
public:
    RAIIManager() : manager_component("raii_manager") {
        TRACE_LIFECYCLE("RAIIManager", "Constructor");
    }
    
    ~RAIIManager() {
        TRACE_LIFECYCLE("RAIIManager", "Destructor (managing " + std::to_string(resources.size()) + " resources)");
        // Resources automatically cleaned up via unique_ptr destructors
        // in reverse order of addition
    }
    
    void add_resource(std::unique_ptr<AbstractResource> resource) {
        std::cout << "Adding resource to manager\n";
        resources.push_back(std::move(resource));
    }
    
    void use_all_resources() {
        for (auto& resource : resources) {
            resource->use_resource();
        }
    }
    
    size_t get_resource_count() const { return resources.size(); }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_basic_construction_order() {
    std::cout << "\n🏗️ BASIC CONSTRUCTION/DESTRUCTION ORDER\n";
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << "\n📋 Creating ComplexObject...\n";
    {
        ComplexObject obj("demo_object");
        obj.display_info();
        std::cout << "\n📋 ComplexObject going out of scope...\n";
    }
    std::cout << "📋 ComplexObject destroyed\n";
}

void demonstrate_inheritance_order() {
    std::cout << "\n👨‍👩‍👧‍👦 INHERITANCE CONSTRUCTION/DESTRUCTION ORDER\n";
    std::cout << std::string(55, '=') << std::endl;
    
    std::cout << "\n📋 Creating Derived object...\n";
    {
        Derived derived_obj("BaseClass", "DerivedClass");
        derived_obj.identify();
        std::cout << "\n📋 Derived object going out of scope...\n";
    }
    std::cout << "📋 Derived object destroyed\n";
    
    std::cout << "\n📋 Creating MultipleInheritance object...\n";
    {
        MultipleInheritance multi_obj;
        std::cout << "\n📋 MultipleInheritance object going out of scope...\n";
    }
    std::cout << "📋 MultipleInheritance object destroyed\n";
}

void demonstrate_polymorphic_destruction() {
    std::cout << "\n🔄 POLYMORPHIC DESTRUCTION\n";
    std::cout << std::string(35, '=') << std::endl;
    
    std::cout << "\n📋 Creating resources via base pointer...\n";
    {
        std::vector<std::unique_ptr<AbstractResource>> resources;
        resources.push_back(std::make_unique<FileResource>("config.txt"));
        resources.push_back(std::make_unique<NetworkResource>("https://api.example.com"));
        
        for (auto& resource : resources) {
            resource->use_resource();
        }
        
        std::cout << "\n📋 Resources going out of scope (polymorphic destruction)...\n";
    }
    std::cout << "📋 All resources destroyed\n";
}

void demonstrate_exception_safety() {
    std::cout << "\n⚠️ EXCEPTION SAFETY DURING CONSTRUCTION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n📋 Successful construction...\n";
    try {
        ExceptionSafetyDemo safe_obj(false);
        std::cout << "✅ Object constructed successfully\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Exception: " << e.what() << std::endl;
    }
    
    std::cout << "\n📋 Construction with exception...\n";
    try {
        ExceptionSafetyDemo unsafe_obj(true);
        std::cout << "✅ Object constructed successfully\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Exception caught: " << e.what() << std::endl;
        std::cout << "🔍 Note: Only successfully constructed members are destroyed\n";
    }
}

void demonstrate_raii_pattern() {
    std::cout << "\n🔒 RAII PATTERN DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    std::cout << "\n📋 Creating RAII manager...\n";
    {
        RAIIManager manager;
        
        manager.add_resource(std::make_unique<FileResource>("data.bin"));
        manager.add_resource(std::make_unique<NetworkResource>("https://service.com/api"));
        manager.add_resource(std::make_unique<FileResource>("log.txt"));
        
        std::cout << "\n📋 Using all managed resources...\n";
        manager.use_all_resources();
        
        std::cout << "\n📋 Manager going out of scope (RAII cleanup)...\n";
    }
    std::cout << "📋 All resources automatically cleaned up\n";
}

void demonstrate_static_initialization() {
    std::cout << "\n🌐 STATIC MEMBER INITIALIZATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    std::cout << "\n📋 Accessing static member before creating instances...\n";
    StaticExample::access_static();
    
    std::cout << "\n📋 Creating StaticExample instances...\n";
    {
        StaticExample obj1;
        StaticExample obj2;
        
        std::cout << "\n📋 Instances going out of scope...\n";
    }
    std::cout << "📋 Instances destroyed (static member remains)\n";
    
    std::cout << "\n📋 Accessing static member after instances destroyed...\n";
    StaticExample::access_static();
}

void demonstrate_construction_performance() {
    std::cout << "\n⚡ CONSTRUCTION/DESTRUCTION PERFORMANCE\n";
    std::cout << std::string(50, '=') << std::endl;
    
    const int NUM_OBJECTS = 100000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Measure construction/destruction overhead
    {
        std::vector<ComplexObject> objects;
        objects.reserve(NUM_OBJECTS);
        
        for (int i = 0; i < NUM_OBJECTS; ++i) {
            objects.emplace_back("object_" + std::to_string(i));
        }
        
        // Objects destroyed when vector goes out of scope
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "⏱️ Created and destroyed " << NUM_OBJECTS << " complex objects in " 
              << duration.count() << " μs" << std::endl;
    std::cout << "📊 Average: " << (duration.count() / NUM_OBJECTS) << " μs per object" << std::endl;
}

void demonstrate_best_practices() {
    std::cout << "\n📚 BEST PRACTICES AND COMMON PITFALLS\n";
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << "\n✅ BEST PRACTICES:\n";
    std::cout << "1. Always use virtual destructors in base classes\n";
    std::cout << "2. Initialize members in declaration order\n";
    std::cout << "3. Use RAII for automatic resource management\n";
    std::cout << "4. Handle exceptions during construction properly\n";
    std::cout << "5. Prefer initialization over assignment in constructors\n";
    std::cout << "6. Use smart pointers for automatic memory management\n";
    
    std::cout << "\n⚠️ COMMON PITFALLS:\n";
    std::cout << "1. Forgetting virtual destructors in base classes\n";
    std::cout << "2. Assuming initializer list order matches execution order\n";
    std::cout << "3. Not handling construction exceptions properly\n";
    std::cout << "4. Circular dependencies in static initialization\n";
    std::cout << "5. Resource leaks when exceptions occur during construction\n";
    std::cout << "6. Calling virtual functions in constructors/destructors\n";
    
    std::cout << "\n🔍 DEBUGGING TIPS:\n";
    std::cout << "1. Use TRACE macros to track construction/destruction order\n";
    std::cout << "2. Set breakpoints in constructors and destructors\n";
    std::cout << "3. Use memory debugging tools (valgrind, AddressSanitizer)\n";
    std::cout << "4. Check for proper cleanup in exception scenarios\n";
    std::cout << "5. Verify virtual destructor calls with debugger\n";
}

// Main function demonstrating all concepts
int main() {
    std::cout << "🚀 C++ CONSTRUCTOR/DESTRUCTOR ORDER - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(75, '=') << std::endl;
    
    std::cout << "\n🌟 Global and static objects constructed before main()\n";
    std::cout << "Global component: " << global_component.get_name() << std::endl;
    
    try {
        // Run all demonstrations
        demonstrate_basic_construction_order();
        demonstrate_inheritance_order();
        demonstrate_polymorphic_destruction();
        demonstrate_exception_safety();
        demonstrate_raii_pattern();
        demonstrate_static_initialization();
        demonstrate_construction_performance();
        demonstrate_best_practices();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🌟 Global and static objects will be destroyed after main()\n";
    return 0;
}
