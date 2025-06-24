/*
 * =============================================================================
 * DEEP VS SHALLOW COPY - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Shallow copy vs deep copy concepts
 * 2. Copy constructor implementation
 * 3. Copy assignment operator
 * 4. Rule of Three/Five/Zero
 * 5. Move semantics and copy elision
 * 6. Smart pointers and automatic memory management
 * 7. Performance implications
 * 8. Common pitfalls and debugging techniques
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 deep_vs_shallow_copy.cpp -o deep_vs_shallow_copy
 * =============================================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <chrono>

#define TRACE(msg) std::cout << "[TRACE] " << __FUNCTION__ << ": " << msg << std::endl;

// =============================================================================
// SHALLOW COPY EXAMPLE (PROBLEMATIC)
// =============================================================================

class ShallowCopyExample {
private:
    char* data;
    size_t size;
    
public:
    explicit ShallowCopyExample(const std::string& str) : size(str.length()) {
        data = new char[size + 1];
        std::strcpy(data, str.c_str());
        TRACE("Constructor: allocated " + std::to_string(size + 1) + " bytes");
    }
    
    // Compiler-generated copy constructor (SHALLOW COPY - DANGEROUS!)
    // ShallowCopyExample(const ShallowCopyExample& other) = default;
    
    ~ShallowCopyExample() {
        delete[] data;
        TRACE("Destructor: freed memory");
    }
    
    void print() const {
        std::cout << "Data: " << data << " (ptr: " << static_cast<void*>(data) << ")" << std::endl;
    }
    
    const char* get_data() const { return data; }
};

// =============================================================================
// DEEP COPY EXAMPLE (CORRECT)
// =============================================================================

class DeepCopyExample {
private:
    char* data;
    size_t size;
    static int instance_count;
    
public:
    explicit DeepCopyExample(const std::string& str) : size(str.length()) {
        data = new char[size + 1];
        std::strcpy(data, str.c_str());
        ++instance_count;
        TRACE("Constructor: allocated " + std::to_string(size + 1) + " bytes (instance #" + std::to_string(instance_count) + ")");
    }
    
    // Deep copy constructor
    DeepCopyExample(const DeepCopyExample& other) : size(other.size) {
        data = new char[size + 1];
        std::strcpy(data, other.data);
        ++instance_count;
        TRACE("Copy Constructor: deep copy created (instance #" + std::to_string(instance_count) + ")");
    }
    
    // Deep copy assignment operator
    DeepCopyExample& operator=(const DeepCopyExample& other) {
        if (this != &other) {  // Self-assignment check
            // Clean up existing resource
            delete[] data;
            
            // Copy from other
            size = other.size;
            data = new char[size + 1];
            std::strcpy(data, other.data);
            
            TRACE("Copy Assignment: deep copy assigned");
        }
        return *this;
    }
    
    // Move constructor (C++11)
    DeepCopyExample(DeepCopyExample&& other) noexcept 
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
        ++instance_count;
        TRACE("Move Constructor: resource moved (instance #" + std::to_string(instance_count) + ")");
    }
    
    // Move assignment operator (C++11)
    DeepCopyExample& operator=(DeepCopyExample&& other) noexcept {
        if (this != &other) {
            // Clean up existing resource
            delete[] data;
            
            // Move from other
            data = other.data;
            size = other.size;
            
            // Reset other
            other.data = nullptr;
            other.size = 0;
            
            TRACE("Move Assignment: resource moved");
        }
        return *this;
    }
    
    ~DeepCopyExample() {
        delete[] data;
        --instance_count;
        TRACE("Destructor: freed memory (remaining instances: " + std::to_string(instance_count) + ")");
    }
    
    void print() const {
        if (data) {
            std::cout << "Data: " << data << " (ptr: " << static_cast<void*>(data) << ")" << std::endl;
        } else {
            std::cout << "Data: <moved-from state>" << std::endl;
        }
    }
    
    const char* get_data() const { return data; }
    size_t get_size() const { return size; }
};

int DeepCopyExample::instance_count = 0;

// =============================================================================
// SMART POINTER APPROACH (MODERN C++)
// =============================================================================

class SmartPointerExample {
private:
    std::unique_ptr<char[]> data;
    size_t size;
    
public:
    explicit SmartPointerExample(const std::string& str) : size(str.length()) {
        data = std::make_unique<char[]>(size + 1);
        std::strcpy(data.get(), str.c_str());
        TRACE("Constructor: smart pointer allocated " + std::to_string(size + 1) + " bytes");
    }
    
    // Custom copy constructor (deep copy)
    SmartPointerExample(const SmartPointerExample& other) : size(other.size) {
        data = std::make_unique<char[]>(size + 1);
        std::strcpy(data.get(), other.data.get());
        TRACE("Copy Constructor: smart pointer deep copy");
    }
    
    // Custom copy assignment operator
    SmartPointerExample& operator=(const SmartPointerExample& other) {
        if (this != &other) {
            size = other.size;
            data = std::make_unique<char[]>(size + 1);
            std::strcpy(data.get(), other.data.get());
            TRACE("Copy Assignment: smart pointer deep copy");
        }
        return *this;
    }
    
    // Move constructor and assignment are automatically generated and efficient
    SmartPointerExample(SmartPointerExample&&) = default;
    SmartPointerExample& operator=(SmartPointerExample&&) = default;
    
    // Destructor is automatically generated (smart pointer handles cleanup)
    ~SmartPointerExample() {
        TRACE("Destructor: smart pointer automatic cleanup");
    }
    
    void print() const {
        if (data) {
            std::cout << "Data: " << data.get() << " (ptr: " << static_cast<void*>(data.get()) << ")" << std::endl;
        } else {
            std::cout << "Data: <moved-from state>" << std::endl;
        }
    }
    
    const char* get_data() const { return data.get(); }
};

// =============================================================================
// COMPLEX OBJECT WITH MULTIPLE RESOURCES
// =============================================================================

class ComplexResource {
private:
    std::string name;
    std::vector<int> numbers;
    std::unique_ptr<DeepCopyExample> nested_object;
    
public:
    ComplexResource(const std::string& n, const std::vector<int>& nums) 
        : name(n), numbers(nums), nested_object(std::make_unique<DeepCopyExample>(n + "_nested")) {
        TRACE("ComplexResource constructor: " + name);
    }
    
    // Copy constructor
    ComplexResource(const ComplexResource& other) 
        : name(other.name + "_copy"), 
          numbers(other.numbers),
          nested_object(std::make_unique<DeepCopyExample>(*other.nested_object)) {
        TRACE("ComplexResource copy constructor: " + name);
    }
    
    // Copy assignment operator
    ComplexResource& operator=(const ComplexResource& other) {
        if (this != &other) {
            name = other.name + "_assigned";
            numbers = other.numbers;
            nested_object = std::make_unique<DeepCopyExample>(*other.nested_object);
            TRACE("ComplexResource copy assignment: " + name);
        }
        return *this;
    }
    
    // Move constructor
    ComplexResource(ComplexResource&& other) noexcept 
        : name(std::move(other.name)), 
          numbers(std::move(other.numbers)),
          nested_object(std::move(other.nested_object)) {
        TRACE("ComplexResource move constructor: " + name);
    }
    
    // Move assignment operator
    ComplexResource& operator=(ComplexResource&& other) noexcept {
        if (this != &other) {
            name = std::move(other.name);
            numbers = std::move(other.numbers);
            nested_object = std::move(other.nested_object);
            TRACE("ComplexResource move assignment: " + name);
        }
        return *this;
    }
    
    ~ComplexResource() {
        TRACE("ComplexResource destructor: " + name);
    }
    
    void print() const {
        std::cout << "ComplexResource: " << name 
                  << " (numbers: " << numbers.size() << ", nested: ";
        if (nested_object) {
            std::cout << nested_object->get_data();
        } else {
            std::cout << "null";
        }
        std::cout << ")" << std::endl;
    }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_shallow_copy_problem() {
    std::cout << "\n⚠️ SHALLOW COPY PROBLEM DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n⚠️ WARNING: This would cause double-delete if uncommented!\n";
    std::cout << "// ShallowCopyExample obj1(\"Hello\");\n";
    std::cout << "// ShallowCopyExample obj2 = obj1;  // Shallow copy - both point to same memory!\n";
    std::cout << "// When destructors run, double-delete occurs!\n";
    
    std::cout << "\nShallow copy issues:\n";
    std::cout << "1. Multiple objects share the same memory\n";
    std::cout << "2. Double-delete when destructors run\n";
    std::cout << "3. Modifications affect all copies\n";
    std::cout << "4. Dangling pointers after one object is destroyed\n";
}

void demonstrate_deep_copy_solution() {
    std::cout << "\n✅ DEEP COPY SOLUTION DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n📋 Creating original object...\n";
    DeepCopyExample obj1("Original Data");
    obj1.print();
    
    std::cout << "\n📋 Creating copy via copy constructor...\n";
    DeepCopyExample obj2 = obj1;  // Copy constructor
    obj2.print();
    
    std::cout << "\n📋 Creating third object and using copy assignment...\n";
    DeepCopyExample obj3("Temporary");
    obj3 = obj1;  // Copy assignment operator
    obj3.print();
    
    std::cout << "\n🔍 Verifying independent memory addresses:\n";
    std::cout << "obj1 address: " << static_cast<const void*>(obj1.get_data()) << std::endl;
    std::cout << "obj2 address: " << static_cast<const void*>(obj2.get_data()) << std::endl;
    std::cout << "obj3 address: " << static_cast<const void*>(obj3.get_data()) << std::endl;
    
    std::cout << "\n✅ All objects have independent memory - safe destruction!\n";
}

void demonstrate_move_semantics() {
    std::cout << "\n🚀 MOVE SEMANTICS DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    std::cout << "\n📋 Creating object for move operations...\n";
    DeepCopyExample obj1("Move Source");
    obj1.print();
    
    std::cout << "\n📋 Move constructor (efficient resource transfer)...\n";
    DeepCopyExample obj2 = std::move(obj1);
    std::cout << "After move constructor:\n";
    std::cout << "  obj1 (moved-from): ";
    obj1.print();
    std::cout << "  obj2 (moved-to): ";
    obj2.print();
    
    std::cout << "\n📋 Move assignment operator...\n";
    DeepCopyExample obj3("Move Target");
    obj3 = std::move(obj2);
    std::cout << "After move assignment:\n";
    std::cout << "  obj2 (moved-from): ";
    obj2.print();
    std::cout << "  obj3 (moved-to): ";
    obj3.print();
}

void demonstrate_smart_pointer_approach() {
    std::cout << "\n🧠 SMART POINTER APPROACH DEMONSTRATION\n";
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << "\n📋 Creating smart pointer objects...\n";
    SmartPointerExample smart1("Smart Original");
    smart1.print();
    
    std::cout << "\n📋 Copy operations with smart pointers...\n";
    SmartPointerExample smart2 = smart1;  // Copy constructor
    SmartPointerExample smart3("Temp");
    smart3 = smart1;  // Copy assignment
    
    smart1.print();
    smart2.print();
    smart3.print();
    
    std::cout << "\n📋 Move operations with smart pointers...\n";
    SmartPointerExample smart4 = std::move(smart1);  // Move constructor
    std::cout << "After move:\n";
    std::cout << "  smart1 (moved-from): ";
    smart1.print();
    std::cout << "  smart4 (moved-to): ";
    smart4.print();
    
    std::cout << "\n✅ Smart pointers handle memory management automatically!\n";
}

void demonstrate_complex_object_copying() {
    std::cout << "\n🏗️ COMPLEX OBJECT COPYING DEMONSTRATION\n";
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << "\n📋 Creating complex object with multiple resources...\n";
    std::vector<int> data = {1, 2, 3, 4, 5};
    ComplexResource complex1("Original", data);
    complex1.print();
    
    std::cout << "\n📋 Deep copying complex object...\n";
    ComplexResource complex2 = complex1;  // Copy constructor
    complex2.print();
    
    std::cout << "\n📋 Move operations with complex object...\n";
    ComplexResource complex3 = std::move(complex1);  // Move constructor
    std::cout << "After move:\n";
    std::cout << "  complex1 (moved-from): ";
    complex1.print();
    std::cout << "  complex3 (moved-to): ";
    complex3.print();
}

void demonstrate_performance_comparison() {
    std::cout << "\n⚡ PERFORMANCE COMPARISON\n";
    std::cout << std::string(35, '=') << std::endl;
    
    const int NUM_OPERATIONS = 10000;
    const std::string test_data = "Performance Test Data String";
    
    // Deep copy performance
    auto start = std::chrono::high_resolution_clock::now();
    {
        std::vector<DeepCopyExample> objects;
        objects.reserve(NUM_OPERATIONS);
        
        DeepCopyExample original(test_data);
        for (int i = 0; i < NUM_OPERATIONS; ++i) {
            objects.push_back(original);  // Copy constructor calls
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Move performance
    start = std::chrono::high_resolution_clock::now();
    {
        std::vector<DeepCopyExample> objects;
        objects.reserve(NUM_OPERATIONS);
        
        for (int i = 0; i < NUM_OPERATIONS; ++i) {
            objects.push_back(DeepCopyExample(test_data));  // Move constructor calls
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto move_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "📊 Performance Results (" << NUM_OPERATIONS << " operations):\n";
    std::cout << "  Copy operations: " << copy_duration.count() << " μs\n";
    std::cout << "  Move operations: " << move_duration.count() << " μs\n";
    std::cout << "  Speedup: " << (static_cast<double>(copy_duration.count()) / move_duration.count()) << "x\n";
}

void demonstrate_best_practices() {
    std::cout << "\n📚 BEST PRACTICES AND GUIDELINES\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n✅ BEST PRACTICES:\n";
    std::cout << "1. Follow Rule of Five: destructor, copy ctor, copy assign, move ctor, move assign\n";
    std::cout << "2. Use smart pointers for automatic memory management\n";
    std::cout << "3. Implement move semantics for better performance\n";
    std::cout << "4. Always check for self-assignment in copy operators\n";
    std::cout << "5. Use copy-and-swap idiom for exception safety\n";
    std::cout << "6. Prefer Rule of Zero when possible (use standard containers)\n";
    
    std::cout << "\n⚠️ COMMON PITFALLS:\n";
    std::cout << "1. Forgetting to implement deep copy for raw pointers\n";
    std::cout << "2. Not checking for self-assignment\n";
    std::cout << "3. Resource leaks in copy assignment operators\n";
    std::cout << "4. Not implementing move semantics for performance\n";
    std::cout << "5. Mixing shallow and deep copy semantics\n";
    std::cout << "6. Not handling exceptions during copying\n";
    
    std::cout << "\n🔍 DEBUGGING TIPS:\n";
    std::cout << "1. Use TRACE macros to track copy/move operations\n";
    std::cout << "2. Check memory addresses to verify deep copying\n";
    std::cout << "3. Use memory debugging tools (valgrind, AddressSanitizer)\n";
    std::cout << "4. Test copy and move operations separately\n";
    std::cout << "5. Verify exception safety in copy operations\n";
}

int main() {
    std::cout << "🚀 C++ DEEP VS SHALLOW COPY - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(70, '=') << std::endl;
    
    try {
        demonstrate_shallow_copy_problem();
        demonstrate_deep_copy_solution();
        demonstrate_move_semantics();
        demonstrate_smart_pointer_approach();
        demonstrate_complex_object_copying();
        demonstrate_performance_comparison();
        demonstrate_best_practices();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
