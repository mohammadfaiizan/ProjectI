/*
 * =============================================================================
 * INHERITANCE VIRTUAL OVERRIDE - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Basic inheritance concepts
 * 2. Virtual functions and polymorphism
 * 3. Override specifier (C++11)
 * 4. Pure virtual functions and abstract classes
 * 5. Virtual destructors
 * 6. Virtual function tables (vtables)
 * 7. Performance implications
 * 8. Best practices and common pitfalls
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 inheritance_virtual_override.cpp -o inheritance_virtual_override
 * =============================================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <typeinfo>
#include <chrono>

#define TRACE(msg) std::cout << "[TRACE] " << __FUNCTION__ << ": " << msg << std::endl;

// =============================================================================
// BASIC INHERITANCE HIERARCHY
// =============================================================================

class Animal {
protected:
    std::string name;
    int age;
    
public:
    Animal(const std::string& n, int a) : name(n), age(a) {
        TRACE("Animal constructor: " + name);
    }
    
    virtual ~Animal() {  // Virtual destructor - IMPORTANT!
        TRACE("Animal destructor: " + name);
    }
    
    // Non-virtual function
    void sleep() const {
        std::cout << name << " is sleeping..." << std::endl;
    }
    
    // Virtual function with default implementation
    virtual void make_sound() const {
        std::cout << name << " makes a generic animal sound" << std::endl;
    }
    
    // Virtual function to be overridden
    virtual void move() const {
        std::cout << name << " moves in some way" << std::endl;
    }
    
    // Pure virtual function (makes this class abstract)
    virtual void eat() const = 0;
    
    // Getters
    const std::string& get_name() const { return name; }
    int get_age() const { return age; }
    
    // Virtual function for type identification
    virtual std::string get_type() const { return "Animal"; }
};

class Mammal : public Animal {
protected:
    bool has_fur;
    
public:
    Mammal(const std::string& n, int a, bool fur) 
        : Animal(n, a), has_fur(fur) {
        TRACE("Mammal constructor: " + name);
    }
    
    ~Mammal() override {  // override specifier (C++11)
        TRACE("Mammal destructor: " + name);
    }
    
    // Override virtual function
    void move() const override {
        std::cout << name << " walks on legs" << std::endl;
    }
    
    // Implement pure virtual function
    void eat() const override {
        std::cout << name << " eats food suitable for mammals" << std::endl;
    }
    
    // New virtual function
    virtual void regulate_temperature() const {
        std::cout << name << " regulates body temperature" << std::endl;
    }
    
    std::string get_type() const override { return "Mammal"; }
    bool get_has_fur() const { return has_fur; }
};

class Dog : public Mammal {
private:
    std::string breed;
    
public:
    Dog(const std::string& n, int a, const std::string& b) 
        : Mammal(n, a, true), breed(b) {
        TRACE("Dog constructor: " + name + " (" + breed + ")");
    }
    
    ~Dog() override {
        TRACE("Dog destructor: " + name);
    }
    
    // Override virtual functions
    void make_sound() const override {
        std::cout << name << " barks: Woof! Woof!" << std::endl;
    }
    
    void eat() const override {
        std::cout << name << " eats dog food and treats" << std::endl;
    }
    
    void regulate_temperature() const override {
        std::cout << name << " pants to cool down" << std::endl;
    }
    
    // New function specific to Dog
    void fetch() const {
        std::cout << name << " fetches the ball!" << std::endl;
    }
    
    std::string get_type() const override { return "Dog (" + breed + ")"; }
    const std::string& get_breed() const { return breed; }
};

class Cat : public Mammal {
private:
    bool is_indoor;
    
public:
    Cat(const std::string& n, int a, bool indoor) 
        : Mammal(n, a, true), is_indoor(indoor) {
        TRACE("Cat constructor: " + name);
    }
    
    ~Cat() override {
        TRACE("Cat destructor: " + name);
    }
    
    // Override virtual functions
    void make_sound() const override {
        std::cout << name << " meows: Meow! Meow!" << std::endl;
    }
    
    void move() const override {
        std::cout << name << " moves gracefully and silently" << std::endl;
    }
    
    void eat() const override {
        std::cout << name << " eats cat food and catches mice" << std::endl;
    }
    
    // New function specific to Cat
    void purr() const {
        std::cout << name << " purrs contentedly" << std::endl;
    }
    
    std::string get_type() const override { 
        return is_indoor ? "Indoor Cat" : "Outdoor Cat"; 
    }
    
    bool get_is_indoor() const { return is_indoor; }
};

// =============================================================================
// BIRD HIERARCHY (DIFFERENT BRANCH)
// =============================================================================

class Bird : public Animal {
protected:
    double wingspan;
    bool can_fly;
    
public:
    Bird(const std::string& n, int a, double ws, bool fly) 
        : Animal(n, a), wingspan(ws), can_fly(fly) {
        TRACE("Bird constructor: " + name);
    }
    
    ~Bird() override {
        TRACE("Bird destructor: " + name);
    }
    
    void move() const override {
        if (can_fly) {
            std::cout << name << " flies through the air" << std::endl;
        } else {
            std::cout << name << " walks or hops on the ground" << std::endl;
        }
    }
    
    void eat() const override {
        std::cout << name << " eats seeds, insects, or other bird food" << std::endl;
    }
    
    virtual void build_nest() const {
        std::cout << name << " builds a nest for eggs" << std::endl;
    }
    
    std::string get_type() const override { return "Bird"; }
    double get_wingspan() const { return wingspan; }
    bool get_can_fly() const { return can_fly; }
};

class Eagle : public Bird {
public:
    Eagle(const std::string& n, int a) 
        : Bird(n, a, 2.3, true) {  // Eagles have ~2.3m wingspan
        TRACE("Eagle constructor: " + name);
    }
    
    ~Eagle() override {
        TRACE("Eagle destructor: " + name);
    }
    
    void make_sound() const override {
        std::cout << name << " screeches: Screech!" << std::endl;
    }
    
    void eat() const override {
        std::cout << name << " hunts fish and small mammals" << std::endl;
    }
    
    // Eagle-specific behavior
    void soar() const {
        std::cout << name << " soars majestically at high altitudes" << std::endl;
    }
    
    std::string get_type() const override { return "Eagle"; }
};

class Penguin : public Bird {
public:
    Penguin(const std::string& n, int a) 
        : Bird(n, a, 0.0, false) {  // Penguins can't fly
        TRACE("Penguin constructor: " + name);
    }
    
    ~Penguin() override {
        TRACE("Penguin destructor: " + name);
    }
    
    void make_sound() const override {
        std::cout << name << " makes penguin sounds: Honk!" << std::endl;
    }
    
    void move() const override {
        std::cout << name << " waddles on land and swims in water" << std::endl;
    }
    
    void eat() const override {
        std::cout << name << " catches fish and krill" << std::endl;
    }
    
    // Penguin-specific behavior
    void swim() const {
        std::cout << name << " swims gracefully underwater" << std::endl;
    }
    
    std::string get_type() const override { return "Penguin"; }
};

// =============================================================================
// INTERFACE EXAMPLE (PURE ABSTRACT CLASS)
// =============================================================================

class Trainable {
public:
    virtual ~Trainable() = default;
    
    // Pure virtual functions define the interface
    virtual void learn_trick(const std::string& trick) = 0;
    virtual void perform_trick(const std::string& trick) const = 0;
    virtual void get_reward() const = 0;
    virtual std::vector<std::string> list_tricks() const = 0;
};

class TrainableDog : public Dog, public Trainable {
private:
    std::vector<std::string> known_tricks;
    
public:
    TrainableDog(const std::string& n, int a, const std::string& b) 
        : Dog(n, a, b) {
        TRACE("TrainableDog constructor: " + name);
    }
    
    ~TrainableDog() override {
        TRACE("TrainableDog destructor: " + name);
    }
    
    // Implement Trainable interface
    void learn_trick(const std::string& trick) override {
        known_tricks.push_back(trick);
        std::cout << name << " learned: " << trick << std::endl;
    }
    
    void perform_trick(const std::string& trick) const override {
        auto it = std::find(known_tricks.begin(), known_tricks.end(), trick);
        if (it != known_tricks.end()) {
            std::cout << name << " performs: " << trick << "!" << std::endl;
        } else {
            std::cout << name << " doesn't know how to: " << trick << std::endl;
        }
    }
    
    void get_reward() const override {
        std::cout << name << " gets a treat and praise!" << std::endl;
    }
    
    std::vector<std::string> list_tricks() const override {
        return known_tricks;
    }
    
    std::string get_type() const override { 
        return "Trainable " + Dog::get_type(); 
    }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_basic_inheritance() {
    std::cout << "\n🏗️ BASIC INHERITANCE DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n📋 Creating animals through inheritance hierarchy...\n";
    
    Dog buddy("Buddy", 3, "Golden Retriever");
    Cat whiskers("Whiskers", 2, true);
    Eagle freedom("Freedom", 5);
    Penguin wadsworth("Wadsworth", 4);
    
    std::vector<std::unique_ptr<Animal>> animals;
    animals.push_back(std::make_unique<Dog>("Rex", 4, "German Shepherd"));
    animals.push_back(std::make_unique<Cat>("Luna", 1, false));
    animals.push_back(std::make_unique<Eagle>("Soar", 7));
    animals.push_back(std::make_unique<Penguin>("Pip", 2));
    
    std::cout << "\n🔍 Demonstrating polymorphism through base pointer...\n";
    for (const auto& animal : animals) {
        std::cout << "\n--- " << animal->get_type() << " ---\n";
        animal->make_sound();
        animal->move();
        animal->eat();
        animal->sleep();
    }
}

void demonstrate_virtual_function_behavior() {
    std::cout << "\n🔄 VIRTUAL FUNCTION BEHAVIOR\n";
    std::cout << std::string(40, '=') << std::endl;
    
    std::cout << "\n📋 Creating objects and calling through base pointers...\n";
    
    std::unique_ptr<Animal> animal1 = std::make_unique<Dog>("Virtual Dog", 3, "Labrador");
    std::unique_ptr<Animal> animal2 = std::make_unique<Cat>("Virtual Cat", 2, true);
    
    std::cout << "\n🔍 Virtual function calls (runtime polymorphism):\n";
    std::cout << "Dog through Animal pointer:\n";
    animal1->make_sound();  // Calls Dog::make_sound()
    animal1->move();        // Calls Mammal::move()
    animal1->eat();         // Calls Dog::eat()
    
    std::cout << "\nCat through Animal pointer:\n";
    animal2->make_sound();  // Calls Cat::make_sound()
    animal2->move();        // Calls Cat::move()
    animal2->eat();         // Calls Cat::eat()
    
    std::cout << "\n🔍 Non-virtual function call:\n";
    animal1->sleep();  // Calls Animal::sleep() (not virtual)
    animal2->sleep();  // Calls Animal::sleep() (not virtual)
}

void demonstrate_override_specifier() {
    std::cout << "\n✅ OVERRIDE SPECIFIER DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\nThe 'override' specifier (C++11) provides:\n";
    std::cout << "1. Compile-time verification that function is actually overriding\n";
    std::cout << "2. Protection against typos in function signatures\n";
    std::cout << "3. Clear documentation of intent\n";
    std::cout << "4. Better error messages when override fails\n";
    
    std::cout << "\n📋 Example of override in action...\n";
    Dog override_dog("Override Example", 2, "Beagle");
    override_dog.make_sound();  // Uses override
    
    std::cout << "\n💡 Without 'override', typos would create new functions instead of overriding!\n";
}

void demonstrate_pure_virtual_and_abstract() {
    std::cout << "\n🎭 PURE VIRTUAL AND ABSTRACT CLASSES\n";
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << "\n📋 Cannot instantiate abstract base class:\n";
    std::cout << "// Animal abstract_animal;  // ERROR: Cannot instantiate abstract class\n";
    
    std::cout << "\n📋 Must implement all pure virtual functions in derived classes:\n";
    Dog concrete_dog("Concrete", 1, "Poodle");
    concrete_dog.eat();  // Implementation required
    
    std::cout << "\n🔍 Abstract classes can have:\n";
    std::cout << "1. Pure virtual functions (= 0)\n";
    std::cout << "2. Regular virtual functions with implementations\n";
    std::cout << "3. Non-virtual functions\n";
    std::cout << "4. Data members\n";
    std::cout << "5. Constructors and destructors\n";
}

void demonstrate_multiple_inheritance_interface() {
    std::cout << "\n🎯 MULTIPLE INHERITANCE WITH INTERFACES\n";
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << "\n📋 Creating trainable dog (multiple inheritance)...\n";
    TrainableDog smart_dog("Einstein", 2, "Border Collie");
    
    std::cout << "\n📋 Using as Dog:\n";
    smart_dog.make_sound();
    smart_dog.fetch();
    
    std::cout << "\n📋 Using as Trainable:\n";
    smart_dog.learn_trick("sit");
    smart_dog.learn_trick("stay");
    smart_dog.learn_trick("roll over");
    
    smart_dog.perform_trick("sit");
    smart_dog.perform_trick("dance");  // Not learned
    smart_dog.get_reward();
    
    std::cout << "\n📋 Known tricks: ";
    auto tricks = smart_dog.list_tricks();
    for (size_t i = 0; i < tricks.size(); ++i) {
        std::cout << tricks[i];
        if (i < tricks.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    std::cout << "\n📋 Using through interface pointer...\n";
    std::unique_ptr<Trainable> trainable_ptr = std::make_unique<TrainableDog>("Genius", 3, "Australian Shepherd");
    trainable_ptr->learn_trick("play dead");
    trainable_ptr->perform_trick("play dead");
    trainable_ptr->get_reward();
}

void demonstrate_virtual_destructor_importance() {
    std::cout << "\n💀 VIRTUAL DESTRUCTOR IMPORTANCE\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n📋 Proper polymorphic destruction with virtual destructors...\n";
    {
        std::vector<std::unique_ptr<Animal>> zoo;
        zoo.push_back(std::make_unique<Dog>("Destructor Dog", 1, "Husky"));
        zoo.push_back(std::make_unique<Cat>("Destructor Cat", 1, false));
        zoo.push_back(std::make_unique<Eagle>("Destructor Eagle", 1));
        
        std::cout << "Creating animals in zoo...\n";
        std::cout << "Zoo going out of scope - watch destruction order...\n";
    }
    std::cout << "All animals properly destroyed!\n";
    
    std::cout << "\n⚠️ Without virtual destructors:\n";
    std::cout << "1. Only base class destructor would be called\n";
    std::cout << "2. Derived class resources might leak\n";
    std::cout << "3. Undefined behavior in polymorphic destruction\n";
}

void demonstrate_performance_implications() {
    std::cout << "\n⚡ PERFORMANCE IMPLICATIONS\n";
    std::cout << std::string(40, '=') << std::endl;
    
    const int NUM_CALLS = 1000000;
    
    // Direct function call performance
    Dog performance_dog("Speed", 1, "Greyhound");
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_CALLS; ++i) {
        performance_dog.make_sound();  // Direct call
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto direct_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Virtual function call performance
    std::unique_ptr<Animal> virtual_dog = std::make_unique<Dog>("Virtual Speed", 1, "Whippet");
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_CALLS; ++i) {
        virtual_dog->make_sound();  // Virtual call
    }
    end = std::chrono::high_resolution_clock::now();
    auto virtual_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "📊 Performance comparison (" << NUM_CALLS << " calls):\n";
    std::cout << "  Direct calls: " << direct_duration.count() << " μs\n";
    std::cout << "  Virtual calls: " << virtual_duration.count() << " μs\n";
    std::cout << "  Overhead: " << (virtual_duration.count() - direct_duration.count()) << " μs\n";
    std::cout << "  Virtual call overhead: ~" << 
        ((double)(virtual_duration.count() - direct_duration.count()) / NUM_CALLS * 1000) << " ns per call\n";
}

void demonstrate_best_practices() {
    std::cout << "\n📚 BEST PRACTICES AND GUIDELINES\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n✅ BEST PRACTICES:\n";
    std::cout << "1. Always use virtual destructors in base classes\n";
    std::cout << "2. Use 'override' specifier for clarity and safety\n";
    std::cout << "3. Prefer pure virtual functions for interfaces\n";
    std::cout << "4. Keep virtual function signatures consistent\n";
    std::cout << "5. Use 'final' to prevent further overriding\n";
    std::cout << "6. Consider performance implications of virtual calls\n";
    
    std::cout << "\n⚠️ COMMON PITFALLS:\n";
    std::cout << "1. Forgetting virtual destructors in base classes\n";
    std::cout << "2. Typos in override function signatures\n";
    std::cout << "3. Calling virtual functions in constructors/destructors\n";
    std::cout << "4. Slicing objects when copying polymorphic types\n";
    std::cout << "5. Not understanding virtual function resolution\n";
    std::cout << "6. Overusing virtual functions for simple cases\n";
    
    std::cout << "\n🔍 DEBUGGING TIPS:\n";
    std::cout << "1. Use 'typeid' to check actual object types\n";
    std::cout << "2. Set breakpoints in virtual functions to trace calls\n";
    std::cout << "3. Use debugger to examine vtable contents\n";
    std::cout << "4. Check for proper destructor chaining\n";
    std::cout << "5. Verify override relationships with compiler warnings\n";
}

int main() {
    std::cout << "🚀 C++ INHERITANCE VIRTUAL OVERRIDE - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(75, '=') << std::endl;
    
    try {
        demonstrate_basic_inheritance();
        demonstrate_virtual_function_behavior();
        demonstrate_override_specifier();
        demonstrate_pure_virtual_and_abstract();
        demonstrate_multiple_inheritance_interface();
        demonstrate_virtual_destructor_importance();
        demonstrate_performance_implications();
        demonstrate_best_practices();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
