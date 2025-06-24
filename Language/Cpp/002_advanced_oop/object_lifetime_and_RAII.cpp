/*
 * =============================================================================
 * OBJECT LIFETIME AND RAII - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Object lifetime concepts (automatic, static, dynamic)
 * 2. RAII (Resource Acquisition Is Initialization) pattern
 * 3. Stack vs heap allocation
 * 4. Smart pointers and automatic resource management
 * 5. Exception safety with RAII
 * 6. Custom resource managers
 * 7. Best practices and common pitfalls
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 object_lifetime_and_RAII.cpp -o object_lifetime_and_RAII
 * =============================================================================
 */

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include <mutex>
#include <thread>
#include <chrono>

#define TRACE(msg) std::cout << "[TRACE] " << msg << std::endl;

// =============================================================================
// OBJECT LIFETIME DEMONSTRATIONS
// =============================================================================

class LifetimeTracker {
private:
    std::string name;
    static int instance_count;
    
public:
    explicit LifetimeTracker(const std::string& n) : name(n) {
        ++instance_count;
        TRACE("LifetimeTracker '" + name + "' created (total: " + std::to_string(instance_count) + ")");
    }
    
    ~LifetimeTracker() {
        --instance_count;
        TRACE("LifetimeTracker '" + name + "' destroyed (remaining: " + std::to_string(instance_count) + ")");
    }
    
    void identify() const {
        std::cout << "I am " << name << " (address: " << this << ")" << std::endl;
    }
    
    static int get_instance_count() { return instance_count; }
};

int LifetimeTracker::instance_count = 0;

// Global object (static lifetime)
LifetimeTracker global_object("Global");

// =============================================================================
// RAII RESOURCE MANAGERS
// =============================================================================

class FileManager {
private:
    std::string filename;
    std::ofstream file;
    bool is_open;
    
public:
    explicit FileManager(const std::string& fname) : filename(fname), is_open(false) {
        file.open(filename);
        if (file.is_open()) {
            is_open = true;
            TRACE("File '" + filename + "' opened successfully");
        } else {
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }
    
    ~FileManager() {
        if (is_open) {
            file.close();
            TRACE("File '" + filename + "' closed automatically");
        }
    }
    
    // Delete copy operations (RAII objects should not be copied)
    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;
    
    // Move operations allowed
    FileManager(FileManager&& other) noexcept 
        : filename(std::move(other.filename)), file(std::move(other.file)), is_open(other.is_open) {
        other.is_open = false;
        TRACE("FileManager moved");
    }
    
    FileManager& operator=(FileManager&& other) noexcept {
        if (this != &other) {
            if (is_open) file.close();
            filename = std::move(other.filename);
            file = std::move(other.file);
            is_open = other.is_open;
            other.is_open = false;
        }
        return *this;
    }
    
    void write(const std::string& data) {
        if (is_open) {
            file << data << std::endl;
            TRACE("Written to file: " + data);
        } else {
            throw std::runtime_error("File is not open");
        }
    }
    
    bool is_file_open() const { return is_open; }
};

class MutexLock {
private:
    std::mutex& mtx;
    bool locked;
    
public:
    explicit MutexLock(std::mutex& m) : mtx(m), locked(false) {
        mtx.lock();
        locked = true;
        TRACE("Mutex locked");
    }
    
    ~MutexLock() {
        if (locked) {
            mtx.unlock();
            TRACE("Mutex unlocked automatically");
        }
    }
    
    // Non-copyable, non-movable
    MutexLock(const MutexLock&) = delete;
    MutexLock& operator=(const MutexLock&) = delete;
    MutexLock(MutexLock&&) = delete;
    MutexLock& operator=(MutexLock&&) = delete;
    
    void unlock() {
        if (locked) {
            mtx.unlock();
            locked = false;
            TRACE("Mutex unlocked manually");
        }
    }
};

class MemoryPool {
private:
    char* pool;
    size_t size;
    size_t used;
    
public:
    explicit MemoryPool(size_t pool_size) : size(pool_size), used(0) {
        pool = new char[size];
        TRACE("Memory pool allocated: " + std::to_string(size) + " bytes");
    }
    
    ~MemoryPool() {
        delete[] pool;
        TRACE("Memory pool deallocated");
    }
    
    // Non-copyable
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // Movable
    MemoryPool(MemoryPool&& other) noexcept 
        : pool(other.pool), size(other.size), used(other.used) {
        other.pool = nullptr;
        other.size = 0;
        other.used = 0;
        TRACE("Memory pool moved");
    }
    
    void* allocate(size_t bytes) {
        if (used + bytes > size) {
            throw std::bad_alloc();
        }
        void* ptr = pool + used;
        used += bytes;
        TRACE("Allocated " + std::to_string(bytes) + " bytes from pool");
        return ptr;
    }
    
    void reset() {
        used = 0;
        TRACE("Memory pool reset");
    }
    
    size_t available() const { return size - used; }
};

// =============================================================================
// SMART POINTER DEMONSTRATIONS
// =============================================================================

class Resource {
private:
    std::string name;
    int* data;
    
public:
    explicit Resource(const std::string& n) : name(n) {
        data = new int[100];  // Simulate resource allocation
        TRACE("Resource '" + name + "' acquired");
    }
    
    ~Resource() {
        delete[] data;
        TRACE("Resource '" + name + "' released");
    }
    
    void use() const {
        std::cout << "Using resource: " << name << std::endl;
    }
    
    const std::string& get_name() const { return name; }
};

class ResourceManager {
private:
    std::vector<std::unique_ptr<Resource>> resources;
    
public:
    ResourceManager() {
        TRACE("ResourceManager created");
    }
    
    ~ResourceManager() {
        TRACE("ResourceManager destroyed (auto-cleanup " + std::to_string(resources.size()) + " resources)");
    }
    
    void add_resource(const std::string& name) {
        resources.push_back(std::make_unique<Resource>(name));
        TRACE("Added resource to manager: " + name);
    }
    
    void use_all_resources() const {
        for (const auto& resource : resources) {
            resource->use();
        }
    }
    
    size_t resource_count() const { return resources.size(); }
};

// =============================================================================
// EXCEPTION SAFETY WITH RAII
// =============================================================================

class ExceptionSafeOperation {
private:
    std::unique_ptr<FileManager> file1;
    std::unique_ptr<FileManager> file2;
    std::unique_ptr<MemoryPool> memory;
    
public:
    ExceptionSafeOperation() {
        TRACE("Starting exception-safe operation");
        
        try {
            // All resources managed by RAII - automatic cleanup on exception
            file1 = std::make_unique<FileManager>("temp1.txt");
            file2 = std::make_unique<FileManager>("temp2.txt");
            memory = std::make_unique<MemoryPool>(1024);
            
            // Simulate some work that might throw
            perform_risky_operation();
            
            TRACE("Exception-safe operation completed successfully");
        } catch (const std::exception& e) {
            TRACE("Exception caught: " + std::string(e.what()));
            // RAII ensures all resources are cleaned up automatically
            throw;  // Re-throw for caller to handle
        }
    }
    
private:
    void perform_risky_operation() {
        file1->write("Starting operation");
        file2->write("Backup data");
        
        void* ptr = memory->allocate(100);
        (void)ptr;  // Suppress unused variable warning
        
        // Simulate potential exception
        static int call_count = 0;
        if (++call_count % 3 == 0) {
            throw std::runtime_error("Simulated operation failure");
        }
        
        file1->write("Operation completed");
        file2->write("Backup completed");
    }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_object_lifetimes() {
    std::cout << "\n⏰ OBJECT LIFETIME DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    std::cout << "\n📋 Global object exists before main():\n";
    global_object.identify();
    std::cout << "Current instance count: " << LifetimeTracker::get_instance_count() << std::endl;
    
    std::cout << "\n📋 Automatic (stack) objects:\n";
    {
        LifetimeTracker auto1("Auto1");
        LifetimeTracker auto2("Auto2");
        
        auto1.identify();
        auto2.identify();
        std::cout << "Instance count in scope: " << LifetimeTracker::get_instance_count() << std::endl;
        
        std::cout << "Leaving scope...\n";
    }
    std::cout << "After scope: " << LifetimeTracker::get_instance_count() << std::endl;
    
    std::cout << "\n📋 Dynamic (heap) objects:\n";
    auto* dynamic1 = new LifetimeTracker("Dynamic1");
    auto* dynamic2 = new LifetimeTracker("Dynamic2");
    
    dynamic1->identify();
    dynamic2->identify();
    std::cout << "Instance count with dynamic objects: " << LifetimeTracker::get_instance_count() << std::endl;
    
    delete dynamic1;
    delete dynamic2;
    std::cout << "After manual deletion: " << LifetimeTracker::get_instance_count() << std::endl;
    
    std::cout << "\n📋 Static local objects:\n";
    auto create_static = []() {
        static LifetimeTracker static_local("StaticLocal");
        static_local.identify();
        return &static_local;
    };
    
    create_static();  // Created on first call
    create_static();  // Same object on subsequent calls
    std::cout << "Static local survives function calls\n";
}

void demonstrate_raii_pattern() {
    std::cout << "\n🔒 RAII PATTERN DEMONSTRATION\n";
    std::cout << std::string(35, '=') << std::endl;
    
    std::cout << "\n📋 File management with RAII:\n";
    {
        try {
            FileManager file("raii_demo.txt");
            file.write("RAII ensures proper cleanup");
            file.write("Even if exceptions occur");
            
            // File automatically closed when going out of scope
        } catch (const std::exception& e) {
            std::cout << "File operation failed: " << e.what() << std::endl;
        }
    }
    std::cout << "File closed automatically\n";
    
    std::cout << "\n📋 Mutex management with RAII:\n";
    std::mutex shared_mutex;
    {
        MutexLock lock(shared_mutex);
        std::cout << "Critical section protected by RAII lock\n";
        // Mutex automatically unlocked when lock goes out of scope
    }
    std::cout << "Mutex unlocked automatically\n";
    
    std::cout << "\n📋 Memory pool with RAII:\n";
    {
        MemoryPool pool(1024);
        void* ptr1 = pool.allocate(100);
        void* ptr2 = pool.allocate(200);
        (void)ptr1; (void)ptr2;  // Suppress warnings
        
        std::cout << "Available memory: " << pool.available() << " bytes\n";
        // Memory pool automatically deallocated
    }
    std::cout << "Memory pool cleaned up automatically\n";
}

void demonstrate_smart_pointers() {
    std::cout << "\n🧠 SMART POINTER RAII\n";
    std::cout << std::string(30, '=') << std::endl;
    
    std::cout << "\n📋 unique_ptr for exclusive ownership:\n";
    {
        auto resource1 = std::make_unique<Resource>("UniqueResource1");
        auto resource2 = std::make_unique<Resource>("UniqueResource2");
        
        resource1->use();
        resource2->use();
        
        // Transfer ownership
        auto resource3 = std::move(resource1);  // resource1 becomes nullptr
        if (resource3) resource3->use();
        if (!resource1) std::cout << "resource1 is now null after move\n";
        
        // Automatic cleanup when unique_ptrs go out of scope
    }
    std::cout << "All unique_ptr resources cleaned up\n";
    
    std::cout << "\n📋 shared_ptr for shared ownership:\n";
    {
        auto shared1 = std::make_shared<Resource>("SharedResource");
        std::cout << "Reference count: " << shared1.use_count() << std::endl;
        
        {
            auto shared2 = shared1;  // Share ownership
            std::cout << "Reference count after sharing: " << shared1.use_count() << std::endl;
            shared2->use();
        }
        
        std::cout << "Reference count after inner scope: " << shared1.use_count() << std::endl;
        shared1->use();
        
        // Resource destroyed when last shared_ptr goes out of scope
    }
    std::cout << "Shared resource cleaned up\n";
    
    std::cout << "\n📋 Resource manager with RAII:\n";
    {
        ResourceManager manager;
        manager.add_resource("ManagedResource1");
        manager.add_resource("ManagedResource2");
        manager.add_resource("ManagedResource3");
        
        manager.use_all_resources();
        std::cout << "Managing " << manager.resource_count() << " resources\n";
        
        // All resources automatically cleaned up when manager is destroyed
    }
    std::cout << "Resource manager and all resources cleaned up\n";
}

void demonstrate_exception_safety() {
    std::cout << "\n⚠️ EXCEPTION SAFETY WITH RAII\n";
    std::cout << std::string(40, '=') << std::endl;
    
    std::cout << "\n📋 Exception-safe operations:\n";
    
    for (int attempt = 1; attempt <= 4; ++attempt) {
        std::cout << "\nAttempt " << attempt << ":\n";
        try {
            ExceptionSafeOperation operation;
            std::cout << "✅ Operation succeeded\n";
        } catch (const std::exception& e) {
            std::cout << "❌ Operation failed: " << e.what() << std::endl;
            std::cout << "🔒 RAII ensured proper cleanup despite exception\n";
        }
    }
    
    std::cout << "\n💡 Key benefits of RAII with exceptions:\n";
    std::cout << "1. Automatic cleanup even when exceptions occur\n";
    std::cout << "2. No resource leaks\n";
    std::cout << "3. Exception-safe code by design\n";
    std::cout << "4. Deterministic resource management\n";
}

void demonstrate_performance_considerations() {
    std::cout << "\n⚡ PERFORMANCE CONSIDERATIONS\n";
    std::cout << std::string(40, '=') << std::endl;
    
    const int NUM_OBJECTS = 100000;
    
    // Stack allocation performance
    auto start = std::chrono::high_resolution_clock::now();
    {
        std::vector<LifetimeTracker> stack_objects;
        stack_objects.reserve(NUM_OBJECTS);
        for (int i = 0; i < NUM_OBJECTS; ++i) {
            stack_objects.emplace_back("Stack" + std::to_string(i));
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto stack_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Heap allocation with smart pointers
    start = std::chrono::high_resolution_clock::now();
    {
        std::vector<std::unique_ptr<LifetimeTracker>> heap_objects;
        heap_objects.reserve(NUM_OBJECTS);
        for (int i = 0; i < NUM_OBJECTS; ++i) {
            heap_objects.push_back(std::make_unique<LifetimeTracker>("Heap" + std::to_string(i)));
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto heap_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "📊 Performance comparison (" << NUM_OBJECTS << " objects):\n";
    std::cout << "  Stack allocation: " << stack_duration.count() << " μs\n";
    std::cout << "  Heap allocation:  " << heap_duration.count() << " μs\n";
    std::cout << "  Heap overhead:    " << (heap_duration.count() - stack_duration.count()) << " μs\n";
    
    std::cout << "\n💡 Performance guidelines:\n";
    std::cout << "1. Prefer stack allocation when possible\n";
    std::cout << "2. Use smart pointers for heap allocation\n";
    std::cout << "3. RAII has minimal runtime overhead\n";
    std::cout << "4. Exception safety comes at little cost\n";
}

void demonstrate_best_practices() {
    std::cout << "\n📚 BEST PRACTICES AND GUIDELINES\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n✅ RAII BEST PRACTICES:\n";
    std::cout << "1. Acquire resources in constructors\n";
    std::cout << "2. Release resources in destructors\n";
    std::cout << "3. Make RAII objects non-copyable when appropriate\n";
    std::cout << "4. Use smart pointers for automatic memory management\n";
    std::cout << "5. Prefer stack allocation over heap allocation\n";
    std::cout << "6. Use RAII for all resource types (files, mutexes, memory, etc.)\n";
    
    std::cout << "\n⚠️ COMMON PITFALLS:\n";
    std::cout << "1. Forgetting to implement proper destructors\n";
    std::cout << "2. Resource leaks in exception scenarios\n";
    std::cout << "3. Manual resource management alongside RAII\n";
    std::cout << "4. Copying RAII objects inappropriately\n";
    std::cout << "5. Not using smart pointers for dynamic allocation\n";
    std::cout << "6. Circular references with shared_ptr\n";
    
    std::cout << "\n🔍 DEBUGGING TIPS:\n";
    std::cout << "1. Use TRACE macros to track object lifetimes\n";
    std::cout << "2. Check for resource leaks with valgrind/AddressSanitizer\n";
    std::cout << "3. Monitor constructor/destructor pairing\n";
    std::cout << "4. Test exception scenarios thoroughly\n";
    std::cout << "5. Use static analysis tools to detect RAII violations\n";
    
    std::cout << "\n🎯 MODERN C++ RAII PATTERNS:\n";
    std::cout << "1. std::unique_ptr for exclusive ownership\n";
    std::cout << "2. std::shared_ptr for shared ownership\n";
    std::cout << "3. std::weak_ptr to break cycles\n";
    std::cout << "4. std::lock_guard for mutex management\n";
    std::cout << "5. Custom RAII wrappers for C APIs\n";
    std::cout << "6. std::unique_lock for flexible locking\n";
}

int main() {
    std::cout << "🚀 C++ OBJECT LIFETIME AND RAII - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(70, '=') << std::endl;
    
    try {
        demonstrate_object_lifetimes();
        demonstrate_raii_pattern();
        demonstrate_smart_pointers();
        demonstrate_exception_safety();
        demonstrate_performance_considerations();
        demonstrate_best_practices();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🌟 Global object will be destroyed after main()\n";
    return 0;
}
