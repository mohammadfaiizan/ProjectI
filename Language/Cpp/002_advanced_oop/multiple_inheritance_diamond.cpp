/*
 * =============================================================================
 * MULTIPLE INHERITANCE DIAMOND - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Multiple inheritance basics
 * 2. Diamond problem and its solutions
 * 3. Virtual inheritance
 * 4. Name resolution and ambiguity
 * 5. Constructor/destructor order in multiple inheritance
 * 6. Interface segregation using multiple inheritance
 * 7. Common pitfalls and best practices
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 multiple_inheritance_diamond.cpp -o multiple_inheritance_diamond
 * =============================================================================
 */

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#define TRACE(msg) std::cout << "[TRACE] " << __FUNCTION__ << ": " << msg << std::endl;

// =============================================================================
// BASIC MULTIPLE INHERITANCE
// =============================================================================

class Flyable {
public:
    Flyable() { TRACE("Flyable constructor"); }
    virtual ~Flyable() { TRACE("Flyable destructor"); }
    
    virtual void fly() const = 0;
    virtual double get_max_altitude() const = 0;
};

class Swimmable {
public:
    Swimmable() { TRACE("Swimmable constructor"); }
    virtual ~Swimmable() { TRACE("Swimmable destructor"); }
    
    virtual void swim() const = 0;
    virtual double get_max_depth() const = 0;
};

class Duck : public Flyable, public Swimmable {
private:
    std::string name;
    
public:
    explicit Duck(const std::string& n) : name(n) {
        TRACE("Duck constructor: " + name);
    }
    
    ~Duck() override {
        TRACE("Duck destructor: " + name);
    }
    
    // Implement Flyable interface
    void fly() const override {
        std::cout << name << " flies gracefully through the air!" << std::endl;
    }
    
    double get_max_altitude() const override {
        return 1000.0;  // meters
    }
    
    // Implement Swimmable interface
    void swim() const override {
        std::cout << name << " swims on the water surface!" << std::endl;
    }
    
    double get_max_depth() const override {
        return 2.0;  // meters (ducks don't dive deep)
    }
    
    void quack() const {
        std::cout << name << " says: Quack! Quack!" << std::endl;
    }
    
    const std::string& get_name() const { return name; }
};

// =============================================================================
// DIAMOND PROBLEM DEMONSTRATION
// =============================================================================

// Base class
class Vehicle {
protected:
    std::string manufacturer;
    int year;
    
public:
    Vehicle(const std::string& mfg, int yr) : manufacturer(mfg), year(yr) {
        TRACE("Vehicle constructor: " + manufacturer + " " + std::to_string(year));
    }
    
    virtual ~Vehicle() {
        TRACE("Vehicle destructor: " + manufacturer);
    }
    
    virtual void start() const {
        std::cout << "Starting " << manufacturer << " vehicle..." << std::endl;
    }
    
    virtual void stop() const {
        std::cout << "Stopping " << manufacturer << " vehicle..." << std::endl;
    }
    
    const std::string& get_manufacturer() const { return manufacturer; }
    int get_year() const { return year; }
};

// First derived class
class LandVehicle : public Vehicle {
protected:
    int num_wheels;
    
public:
    LandVehicle(const std::string& mfg, int yr, int wheels) 
        : Vehicle(mfg, yr), num_wheels(wheels) {
        TRACE("LandVehicle constructor: " + std::to_string(wheels) + " wheels");
    }
    
    ~LandVehicle() override {
        TRACE("LandVehicle destructor");
    }
    
    void drive() const {
        std::cout << "Driving on land with " << num_wheels << " wheels" << std::endl;
    }
    
    int get_num_wheels() const { return num_wheels; }
};

// Second derived class
class WaterVehicle : public Vehicle {
protected:
    std::string propulsion_type;
    
public:
    WaterVehicle(const std::string& mfg, int yr, const std::string& prop) 
        : Vehicle(mfg, yr), propulsion_type(prop) {
        TRACE("WaterVehicle constructor: " + propulsion_type);
    }
    
    ~WaterVehicle() override {
        TRACE("WaterVehicle destructor");
    }
    
    void sail() const {
        std::cout << "Sailing on water using " << propulsion_type << std::endl;
    }
    
    const std::string& get_propulsion_type() const { return propulsion_type; }
};

// Diamond inheritance - PROBLEMATIC without virtual inheritance
class AmphibiousVehicle : public LandVehicle, public WaterVehicle {
private:
    std::string model;
    
public:
    AmphibiousVehicle(const std::string& mfg, int yr, const std::string& mdl) 
        : LandVehicle(mfg, yr, 4),  // This creates one Vehicle instance
          WaterVehicle(mfg, yr, "propeller"),  // This creates another Vehicle instance!
          model(mdl) {
        TRACE("AmphibiousVehicle constructor: " + model);
    }
    
    ~AmphibiousVehicle() override {
        TRACE("AmphibiousVehicle destructor: " + model);
    }
    
    void operate_on_land() const {
        std::cout << model << " operating on land:" << std::endl;
        drive();
        // start();  // ERROR: Ambiguous! Which Vehicle::start()?
    }
    
    void operate_on_water() const {
        std::cout << model << " operating on water:" << std::endl;
        sail();
        // stop();   // ERROR: Ambiguous! Which Vehicle::stop()?
    }
    
    // Resolve ambiguity by explicitly specifying which base class
    void start_vehicle() const {
        LandVehicle::start();  // Choose one explicitly
    }
    
    void stop_vehicle() const {
        WaterVehicle::stop();  // Choose one explicitly
    }
    
    const std::string& get_model() const { return model; }
};

// =============================================================================
// VIRTUAL INHERITANCE SOLUTION
// =============================================================================

// Base class (same as before)
class Device {
protected:
    std::string brand;
    std::string serial_number;
    
public:
    Device(const std::string& b, const std::string& sn) : brand(b), serial_number(sn) {
        TRACE("Device constructor: " + brand + " (" + serial_number + ")");
    }
    
    virtual ~Device() {
        TRACE("Device destructor: " + brand);
    }
    
    virtual void power_on() const {
        std::cout << "Powering on " << brand << " device..." << std::endl;
    }
    
    virtual void power_off() const {
        std::cout << "Powering off " << brand << " device..." << std::endl;
    }
    
    const std::string& get_brand() const { return brand; }
    const std::string& get_serial_number() const { return serial_number; }
};

// Virtual inheritance prevents diamond problem
class AudioDevice : public virtual Device {
protected:
    int volume_level;
    
public:
    AudioDevice(const std::string& b, const std::string& sn, int vol = 50) 
        : Device(b, sn), volume_level(vol) {
        TRACE("AudioDevice constructor: volume " + std::to_string(vol));
    }
    
    ~AudioDevice() override {
        TRACE("AudioDevice destructor");
    }
    
    virtual void play_audio() const {
        std::cout << "Playing audio at volume " << volume_level << std::endl;
    }
    
    void set_volume(int vol) {
        volume_level = (vol >= 0 && vol <= 100) ? vol : volume_level;
    }
    
    int get_volume() const { return volume_level; }
};

class VideoDevice : public virtual Device {  // Virtual inheritance
protected:
    std::string resolution;
    
public:
    VideoDevice(const std::string& b, const std::string& sn, const std::string& res = "1920x1080") 
        : Device(b, sn), resolution(res) {
        TRACE("VideoDevice constructor: " + resolution);
    }
    
    ~VideoDevice() override {
        TRACE("VideoDevice destructor");
    }
    
    virtual void display_video() const {
        std::cout << "Displaying video at " << resolution << " resolution" << std::endl;
    }
    
    void set_resolution(const std::string& res) {
        resolution = res;
    }
    
    const std::string& get_resolution() const { return resolution; }
};

// No diamond problem with virtual inheritance!
class MultimediaDevice : public AudioDevice, public VideoDevice {
private:
    std::string model_name;
    
public:
    MultimediaDevice(const std::string& b, const std::string& sn, const std::string& model) 
        : Device(b, sn),  // Only one Device constructor call needed!
          AudioDevice(b, sn, 75),
          VideoDevice(b, sn, "4K"),
          model_name(model) {
        TRACE("MultimediaDevice constructor: " + model_name);
    }
    
    ~MultimediaDevice() override {
        TRACE("MultimediaDevice destructor: " + model_name);
    }
    
    void play_multimedia() const {
        std::cout << model_name << " playing multimedia content:" << std::endl;
        power_on();      // No ambiguity!
        play_audio();
        display_video();
    }
    
    void stop_multimedia() const {
        std::cout << model_name << " stopping multimedia:" << std::endl;
        power_off();     // No ambiguity!
    }
    
    const std::string& get_model_name() const { return model_name; }
};

// =============================================================================
// INTERFACE SEGREGATION WITH MULTIPLE INHERITANCE
// =============================================================================

class Readable {
public:
    virtual ~Readable() = default;
    virtual void read() const = 0;
    virtual bool can_read() const = 0;
};

class Writable {
public:
    virtual ~Writable() = default;
    virtual void write(const std::string& data) = 0;
    virtual bool can_write() const = 0;
};

class Executable {
public:
    virtual ~Executable() = default;
    virtual void execute() const = 0;
    virtual bool can_execute() const = 0;
};

class File : public Readable, public Writable, public Executable {
private:
    std::string filename;
    std::string content;
    bool read_permission;
    bool write_permission;
    bool execute_permission;
    
public:
    File(const std::string& name, bool r = true, bool w = true, bool x = false) 
        : filename(name), read_permission(r), write_permission(w), execute_permission(x) {
        TRACE("File constructor: " + filename);
    }
    
    ~File() override {
        TRACE("File destructor: " + filename);
    }
    
    // Implement Readable
    void read() const override {
        if (can_read()) {
            std::cout << "Reading file " << filename << ": " << content << std::endl;
        } else {
            std::cout << "Permission denied: Cannot read " << filename << std::endl;
        }
    }
    
    bool can_read() const override {
        return read_permission;
    }
    
    // Implement Writable
    void write(const std::string& data) override {
        if (can_write()) {
            content = data;
            std::cout << "Writing to file " << filename << ": " << data << std::endl;
        } else {
            std::cout << "Permission denied: Cannot write to " << filename << std::endl;
        }
    }
    
    bool can_write() const override {
        return write_permission;
    }
    
    // Implement Executable
    void execute() const override {
        if (can_execute()) {
            std::cout << "Executing file " << filename << std::endl;
        } else {
            std::cout << "Permission denied: Cannot execute " << filename << std::endl;
        }
    }
    
    bool can_execute() const override {
        return execute_permission;
    }
    
    // File-specific methods
    const std::string& get_filename() const { return filename; }
    const std::string& get_content() const { return content; }
    
    void set_permissions(bool r, bool w, bool x) {
        read_permission = r;
        write_permission = w;
        execute_permission = x;
    }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_basic_multiple_inheritance() {
    std::cout << "\n🦆 BASIC MULTIPLE INHERITANCE\n";
    std::cout << std::string(40, '=') << std::endl;
    
    std::cout << "\n📋 Creating duck with multiple capabilities...\n";
    Duck mallard("Mallard");
    
    std::cout << "\n🔍 Using as Flyable:\n";
    Flyable* flyable_duck = &mallard;
    flyable_duck->fly();
    std::cout << "Max altitude: " << flyable_duck->get_max_altitude() << "m" << std::endl;
    
    std::cout << "\n🔍 Using as Swimmable:\n";
    Swimmable* swimmable_duck = &mallard;
    swimmable_duck->swim();
    std::cout << "Max depth: " << swimmable_duck->get_max_depth() << "m" << std::endl;
    
    std::cout << "\n🔍 Using as Duck:\n";
    mallard.quack();
    mallard.fly();
    mallard.swim();
    
    std::cout << "\n📋 Polymorphic usage:\n";
    std::vector<std::unique_ptr<Flyable>> flying_things;
    flying_things.push_back(std::make_unique<Duck>("Flying Duck"));
    
    for (auto& flyer : flying_things) {
        flyer->fly();
    }
}

void demonstrate_diamond_problem() {
    std::cout << "\n💎 DIAMOND PROBLEM DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n⚠️ Creating amphibious vehicle (diamond inheritance)...\n";
    AmphibiousVehicle amphicar("Amphicar", 1965, "Model 770");
    
    std::cout << "\n🔍 Operating on different terrains:\n";
    amphicar.operate_on_land();
    amphicar.operate_on_water();
    
    std::cout << "\n⚠️ Ambiguity resolution required:\n";
    amphicar.start_vehicle();  // Must specify which base class
    amphicar.stop_vehicle();
    
    std::cout << "\n🔍 Memory layout shows duplicate Vehicle instances:\n";
    std::cout << "LandVehicle manufacturer: " << static_cast<LandVehicle&>(amphicar).get_manufacturer() << std::endl;
    std::cout << "WaterVehicle manufacturer: " << static_cast<WaterVehicle&>(amphicar).get_manufacturer() << std::endl;
    
    // These might be different memory addresses due to diamond problem!
    std::cout << "LandVehicle address: " << static_cast<Vehicle*>(static_cast<LandVehicle*>(&amphicar)) << std::endl;
    std::cout << "WaterVehicle address: " << static_cast<Vehicle*>(static_cast<WaterVehicle*>(&amphicar)) << std::endl;
}

void demonstrate_virtual_inheritance_solution() {
    std::cout << "\n✅ VIRTUAL INHERITANCE SOLUTION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    std::cout << "\n📋 Creating multimedia device (virtual inheritance)...\n";
    MultimediaDevice smart_tv("Samsung", "SM-TV-2023", "Smart TV Pro");
    
    std::cout << "\n🔍 No ambiguity with virtual inheritance:\n";
    smart_tv.play_multimedia();
    smart_tv.set_volume(80);
    smart_tv.set_resolution("8K");
    smart_tv.stop_multimedia();
    
    std::cout << "\n🔍 Single Device instance shared:\n";
    std::cout << "Device brand: " << smart_tv.get_brand() << std::endl;
    std::cout << "Audio volume: " << smart_tv.get_volume() << std::endl;
    std::cout << "Video resolution: " << smart_tv.get_resolution() << std::endl;
    
    std::cout << "\n🔍 Memory layout shows single Device instance:\n";
    std::cout << "AudioDevice->Device address: " << static_cast<Device*>(static_cast<AudioDevice*>(&smart_tv)) << std::endl;
    std::cout << "VideoDevice->Device address: " << static_cast<Device*>(static_cast<VideoDevice*>(&smart_tv)) << std::endl;
    std::cout << "Same address = single instance!" << std::endl;
}

void demonstrate_interface_segregation() {
    std::cout << "\n📁 INTERFACE SEGREGATION WITH MULTIPLE INHERITANCE\n";
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << "\n📋 Creating files with different permissions...\n";
    File document("document.txt", true, true, false);    // Read/Write only
    File script("script.sh", true, false, true);         // Read/Execute only
    File config("config.conf", true, true, true);        // Full permissions
    
    std::cout << "\n🔍 Testing document file:\n";
    document.write("This is a document content.");
    document.read();
    document.execute();  // Should fail
    
    std::cout << "\n🔍 Testing script file:\n";
    script.read();
    script.write("malicious code");  // Should fail
    script.execute();
    
    std::cout << "\n🔍 Testing config file:\n";
    config.write("server_port=8080");
    config.read();
    config.execute();
    
    std::cout << "\n📋 Using through interface pointers:\n";
    std::vector<std::unique_ptr<Readable>> readable_files;
    readable_files.push_back(std::make_unique<File>("readable1.txt", true, false, false));
    readable_files.push_back(std::make_unique<File>("readable2.txt", true, true, true));
    
    for (auto& file : readable_files) {
        file->read();
    }
}

void demonstrate_constructor_destructor_order() {
    std::cout << "\n🏗️ CONSTRUCTOR/DESTRUCTOR ORDER IN MULTIPLE INHERITANCE\n";
    std::cout << std::string(65, '=') << std::endl;
    
    std::cout << "\n📋 Creating objects to observe construction/destruction order...\n";
    
    std::cout << "\n--- Basic Multiple Inheritance ---\n";
    {
        Duck order_duck("Order Test Duck");
        std::cout << "Duck created, going out of scope...\n";
    }
    
    std::cout << "\n--- Virtual Inheritance ---\n";
    {
        MultimediaDevice order_device("TestBrand", "TEST-001", "Order Test Device");
        std::cout << "MultimediaDevice created, going out of scope...\n";
    }
    
    std::cout << "\nConstruction order: Base classes in order of inheritance, then derived\n";
    std::cout << "Destruction order: Reverse of construction\n";
    std::cout << "Virtual inheritance: Virtual base constructed first, only once\n";
}

void demonstrate_best_practices() {
    std::cout << "\n📚 BEST PRACTICES AND COMMON PITFALLS\n";
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << "\n✅ BEST PRACTICES:\n";
    std::cout << "1. Use virtual inheritance to solve diamond problem\n";
    std::cout << "2. Keep multiple inheritance hierarchies shallow\n";
    std::cout << "3. Prefer composition over multiple inheritance when possible\n";
    std::cout << "4. Use multiple inheritance mainly for interfaces/mixins\n";
    std::cout << "5. Be explicit about ambiguity resolution\n";
    std::cout << "6. Document inheritance relationships clearly\n";
    
    std::cout << "\n⚠️ COMMON PITFALLS:\n";
    std::cout << "1. Diamond problem without virtual inheritance\n";
    std::cout << "2. Ambiguous function calls in diamond inheritance\n";
    std::cout << "3. Complex constructor initialization in virtual inheritance\n";
    std::cout << "4. Performance overhead of virtual inheritance\n";
    std::cout << "5. Name hiding in multiple inheritance\n";
    std::cout << "6. Overly complex inheritance hierarchies\n";
    
    std::cout << "\n🔍 DEBUGGING TIPS:\n";
    std::cout << "1. Use qualified names to resolve ambiguity\n";
    std::cout << "2. Check memory addresses to verify single/multiple instances\n";
    std::cout << "3. Use debugger to trace constructor/destructor calls\n";
    std::cout << "4. Draw inheritance diagrams for complex hierarchies\n";
    std::cout << "5. Use static_cast to navigate inheritance tree\n";
    
    std::cout << "\n💡 ALTERNATIVES TO CONSIDER:\n";
    std::cout << "1. Composition instead of inheritance\n";
    std::cout << "2. Strategy pattern for behavior selection\n";
    std::cout << "3. Interface segregation with single inheritance\n";
    std::cout << "4. Template-based solutions (CRTP)\n";
    std::cout << "5. Modern C++ concepts for interface definition\n";
}

int main() {
    std::cout << "🚀 C++ MULTIPLE INHERITANCE DIAMOND - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(75, '=') << std::endl;
    
    try {
        demonstrate_basic_multiple_inheritance();
        demonstrate_diamond_problem();
        demonstrate_virtual_inheritance_solution();
        demonstrate_interface_segregation();
        demonstrate_constructor_destructor_order();
        demonstrate_best_practices();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
