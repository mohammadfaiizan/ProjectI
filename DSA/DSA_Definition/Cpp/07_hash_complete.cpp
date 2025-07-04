/*
 * =============================================================================
 * COMPLETE HASH TABLE GUIDE - All Implementations & Techniques
 * =============================================================================
 * 
 * This file covers:
 * 1. Hash Table with Linear Probing
 * 2. Hash Table with Quadratic Probing
 * 3. Hash Table with Chaining (Separate Chaining)
 * 4. STL unordered_map and unordered_set operations
 * 5. Hash functions and collision handling
 * 6. Hash table applications and algorithms
 * 
 * =============================================================================
 */

#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>
using namespace std;

// =============================================================================
// HASH TABLE WITH LINEAR PROBING
// =============================================================================

template<typename K, typename V>
class LinearProbingHashTable {
private:
    struct HashEntry {
        K key;
        V value;
        bool is_occupied;
        bool is_deleted;
        
        HashEntry() : is_occupied(false), is_deleted(false) {}
        HashEntry(const K& k, const V& v) : key(k), value(v), is_occupied(true), is_deleted(false) {}
    };
    
    vector<HashEntry> table;
    int capacity;
    int size;
    int deleted_count;
    
    // Hash function for integers
    int hash(int key) const {
        return key % capacity;
    }
    
    // Hash function for strings
    int hash(const string& key) const {
        int hash_value = 0;
        for (char c : key) {
            hash_value = (hash_value * 31 + c) % capacity;
        }
        return hash_value;
    }
    
    // Generic hash function
    int hash(const K& key) const {
        if constexpr (is_same_v<K, int>) {
            return key % capacity;
        } else if constexpr (is_same_v<K, string>) {
            int hash_value = 0;
            for (char c : key) {
                hash_value = (hash_value * 31 + c) % capacity;
            }
            return hash_value;
        }
        return 0;
    }
    
    void rehash() {
        vector<HashEntry> old_table = table;
        int old_capacity = capacity;
        
        capacity *= 2;
        table.clear();
        table.resize(capacity);
        size = 0;
        deleted_count = 0;
        
        cout << "Rehashing from capacity " << old_capacity << " to " << capacity << endl;
        
        for (const auto& entry : old_table) {
            if (entry.is_occupied && !entry.is_deleted) {
                insert(entry.key, entry.value);
            }
        }
    }

public:
    // Constructor
    LinearProbingHashTable(int initial_capacity = 11) 
        : capacity(initial_capacity), size(0), deleted_count(0) {
        table.resize(capacity);
        cout << "Linear probing hash table created with capacity: " << capacity << endl;
    }
    
    // Destructor
    ~LinearProbingHashTable() {
        cout << "Linear probing hash table destroyed" << endl;
    }
    
    // Insert operation
    void insert(const K& key, const V& value) {
        if (size + deleted_count >= capacity * 0.7) {
            rehash();
        }
        
        int index = hash(key);
        int original_index = index;
        
        while (table[index].is_occupied && !table[index].is_deleted) {
            if (table[index].key == key) {
                table[index].value = value;
                cout << "Updated key " << key << " with value " << value << endl;
                return;
            }
            index = (index + 1) % capacity;
            
            if (index == original_index) {
                cout << "Hash table is full!" << endl;
                return;
            }
        }
        
        if (table[index].is_deleted) {
            deleted_count--;
        }
        
        table[index] = HashEntry(key, value);
        size++;
        cout << "Inserted (" << key << ", " << value << ") at index " << index << endl;
    }
    
    // Search operation
    bool search(const K& key, V& value) const {
        int index = hash(key);
        int original_index = index;
        
        while (table[index].is_occupied || table[index].is_deleted) {
            if (table[index].is_occupied && !table[index].is_deleted && table[index].key == key) {
                value = table[index].value;
                return true;
            }
            index = (index + 1) % capacity;
            
            if (index == original_index) {
                break;
            }
        }
        
        return false;
    }
    
    // Delete operation
    bool remove(const K& key) {
        int index = hash(key);
        int original_index = index;
        
        while (table[index].is_occupied || table[index].is_deleted) {
            if (table[index].is_occupied && !table[index].is_deleted && table[index].key == key) {
                table[index].is_deleted = true;
                table[index].is_occupied = false;
                size--;
                deleted_count++;
                cout << "Deleted key " << key << " from index " << index << endl;
                return true;
            }
            index = (index + 1) % capacity;
            
            if (index == original_index) {
                break;
            }
        }
        
        cout << "Key " << key << " not found for deletion" << endl;
        return false;
    }
    
    // Display methods
    void display() const {
        cout << "Linear Probing Hash Table:" << endl;
        for (int i = 0; i < capacity; i++) {
            cout << "Index " << i << ": ";
            if (table[i].is_occupied && !table[i].is_deleted) {
                cout << "(" << table[i].key << ", " << table[i].value << ")";
            } else if (table[i].is_deleted) {
                cout << "DELETED";
            } else {
                cout << "EMPTY";
            }
            cout << endl;
        }
    }
    
    void printInfo() const {
        double load_factor = static_cast<double>(size) / capacity;
        cout << "Linear Probing Hash Table Info:" << endl;
        cout << "Size: " << size << endl;
        cout << "Capacity: " << capacity << endl;
        cout << "Deleted entries: " << deleted_count << endl;
        cout << "Load factor: " << load_factor << endl;
        cout << "Empty slots: " << capacity - size - deleted_count << endl;
    }
    
    // Utility methods
    int getSize() const { return size; }
    int getCapacity() const { return capacity; }
    bool isEmpty() const { return size == 0; }
    double getLoadFactor() const { return static_cast<double>(size) / capacity; }
};

// =============================================================================
// HASH TABLE WITH QUADRATIC PROBING
// =============================================================================

template<typename K, typename V>
class QuadraticProbingHashTable {
private:
    struct HashEntry {
        K key;
        V value;
        bool is_occupied;
        bool is_deleted;
        
        HashEntry() : is_occupied(false), is_deleted(false) {}
        HashEntry(const K& k, const V& v) : key(k), value(v), is_occupied(true), is_deleted(false) {}
    };
    
    vector<HashEntry> table;
    int capacity;
    int size;
    
    int hash(const K& key) const {
        if constexpr (is_same_v<K, int>) {
            return key % capacity;
        } else if constexpr (is_same_v<K, string>) {
            int hash_value = 0;
            for (char c : key) {
                hash_value = (hash_value * 31 + c) % capacity;
            }
            return hash_value;
        }
        return 0;
    }

public:
    // Constructor
    QuadraticProbingHashTable(int initial_capacity = 11) 
        : capacity(initial_capacity), size(0) {
        table.resize(capacity);
        cout << "Quadratic probing hash table created with capacity: " << capacity << endl;
    }
    
    // Insert operation
    void insert(const K& key, const V& value) {
        if (size >= capacity * 0.5) {
            cout << "Hash table load factor too high for quadratic probing!" << endl;
            return;
        }
        
        int index = hash(key);
        int i = 0;
        
        while (table[index].is_occupied && !table[index].is_deleted) {
            if (table[index].key == key) {
                table[index].value = value;
                cout << "Updated key " << key << " with value " << value << endl;
                return;
            }
            i++;
            index = (hash(key) + i * i) % capacity;
            
            if (i >= capacity) {
                cout << "Could not insert - table full or no suitable position found" << endl;
                return;
            }
        }
        
        table[index] = HashEntry(key, value);
        size++;
        cout << "Inserted (" << key << ", " << value << ") at index " << index 
             << " (probe sequence: " << i << ")" << endl;
    }
    
    // Search operation
    bool search(const K& key, V& value) const {
        int index = hash(key);
        int i = 0;
        
        while (i < capacity) {
            if (!table[index].is_occupied && !table[index].is_deleted) {
                break;
            }
            
            if (table[index].is_occupied && !table[index].is_deleted && table[index].key == key) {
                value = table[index].value;
                return true;
            }
            
            i++;
            index = (hash(key) + i * i) % capacity;
        }
        
        return false;
    }
    
    // Display methods
    void display() const {
        cout << "Quadratic Probing Hash Table:" << endl;
        for (int i = 0; i < capacity; i++) {
            cout << "Index " << i << ": ";
            if (table[i].is_occupied && !table[i].is_deleted) {
                cout << "(" << table[i].key << ", " << table[i].value << ")";
            } else if (table[i].is_deleted) {
                cout << "DELETED";
            } else {
                cout << "EMPTY";
            }
            cout << endl;
        }
    }
    
    void printInfo() const {
        double load_factor = static_cast<double>(size) / capacity;
        cout << "Quadratic Probing Hash Table Info:" << endl;
        cout << "Size: " << size << endl;
        cout << "Capacity: " << capacity << endl;
        cout << "Load factor: " << load_factor << endl;
    }
};

// =============================================================================
// HASH TABLE WITH CHAINING (SEPARATE CHAINING)
// =============================================================================

template<typename K, typename V>
class ChainingHashTable {
private:
    struct KeyValuePair {
        K key;
        V value;
        
        KeyValuePair(const K& k, const V& v) : key(k), value(v) {}
    };
    
    vector<list<KeyValuePair>> table;
    int capacity;
    int size;
    
    int hash(const K& key) const {
        if constexpr (is_same_v<K, int>) {
            return key % capacity;
        } else if constexpr (is_same_v<K, string>) {
            int hash_value = 0;
            for (char c : key) {
                hash_value = (hash_value * 31 + c) % capacity;
            }
            return hash_value;
        }
        return 0;
    }
    
    void rehash() {
        vector<list<KeyValuePair>> old_table = table;
        int old_capacity = capacity;
        
        capacity *= 2;
        table.clear();
        table.resize(capacity);
        size = 0;
        
        cout << "Rehashing chaining hash table from " << old_capacity << " to " << capacity << endl;
        
        for (const auto& chain : old_table) {
            for (const auto& pair : chain) {
                insert(pair.key, pair.value);
            }
        }
    }

public:
    // Constructor
    ChainingHashTable(int initial_capacity = 11) 
        : capacity(initial_capacity), size(0) {
        table.resize(capacity);
        cout << "Chaining hash table created with capacity: " << capacity << endl;
    }
    
    // Insert operation
    void insert(const K& key, const V& value) {
        if (size >= capacity * 2) {
            rehash();
        }
        
        int index = hash(key);
        
        // Check if key already exists
        for (auto& pair : table[index]) {
            if (pair.key == key) {
                pair.value = value;
                cout << "Updated key " << key << " with value " << value << " in chain " << index << endl;
                return;
            }
        }
        
        // Add new key-value pair
        table[index].emplace_back(key, value);
        size++;
        cout << "Inserted (" << key << ", " << value << ") in chain " << index 
             << " (chain size: " << table[index].size() << ")" << endl;
    }
    
    // Search operation
    bool search(const K& key, V& value) const {
        int index = hash(key);
        
        for (const auto& pair : table[index]) {
            if (pair.key == key) {
                value = pair.value;
                return true;
            }
        }
        
        return false;
    }
    
    // Delete operation
    bool remove(const K& key) {
        int index = hash(key);
        
        auto& chain = table[index];
        for (auto it = chain.begin(); it != chain.end(); ++it) {
            if (it->key == key) {
                chain.erase(it);
                size--;
                cout << "Deleted key " << key << " from chain " << index << endl;
                return true;
            }
        }
        
        cout << "Key " << key << " not found for deletion" << endl;
        return false;
    }
    
    // Display methods
    void display() const {
        cout << "Chaining Hash Table:" << endl;
        for (int i = 0; i < capacity; i++) {
            cout << "Index " << i << ": ";
            if (table[i].empty()) {
                cout << "EMPTY";
            } else {
                bool first = true;
                for (const auto& pair : table[i]) {
                    if (!first) cout << " -> ";
                    cout << "(" << pair.key << ", " << pair.value << ")";
                    first = false;
                }
            }
            cout << endl;
        }
    }
    
    void displayChainLengths() const {
        cout << "Chain lengths:" << endl;
        int max_chain = 0;
        int empty_chains = 0;
        
        for (int i = 0; i < capacity; i++) {
            int chain_length = table[i].size();
            cout << "Chain " << i << ": " << chain_length << " elements" << endl;
            max_chain = max(max_chain, chain_length);
            if (chain_length == 0) empty_chains++;
        }
        
        cout << "Max chain length: " << max_chain << endl;
        cout << "Empty chains: " << empty_chains << endl;
    }
    
    void printInfo() const {
        double load_factor = static_cast<double>(size) / capacity;
        cout << "Chaining Hash Table Info:" << endl;
        cout << "Size: " << size << endl;
        cout << "Capacity: " << capacity << endl;
        cout << "Load factor: " << load_factor << endl;
        
        // Calculate average chain length
        int non_empty_chains = 0;
        int total_elements = 0;
        for (const auto& chain : table) {
            if (!chain.empty()) {
                non_empty_chains++;
                total_elements += chain.size();
            }
        }
        
        if (non_empty_chains > 0) {
            double avg_chain_length = static_cast<double>(total_elements) / non_empty_chains;
            cout << "Average chain length: " << avg_chain_length << endl;
        }
    }
    
    // Utility methods
    int getSize() const { return size; }
    int getCapacity() const { return capacity; }
    bool isEmpty() const { return size == 0; }
    double getLoadFactor() const { return static_cast<double>(size) / capacity; }
};

// =============================================================================
// STL HASH CONTAINERS OPERATIONS
// =============================================================================

void demonstrate_stl_hash_containers() {
    cout << "\n=== STL HASH CONTAINERS DEMONSTRATION ===" << endl;
    
    // unordered_map operations
    cout << "\n--- unordered_map Operations ---" << endl;
    
    // Different ways to create unordered_map
    unordered_map<string, int> word_count;
    unordered_map<int, string> id_to_name = {{1, "Alice"}, {2, "Bob"}, {3, "Charlie"}};
    
    // Insert operations
    word_count["hello"] = 1;
    word_count["world"] = 2;
    word_count.insert({"hash", 3});
    word_count.emplace("table", 4);
    
    // Access operations
    cout << "word_count['hello']: " << word_count["hello"] << endl;
    cout << "word_count.at('world'): " << word_count.at("world") << endl;
    
    // Search operations
    auto it = word_count.find("hash");
    if (it != word_count.end()) {
        cout << "Found 'hash' with value: " << it->second << endl;
    }
    
    // Check if key exists
    if (word_count.count("table") > 0) {
        cout << "'table' exists in map" << endl;
    }
    
    // Iterate through map
    cout << "All key-value pairs:" << endl;
    for (const auto& pair : word_count) {
        cout << "'" << pair.first << "': " << pair.second << endl;
    }
    
    // Map properties
    cout << "Map size: " << word_count.size() << endl;
    cout << "Map empty: " << (word_count.empty() ? "Yes" : "No") << endl;
    cout << "Bucket count: " << word_count.bucket_count() << endl;
    cout << "Load factor: " << word_count.load_factor() << endl;
    cout << "Max load factor: " << word_count.max_load_factor() << endl;
    
    // unordered_set operations
    cout << "\n--- unordered_set Operations ---" << endl;
    
    unordered_set<int> number_set = {1, 2, 3, 4, 5};
    
    // Insert operations
    number_set.insert(6);
    number_set.emplace(7);
    
    // Search operations
    if (number_set.find(3) != number_set.end()) {
        cout << "Found 3 in set" << endl;
    }
    
    if (number_set.count(10) == 0) {
        cout << "10 not found in set" << endl;
    }
    
    // Iterate through set
    cout << "Set elements: ";
    for (int num : number_set) {
        cout << num << " ";
    }
    cout << endl;
    
    // Set operations
    number_set.erase(2);
    cout << "After erasing 2, size: " << number_set.size() << endl;
    
    // Custom hash function example
    cout << "\n--- Custom Hash Function ---" << endl;
    
    struct Person {
        string name;
        int age;
        
        Person(const string& n, int a) : name(n), age(a) {}
        
        bool operator==(const Person& other) const {
            return name == other.name && age == other.age;
        }
    };
    
    struct PersonHash {
        size_t operator()(const Person& p) const {
            return hash<string>{}(p.name) ^ (hash<int>{}(p.age) << 1);
        }
    };
    
    unordered_set<Person, PersonHash> person_set;
    person_set.emplace("Alice", 25);
    person_set.emplace("Bob", 30);
    
    cout << "Person set size: " << person_set.size() << endl;
}

// =============================================================================
// HASH TABLE APPLICATIONS
// =============================================================================

class HashApplications {
public:
    // Application 1: Two Sum Problem
    static vector<int> twoSum(const vector<int>& nums, int target) {
        unordered_map<int, int> num_to_index;
        
        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            
            if (num_to_index.find(complement) != num_to_index.end()) {
                return {num_to_index[complement], i};
            }
            
            num_to_index[nums[i]] = i;
        }
        
        return {};
    }
    
    // Application 2: Character Frequency Counter
    static unordered_map<char, int> characterFrequency(const string& str) {
        unordered_map<char, int> freq;
        
        for (char c : str) {
            freq[c]++;
        }
        
        return freq;
    }
    
    // Application 3: First Non-Repeating Character
    static char firstNonRepeatingChar(const string& str) {
        unordered_map<char, int> freq;
        
        // Count frequencies
        for (char c : str) {
            freq[c]++;
        }
        
        // Find first non-repeating
        for (char c : str) {
            if (freq[c] == 1) {
                return c;
            }
        }
        
        return '\0';
    }
    
    // Application 4: Group Anagrams
    static vector<vector<string>> groupAnagrams(const vector<string>& strs) {
        unordered_map<string, vector<string>> groups;
        
        for (const string& str : strs) {
            string sorted_str = str;
            sort(sorted_str.begin(), sorted_str.end());
            groups[sorted_str].push_back(str);
        }
        
        vector<vector<string>> result;
        for (const auto& group : groups) {
            result.push_back(group.second);
        }
        
        return result;
    }
    
    // Application 5: LRU Cache
    class LRUCache {
    private:
        struct Node {
            int key, value;
            Node* prev;
            Node* next;
            
            Node(int k = 0, int v = 0) : key(k), value(v), prev(nullptr), next(nullptr) {}
        };
        
        int capacity;
        unordered_map<int, Node*> cache;
        Node* head;
        Node* tail;
        
        void addToHead(Node* node) {
            node->prev = head;
            node->next = head->next;
            head->next->prev = node;
            head->next = node;
        }
        
        void removeNode(Node* node) {
            node->prev->next = node->next;
            node->next->prev = node->prev;
        }
        
        void moveToHead(Node* node) {
            removeNode(node);
            addToHead(node);
        }
        
        Node* removeTail() {
            Node* last = tail->prev;
            removeNode(last);
            return last;
        }
        
    public:
        LRUCache(int cap) : capacity(cap) {
            head = new Node();
            tail = new Node();
            head->next = tail;
            tail->prev = head;
        }
        
        ~LRUCache() {
            Node* current = head;
            while (current) {
                Node* next = current->next;
                delete current;
                current = next;
            }
        }
        
        int get(int key) {
            if (cache.find(key) != cache.end()) {
                Node* node = cache[key];
                moveToHead(node);
                return node->value;
            }
            return -1;
        }
        
        void put(int key, int value) {
            if (cache.find(key) != cache.end()) {
                Node* node = cache[key];
                node->value = value;
                moveToHead(node);
            } else {
                Node* newNode = new Node(key, value);
                
                if (cache.size() >= capacity) {
                    Node* tail_node = removeTail();
                    cache.erase(tail_node->key);
                    delete tail_node;
                }
                
                cache[key] = newNode;
                addToHead(newNode);
            }
        }
        
        void display() const {
            cout << "LRU Cache (most recent first): ";
            Node* current = head->next;
            while (current != tail) {
                cout << "(" << current->key << "," << current->value << ") ";
                current = current->next;
            }
            cout << endl;
        }
    };
    
    // Application 6: Detect Cycle in Array
    static bool hasCycle(const vector<int>& nums) {
        unordered_set<int> visited;
        
        for (int i = 0; i < nums.size(); i++) {
            if (visited.count(i)) continue;
            
            unordered_set<int> current_path;
            int index = i;
            
            while (visited.find(index) == visited.end()) {
                if (current_path.count(index)) {
                    return true;
                }
                
                current_path.insert(index);
                visited.insert(index);
                index = (index + nums[index]) % nums.size();
                if (index < 0) index += nums.size();
            }
        }
        
        return false;
    }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_linear_probing() {
    cout << "\n=== LINEAR PROBING HASH TABLE DEMONSTRATION ===" << endl;
    
    LinearProbingHashTable<int, string> hash_table(7);
    
    // Insert elements
    hash_table.insert(10, "ten");
    hash_table.insert(22, "twenty-two");
    hash_table.insert(31, "thirty-one");
    hash_table.insert(4, "four");
    hash_table.insert(15, "fifteen");
    hash_table.insert(28, "twenty-eight");
    hash_table.insert(17, "seventeen");
    
    hash_table.display();
    hash_table.printInfo();
    
    // Search operations
    cout << "\n--- Search Operations ---" << endl;
    string value;
    if (hash_table.search(22, value)) {
        cout << "Found key 22 with value: " << value << endl;
    }
    
    if (!hash_table.search(99, value)) {
        cout << "Key 99 not found" << endl;
    }
    
    // Delete operations
    cout << "\n--- Delete Operations ---" << endl;
    hash_table.remove(22);
    hash_table.remove(4);
    hash_table.display();
    hash_table.printInfo();
}

void demonstrate_quadratic_probing() {
    cout << "\n=== QUADRATIC PROBING HASH TABLE DEMONSTRATION ===" << endl;
    
    QuadraticProbingHashTable<int, string> quad_table(11);
    
    // Insert elements
    quad_table.insert(10, "ten");
    quad_table.insert(22, "twenty-two");
    quad_table.insert(31, "thirty-one");
    quad_table.insert(4, "four");
    quad_table.insert(15, "fifteen");
    
    quad_table.display();
    quad_table.printInfo();
    
    // Search operations
    string value;
    if (quad_table.search(31, value)) {
        cout << "Found key 31 with value: " << value << endl;
    }
}

void demonstrate_chaining() {
    cout << "\n=== CHAINING HASH TABLE DEMONSTRATION ===" << endl;
    
    ChainingHashTable<string, int> chain_table(5);
    
    // Insert elements
    chain_table.insert("apple", 5);
    chain_table.insert("banana", 7);
    chain_table.insert("orange", 6);
    chain_table.insert("grape", 5);
    chain_table.insert("kiwi", 4);
    chain_table.insert("mango", 5);
    
    chain_table.display();
    chain_table.displayChainLengths();
    chain_table.printInfo();
    
    // Search and delete operations
    cout << "\n--- Search and Delete ---" << endl;
    int value;
    if (chain_table.search("banana", value)) {
        cout << "Found 'banana' with value: " << value << endl;
    }
    
    chain_table.remove("orange");
    chain_table.display();
}

void demonstrate_hash_applications() {
    cout << "\n=== HASH TABLE APPLICATIONS DEMONSTRATION ===" << endl;
    
    // Two Sum
    cout << "\n--- Two Sum Problem ---" << endl;
    vector<int> nums = {2, 7, 11, 15};
    int target = 9;
    vector<int> indices = HashApplications::twoSum(nums, target);
    cout << "Two sum indices for target " << target << ": ";
    for (int idx : indices) cout << idx << " ";
    cout << endl;
    
    // Character frequency
    cout << "\n--- Character Frequency ---" << endl;
    string text = "hello world";
    auto freq = HashApplications::characterFrequency(text);
    cout << "Character frequencies in '" << text << "':" << endl;
    for (const auto& pair : freq) {
        cout << "'" << pair.first << "': " << pair.second << endl;
    }
    
    // First non-repeating character
    cout << "\n--- First Non-Repeating Character ---" << endl;
    string str = "abccba";
    char first_non_rep = HashApplications::firstNonRepeatingChar(str);
    cout << "First non-repeating character in '" << str << "': " 
         << (first_non_rep ? first_non_rep : '0') << endl;
    
    // Group anagrams
    cout << "\n--- Group Anagrams ---" << endl;
    vector<string> words = {"eat", "tea", "tan", "ate", "nat", "bat"};
    auto groups = HashApplications::groupAnagrams(words);
    cout << "Anagram groups:" << endl;
    for (int i = 0; i < groups.size(); i++) {
        cout << "Group " << i + 1 << ": ";
        for (const string& word : groups[i]) {
            cout << word << " ";
        }
        cout << endl;
    }
    
    // LRU Cache
    cout << "\n--- LRU Cache ---" << endl;
    HashApplications::LRUCache lru(3);
    lru.put(1, 10);
    lru.put(2, 20);
    lru.put(3, 30);
    lru.display();
    
    cout << "Get 2: " << lru.get(2) << endl;
    lru.display();
    
    lru.put(4, 40);
    lru.display();
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    cout << "=== COMPLETE HASH TABLE GUIDE ===" << endl;
    
    demonstrate_linear_probing();
    demonstrate_quadratic_probing();
    demonstrate_chaining();
    demonstrate_stl_hash_containers();
    demonstrate_hash_applications();
    
    cout << "\n=== SUMMARY ===" << endl;
    cout << "1. Linear Probing: Simple, can have clustering issues" << endl;
    cout << "2. Quadratic Probing: Reduces clustering, requires careful capacity" << endl;
    cout << "3. Chaining: Handles collisions well, uses extra memory" << endl;
    cout << "4. STL Containers: unordered_map, unordered_set - ready to use" << endl;
    cout << "5. Applications: Two sum, frequency counting, caching, grouping" << endl;
    cout << "6. Time Complexity: Average O(1), Worst O(n)" << endl;
    
    return 0;
} 