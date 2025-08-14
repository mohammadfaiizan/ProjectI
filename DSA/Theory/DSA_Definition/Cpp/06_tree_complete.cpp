/*
 * =============================================================================
 * COMPLETE TREE GUIDE - Binary Tree & BST Operations
 * =============================================================================
 * 
 * This file covers:
 * 1. Binary Tree implementation
 * 2. Binary Search Tree (BST) implementation
 * 3. Tree traversals (recursive and iterative)
 * 4. Tree construction methods
 * 5. Tree algorithms and operations
 * 6. Tree properties and analysis
 * 
 * =============================================================================
 */

#include <iostream>
#include <queue>
#include <stack>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;

// =============================================================================
// BINARY TREE NODE STRUCTURE
// =============================================================================

struct TreeNode {
    int data;
    TreeNode* left;
    TreeNode* right;
    
    TreeNode(int value) : data(value), left(nullptr), right(nullptr) {}
};

// =============================================================================
// BINARY TREE IMPLEMENTATION
// =============================================================================

class BinaryTree {
private:
    TreeNode* root;
    
    // Helper functions for recursive operations
    void destroyTree(TreeNode* node) {
        if (node) {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }
    
    TreeNode* copyTree(TreeNode* node) {
        if (!node) return nullptr;
        
        TreeNode* new_node = new TreeNode(node->data);
        new_node->left = copyTree(node->left);
        new_node->right = copyTree(node->right);
        return new_node;
    }
    
    void inorderHelper(TreeNode* node) const {
        if (node) {
            inorderHelper(node->left);
            cout << node->data << " ";
            inorderHelper(node->right);
        }
    }
    
    void preorderHelper(TreeNode* node) const {
        if (node) {
            cout << node->data << " ";
            preorderHelper(node->left);
            preorderHelper(node->right);
        }
    }
    
    void postorderHelper(TreeNode* node) const {
        if (node) {
            postorderHelper(node->left);
            postorderHelper(node->right);
            cout << node->data << " ";
        }
    }
    
    int heightHelper(TreeNode* node) const {
        if (!node) return -1;
        return 1 + max(heightHelper(node->left), heightHelper(node->right));
    }
    
    int sizeHelper(TreeNode* node) const {
        if (!node) return 0;
        return 1 + sizeHelper(node->left) + sizeHelper(node->right);
    }
    
    bool searchHelper(TreeNode* node, int value) const {
        if (!node) return false;
        if (node->data == value) return true;
        return searchHelper(node->left, value) || searchHelper(node->right, value);
    }

public:
    // Constructor
    BinaryTree() : root(nullptr) {
        cout << "Binary tree created" << endl;
    }
    
    // Destructor
    ~BinaryTree() {
        destroyTree(root);
        cout << "Binary tree destroyed" << endl;
    }
    
    // Copy constructor
    BinaryTree(const BinaryTree& other) : root(copyTree(other.root)) {
        cout << "Binary tree copied" << endl;
    }
    
    // Assignment operator
    BinaryTree& operator=(const BinaryTree& other) {
        if (this != &other) {
            destroyTree(root);
            root = copyTree(other.root);
        }
        return *this;
    }
    
    // Basic operations
    void setRoot(int value) {
        if (root) {
            root->data = value;
        } else {
            root = new TreeNode(value);
        }
        cout << "Root set to " << value << endl;
    }
    
    TreeNode* getRoot() { return root; }
    const TreeNode* getRoot() const { return root; }
    
    // Tree construction methods
    void insertLevelOrder(const vector<int>& values) {
        if (values.empty()) return;
        
        root = new TreeNode(values[0]);
        queue<TreeNode*> q;
        q.push(root);
        
        for (int i = 1; i < values.size(); i += 2) {
            TreeNode* current = q.front();
            q.pop();
            
            // Add left child
            if (i < values.size() && values[i] != -1) {
                current->left = new TreeNode(values[i]);
                q.push(current->left);
            }
            
            // Add right child
            if (i + 1 < values.size() && values[i + 1] != -1) {
                current->right = new TreeNode(values[i + 1]);
                q.push(current->right);
            }
        }
        
        cout << "Tree constructed from level order array" << endl;
    }
    
    TreeNode* insertRecursive(TreeNode* node, int value) {
        if (!node) {
            return new TreeNode(value);
        }
        
        // Insert in left subtree if value is smaller
        if (value <= node->data) {
            node->left = insertRecursive(node->left, value);
        } else {
            node->right = insertRecursive(node->right, value);
        }
        
        return node;
    }
    
    void insert(int value) {
        root = insertRecursive(root, value);
        cout << "Inserted " << value << " into tree" << endl;
    }
    
    // Tree traversal methods (recursive)
    void inorderTraversal() const {
        cout << "Inorder traversal: ";
        inorderHelper(root);
        cout << endl;
    }
    
    void preorderTraversal() const {
        cout << "Preorder traversal: ";
        preorderHelper(root);
        cout << endl;
    }
    
    void postorderTraversal() const {
        cout << "Postorder traversal: ";
        postorderHelper(root);
        cout << endl;
    }
    
    // Tree traversal methods (iterative)
    void inorderIterative() const {
        cout << "Inorder iterative: ";
        stack<TreeNode*> s;
        TreeNode* current = root;
        
        while (current || !s.empty()) {
            while (current) {
                s.push(current);
                current = current->left;
            }
            current = s.top();
            s.pop();
            cout << current->data << " ";
            current = current->right;
        }
        cout << endl;
    }
    
    void preorderIterative() const {
        cout << "Preorder iterative: ";
        if (!root) {
            cout << endl;
            return;
        }
        
        stack<TreeNode*> s;
        s.push(root);
        
        while (!s.empty()) {
            TreeNode* current = s.top();
            s.pop();
            cout << current->data << " ";
            
            if (current->right) s.push(current->right);
            if (current->left) s.push(current->left);
        }
        cout << endl;
    }
    
    void postorderIterative() const {
        cout << "Postorder iterative: ";
        if (!root) {
            cout << endl;
            return;
        }
        
        stack<TreeNode*> s1, s2;
        s1.push(root);
        
        while (!s1.empty()) {
            TreeNode* current = s1.top();
            s1.pop();
            s2.push(current);
            
            if (current->left) s1.push(current->left);
            if (current->right) s1.push(current->right);
        }
        
        while (!s2.empty()) {
            cout << s2.top()->data << " ";
            s2.pop();
        }
        cout << endl;
    }
    
    void levelOrderTraversal() const {
        cout << "Level order traversal: ";
        if (!root) {
            cout << endl;
            return;
        }
        
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            TreeNode* current = q.front();
            q.pop();
            cout << current->data << " ";
            
            if (current->left) q.push(current->left);
            if (current->right) q.push(current->right);
        }
        cout << endl;
    }
    
    void levelOrderWithLevels() const {
        cout << "Level order with levels:" << endl;
        if (!root) return;
        
        queue<TreeNode*> q;
        q.push(root);
        int level = 0;
        
        while (!q.empty()) {
            int size = q.size();
            cout << "Level " << level << ": ";
            
            for (int i = 0; i < size; i++) {
                TreeNode* current = q.front();
                q.pop();
                cout << current->data << " ";
                
                if (current->left) q.push(current->left);
                if (current->right) q.push(current->right);
            }
            cout << endl;
            level++;
        }
    }
    
    // Tree properties
    int height() const {
        return heightHelper(root);
    }
    
    int size() const {
        return sizeHelper(root);
    }
    
    bool isEmpty() const {
        return root == nullptr;
    }
    
    bool search(int value) const {
        return searchHelper(root, value);
    }
    
    // Tree analysis
    bool isBalanced() const {
        return isBalancedHelper(root) != -1;
    }
    
    int isBalancedHelper(TreeNode* node) const {
        if (!node) return 0;
        
        int left_height = isBalancedHelper(node->left);
        if (left_height == -1) return -1;
        
        int right_height = isBalancedHelper(node->right);
        if (right_height == -1) return -1;
        
        if (abs(left_height - right_height) > 1) return -1;
        
        return 1 + max(left_height, right_height);
    }
    
    bool isSymmetric() const {
        return isSymmetricHelper(root, root);
    }
    
    bool isSymmetricHelper(TreeNode* left, TreeNode* right) const {
        if (!left && !right) return true;
        if (!left || !right) return false;
        
        return (left->data == right->data) &&
               isSymmetricHelper(left->left, right->right) &&
               isSymmetricHelper(left->right, right->left);
    }
    
    // Tree algorithms
    int findMaximum() const {
        if (!root) throw runtime_error("Tree is empty!");
        
        int max_val = root->data;
        findMaxHelper(root, max_val);
        return max_val;
    }
    
    void findMaxHelper(TreeNode* node, int& max_val) const {
        if (node) {
            max_val = max(max_val, node->data);
            findMaxHelper(node->left, max_val);
            findMaxHelper(node->right, max_val);
        }
    }
    
    int findMinimum() const {
        if (!root) throw runtime_error("Tree is empty!");
        
        int min_val = root->data;
        findMinHelper(root, min_val);
        return min_val;
    }
    
    void findMinHelper(TreeNode* node, int& min_val) const {
        if (node) {
            min_val = min(min_val, node->data);
            findMinHelper(node->left, min_val);
            findMinHelper(node->right, min_val);
        }
    }
    
    vector<int> rightView() const {
        vector<int> result;
        if (!root) return result;
        
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            int size = q.size();
            
            for (int i = 0; i < size; i++) {
                TreeNode* current = q.front();
                q.pop();
                
                if (i == size - 1) {
                    result.push_back(current->data);
                }
                
                if (current->left) q.push(current->left);
                if (current->right) q.push(current->right);
            }
        }
        
        return result;
    }
    
    vector<int> leftView() const {
        vector<int> result;
        if (!root) return result;
        
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            int size = q.size();
            
            for (int i = 0; i < size; i++) {
                TreeNode* current = q.front();
                q.pop();
                
                if (i == 0) {
                    result.push_back(current->data);
                }
                
                if (current->left) q.push(current->left);
                if (current->right) q.push(current->right);
            }
        }
        
        return result;
    }
    
    void printInfo() const {
        cout << "Binary Tree Info:" << endl;
        cout << "Size: " << size() << endl;
        cout << "Height: " << height() << endl;
        cout << "Is empty: " << (isEmpty() ? "Yes" : "No") << endl;
        cout << "Is balanced: " << (isBalanced() ? "Yes" : "No") << endl;
        cout << "Is symmetric: " << (isSymmetric() ? "Yes" : "No") << endl;
        if (!isEmpty()) {
            cout << "Maximum value: " << findMaximum() << endl;
            cout << "Minimum value: " << findMinimum() << endl;
        }
    }
};

// =============================================================================
// BINARY SEARCH TREE IMPLEMENTATION
// =============================================================================

class BinarySearchTree {
private:
    TreeNode* root;
    
    TreeNode* insertHelper(TreeNode* node, int value) {
        if (!node) {
            return new TreeNode(value);
        }
        
        if (value < node->data) {
            node->left = insertHelper(node->left, value);
        } else if (value > node->data) {
            node->right = insertHelper(node->right, value);
        }
        // If value equals node->data, don't insert (no duplicates)
        
        return node;
    }
    
    TreeNode* deleteHelper(TreeNode* node, int value) {
        if (!node) return node;
        
        if (value < node->data) {
            node->left = deleteHelper(node->left, value);
        } else if (value > node->data) {
            node->right = deleteHelper(node->right, value);
        } else {
            // Node to be deleted found
            if (!node->left) {
                TreeNode* temp = node->right;
                delete node;
                return temp;
            } else if (!node->right) {
                TreeNode* temp = node->left;
                delete node;
                return temp;
            }
            
            // Node with two children
            TreeNode* temp = findMin(node->right);
            node->data = temp->data;
            node->right = deleteHelper(node->right, temp->data);
        }
        
        return node;
    }
    
    TreeNode* findMin(TreeNode* node) const {
        while (node && node->left) {
            node = node->left;
        }
        return node;
    }
    
    TreeNode* findMax(TreeNode* node) const {
        while (node && node->right) {
            node = node->right;
        }
        return node;
    }
    
    bool searchHelper(TreeNode* node, int value) const {
        if (!node) return false;
        if (node->data == value) return true;
        
        if (value < node->data) {
            return searchHelper(node->left, value);
        } else {
            return searchHelper(node->right, value);
        }
    }
    
    void inorderHelper(TreeNode* node, vector<int>& result) const {
        if (node) {
            inorderHelper(node->left, result);
            result.push_back(node->data);
            inorderHelper(node->right, result);
        }
    }
    
    bool isBSTHelper(TreeNode* node, int min_val, int max_val) const {
        if (!node) return true;
        
        if (node->data <= min_val || node->data >= max_val) {
            return false;
        }
        
        return isBSTHelper(node->left, min_val, node->data) &&
               isBSTHelper(node->right, node->data, max_val);
    }
    
    void destroyTree(TreeNode* node) {
        if (node) {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }

public:
    // Constructor
    BinarySearchTree() : root(nullptr) {
        cout << "Binary Search Tree created" << endl;
    }
    
    // Destructor
    ~BinarySearchTree() {
        destroyTree(root);
        cout << "Binary Search Tree destroyed" << endl;
    }
    
    // Basic operations
    void insert(int value) {
        root = insertHelper(root, value);
        cout << "Inserted " << value << " into BST" << endl;
    }
    
    void remove(int value) {
        root = deleteHelper(root, value);
        cout << "Removed " << value << " from BST" << endl;
    }
    
    bool search(int value) const {
        return searchHelper(root, value);
    }
    
    // Tree traversals
    void inorderTraversal() const {
        cout << "BST Inorder (sorted): ";
        vector<int> result;
        inorderHelper(root, result);
        for (int value : result) {
            cout << value << " ";
        }
        cout << endl;
    }
    
    void levelOrderTraversal() const {
        cout << "BST Level order: ";
        if (!root) {
            cout << endl;
            return;
        }
        
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            TreeNode* current = q.front();
            q.pop();
            cout << current->data << " ";
            
            if (current->left) q.push(current->left);
            if (current->right) q.push(current->right);
        }
        cout << endl;
    }
    
    // BST specific operations
    int findMinimum() const {
        if (!root) throw runtime_error("BST is empty!");
        return findMin(root)->data;
    }
    
    int findMaximum() const {
        if (!root) throw runtime_error("BST is empty!");
        return findMax(root)->data;
    }
    
    int findKthSmallest(int k) const {
        vector<int> sorted;
        inorderHelper(root, sorted);
        if (k <= 0 || k > sorted.size()) {
            throw runtime_error("Invalid k value!");
        }
        return sorted[k - 1];
    }
    
    int findKthLargest(int k) const {
        vector<int> sorted;
        inorderHelper(root, sorted);
        if (k <= 0 || k > sorted.size()) {
            throw runtime_error("Invalid k value!");
        }
        return sorted[sorted.size() - k];
    }
    
    vector<int> getRangeValues(int min_val, int max_val) const {
        vector<int> result;
        getRangeHelper(root, min_val, max_val, result);
        return result;
    }
    
    void getRangeHelper(TreeNode* node, int min_val, int max_val, vector<int>& result) const {
        if (!node) return;
        
        if (node->data >= min_val && node->data <= max_val) {
            result.push_back(node->data);
        }
        
        if (node->data > min_val) {
            getRangeHelper(node->left, min_val, max_val, result);
        }
        
        if (node->data < max_val) {
            getRangeHelper(node->right, min_val, max_val, result);
        }
    }
    
    // Validation
    bool isBST() const {
        return isBSTHelper(root, INT_MIN, INT_MAX);
    }
    
    // Tree properties
    bool isEmpty() const {
        return root == nullptr;
    }
    
    int size() const {
        return sizeHelper(root);
    }
    
    int sizeHelper(TreeNode* node) const {
        if (!node) return 0;
        return 1 + sizeHelper(node->left) + sizeHelper(node->right);
    }
    
    int height() const {
        return heightHelper(root);
    }
    
    int heightHelper(TreeNode* node) const {
        if (!node) return -1;
        return 1 + max(heightHelper(node->left), heightHelper(node->right));
    }
    
    void printInfo() const {
        cout << "BST Info:" << endl;
        cout << "Size: " << size() << endl;
        cout << "Height: " << height() << endl;
        cout << "Is empty: " << (isEmpty() ? "Yes" : "No") << endl;
        cout << "Is valid BST: " << (isBST() ? "Yes" : "No") << endl;
        if (!isEmpty()) {
            cout << "Minimum value: " << findMinimum() << endl;
            cout << "Maximum value: " << findMaximum() << endl;
        }
    }
};

// =============================================================================
// TREE ALGORITHMS AND PATTERNS
// =============================================================================

class TreeAlgorithms {
public:
    // Find Lowest Common Ancestor
    static TreeNode* findLCA(TreeNode* root, int p, int q) {
        if (!root) return nullptr;
        
        if (root->data == p || root->data == q) {
            return root;
        }
        
        TreeNode* left_lca = findLCA(root->left, p, q);
        TreeNode* right_lca = findLCA(root->right, p, q);
        
        if (left_lca && right_lca) {
            return root;
        }
        
        return left_lca ? left_lca : right_lca;
    }
    
    // Check if two trees are identical
    static bool areIdentical(TreeNode* tree1, TreeNode* tree2) {
        if (!tree1 && !tree2) return true;
        if (!tree1 || !tree2) return false;
        
        return (tree1->data == tree2->data) &&
               areIdentical(tree1->left, tree2->left) &&
               areIdentical(tree1->right, tree2->right);
    }
    
    // Find diameter of tree
    static int findDiameter(TreeNode* root) {
        int diameter = 0;
        findHeight(root, diameter);
        return diameter;
    }
    
    static int findHeight(TreeNode* node, int& diameter) {
        if (!node) return 0;
        
        int left_height = findHeight(node->left, diameter);
        int right_height = findHeight(node->right, diameter);
        
        diameter = max(diameter, left_height + right_height);
        
        return 1 + max(left_height, right_height);
    }
    
    // Convert BST to sorted array
    static vector<int> bstToArray(TreeNode* root) {
        vector<int> result;
        bstToArrayHelper(root, result);
        return result;
    }
    
    static void bstToArrayHelper(TreeNode* node, vector<int>& result) {
        if (node) {
            bstToArrayHelper(node->left, result);
            result.push_back(node->data);
            bstToArrayHelper(node->right, result);
        }
    }
    
    // Convert sorted array to BST
    static TreeNode* arrayToBST(const vector<int>& arr) {
        return arrayToBSTHelper(arr, 0, arr.size() - 1);
    }
    
    static TreeNode* arrayToBSTHelper(const vector<int>& arr, int start, int end) {
        if (start > end) return nullptr;
        
        int mid = start + (end - start) / 2;
        TreeNode* root = new TreeNode(arr[mid]);
        
        root->left = arrayToBSTHelper(arr, start, mid - 1);
        root->right = arrayToBSTHelper(arr, mid + 1, end);
        
        return root;
    }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_binary_tree() {
    cout << "\n=== BINARY TREE DEMONSTRATION ===" << endl;
    
    BinaryTree bt;
    
    // Method 1: Level order construction
    cout << "\n--- Level Order Construction ---" << endl;
    vector<int> level_order = {1, 2, 3, 4, 5, 6, 7};
    bt.insertLevelOrder(level_order);
    
    // Different traversals
    cout << "\n--- Tree Traversals ---" << endl;
    bt.inorderTraversal();
    bt.preorderTraversal();
    bt.postorderTraversal();
    bt.levelOrderTraversal();
    
    cout << "\n--- Iterative Traversals ---" << endl;
    bt.inorderIterative();
    bt.preorderIterative();
    bt.postorderIterative();
    
    cout << "\n--- Level Order with Levels ---" << endl;
    bt.levelOrderWithLevels();
    
    // Tree properties and analysis
    cout << "\n--- Tree Analysis ---" << endl;
    bt.printInfo();
    
    // Tree views
    cout << "\n--- Tree Views ---" << endl;
    vector<int> right_view = bt.rightView();
    cout << "Right view: ";
    for (int val : right_view) cout << val << " ";
    cout << endl;
    
    vector<int> left_view = bt.leftView();
    cout << "Left view: ";
    for (int val : left_view) cout << val << " ";
    cout << endl;
}

void demonstrate_binary_search_tree() {
    cout << "\n=== BINARY SEARCH TREE DEMONSTRATION ===" << endl;
    
    BinarySearchTree bst;
    
    // Insert elements
    cout << "\n--- Insertion ---" << endl;
    vector<int> values = {50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45};
    for (int val : values) {
        bst.insert(val);
    }
    
    // Traversals
    cout << "\n--- Traversals ---" << endl;
    bst.inorderTraversal();
    bst.levelOrderTraversal();
    
    // Search operations
    cout << "\n--- Search Operations ---" << endl;
    cout << "Search 40: " << (bst.search(40) ? "Found" : "Not found") << endl;
    cout << "Search 90: " << (bst.search(90) ? "Found" : "Not found") << endl;
    
    // BST specific operations
    cout << "\n--- BST Specific Operations ---" << endl;
    cout << "Minimum value: " << bst.findMinimum() << endl;
    cout << "Maximum value: " << bst.findMaximum() << endl;
    cout << "3rd smallest: " << bst.findKthSmallest(3) << endl;
    cout << "2nd largest: " << bst.findKthLargest(2) << endl;
    
    // Range query
    vector<int> range_values = bst.getRangeValues(25, 65);
    cout << "Values in range [25, 65]: ";
    for (int val : range_values) cout << val << " ";
    cout << endl;
    
    // Tree analysis
    cout << "\n--- BST Analysis ---" << endl;
    bst.printInfo();
    
    // Deletion
    cout << "\n--- Deletion ---" << endl;
    bst.remove(20);
    bst.remove(30);
    bst.remove(50);
    bst.inorderTraversal();
}

void demonstrate_tree_algorithms() {
    cout << "\n=== TREE ALGORITHMS DEMONSTRATION ===" << endl;
    
    // Create a sample tree
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(7);
    
    // Find LCA
    cout << "\n--- Lowest Common Ancestor ---" << endl;
    TreeNode* lca = TreeAlgorithms::findLCA(root, 4, 5);
    cout << "LCA of 4 and 5: " << (lca ? to_string(lca->data) : "Not found") << endl;
    
    // Find diameter
    cout << "\n--- Tree Diameter ---" << endl;
    int diameter = TreeAlgorithms::findDiameter(root);
    cout << "Diameter of tree: " << diameter << endl;
    
    // BST to Array and back
    cout << "\n--- BST Array Conversion ---" << endl;
    vector<int> sorted_array = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    TreeNode* bst_root = TreeAlgorithms::arrayToBST(sorted_array);
    
    cout << "Created BST from sorted array: ";
    vector<int> bst_array = TreeAlgorithms::bstToArray(bst_root);
    for (int val : bst_array) cout << val << " ";
    cout << endl;
    
    // Check if trees are identical
    TreeNode* root2 = new TreeNode(1);
    root2->left = new TreeNode(2);
    root2->right = new TreeNode(3);
    
    bool identical = TreeAlgorithms::areIdentical(root, root2);
    cout << "Trees are identical: " << (identical ? "Yes" : "No") << endl;
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    cout << "=== COMPLETE TREE GUIDE ===" << endl;
    
    demonstrate_binary_tree();
    demonstrate_binary_search_tree();
    demonstrate_tree_algorithms();
    
    cout << "\n=== SUMMARY ===" << endl;
    cout << "1. Binary Tree: Hierarchical structure, no ordering constraint" << endl;
    cout << "2. BST: Ordered binary tree, efficient search/insert/delete" << endl;
    cout << "3. Traversals: Inorder, Preorder, Postorder, Level order" << endl;
    cout << "4. Applications: Expression trees, file systems, decision trees" << endl;
    cout << "5. Algorithms: LCA, diameter, tree construction, validation" << endl;
    
    return 0;
} 