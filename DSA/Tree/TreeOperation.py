class TreeOperation:
    def __init__(self):
        self.diameter = 0
    
    def sum_of_tree(self, root):
        if root is None:
            return 0
        
        return root.val + self.sum_of_tree(root.left) + self.sum_of_tree(root.right)
    
    def sum_of_leaf_nodes(self, root):
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return root.val
        
        return self.sum_of_leaf_nodes(root.left) + self.sum_of_leaf_nodes(root.right)

    
