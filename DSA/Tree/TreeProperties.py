class TreeProperties:
    def __init__(self):
        self.diameter = 0
    
    def height_of_tree(self, root):
        if root is None:
            return 0
        left_height = self.height_of_tree(root.left)
        right_height = self.height_of_tree(root.right)

        self.diameter = max(self.diameter, left_height + right_height)
        
        return 1 + max(left_height, right_height) 
    
    def CountOfNode(self, root):
        if root is None:
            return 0
        return 1 + self.CountOfNode(root.left) + self.CountOfNode(root.right)
    
    def diameter_of_tree(self,root):
        if root is None:
            return 0
        self.height_of_tree(root)
        return self.diameter
        
    def isIdentical(self, r1, r2):
        if r1 is None and r2 is None:
            return True
        if r1 is None or r2 is None:
            return False        
        return r1.val == r2.val and self.isIdentical(r1.left, r2.left) and self.isIdentical(r1.right, r2.right)
    
    def isMirror(self, left, right):
        if left is None and right is None:
            return True
        if left is None or right is None:
            return False

        return left.val == right.val and self.isMirror(left.left, right.right) and self.isMirror(left.right, right.left)    

    def isHeightBalanced(self, root):
        def helper(root):
            if root is None:
                return 0, True
            
            lh, lb = helper(root.left)
            rh, rb = helper(root.right)
            isBal = lb and rb and abs(lh -rh) <= 1
            return 1 + max(lh, rh), isBal
        height, balanced = helper(root)
        return balanced
    
    def isSumProperty(self, root):
        if root is None:
            return True
            
        if(root.left is None and root.right is None):
            return True
        
        left_data = root.left.val if root.left else 0
        right_data = root.right.val if root.right else 0
        
        if root.val == left_data + right_data:
            lc= self.isSumProperty(root.left)
            rc = self.isSumProperty(root.right)
            return True if lc and rc else False
        else:
            return False
