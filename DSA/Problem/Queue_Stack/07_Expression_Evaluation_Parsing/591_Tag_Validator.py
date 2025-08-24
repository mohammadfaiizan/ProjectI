"""
591. Tag Validator - Multiple Approaches
Difficulty: Hard

Given a string representing a code snippet, implement a tag validator to parse and return whether it is valid.

A code snippet is valid if all the following rules hold:

1. The code must be wrapped in a valid closed tag. Otherwise, the code is invalid.
2. A closed tag (not necessarily valid) has exactly the following format : <TAG_NAME>TAG_CONTENT</TAG_NAME>. Among them, <TAG_NAME> is the start tag, and </TAG_NAME> is the end tag. The TAG_NAME in start and end tags should be the same. A closed tag is valid if and only if the TAG_NAME and TAG_CONTENT are valid.
3. A valid TAG_NAME contains only upper-case letters, and has length in range [1,9]. Otherwise, the TAG_NAME is invalid.
4. A valid TAG_CONTENT may contain other valid closed tags, cdata and any characters (see note1) EXCEPT unmatched <, unmatched start and end tag, and unmatched or closed tags with invalid TAG_NAME. Otherwise, the TAG_CONTENT is invalid.
5. A cdata has the following format : <![CDATA[CDATA_CONTENT]]>. The range of CDATA_CONTENT is defined as the characters between <![CDATA[ and the first subsequent ]]>.
6. CDATA_CONTENT may contain any characters. The function of cdata is to forbid the validator to parse CDATA_CONTENT, so even it has some characters that can be parsed as tag (no matter valid or invalid), you should treat it as regular characters.
"""

from typing import List, Tuple
import re

class TagValidator:
    """Multiple approaches to validate XML-like tags"""
    
    def isValid_stack_approach(self, code: str) -> bool:
        """
        Approach 1: Stack-based Validation (Optimal)
        
        Use stack to track open tags and validate structure.
        
        Time: O(n), Space: O(n)
        """
        if not code or code[0] != '<':
            return False
        
        stack = []
        i = 0
        
        while i < len(code):
            if i == 0 or stack:  # Must be inside a tag or at the beginning
                if code[i:i+9] == '<![CDATA[':
                    # Handle CDATA
                    i += 9
                    cdata_end = code.find(']]>', i)
                    if cdata_end == -1:
                        return False
                    i = cdata_end + 3
                elif code[i] == '<':
                    # Handle tag
                    if i + 1 < len(code) and code[i+1] == '/':
                        # Closing tag
                        i += 2
                        tag_end = code.find('>', i)
                        if tag_end == -1:
                            return False
                        
                        tag_name = code[i:tag_end]
                        if not self._is_valid_tag_name(tag_name):
                            return False
                        
                        if not stack or stack[-1] != tag_name:
                            return False
                        
                        stack.pop()
                        i = tag_end + 1
                    else:
                        # Opening tag
                        i += 1
                        tag_end = code.find('>', i)
                        if tag_end == -1:
                            return False
                        
                        tag_name = code[i:tag_end]
                        if not self._is_valid_tag_name(tag_name):
                            return False
                        
                        stack.append(tag_name)
                        i = tag_end + 1
                else:
                    # Regular character
                    i += 1
            else:
                # Outside of any tag (invalid)
                return False
        
        return len(stack) == 0
    
    def isValid_regex_approach(self, code: str) -> bool:
        """
        Approach 2: Regex-based Validation
        
        Use regular expressions to validate structure.
        
        Time: O(n), Space: O(n)
        """
        if not code:
            return False
        
        # Remove CDATA sections first
        code = self._remove_cdata(code)
        
        # Check if wrapped in valid tags
        if not re.match(r'^<[A-Z]{1,9}>.*</[A-Z]{1,9}>$', code):
            return False
        
        # Extract outer tag
        outer_match = re.match(r'^<([A-Z]{1,9})>(.*)</([A-Z]{1,9})>$', code)
        if not outer_match:
            return False
        
        start_tag, content, end_tag = outer_match.groups()
        if start_tag != end_tag:
            return False
        
        # Validate inner content
        return self._validate_content(content)
    
    def isValid_recursive_descent(self, code: str) -> bool:
        """
        Approach 3: Recursive Descent Parser
        
        Use recursive parsing to validate structure.
        
        Time: O(n), Space: O(n)
        """
        if not code:
            return False
        
        self.pos = 0
        self.code = code
        
        try:
            # Must start with a tag
            if not self._parse_tag():
                return False
            
            # Must consume entire input
            return self.pos == len(code)
        except:
            return False
    
    def isValid_state_machine(self, code: str) -> bool:
        """
        Approach 4: State Machine Approach
        
        Use finite state machine to validate structure.
        
        Time: O(n), Space: O(n)
        """
        if not code:
            return False
        
        stack = []
        state = 'EXPECT_OPEN_TAG'
        i = 0
        
        while i < len(code):
            if state == 'EXPECT_OPEN_TAG':
                if code[i] != '<':
                    return False
                
                # Check for CDATA
                if code[i:i+9] == '<![CDATA[':
                    return False  # CDATA not allowed at root level
                
                # Parse opening tag
                tag_end = code.find('>', i + 1)
                if tag_end == -1:
                    return False
                
                tag_name = code[i+1:tag_end]
                if not self._is_valid_tag_name(tag_name):
                    return False
                
                stack.append(tag_name)
                state = 'IN_CONTENT'
                i = tag_end + 1
            
            elif state == 'IN_CONTENT':
                if not stack:
                    return False
                
                if code[i:i+9] == '<![CDATA[':
                    # Handle CDATA
                    cdata_end = code.find(']]>', i + 9)
                    if cdata_end == -1:
                        return False
                    i = cdata_end + 3
                elif code[i] == '<':
                    if i + 1 < len(code) and code[i+1] == '/':
                        # Closing tag
                        tag_end = code.find('>', i + 2)
                        if tag_end == -1:
                            return False
                        
                        tag_name = code[i+2:tag_end]
                        if not self._is_valid_tag_name(tag_name):
                            return False
                        
                        if not stack or stack[-1] != tag_name:
                            return False
                        
                        stack.pop()
                        i = tag_end + 1
                        
                        if not stack:
                            state = 'EXPECT_END'
                    else:
                        # Opening tag
                        tag_end = code.find('>', i + 1)
                        if tag_end == -1:
                            return False
                        
                        tag_name = code[i+1:tag_end]
                        if not self._is_valid_tag_name(tag_name):
                            return False
                        
                        stack.append(tag_name)
                        i = tag_end + 1
                else:
                    # Regular character
                    i += 1
            
            elif state == 'EXPECT_END':
                # Should not have any more content
                return False
        
        return len(stack) == 0 and state != 'EXPECT_OPEN_TAG'
    
    def _is_valid_tag_name(self, tag_name: str) -> bool:
        """Check if tag name is valid"""
        return (1 <= len(tag_name) <= 9 and 
                tag_name.isalpha() and 
                tag_name.isupper())
    
    def _remove_cdata(self, code: str) -> str:
        """Remove CDATA sections from code"""
        while '<![CDATA[' in code:
            start = code.find('<![CDATA[')
            end = code.find(']]>', start)
            if end == -1:
                break
            code = code[:start] + code[end+3:]
        return code
    
    def _validate_content(self, content: str) -> bool:
        """Validate tag content recursively"""
        if not content:
            return True
        
        # Remove CDATA sections
        content = self._remove_cdata(content)
        
        # Find all tags
        i = 0
        while i < len(content):
            if content[i] == '<':
                if content[i+1:i+2] == '/':
                    return False  # Unmatched closing tag
                
                # Find matching closing tag
                tag_end = content.find('>', i + 1)
                if tag_end == -1:
                    return False
                
                tag_name = content[i+1:tag_end]
                if not self._is_valid_tag_name(tag_name):
                    return False
                
                # Find matching closing tag
                close_tag = f'</{tag_name}>'
                close_pos = content.find(close_tag, tag_end + 1)
                if close_pos == -1:
                    return False
                
                # Recursively validate inner content
                inner_content = content[tag_end+1:close_pos]
                if not self._validate_content(inner_content):
                    return False
                
                i = close_pos + len(close_tag)
            else:
                i += 1
        
        return True
    
    def _parse_tag(self) -> bool:
        """Parse a complete tag (recursive descent helper)"""
        if self.pos >= len(self.code) or self.code[self.pos] != '<':
            return False
        
        # Check for CDATA
        if self.code[self.pos:self.pos+9] == '<![CDATA[':
            return self._parse_cdata()
        
        # Parse opening tag
        self.pos += 1  # Skip '<'
        tag_start = self.pos
        
        while (self.pos < len(self.code) and 
               self.code[self.pos] != '>' and 
               self.code[self.pos] != '<'):
            self.pos += 1
        
        if self.pos >= len(self.code) or self.code[self.pos] != '>':
            return False
        
        tag_name = self.code[tag_start:self.pos]
        if not self._is_valid_tag_name(tag_name):
            return False
        
        self.pos += 1  # Skip '>'
        
        # Parse content
        while self.pos < len(self.code):
            if self.code[self.pos:self.pos+2] == '</':
                # Check closing tag
                self.pos += 2  # Skip '</'
                close_start = self.pos
                
                while (self.pos < len(self.code) and 
                       self.code[self.pos] != '>'):
                    self.pos += 1
                
                if self.pos >= len(self.code):
                    return False
                
                close_tag = self.code[close_start:self.pos]
                if close_tag != tag_name:
                    return False
                
                self.pos += 1  # Skip '>'
                return True
            
            elif self.code[self.pos] == '<':
                # Nested tag or CDATA
                if not self._parse_tag():
                    return False
            else:
                # Regular character
                self.pos += 1
        
        return False  # No closing tag found
    
    def _parse_cdata(self) -> bool:
        """Parse CDATA section"""
        if self.code[self.pos:self.pos+9] != '<![CDATA[':
            return False
        
        self.pos += 9
        end_pos = self.code.find(']]>', self.pos)
        
        if end_pos == -1:
            return False
        
        self.pos = end_pos + 3
        return True


def test_tag_validator():
    """Test tag validator algorithms"""
    validator = TagValidator()
    
    test_cases = [
        ("<DIV>This is the first line <![CDATA[<div>]]></DIV>", True, "Example 1"),
        ("<DIV>>>  ![cdata[]] <![CDATA[<div>content</div>]]>></DIV>", True, "Example 2"),
        ("<A>  <B> </A>   </B>", False, "Example 3"),
        ("<DIV>  div tag is not closed  <DIV>", False, "Example 4"),
        ("<DIV>  unmatched <  </DIV>", False, "Example 5"),
        ("<DIV> closed tags with invalid tag name  <b>123</b> </DIV>", False, "Example 6"),
        ("<DIV> unmatched tags with invalid tag name  </1234567890> and <CDATA[[]]>  </DIV>", False, "Example 7"),
        ("<DIV>  <![CDATA[ <DIV> ]]>  </DIV>", True, "Example 8"),
        ("<A></A>", True, "Simple valid tag"),
        ("<A><B></B></A>", True, "Nested valid tags"),
        ("<A><B></B><C></C></A>", True, "Multiple nested tags"),
        ("<ABCDEFGHI></ABCDEFGHI>", True, "Max length tag name"),
        ("<ABCDEFGHIJ></ABCDEFGHIJ>", False, "Too long tag name"),
        ("<a></a>", False, "Lowercase tag name"),
        ("<A><B></A></B>", False, "Mismatched nesting"),
        ("", False, "Empty string"),
        ("text", False, "No tags"),
        ("<A>", False, "Unclosed tag"),
        ("</A>", False, "Only closing tag"),
    ]
    
    algorithms = [
        ("Stack Approach", validator.isValid_stack_approach),
        ("Regex Approach", validator.isValid_regex_approach),
        ("Recursive Descent", validator.isValid_recursive_descent),
        ("State Machine", validator.isValid_state_machine),
    ]
    
    print("=== Testing Tag Validator ===")
    
    for code, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Code: '{code[:50]}{'...' if len(code) > 50 else ''}'")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(code)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    code = "<A><B></B></A>"
    print(f"Code: '{code}'")
    print("Strategy: Use stack to track open tags and validate nesting")
    
    stack = []
    i = 0
    
    print(f"\nStep-by-step processing:")
    
    while i < len(code):
        print(f"\nStep: At position {i}, character '{code[i] if i < len(code) else 'EOF'}'")
        print(f"  Stack: {stack}")
        
        if code[i] == '<':
            if i + 1 < len(code) and code[i+1] == '/':
                # Closing tag
                i += 2
                tag_end = code.find('>', i)
                tag_name = code[i:tag_end]
                
                print(f"  Found closing tag: '{tag_name}'")
                
                if stack and stack[-1] == tag_name:
                    stack.pop()
                    print(f"  Matched with open tag, popped from stack")
                else:
                    print(f"  ERROR: No matching open tag")
                    break
                
                i = tag_end + 1
            else:
                # Opening tag
                i += 1
                tag_end = code.find('>', i)
                tag_name = code[i:tag_end]
                
                print(f"  Found opening tag: '{tag_name}'")
                
                if 1 <= len(tag_name) <= 9 and tag_name.isupper():
                    stack.append(tag_name)
                    print(f"  Valid tag, pushed to stack")
                else:
                    print(f"  ERROR: Invalid tag name")
                    break
                
                i = tag_end + 1
        else:
            # Regular character
            print(f"  Regular character: '{code[i]}'")
            i += 1
    
    print(f"\nFinal stack: {stack}")
    print(f"Valid: {len(stack) == 0}")


def demonstrate_cdata_handling():
    """Demonstrate CDATA handling"""
    print("\n=== CDATA Handling Demonstration ===")
    
    examples = [
        "<A><![CDATA[<B>content</B>]]></A>",
        "<A><![CDATA[invalid < > tags]]></A>",
        "<A>before<![CDATA[middle]]>after</A>",
    ]
    
    validator = TagValidator()
    
    for code in examples:
        print(f"\nCode: '{code}'")
        
        # Show CDATA extraction
        cdata_start = code.find('<![CDATA[')
        if cdata_start != -1:
            cdata_end = code.find(']]>', cdata_start)
            if cdata_end != -1:
                cdata_content = code[cdata_start+9:cdata_end]
                print(f"  CDATA content: '{cdata_content}'")
                print(f"  Note: CDATA content is treated as literal text")
        
        result = validator.isValid_stack_approach(code)
        print(f"  Valid: {result}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    validator = TagValidator()
    
    # Application 1: XML validation
    print("1. XML Document Validation:")
    xml_examples = [
        ("<ROOT><HEADER>Title</HEADER><BODY>Content</BODY></ROOT>", "Simple XML document"),
        ("<CONFIG><DATABASE><HOST>localhost</HOST><PORT>3306</PORT></DATABASE></CONFIG>", "Configuration file"),
        ("<HTML><HEAD><TITLE>Page</TITLE></HEAD><BODY>Content</BODY></HTML>", "HTML-like structure"),
    ]
    
    for xml, description in xml_examples:
        result = validator.isValid_stack_approach(xml)
        print(f"  {description}: {'Valid' if result else 'Invalid'}")
        print(f"    {xml[:50]}{'...' if len(xml) > 50 else ''}")
    
    # Application 2: Template validation
    print(f"\n2. Template System Validation:")
    templates = [
        ("<TEMPLATE><HEADER>{{title}}</HEADER><CONTENT>{{body}}</CONTENT></TEMPLATE>", "Template structure"),
        ("<FORM><INPUT>name</INPUT><BUTTON>Submit</BUTTON></FORM>", "Form template"),
        ("<LAYOUT><SIDEBAR>menu</SIDEBAR><MAIN>content</MAIN></LAYOUT>", "Layout template"),
    ]
    
    for template, description in templates:
        result = validator.isValid_stack_approach(template)
        print(f"  {description}: {'Valid' if result else 'Invalid'}")
    
    # Application 3: Data serialization validation
    print(f"\n3. Data Serialization Format:")
    data_formats = [
        ("<DATA><USER><NAME>John</NAME><AGE>30</AGE></USER></DATA>", "User data"),
        ("<RESPONSE><STATUS>OK</STATUS><RESULT>Success</RESULT></RESPONSE>", "API response"),
        ("<LOG><TIMESTAMP>2023-01-01</TIMESTAMP><MESSAGE>Info</MESSAGE></LOG>", "Log entry"),
    ]
    
    for data, description in data_formats:
        result = validator.isValid_stack_approach(data)
        print(f"  {description}: {'Valid' if result else 'Invalid'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(n)", "Single pass with stack tracking"),
        ("Regex Approach", "O(n²)", "O(n)", "Multiple regex operations"),
        ("Recursive Descent", "O(n)", "O(n)", "Recursive parsing"),
        ("State Machine", "O(n)", "O(n)", "State-based validation"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nStack and State Machine approaches are most efficient")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    validator = TagValidator()
    
    edge_cases = [
        ("", False, "Empty string"),
        ("<>", False, "Empty tag name"),
        ("<A", False, "Unclosed opening bracket"),
        ("A>", False, "No opening bracket"),
        ("<A></B>", False, "Mismatched tag names"),
        ("<A><B></B></A>", True, "Proper nesting"),
        ("<A><B></A></B>", False, "Improper nesting"),
        ("<![CDATA[content]]>", False, "CDATA without wrapper tag"),
        ("<A><![CDATA[]]></A>", True, "Empty CDATA"),
        ("<A><![CDATA[<invalid>]]></A>", True, "Invalid tags in CDATA"),
        ("<ABCDEFGHI></ABCDEFGHI>", True, "Maximum tag length"),
        ("<ABCDEFGHIJ></ABCDEFGHIJ>", False, "Exceeds maximum tag length"),
        ("<a></a>", False, "Lowercase tag"),
        ("<A1></A1>", False, "Tag with number"),
        ("<A_B></A_B>", False, "Tag with underscore"),
    ]
    
    for code, expected, description in edge_cases:
        try:
            result = validator.isValid_stack_approach(code)
            status = "✓" if result == expected else "✗"
            print(f"{description:30} | {status} | '{code[:20]}{'...' if len(code) > 20 else ''}' -> {result}")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_tag_validator()
    demonstrate_stack_approach()
    demonstrate_cdata_handling()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()

"""
Tag Validator demonstrates advanced expression parsing for XML-like
structures, including stack-based validation, recursive descent parsing,
and multiple approaches for handling nested tags and CDATA sections.
"""
