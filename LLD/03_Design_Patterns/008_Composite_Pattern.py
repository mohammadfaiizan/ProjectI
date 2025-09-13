"""
COMPOSITE PATTERN - Structural Design Pattern
=============================================

Problem Statement:
Implement the Composite pattern to treat individual objects and compositions
of objects uniformly:
- Tree structures with leaf and composite nodes
- Uniform interface for individual and composite objects
- Recursive operations on hierarchical structures
- File system and UI component hierarchies
- Mathematical expression trees

Learning Objectives:
- Understand when to use Composite pattern
- Design uniform interfaces for leaf and composite objects
- Implement recursive operations on tree structures
- Handle complex hierarchical data structures
- Build flexible component-based systems
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Iterator, Dict, Any, Union
import json
from datetime import datetime
from enum import Enum


# ============================================================================
# COMPONENT INTERFACE
# ============================================================================

class FileSystemComponent(ABC):
    """Abstract component for file system elements."""
    
    def __init__(self, name: str):
        self.name = name
        self.parent: Optional['FileSystemComponent'] = None
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
    
    @abstractmethod
    def get_size(self) -> int:
        """Get size in bytes."""
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """Get component type."""
        pass
    
    @abstractmethod
    def display(self, indent: int = 0) -> str:
        """Display component with indentation."""
        pass
    
    def get_path(self) -> str:
        """Get full path of component."""
        if self.parent is None:
            return self.name
        return f"{self.parent.get_path()}/{self.name}"
    
    def get_depth(self) -> int:
        """Get depth in hierarchy."""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1
    
    # Default implementations for composite operations (will be overridden in Composite)
    def add(self, component: 'FileSystemComponent') -> None:
        """Add child component (only for composites)."""
        raise NotImplementedError("Cannot add to leaf component")
    
    def remove(self, component: 'FileSystemComponent') -> bool:
        """Remove child component (only for composites)."""
        raise NotImplementedError("Cannot remove from leaf component")
    
    def get_child(self, index: int) -> Optional['FileSystemComponent']:
        """Get child by index (only for composites)."""
        raise NotImplementedError("Leaf component has no children")
    
    def get_children(self) -> List['FileSystemComponent']:
        """Get all children (only for composites)."""
        return []


class UIComponent(ABC):
    """Abstract component for UI elements."""
    
    def __init__(self, name: str):
        self.name = name
        self.visible = True
        self.enabled = True
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.parent: Optional['UIComponent'] = None
    
    @abstractmethod
    def render(self) -> str:
        """Render the component."""
        pass
    
    @abstractmethod
    def get_bounds(self) -> Dict[str, int]:
        """Get component bounds."""
        pass
    
    def set_position(self, x: int, y: int) -> None:
        """Set component position."""
        self.x = x
        self.y = y
    
    def set_size(self, width: int, height: int) -> None:
        """Set component size."""
        self.width = width
        self.height = height
    
    def set_visible(self, visible: bool) -> None:
        """Set visibility."""
        self.visible = visible
    
    def set_enabled(self, enabled: bool) -> None:
        """Set enabled state."""
        self.enabled = enabled
    
    def get_absolute_position(self) -> Dict[str, int]:
        """Get absolute position in screen coordinates."""
        if self.parent is None:
            return {'x': self.x, 'y': self.y}
        
        parent_pos = self.parent.get_absolute_position()
        return {
            'x': parent_pos['x'] + self.x,
            'y': parent_pos['y'] + self.y
        }
    
    # Default implementations for composite operations
    def add_child(self, component: 'UIComponent') -> None:
        """Add child component (only for composites)."""
        raise NotImplementedError("Cannot add child to leaf component")
    
    def remove_child(self, component: 'UIComponent') -> bool:
        """Remove child component (only for composites)."""
        raise NotImplementedError("Cannot remove child from leaf component")
    
    def get_children(self) -> List['UIComponent']:
        """Get all children (only for composites)."""
        return []


class Expression(ABC):
    """Abstract component for mathematical expressions."""
    
    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the expression."""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert expression to string representation."""
        pass
    
    @abstractmethod
    def get_variables(self) -> set:
        """Get all variables in the expression."""
        pass


# ============================================================================
# LEAF COMPONENTS
# ============================================================================

class File(FileSystemComponent):
    """Leaf component representing a file."""
    
    def __init__(self, name: str, content: str = ""):
        super().__init__(name)
        self.content = content
        self.file_extension = name.split('.')[-1] if '.' in name else ''
    
    def get_size(self) -> int:
        """Get file size in bytes."""
        return len(self.content.encode('utf-8'))
    
    def get_type(self) -> str:
        """Get file type."""
        return "file"
    
    def display(self, indent: int = 0) -> str:
        """Display file information."""
        spaces = "  " * indent
        size = self.get_size()
        return f"{spaces}📄 {self.name} ({size} bytes)"
    
    def read_content(self) -> str:
        """Read file content."""
        return self.content
    
    def write_content(self, content: str) -> None:
        """Write content to file."""
        self.content = content
        self.modified_at = datetime.now()
    
    def append_content(self, content: str) -> None:
        """Append content to file."""
        self.content += content
        self.modified_at = datetime.now()
    
    def get_line_count(self) -> int:
        """Get number of lines in file."""
        return len(self.content.split('\n')) if self.content else 0


class Button(UIComponent):
    """Leaf component representing a button."""
    
    def __init__(self, name: str, text: str):
        super().__init__(name)
        self.text = text
        self.width = 100
        self.height = 30
        self.background_color = "#f0f0f0"
        self.text_color = "#000000"
        self.border_width = 1
    
    def render(self) -> str:
        """Render button."""
        if not self.visible:
            return ""
        
        status = "enabled" if self.enabled else "disabled"
        pos = self.get_absolute_position()
        
        return f'<button id="{self.name}" x="{pos["x"]}" y="{pos["y"]}" ' \
               f'width="{self.width}" height="{self.height}" ' \
               f'status="{status}" bg="{self.background_color}">{self.text}</button>'
    
    def get_bounds(self) -> Dict[str, int]:
        """Get button bounds."""
        pos = self.get_absolute_position()
        return {
            'x': pos['x'],
            'y': pos['y'],
            'width': self.width,
            'height': self.height
        }
    
    def set_text(self, text: str) -> None:
        """Set button text."""
        self.text = text
    
    def set_colors(self, background: str, text_color: str) -> None:
        """Set button colors."""
        self.background_color = background
        self.text_color = text_color


class TextBox(UIComponent):
    """Leaf component representing a text box."""
    
    def __init__(self, name: str, placeholder: str = ""):
        super().__init__(name)
        self.placeholder = placeholder
        self.value = ""
        self.width = 200
        self.height = 25
        self.max_length = 255
        self.readonly = False
    
    def render(self) -> str:
        """Render text box."""
        if not self.visible:
            return ""
        
        status = "enabled" if self.enabled else "disabled"
        readonly_attr = "readonly" if self.readonly else ""
        pos = self.get_absolute_position()
        
        return f'<input type="text" id="{self.name}" x="{pos["x"]}" y="{pos["y"]}" ' \
               f'width="{self.width}" height="{self.height}" ' \
               f'placeholder="{self.placeholder}" value="{self.value}" ' \
               f'maxlength="{self.max_length}" {readonly_attr} status="{status}"/>'
    
    def get_bounds(self) -> Dict[str, int]:
        """Get text box bounds."""
        pos = self.get_absolute_position()
        return {
            'x': pos['x'],
            'y': pos['y'],
            'width': self.width,
            'height': self.height
        }
    
    def set_value(self, value: str) -> None:
        """Set text box value."""
        if len(value) <= self.max_length:
            self.value = value
    
    def get_value(self) -> str:
        """Get text box value."""
        return self.value


class Number(Expression):
    """Leaf component representing a number in expression."""
    
    def __init__(self, value: float):
        self.value = value
    
    def evaluate(self) -> float:
        """Evaluate number (returns itself)."""
        return self.value
    
    def to_string(self) -> str:
        """Convert number to string."""
        return str(self.value)
    
    def get_variables(self) -> set:
        """Numbers have no variables."""
        return set()


class Variable(Expression):
    """Leaf component representing a variable in expression."""
    
    def __init__(self, name: str, value: float = 0.0):
        self.name = name
        self.value = value
    
    def evaluate(self) -> float:
        """Evaluate variable (returns its value)."""
        return self.value
    
    def to_string(self) -> str:
        """Convert variable to string."""
        return self.name
    
    def get_variables(self) -> set:
        """Return variable name."""
        return {self.name}
    
    def set_value(self, value: float) -> None:
        """Set variable value."""
        self.value = value


# ============================================================================
# COMPOSITE COMPONENTS
# ============================================================================

class Directory(FileSystemComponent):
    """Composite component representing a directory."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.children: List[FileSystemComponent] = []
    
    def get_size(self) -> int:
        """Get total size of directory and all contents."""
        total_size = 0
        for child in self.children:
            total_size += child.get_size()
        return total_size
    
    def get_type(self) -> str:
        """Get directory type."""
        return "directory"
    
    def display(self, indent: int = 0) -> str:
        """Display directory and all contents."""
        spaces = "  " * indent
        result = f"{spaces}📁 {self.name}/ ({len(self.children)} items)\n"
        
        for child in self.children:
            result += child.display(indent + 1) + "\n"
        
        return result.rstrip()
    
    def add(self, component: FileSystemComponent) -> None:
        """Add child component to directory."""
        component.parent = self
        self.children.append(component)
        self.modified_at = datetime.now()
        print(f"Added {component.name} to directory {self.name}")
    
    def remove(self, component: FileSystemComponent) -> bool:
        """Remove child component from directory."""
        if component in self.children:
            component.parent = None
            self.children.remove(component)
            self.modified_at = datetime.now()
            print(f"Removed {component.name} from directory {self.name}")
            return True
        return False
    
    def get_child(self, index: int) -> Optional[FileSystemComponent]:
        """Get child by index."""
        if 0 <= index < len(self.children):
            return self.children[index]
        return None
    
    def get_children(self) -> List[FileSystemComponent]:
        """Get all children."""
        return self.children.copy()
    
    def find(self, name: str) -> Optional[FileSystemComponent]:
        """Find child by name (recursive search)."""
        for child in self.children:
            if child.name == name:
                return child
            
            # If child is a directory, search recursively
            if isinstance(child, Directory):
                found = child.find(name)
                if found:
                    return found
        
        return None
    
    def get_file_count(self) -> int:
        """Get total number of files in directory tree."""
        count = 0
        for child in self.children:
            if isinstance(child, File):
                count += 1
            elif isinstance(child, Directory):
                count += child.get_file_count()
        return count
    
    def get_directory_count(self) -> int:
        """Get total number of subdirectories."""
        count = 0
        for child in self.children:
            if isinstance(child, Directory):
                count += 1 + child.get_directory_count()
        return count
    
    def list_all_files(self) -> List[File]:
        """Get list of all files in directory tree."""
        files = []
        for child in self.children:
            if isinstance(child, File):
                files.append(child)
            elif isinstance(child, Directory):
                files.extend(child.list_all_files())
        return files


class Panel(UIComponent):
    """Composite component representing a UI panel."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.children: List[UIComponent] = []
        self.background_color = "#ffffff"
        self.border_color = "#cccccc"
        self.border_width = 1
        self.padding = 5
    
    def render(self) -> str:
        """Render panel and all children."""
        if not self.visible:
            return ""
        
        pos = self.get_absolute_position()
        result = f'<panel id="{self.name}" x="{pos["x"]}" y="{pos["y"]}" ' \
                f'width="{self.width}" height="{self.height}" ' \
                f'bg="{self.background_color}" border="{self.border_color}">\n'
        
        for child in self.children:
            if child.visible:
                child_render = child.render()
                if child_render:
                    result += "  " + child_render.replace('\n', '\n  ') + "\n"
        
        result += "</panel>"
        return result
    
    def get_bounds(self) -> Dict[str, int]:
        """Get panel bounds."""
        pos = self.get_absolute_position()
        return {
            'x': pos['x'],
            'y': pos['y'],
            'width': self.width,
            'height': self.height
        }
    
    def add_child(self, component: UIComponent) -> None:
        """Add child component to panel."""
        component.parent = self
        self.children.append(component)
        print(f"Added {component.name} to panel {self.name}")
    
    def remove_child(self, component: UIComponent) -> bool:
        """Remove child component from panel."""
        if component in self.children:
            component.parent = None
            self.children.remove(component)
            print(f"Removed {component.name} from panel {self.name}")
            return True
        return False
    
    def get_children(self) -> List[UIComponent]:
        """Get all children."""
        return self.children.copy()
    
    def find_component(self, name: str) -> Optional[UIComponent]:
        """Find child component by name (recursive search)."""
        for child in self.children:
            if child.name == name:
                return child
            
            # If child is a panel, search recursively
            if isinstance(child, Panel):
                found = child.find_component(name)
                if found:
                    return found
        
        return None
    
    def auto_layout(self, layout_type: str = "vertical") -> None:
        """Auto-layout children components."""
        if not self.children:
            return
        
        if layout_type == "vertical":
            current_y = self.padding
            for child in self.children:
                child.set_position(self.padding, current_y)
                current_y += child.height + self.padding
        
        elif layout_type == "horizontal":
            current_x = self.padding
            for child in self.children:
                child.set_position(current_x, self.padding)
                current_x += child.width + self.padding
        
        elif layout_type == "grid":
            cols = int(len(self.children) ** 0.5) + 1
            for i, child in enumerate(self.children):
                row = i // cols
                col = i % cols
                x = self.padding + col * (child.width + self.padding)
                y = self.padding + row * (child.height + self.padding)
                child.set_position(x, y)
    
    def get_total_child_count(self) -> int:
        """Get total number of child components (recursive)."""
        count = len(self.children)
        for child in self.children:
            if isinstance(child, Panel):
                count += child.get_total_child_count()
        return count


class BinaryOperation(Expression):
    """Composite component representing a binary operation."""
    
    def __init__(self, operator: str, left: Expression, right: Expression):
        self.operator = operator
        self.left = left
        self.right = right
    
    def evaluate(self) -> float:
        """Evaluate binary operation."""
        left_val = self.left.evaluate()
        right_val = self.right.evaluate()
        
        if self.operator == '+':
            return left_val + right_val
        elif self.operator == '-':
            return left_val - right_val
        elif self.operator == '*':
            return left_val * right_val
        elif self.operator == '/':
            if right_val == 0:
                raise ValueError("Division by zero")
            return left_val / right_val
        elif self.operator == '^' or self.operator == '**':
            return left_val ** right_val
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def to_string(self) -> str:
        """Convert binary operation to string."""
        return f"({self.left.to_string()} {self.operator} {self.right.to_string()})"
    
    def get_variables(self) -> set:
        """Get all variables in the operation."""
        return self.left.get_variables().union(self.right.get_variables())


class UnaryOperation(Expression):
    """Composite component representing a unary operation."""
    
    def __init__(self, operator: str, operand: Expression):
        self.operator = operator
        self.operand = operand
    
    def evaluate(self) -> float:
        """Evaluate unary operation."""
        operand_val = self.operand.evaluate()
        
        if self.operator == '-':
            return -operand_val
        elif self.operator == '+':
            return operand_val
        elif self.operator == 'sqrt':
            if operand_val < 0:
                raise ValueError("Square root of negative number")
            return operand_val ** 0.5
        elif self.operator == 'abs':
            return abs(operand_val)
        else:
            raise ValueError(f"Unknown unary operator: {self.operator}")
    
    def to_string(self) -> str:
        """Convert unary operation to string."""
        if self.operator in ['-', '+']:
            return f"{self.operator}{self.operand.to_string()}"
        else:
            return f"{self.operator}({self.operand.to_string()})"
    
    def get_variables(self) -> set:
        """Get all variables in the operation."""
        return self.operand.get_variables()


# ============================================================================
# COMPOSITE PATTERN UTILITIES
# ============================================================================

class FileSystemManager:
    """Manager for file system operations using composite pattern."""
    
    def __init__(self):
        self.root = Directory("root")
    
    def create_file_structure(self) -> Directory:
        """Create a sample file structure."""
        # Create directories
        documents = Directory("Documents")
        projects = Directory("Projects")
        images = Directory("Images")
        
        # Create files
        readme = File("README.md", "# Project Documentation\n\nThis is a sample project.")
        config = File("config.json", '{"debug": true, "port": 8080}')
        script = File("script.py", "print('Hello, World!')\n")
        photo1 = File("photo1.jpg", "binary_image_data_1")
        photo2 = File("photo2.png", "binary_image_data_2")
        
        # Build structure
        self.root.add(documents)
        self.root.add(projects)
        self.root.add(images)
        
        documents.add(readme)
        projects.add(config)
        projects.add(script)
        images.add(photo1)
        images.add(photo2)
        
        # Create nested structure
        web_project = Directory("WebProject")
        projects.add(web_project)
        
        html_file = File("index.html", "<html><body>Hello World</body></html>")
        css_file = File("style.css", "body { margin: 0; }")
        js_file = File("app.js", "console.log('App started');")
        
        web_project.add(html_file)
        web_project.add(css_file)
        web_project.add(js_file)
        
        return self.root
    
    def get_statistics(self, component: FileSystemComponent) -> Dict[str, Any]:
        """Get comprehensive statistics for a file system component."""
        stats = {
            'name': component.name,
            'type': component.get_type(),
            'size': component.get_size(),
            'path': component.get_path(),
            'depth': component.get_depth()
        }
        
        if isinstance(component, Directory):
            stats.update({
                'children_count': len(component.children),
                'file_count': component.get_file_count(),
                'directory_count': component.get_directory_count(),
                'total_items': component.get_file_count() + component.get_directory_count()
            })
        elif isinstance(component, File):
            stats.update({
                'extension': component.file_extension,
                'line_count': component.get_line_count()
            })
        
        return stats


class UIBuilder:
    """Builder for creating UI hierarchies using composite pattern."""
    
    def __init__(self):
        self.root = Panel("MainWindow")
        self.root.set_size(800, 600)
    
    def create_login_form(self) -> Panel:
        """Create a login form UI."""
        login_panel = Panel("LoginPanel")
        login_panel.set_size(300, 200)
        login_panel.set_position(250, 200)
        
        # Create form components
        title_label = Button("TitleLabel", "Login")  # Using button as label
        title_label.set_size(200, 30)
        title_label.background_color = "#e0e0e0"
        
        username_box = TextBox("UsernameBox", "Enter username")
        username_box.set_size(250, 25)
        
        password_box = TextBox("PasswordBox", "Enter password")
        password_box.set_size(250, 25)
        
        login_button = Button("LoginButton", "Login")
        login_button.set_size(100, 30)
        login_button.background_color = "#4CAF50"
        
        cancel_button = Button("CancelButton", "Cancel")
        cancel_button.set_size(100, 30)
        cancel_button.background_color = "#f44336"
        
        # Add components to panel
        login_panel.add_child(title_label)
        login_panel.add_child(username_box)
        login_panel.add_child(password_box)
        login_panel.add_child(login_button)
        login_panel.add_child(cancel_button)
        
        # Auto-layout components
        login_panel.auto_layout("vertical")
        
        return login_panel
    
    def create_dashboard(self) -> Panel:
        """Create a dashboard UI with nested panels."""
        dashboard = Panel("Dashboard")
        dashboard.set_size(780, 580)
        dashboard.set_position(10, 10)
        
        # Create header panel
        header = Panel("Header")
        header.set_size(760, 60)
        header.background_color = "#2196F3"
        
        title = Button("DashboardTitle", "Application Dashboard")
        title.set_size(300, 40)
        title.background_color = "#2196F3"
        title.text_color = "#ffffff"
        
        logout_btn = Button("LogoutButton", "Logout")
        logout_btn.set_size(80, 30)
        logout_btn.background_color = "#f44336"
        
        header.add_child(title)
        header.add_child(logout_btn)
        header.auto_layout("horizontal")
        
        # Create content area
        content = Panel("ContentArea")
        content.set_size(760, 450)
        content.background_color = "#f5f5f5"
        
        # Create sidebar
        sidebar = Panel("Sidebar")
        sidebar.set_size(200, 430)
        sidebar.background_color = "#e0e0e0"
        
        nav_buttons = ["Home", "Users", "Reports", "Settings"]
        for btn_text in nav_buttons:
            nav_btn = Button(f"Nav{btn_text}", btn_text)
            nav_btn.set_size(180, 35)
            sidebar.add_child(nav_btn)
        
        sidebar.auto_layout("vertical")
        
        # Create main content
        main_content = Panel("MainContent")
        main_content.set_size(540, 430)
        main_content.background_color = "#ffffff"
        
        welcome_msg = Button("WelcomeMessage", "Welcome to the Dashboard!")
        welcome_msg.set_size(500, 40)
        welcome_msg.background_color = "#ffffff"
        
        stats_panel = Panel("StatsPanel")
        stats_panel.set_size(500, 100)
        stats_panel.background_color = "#f0f0f0"
        
        stat_items = ["Users: 1,234", "Orders: 567", "Revenue: $45,678"]
        for stat_text in stat_items:
            stat_btn = Button(f"Stat{stat_text.split(':')[0]}", stat_text)
            stat_btn.set_size(150, 30)
            stat_btn.background_color = "#4CAF50"
            stats_panel.add_child(stat_btn)
        
        stats_panel.auto_layout("horizontal")
        
        main_content.add_child(welcome_msg)
        main_content.add_child(stats_panel)
        main_content.auto_layout("vertical")
        
        content.add_child(sidebar)
        content.add_child(main_content)
        content.auto_layout("horizontal")
        
        # Create footer
        footer = Panel("Footer")
        footer.set_size(760, 50)
        footer.background_color = "#9E9E9E"
        
        footer_text = Button("FooterText", "© 2024 Application Name. All rights reserved.")
        footer_text.set_size(400, 30)
        footer_text.background_color = "#9E9E9E"
        
        footer.add_child(footer_text)
        
        # Assemble dashboard
        dashboard.add_child(header)
        dashboard.add_child(content)
        dashboard.add_child(footer)
        dashboard.auto_layout("vertical")
        
        return dashboard


class ExpressionBuilder:
    """Builder for creating mathematical expressions using composite pattern."""
    
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
    
    def create_variable(self, name: str, value: float = 0.0) -> Variable:
        """Create or get a variable."""
        if name not in self.variables:
            self.variables[name] = Variable(name, value)
        else:
            self.variables[name].set_value(value)
        return self.variables[name]
    
    def create_number(self, value: float) -> Number:
        """Create a number."""
        return Number(value)
    
    def create_binary_op(self, operator: str, left: Expression, right: Expression) -> BinaryOperation:
        """Create a binary operation."""
        return BinaryOperation(operator, left, right)
    
    def create_unary_op(self, operator: str, operand: Expression) -> UnaryOperation:
        """Create a unary operation."""
        return UnaryOperation(operator, operand)
    
    def parse_simple_expression(self, expr_str: str) -> Expression:
        """Parse a simple mathematical expression (simplified parser)."""
        expr_str = expr_str.strip()
        
        # Handle parentheses (simplified)
        if expr_str.startswith('(') and expr_str.endswith(')'):
            return self.parse_simple_expression(expr_str[1:-1])
        
        # Handle binary operations (simplified - only handles basic cases)
        for op in ['+', '-', '*', '/', '^']:
            if op in expr_str:
                parts = expr_str.split(op, 1)
                if len(parts) == 2:
                    left = self.parse_simple_expression(parts[0].strip())
                    right = self.parse_simple_expression(parts[1].strip())
                    return self.create_binary_op(op, left, right)
        
        # Handle numbers
        try:
            value = float(expr_str)
            return self.create_number(value)
        except ValueError:
            pass
        
        # Handle variables
        if expr_str.isalpha():
            return self.create_variable(expr_str)
        
        raise ValueError(f"Cannot parse expression: {expr_str}")


def demonstrate_composite_pattern():
    """
    Demonstrate Composite pattern implementations.
    """
    print("=== COMPOSITE PATTERN DEMONSTRATION ===\n")
    
    # 1. File System Composite
    print("1. FILE SYSTEM COMPOSITE:")
    
    fs_manager = FileSystemManager()
    root = fs_manager.create_file_structure()
    
    print("   File system structure:")
    print(root.display())
    
    # Show statistics
    stats = fs_manager.get_statistics(root)
    print(f"\n   Root directory statistics:")
    print(f"     Total size: {stats['size']} bytes")
    print(f"     Files: {stats['file_count']}")
    print(f"     Directories: {stats['directory_count']}")
    print(f"     Total items: {stats['total_items']}")
    
    # Demonstrate uniform interface
    print(f"\n   Uniform interface demonstration:")
    all_components = [root] + root.list_all_files()
    for component in all_components[:5]:  # Show first 5
        print(f"     {component.name}: {component.get_size()} bytes ({component.get_type()})")
    
    print()
    
    # 2. File Operations
    print("2. FILE SYSTEM OPERATIONS:")
    
    # Find and modify files
    readme_file = root.find("README.md")
    if readme_file and isinstance(readme_file, File):
        print(f"   Found README.md at: {readme_file.get_path()}")
        print(f"   Original content length: {len(readme_file.read_content())}")
        
        readme_file.append_content("\n\n## Additional Information\nThis file was modified.")
        print(f"   Updated content length: {len(readme_file.read_content())}")
        print(f"   Line count: {readme_file.get_line_count()}")
    
    # Add new files
    projects_dir = root.find("Projects")
    if projects_dir and isinstance(projects_dir, Directory):
        new_file = File("notes.txt", "Important project notes")
        projects_dir.add(new_file)
        
        print(f"   Added new file to Projects directory")
        print(f"   Projects directory now has {len(projects_dir.get_children())} items")
    
    print()
    
    # 3. UI Component Composite
    print("3. UI COMPONENT COMPOSITE:")
    
    ui_builder = UIBuilder()
    
    # Create login form
    login_form = ui_builder.create_login_form()
    print("   Login form created:")
    print(f"     Components: {len(login_form.get_children())}")
    print(f"     Total child count (recursive): {login_form.get_total_child_count()}")
    
    # Render login form
    print("\n   Login form rendering:")
    rendered = login_form.render()
    lines = rendered.split('\n')
    for line in lines[:10]:  # Show first 10 lines
        print(f"     {line}")
    if len(lines) > 10:
        print(f"     ... ({len(lines) - 10} more lines)")
    
    print()
    
    # 4. Complex UI Dashboard
    print("4. COMPLEX UI DASHBOARD:")
    
    dashboard = ui_builder.create_dashboard()
    print("   Dashboard created:")
    print(f"     Main panels: {len(dashboard.get_children())}")
    print(f"     Total components: {dashboard.get_total_child_count()}")
    
    # Find specific components
    logout_btn = dashboard.find_component("LogoutButton")
    if logout_btn:
        print(f"     Found logout button at: {logout_btn.get_absolute_position()}")
    
    welcome_msg = dashboard.find_component("WelcomeMessage")
    if welcome_msg:
        bounds = welcome_msg.get_bounds()
        print(f"     Welcome message bounds: {bounds}")
    
    # Show component hierarchy
    print("\n   Component hierarchy (first level):")
    for child in dashboard.get_children():
        print(f"     {child.name} ({child.__class__.__name__})")
        if isinstance(child, Panel):
            for grandchild in child.get_children():
                print(f"       └─ {grandchild.name} ({grandchild.__class__.__name__})")
    
    print()
    
    # 5. Mathematical Expression Composite
    print("5. MATHEMATICAL EXPRESSION COMPOSITE:")
    
    expr_builder = ExpressionBuilder()
    
    # Create variables
    x = expr_builder.create_variable("x", 5.0)
    y = expr_builder.create_variable("y", 3.0)
    
    # Create numbers
    num2 = expr_builder.create_number(2.0)
    num10 = expr_builder.create_number(10.0)
    
    # Build complex expression: (x + y) * 2 + 10
    sum_expr = expr_builder.create_binary_op("+", x, y)
    mult_expr = expr_builder.create_binary_op("*", sum_expr, num2)
    final_expr = expr_builder.create_binary_op("+", mult_expr, num10)
    
    print("   Mathematical expression tree:")
    print(f"     Expression: {final_expr.to_string()}")
    print(f"     Variables: {final_expr.get_variables()}")
    print(f"     Result: {final_expr.evaluate()}")
    
    # Change variable values and re-evaluate
    x.set_value(10.0)
    y.set_value(7.0)
    print(f"\n   After changing variables (x=10, y=7):")
    print(f"     Result: {final_expr.evaluate()}")
    
    # Create more complex expression with unary operations
    sqrt_expr = expr_builder.create_unary_op("sqrt", expr_builder.create_number(16.0))
    neg_expr = expr_builder.create_unary_op("-", x)
    complex_expr = expr_builder.create_binary_op("+", sqrt_expr, neg_expr)
    
    print(f"\n   Complex expression: {complex_expr.to_string()}")
    print(f"     Result: {complex_expr.evaluate()}")
    
    print()
    
    # 6. Expression Parser
    print("6. EXPRESSION PARSER:")
    
    # Parse simple expressions
    expressions_to_parse = [
        "5 + 3",
        "x * 2",
        "10 - y",
        "a + b"
    ]
    
    for expr_str in expressions_to_parse:
        try:
            parsed_expr = expr_builder.parse_simple_expression(expr_str)
            print(f"   Parsed '{expr_str}':")
            print(f"     String form: {parsed_expr.to_string()}")
            print(f"     Variables: {parsed_expr.get_variables()}")
            
            # Set default values for any new variables
            for var_name in parsed_expr.get_variables():
                if var_name not in expr_builder.variables:
                    expr_builder.create_variable(var_name, 1.0)
            
            result = parsed_expr.evaluate()
            print(f"     Result: {result}")
            
        except Exception as e:
            print(f"   Failed to parse '{expr_str}': {e}")
        
        print()
    
    # 7. Composite Operations
    print("7. COMPOSITE OPERATIONS:")
    
    # File system operations
    print("   File system operations:")
    
    # Copy directory structure (simplified)
    def copy_structure(source: FileSystemComponent, target_parent: Directory) -> FileSystemComponent:
        """Recursively copy file system structure."""
        if isinstance(source, File):
            copy = File(f"copy_of_{source.name}", source.content)
            target_parent.add(copy)
            return copy
        elif isinstance(source, Directory):
            copy = Directory(f"copy_of_{source.name}")
            target_parent.add(copy)
            for child in source.get_children():
                copy_structure(child, copy)
            return copy
    
    # Create backup directory
    backup_dir = Directory("backup")
    root.add(backup_dir)
    
    # Copy Projects directory
    projects_dir = root.find("Projects")
    if projects_dir:
        copy_structure(projects_dir, backup_dir)
        print(f"     Copied Projects directory to backup")
        print(f"     Backup directory size: {backup_dir.get_size()} bytes")
    
    # UI operations
    print("\n   UI operations:")
    
    # Clone UI components (simplified)
    def clone_ui_component(source: UIComponent, name_suffix: str = "_clone") -> UIComponent:
        """Clone UI component."""
        if isinstance(source, Button):
            clone = Button(source.name + name_suffix, source.text)
            clone.set_size(source.width, source.height)
            clone.background_color = source.background_color
            return clone
        elif isinstance(source, TextBox):
            clone = TextBox(source.name + name_suffix, source.placeholder)
            clone.set_size(source.width, source.height)
            clone.set_value(source.value)
            return clone
        elif isinstance(source, Panel):
            clone = Panel(source.name + name_suffix)
            clone.set_size(source.width, source.height)
            clone.background_color = source.background_color
            for child in source.get_children():
                child_clone = clone_ui_component(child, name_suffix)
                clone.add_child(child_clone)
            return clone
    
    # Clone login form
    login_clone = clone_ui_component(login_form, "_backup")
    print(f"     Cloned login form: {login_clone.name}")
    print(f"     Clone has {len(login_clone.get_children())} components")
    
    print()
    
    # 8. Composite Pattern Benefits
    print("8. COMPOSITE PATTERN BENEFITS:")
    print("   ✓ Uniform Interface: Same operations work on leaf and composite objects")
    print("   ✓ Hierarchical Structures: Natural representation of tree structures")
    print("   ✓ Recursive Operations: Operations automatically work on entire hierarchies")
    print("   ✓ Flexibility: Easy to add new types of components")
    print("   ✓ Client Simplicity: Clients don't need to distinguish between leaf and composite")
    print("   ✓ Extensibility: New composite and leaf classes can be added easily")
    print("   ✓ Transparency: All components implement the same interface")
    print("   ✓ Recursive Composition: Composites can contain other composites")
    print()
    
    print("=== COMPOSITE PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_composite_pattern()
