"""
ABSTRACT FACTORY PATTERN - Creational Design Pattern
====================================================

Problem Statement:
Implement the Abstract Factory pattern to create families of related objects
without specifying their concrete classes:
- Abstract factory interfaces for product families
- Concrete factories for specific product families
- Consistent product creation across families
- Cross-platform UI component creation
- Theme-based object creation

Learning Objectives:
- Understand Abstract Factory vs Factory Method
- Design product families and their relationships
- Implement cross-platform compatibility layers
- Handle complex object family creation
- Ensure consistency within product families
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Protocol
from enum import Enum
import datetime


# ============================================================================
# PRODUCT FAMILY ENUMS AND TYPES
# ============================================================================

class UITheme(Enum):
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"
    CUSTOM = "custom"


class Platform(Enum):
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    WEB = "web"
    MOBILE = "mobile"


class ComponentSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTRA_LARGE = "xl"


# ============================================================================
# ABSTRACT PRODUCT INTERFACES
# ============================================================================

class Button(ABC):
    """Abstract button component."""
    
    def __init__(self, text: str, size: ComponentSize = ComponentSize.MEDIUM):
        self.text = text
        self.size = size
        self.enabled = True
        self.visible = True
        self.click_handlers = []
    
    @abstractmethod
    def render(self) -> str:
        """Render the button."""
        pass
    
    @abstractmethod
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set button styling."""
        pass
    
    @abstractmethod
    def get_default_style(self) -> Dict[str, Any]:
        """Get default style properties."""
        pass
    
    def add_click_handler(self, handler) -> None:
        """Add click event handler."""
        self.click_handlers.append(handler)
    
    def click(self) -> None:
        """Simulate button click."""
        if self.enabled:
            for handler in self.click_handlers:
                handler()
            print(f"Button '{self.text}' clicked")
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable button."""
        self.enabled = enabled
    
    def set_visible(self, visible: bool) -> None:
        """Show/hide button."""
        self.visible = visible


class TextField(ABC):
    """Abstract text field component."""
    
    def __init__(self, placeholder: str = "", size: ComponentSize = ComponentSize.MEDIUM):
        self.placeholder = placeholder
        self.size = size
        self.value = ""
        self.enabled = True
        self.visible = True
        self.validators = []
        self.change_handlers = []
    
    @abstractmethod
    def render(self) -> str:
        """Render the text field."""
        pass
    
    @abstractmethod
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set text field styling."""
        pass
    
    @abstractmethod
    def get_default_style(self) -> Dict[str, Any]:
        """Get default style properties."""
        pass
    
    def set_value(self, value: str) -> bool:
        """Set text field value with validation."""
        if self.validate(value):
            old_value = self.value
            self.value = value
            for handler in self.change_handlers:
                handler(old_value, value)
            return True
        return False
    
    def get_value(self) -> str:
        """Get text field value."""
        return self.value
    
    def add_validator(self, validator) -> None:
        """Add value validator."""
        self.validators.append(validator)
    
    def validate(self, value: str) -> bool:
        """Validate value against all validators."""
        return all(validator(value) for validator in self.validators)
    
    def add_change_handler(self, handler) -> None:
        """Add value change handler."""
        self.change_handlers.append(handler)


class CheckBox(ABC):
    """Abstract checkbox component."""
    
    def __init__(self, label: str, size: ComponentSize = ComponentSize.MEDIUM):
        self.label = label
        self.size = size
        self.checked = False
        self.enabled = True
        self.visible = True
        self.change_handlers = []
    
    @abstractmethod
    def render(self) -> str:
        """Render the checkbox."""
        pass
    
    @abstractmethod
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set checkbox styling."""
        pass
    
    @abstractmethod
    def get_default_style(self) -> Dict[str, Any]:
        """Get default style properties."""
        pass
    
    def toggle(self) -> None:
        """Toggle checkbox state."""
        if self.enabled:
            old_state = self.checked
            self.checked = not self.checked
            for handler in self.change_handlers:
                handler(old_state, self.checked)
            print(f"Checkbox '{self.label}' {'checked' if self.checked else 'unchecked'}")
    
    def set_checked(self, checked: bool) -> None:
        """Set checkbox state."""
        if self.enabled and self.checked != checked:
            old_state = self.checked
            self.checked = checked
            for handler in self.change_handlers:
                handler(old_state, self.checked)
    
    def is_checked(self) -> bool:
        """Get checkbox state."""
        return self.checked
    
    def add_change_handler(self, handler) -> None:
        """Add state change handler."""
        self.change_handlers.append(handler)


class Panel(ABC):
    """Abstract panel/container component."""
    
    def __init__(self, title: str = ""):
        self.title = title
        self.children = []
        self.visible = True
        self.layout_properties = {}
    
    @abstractmethod
    def render(self) -> str:
        """Render the panel."""
        pass
    
    @abstractmethod
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set panel styling."""
        pass
    
    @abstractmethod
    def get_default_style(self) -> Dict[str, Any]:
        """Get default style properties."""
        pass
    
    def add_child(self, child) -> None:
        """Add child component."""
        self.children.append(child)
    
    def remove_child(self, child) -> bool:
        """Remove child component."""
        if child in self.children:
            self.children.remove(child)
            return True
        return False
    
    def get_children(self) -> List[Any]:
        """Get all child components."""
        return self.children.copy()
    
    def set_layout(self, layout_type: str, properties: Dict[str, Any]) -> None:
        """Set layout properties."""
        self.layout_properties = {'type': layout_type, **properties}


# ============================================================================
# LIGHT THEME CONCRETE PRODUCTS
# ============================================================================

class LightButton(Button):
    """Light theme button implementation."""
    
    def render(self) -> str:
        """Render light theme button."""
        style = self.get_default_style()
        size_class = f"btn-{self.size.value}"
        enabled_class = "" if self.enabled else "disabled"
        visible_style = "" if self.visible else "display: none;"
        
        return f'<button class="light-btn {size_class} {enabled_class}" style="{visible_style}">{self.text}</button>'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set light button styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to light button: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get light theme button default style."""
        return {
            'background_color': '#ffffff',
            'text_color': '#333333',
            'border_color': '#cccccc',
            'border_width': '1px',
            'border_radius': '4px',
            'padding': '8px 16px',
            'font_family': 'Arial, sans-serif',
            'font_size': '14px'
        }


class LightTextField(TextField):
    """Light theme text field implementation."""
    
    def render(self) -> str:
        """Render light theme text field."""
        size_class = f"input-{self.size.value}"
        enabled_class = "" if self.enabled else "disabled"
        visible_style = "" if self.visible else "display: none;"
        
        return f'<input type="text" class="light-input {size_class} {enabled_class}" placeholder="{self.placeholder}" value="{self.value}" style="{visible_style}">'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set light text field styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to light text field: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get light theme text field default style."""
        return {
            'background_color': '#ffffff',
            'text_color': '#333333',
            'border_color': '#cccccc',
            'border_width': '1px',
            'border_radius': '4px',
            'padding': '8px 12px',
            'font_family': 'Arial, sans-serif',
            'font_size': '14px'
        }


class LightCheckBox(CheckBox):
    """Light theme checkbox implementation."""
    
    def render(self) -> str:
        """Render light theme checkbox."""
        size_class = f"checkbox-{self.size.value}"
        enabled_class = "" if self.enabled else "disabled"
        visible_style = "" if self.visible else "display: none;"
        checked_attr = "checked" if self.checked else ""
        
        return f'<label class="light-checkbox {size_class} {enabled_class}" style="{visible_style}"><input type="checkbox" {checked_attr}> {self.label}</label>'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set light checkbox styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to light checkbox: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get light theme checkbox default style."""
        return {
            'text_color': '#333333',
            'accent_color': '#007bff',
            'font_family': 'Arial, sans-serif',
            'font_size': '14px'
        }


class LightPanel(Panel):
    """Light theme panel implementation."""
    
    def render(self) -> str:
        """Render light theme panel."""
        visible_style = "" if self.visible else "display: none;"
        title_html = f"<h3>{self.title}</h3>" if self.title else ""
        
        children_html = ""
        for child in self.children:
            if hasattr(child, 'render'):
                children_html += child.render() + "\n"
        
        return f'<div class="light-panel" style="{visible_style}">{title_html}{children_html}</div>'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set light panel styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to light panel: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get light theme panel default style."""
        return {
            'background_color': '#f8f9fa',
            'border_color': '#dee2e6',
            'border_width': '1px',
            'border_radius': '8px',
            'padding': '16px',
            'margin': '8px'
        }


# ============================================================================
# DARK THEME CONCRETE PRODUCTS
# ============================================================================

class DarkButton(Button):
    """Dark theme button implementation."""
    
    def render(self) -> str:
        """Render dark theme button."""
        size_class = f"btn-{self.size.value}"
        enabled_class = "" if self.enabled else "disabled"
        visible_style = "" if self.visible else "display: none;"
        
        return f'<button class="dark-btn {size_class} {enabled_class}" style="{visible_style}">{self.text}</button>'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set dark button styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to dark button: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get dark theme button default style."""
        return {
            'background_color': '#343a40',
            'text_color': '#ffffff',
            'border_color': '#6c757d',
            'border_width': '1px',
            'border_radius': '4px',
            'padding': '8px 16px',
            'font_family': 'Arial, sans-serif',
            'font_size': '14px'
        }


class DarkTextField(TextField):
    """Dark theme text field implementation."""
    
    def render(self) -> str:
        """Render dark theme text field."""
        size_class = f"input-{self.size.value}"
        enabled_class = "" if self.enabled else "disabled"
        visible_style = "" if self.visible else "display: none;"
        
        return f'<input type="text" class="dark-input {size_class} {enabled_class}" placeholder="{self.placeholder}" value="{self.value}" style="{visible_style}">'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set dark text field styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to dark text field: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get dark theme text field default style."""
        return {
            'background_color': '#495057',
            'text_color': '#ffffff',
            'border_color': '#6c757d',
            'border_width': '1px',
            'border_radius': '4px',
            'padding': '8px 12px',
            'font_family': 'Arial, sans-serif',
            'font_size': '14px'
        }


class DarkCheckBox(CheckBox):
    """Dark theme checkbox implementation."""
    
    def render(self) -> str:
        """Render dark theme checkbox."""
        size_class = f"checkbox-{self.size.value}"
        enabled_class = "" if self.enabled else "disabled"
        visible_style = "" if self.visible else "display: none;"
        checked_attr = "checked" if self.checked else ""
        
        return f'<label class="dark-checkbox {size_class} {enabled_class}" style="{visible_style}"><input type="checkbox" {checked_attr}> {self.label}</label>'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set dark checkbox styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to dark checkbox: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get dark theme checkbox default style."""
        return {
            'text_color': '#ffffff',
            'accent_color': '#17a2b8',
            'font_family': 'Arial, sans-serif',
            'font_size': '14px'
        }


class DarkPanel(Panel):
    """Dark theme panel implementation."""
    
    def render(self) -> str:
        """Render dark theme panel."""
        visible_style = "" if self.visible else "display: none;"
        title_html = f"<h3 style='color: #ffffff'>{self.title}</h3>" if self.title else ""
        
        children_html = ""
        for child in self.children:
            if hasattr(child, 'render'):
                children_html += child.render() + "\n"
        
        return f'<div class="dark-panel" style="{visible_style}">{title_html}{children_html}</div>'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set dark panel styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to dark panel: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get dark theme panel default style."""
        return {
            'background_color': '#212529',
            'border_color': '#495057',
            'border_width': '1px',
            'border_radius': '8px',
            'padding': '16px',
            'margin': '8px'
        }


# ============================================================================
# HIGH CONTRAST THEME CONCRETE PRODUCTS
# ============================================================================

class HighContrastButton(Button):
    """High contrast theme button implementation."""
    
    def render(self) -> str:
        """Render high contrast theme button."""
        size_class = f"btn-{self.size.value}"
        enabled_class = "" if self.enabled else "disabled"
        visible_style = "" if self.visible else "display: none;"
        
        return f'<button class="hc-btn {size_class} {enabled_class}" style="{visible_style}">{self.text}</button>'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set high contrast button styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to high contrast button: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get high contrast theme button default style."""
        return {
            'background_color': '#000000',
            'text_color': '#ffffff',
            'border_color': '#ffffff',
            'border_width': '3px',
            'border_radius': '0px',
            'padding': '12px 20px',
            'font_family': 'Arial, sans-serif',
            'font_size': '16px',
            'font_weight': 'bold'
        }


class HighContrastTextField(TextField):
    """High contrast theme text field implementation."""
    
    def render(self) -> str:
        """Render high contrast theme text field."""
        size_class = f"input-{self.size.value}"
        enabled_class = "" if self.enabled else "disabled"
        visible_style = "" if self.visible else "display: none;"
        
        return f'<input type="text" class="hc-input {size_class} {enabled_class}" placeholder="{self.placeholder}" value="{self.value}" style="{visible_style}">'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set high contrast text field styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to high contrast text field: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get high contrast theme text field default style."""
        return {
            'background_color': '#ffffff',
            'text_color': '#000000',
            'border_color': '#000000',
            'border_width': '3px',
            'border_radius': '0px',
            'padding': '12px 16px',
            'font_family': 'Arial, sans-serif',
            'font_size': '16px',
            'font_weight': 'bold'
        }


class HighContrastCheckBox(CheckBox):
    """High contrast theme checkbox implementation."""
    
    def render(self) -> str:
        """Render high contrast theme checkbox."""
        size_class = f"checkbox-{self.size.value}"
        enabled_class = "" if self.enabled else "disabled"
        visible_style = "" if self.visible else "display: none;"
        checked_attr = "checked" if self.checked else ""
        
        return f'<label class="hc-checkbox {size_class} {enabled_class}" style="{visible_style}"><input type="checkbox" {checked_attr}> {self.label}</label>'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set high contrast checkbox styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to high contrast checkbox: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get high contrast theme checkbox default style."""
        return {
            'text_color': '#000000',
            'accent_color': '#000000',
            'font_family': 'Arial, sans-serif',
            'font_size': '16px',
            'font_weight': 'bold'
        }


class HighContrastPanel(Panel):
    """High contrast theme panel implementation."""
    
    def render(self) -> str:
        """Render high contrast theme panel."""
        visible_style = "" if self.visible else "display: none;"
        title_html = f"<h3 style='color: #000000; font-weight: bold;'>{self.title}</h3>" if self.title else ""
        
        children_html = ""
        for child in self.children:
            if hasattr(child, 'render'):
                children_html += child.render() + "\n"
        
        return f'<div class="hc-panel" style="{visible_style}">{title_html}{children_html}</div>'
    
    def set_style(self, style_properties: Dict[str, Any]) -> None:
        """Set high contrast panel styling."""
        self.custom_style = style_properties
        print(f"Applied custom style to high contrast panel: {style_properties}")
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get high contrast theme panel default style."""
        return {
            'background_color': '#ffffff',
            'border_color': '#000000',
            'border_width': '3px',
            'border_radius': '0px',
            'padding': '20px',
            'margin': '10px'
        }


# ============================================================================
# ABSTRACT FACTORY INTERFACE
# ============================================================================

class UIComponentFactory(ABC):
    """Abstract factory for creating UI components."""
    
    @abstractmethod
    def create_button(self, text: str, size: ComponentSize = ComponentSize.MEDIUM) -> Button:
        """Create a button component."""
        pass
    
    @abstractmethod
    def create_text_field(self, placeholder: str = "", size: ComponentSize = ComponentSize.MEDIUM) -> TextField:
        """Create a text field component."""
        pass
    
    @abstractmethod
    def create_checkbox(self, label: str, size: ComponentSize = ComponentSize.MEDIUM) -> CheckBox:
        """Create a checkbox component."""
        pass
    
    @abstractmethod
    def create_panel(self, title: str = "") -> Panel:
        """Create a panel component."""
        pass
    
    @abstractmethod
    def get_theme_name(self) -> str:
        """Get the theme name."""
        pass
    
    @abstractmethod
    def get_theme_properties(self) -> Dict[str, Any]:
        """Get theme-specific properties."""
        pass
    
    def create_form(self, title: str, fields: List[Dict[str, Any]]) -> Panel:
        """Create a complete form with multiple components."""
        form_panel = self.create_panel(title)
        
        for field_config in fields:
            field_type = field_config.get('type', 'text')
            
            if field_type == 'button':
                component = self.create_button(
                    field_config.get('text', 'Button'),
                    ComponentSize(field_config.get('size', 'medium'))
                )
            elif field_type == 'text':
                component = self.create_text_field(
                    field_config.get('placeholder', ''),
                    ComponentSize(field_config.get('size', 'medium'))
                )
            elif field_type == 'checkbox':
                component = self.create_checkbox(
                    field_config.get('label', 'Checkbox'),
                    ComponentSize(field_config.get('size', 'medium'))
                )
            else:
                continue
            
            form_panel.add_child(component)
        
        return form_panel


# ============================================================================
# CONCRETE FACTORIES
# ============================================================================

class LightThemeFactory(UIComponentFactory):
    """Factory for creating light theme UI components."""
    
    def create_button(self, text: str, size: ComponentSize = ComponentSize.MEDIUM) -> Button:
        """Create light theme button."""
        return LightButton(text, size)
    
    def create_text_field(self, placeholder: str = "", size: ComponentSize = ComponentSize.MEDIUM) -> TextField:
        """Create light theme text field."""
        return LightTextField(placeholder, size)
    
    def create_checkbox(self, label: str, size: ComponentSize = ComponentSize.MEDIUM) -> CheckBox:
        """Create light theme checkbox."""
        return LightCheckBox(label, size)
    
    def create_panel(self, title: str = "") -> Panel:
        """Create light theme panel."""
        return LightPanel(title)
    
    def get_theme_name(self) -> str:
        """Get theme name."""
        return "Light Theme"
    
    def get_theme_properties(self) -> Dict[str, Any]:
        """Get light theme properties."""
        return {
            'primary_color': '#007bff',
            'secondary_color': '#6c757d',
            'background_color': '#ffffff',
            'text_color': '#333333',
            'border_color': '#cccccc',
            'accent_color': '#007bff'
        }


class DarkThemeFactory(UIComponentFactory):
    """Factory for creating dark theme UI components."""
    
    def create_button(self, text: str, size: ComponentSize = ComponentSize.MEDIUM) -> Button:
        """Create dark theme button."""
        return DarkButton(text, size)
    
    def create_text_field(self, placeholder: str = "", size: ComponentSize = ComponentSize.MEDIUM) -> TextField:
        """Create dark theme text field."""
        return DarkTextField(placeholder, size)
    
    def create_checkbox(self, label: str, size: ComponentSize = ComponentSize.MEDIUM) -> CheckBox:
        """Create dark theme checkbox."""
        return DarkCheckBox(label, size)
    
    def create_panel(self, title: str = "") -> Panel:
        """Create dark theme panel."""
        return DarkPanel(title)
    
    def get_theme_name(self) -> str:
        """Get theme name."""
        return "Dark Theme"
    
    def get_theme_properties(self) -> Dict[str, Any]:
        """Get dark theme properties."""
        return {
            'primary_color': '#17a2b8',
            'secondary_color': '#6c757d',
            'background_color': '#212529',
            'text_color': '#ffffff',
            'border_color': '#495057',
            'accent_color': '#17a2b8'
        }


class HighContrastThemeFactory(UIComponentFactory):
    """Factory for creating high contrast theme UI components."""
    
    def create_button(self, text: str, size: ComponentSize = ComponentSize.MEDIUM) -> Button:
        """Create high contrast theme button."""
        return HighContrastButton(text, size)
    
    def create_text_field(self, placeholder: str = "", size: ComponentSize = ComponentSize.MEDIUM) -> TextField:
        """Create high contrast theme text field."""
        return HighContrastTextField(placeholder, size)
    
    def create_checkbox(self, label: str, size: ComponentSize = ComponentSize.MEDIUM) -> CheckBox:
        """Create high contrast theme checkbox."""
        return HighContrastCheckBox(label, size)
    
    def create_panel(self, title: str = "") -> Panel:
        """Create high contrast theme panel."""
        return HighContrastPanel(title)
    
    def get_theme_name(self) -> str:
        """Get theme name."""
        return "High Contrast Theme"
    
    def get_theme_properties(self) -> Dict[str, Any]:
        """Get high contrast theme properties."""
        return {
            'primary_color': '#000000',
            'secondary_color': '#ffffff',
            'background_color': '#ffffff',
            'text_color': '#000000',
            'border_color': '#000000',
            'accent_color': '#000000'
        }


# ============================================================================
# FACTORY PROVIDER AND REGISTRY
# ============================================================================

class UIFactoryProvider:
    """Provider for UI component factories."""
    
    def __init__(self):
        self._factories: Dict[UITheme, UIComponentFactory] = {}
        self._current_theme = UITheme.LIGHT
        self._register_default_factories()
    
    def _register_default_factories(self) -> None:
        """Register default theme factories."""
        self.register_factory(UITheme.LIGHT, LightThemeFactory())
        self.register_factory(UITheme.DARK, DarkThemeFactory())
        self.register_factory(UITheme.HIGH_CONTRAST, HighContrastThemeFactory())
    
    def register_factory(self, theme: UITheme, factory: UIComponentFactory) -> None:
        """Register a factory for a theme."""
        self._factories[theme] = factory
        print(f"Registered factory for {theme.value}: {factory.__class__.__name__}")
    
    def get_factory(self, theme: UITheme) -> Optional[UIComponentFactory]:
        """Get factory for theme."""
        return self._factories.get(theme)
    
    def get_current_factory(self) -> Optional[UIComponentFactory]:
        """Get factory for current theme."""
        return self._factories.get(self._current_theme)
    
    def set_current_theme(self, theme: UITheme) -> bool:
        """Set current theme."""
        if theme in self._factories:
            self._current_theme = theme
            print(f"Current theme set to: {theme.value}")
            return True
        return False
    
    def get_current_theme(self) -> UITheme:
        """Get current theme."""
        return self._current_theme
    
    def get_available_themes(self) -> List[UITheme]:
        """Get list of available themes."""
        return list(self._factories.keys())
    
    def create_themed_components(self, theme: UITheme, components_config: List[Dict[str, Any]]) -> List[Any]:
        """Create multiple components with specified theme."""
        factory = self.get_factory(theme)
        if not factory:
            return []
        
        components = []
        for config in components_config:
            component_type = config.get('type')
            
            if component_type == 'button':
                component = factory.create_button(
                    config.get('text', 'Button'),
                    ComponentSize(config.get('size', 'medium'))
                )
            elif component_type == 'text_field':
                component = factory.create_text_field(
                    config.get('placeholder', ''),
                    ComponentSize(config.get('size', 'medium'))
                )
            elif component_type == 'checkbox':
                component = factory.create_checkbox(
                    config.get('label', 'Checkbox'),
                    ComponentSize(config.get('size', 'medium'))
                )
            elif component_type == 'panel':
                component = factory.create_panel(config.get('title', ''))
            else:
                continue
            
            components.append(component)
        
        return components


# ============================================================================
# APPLICATION BUILDER USING ABSTRACT FACTORY
# ============================================================================

class ApplicationBuilder:
    """Builder for creating complete applications using abstract factory."""
    
    def __init__(self, factory_provider: UIFactoryProvider):
        self.factory_provider = factory_provider
        self.applications = {}
    
    def create_login_form(self, theme: UITheme) -> Panel:
        """Create a login form with specified theme."""
        factory = self.factory_provider.get_factory(theme)
        if not factory:
            raise ValueError(f"No factory available for theme: {theme}")
        
        # Create login form components
        login_panel = factory.create_panel("User Login")
        
        username_field = factory.create_text_field("Enter username", ComponentSize.LARGE)
        password_field = factory.create_text_field("Enter password", ComponentSize.LARGE)
        remember_checkbox = factory.create_checkbox("Remember me", ComponentSize.MEDIUM)
        login_button = factory.create_button("Login", ComponentSize.LARGE)
        cancel_button = factory.create_button("Cancel", ComponentSize.MEDIUM)
        
        # Add components to panel
        login_panel.add_child(username_field)
        login_panel.add_child(password_field)
        login_panel.add_child(remember_checkbox)
        login_panel.add_child(login_button)
        login_panel.add_child(cancel_button)
        
        return login_panel
    
    def create_settings_form(self, theme: UITheme) -> Panel:
        """Create a settings form with specified theme."""
        factory = self.factory_provider.get_factory(theme)
        if not factory:
            raise ValueError(f"No factory available for theme: {theme}")
        
        # Create settings form
        settings_panel = factory.create_panel("Application Settings")
        
        # Theme selection
        theme_panel = factory.create_panel("Theme Settings")
        light_theme_checkbox = factory.create_checkbox("Light Theme")
        dark_theme_checkbox = factory.create_checkbox("Dark Theme")
        hc_theme_checkbox = factory.create_checkbox("High Contrast Theme")
        
        theme_panel.add_child(light_theme_checkbox)
        theme_panel.add_child(dark_theme_checkbox)
        theme_panel.add_child(hc_theme_checkbox)
        
        # General settings
        general_panel = factory.create_panel("General Settings")
        notifications_checkbox = factory.create_checkbox("Enable Notifications")
        auto_save_checkbox = factory.create_checkbox("Auto Save")
        language_field = factory.create_text_field("Language")
        
        general_panel.add_child(notifications_checkbox)
        general_panel.add_child(auto_save_checkbox)
        general_panel.add_child(language_field)
        
        # Action buttons
        save_button = factory.create_button("Save Settings", ComponentSize.LARGE)
        reset_button = factory.create_button("Reset to Defaults", ComponentSize.MEDIUM)
        
        # Assemble form
        settings_panel.add_child(theme_panel)
        settings_panel.add_child(general_panel)
        settings_panel.add_child(save_button)
        settings_panel.add_child(reset_button)
        
        return settings_panel
    
    def create_dashboard(self, theme: UITheme) -> Panel:
        """Create a dashboard with specified theme."""
        factory = self.factory_provider.get_factory(theme)
        if not factory:
            raise ValueError(f"No factory available for theme: {theme}")
        
        # Create main dashboard
        dashboard = factory.create_panel("Dashboard")
        
        # Statistics panel
        stats_panel = factory.create_panel("Statistics")
        users_field = factory.create_text_field("Total Users: 1,234")
        revenue_field = factory.create_text_field("Revenue: $45,678")
        orders_field = factory.create_text_field("Orders: 567")
        
        stats_panel.add_child(users_field)
        stats_panel.add_child(revenue_field)
        stats_panel.add_child(orders_field)
        
        # Actions panel
        actions_panel = factory.create_panel("Quick Actions")
        new_user_button = factory.create_button("Add New User")
        generate_report_button = factory.create_button("Generate Report")
        export_data_button = factory.create_button("Export Data")
        
        actions_panel.add_child(new_user_button)
        actions_panel.add_child(generate_report_button)
        actions_panel.add_child(export_data_button)
        
        # Settings panel
        settings_panel = factory.create_panel("Settings")
        maintenance_checkbox = factory.create_checkbox("Maintenance Mode")
        debug_checkbox = factory.create_checkbox("Debug Mode")
        
        settings_panel.add_child(maintenance_checkbox)
        settings_panel.add_child(debug_checkbox)
        
        # Assemble dashboard
        dashboard.add_child(stats_panel)
        dashboard.add_child(actions_panel)
        dashboard.add_child(settings_panel)
        
        return dashboard


def demonstrate_abstract_factory_pattern():
    """
    Demonstrate Abstract Factory pattern implementations.
    """
    print("=== ABSTRACT FACTORY PATTERN DEMONSTRATION ===\n")
    
    # 1. Factory Provider Setup
    print("1. FACTORY PROVIDER SETUP:")
    
    provider = UIFactoryProvider()
    available_themes = provider.get_available_themes()
    
    print(f"   Available themes: {[theme.value for theme in available_themes]}")
    print(f"   Current theme: {provider.get_current_theme().value}")
    print()
    
    # 2. Creating Components with Different Themes
    print("2. CREATING COMPONENTS WITH DIFFERENT THEMES:")
    
    themes_to_test = [UITheme.LIGHT, UITheme.DARK, UITheme.HIGH_CONTRAST]
    
    for theme in themes_to_test:
        print(f"\n   {theme.value.upper()} THEME:")
        factory = provider.get_factory(theme)
        
        if factory:
            # Create components
            button = factory.create_button("Test Button", ComponentSize.MEDIUM)
            text_field = factory.create_text_field("Enter text here", ComponentSize.MEDIUM)
            checkbox = factory.create_checkbox("Test Option", ComponentSize.MEDIUM)
            panel = factory.create_panel("Test Panel")
            
            # Show theme properties
            theme_props = factory.get_theme_properties()
            print(f"     Theme: {factory.get_theme_name()}")
            print(f"     Primary Color: {theme_props.get('primary_color')}")
            print(f"     Background: {theme_props.get('background_color')}")
            print(f"     Text Color: {theme_props.get('text_color')}")
            
            # Show component rendering (first 60 chars)
            print(f"     Button HTML: {button.render()[:60]}...")
            print(f"     TextField HTML: {text_field.render()[:60]}...")
            print(f"     CheckBox HTML: {checkbox.render()[:60]}...")
    
    print()
    
    # 3. Component Interaction
    print("3. COMPONENT INTERACTION:")
    
    provider.set_current_theme(UITheme.LIGHT)
    factory = provider.get_current_factory()
    
    # Create interactive components
    interactive_button = factory.create_button("Click Me!")
    interactive_text = factory.create_text_field("Type here...")
    interactive_checkbox = factory.create_checkbox("Enable feature")
    
    # Add event handlers
    def button_clicked():
        print("     Button was clicked!")
    
    def text_changed(old_value, new_value):
        print(f"     Text changed from '{old_value}' to '{new_value}'")
    
    def checkbox_changed(old_state, new_state):
        print(f"     Checkbox changed from {old_state} to {new_state}")
    
    interactive_button.add_click_handler(button_clicked)
    interactive_text.add_change_handler(text_changed)
    interactive_checkbox.add_change_handler(checkbox_changed)
    
    # Simulate interactions
    print("   Simulating component interactions:")
    interactive_button.click()
    interactive_text.set_value("Hello World!")
    interactive_checkbox.toggle()
    interactive_checkbox.toggle()
    print()
    
    # 4. Form Creation
    print("4. FORM CREATION WITH ABSTRACT FACTORY:")
    
    form_config = [
        {'type': 'text', 'placeholder': 'First Name', 'size': 'medium'},
        {'type': 'text', 'placeholder': 'Last Name', 'size': 'medium'},
        {'type': 'text', 'placeholder': 'Email Address', 'size': 'large'},
        {'type': 'checkbox', 'label': 'Subscribe to newsletter', 'size': 'medium'},
        {'type': 'button', 'text': 'Submit', 'size': 'large'},
        {'type': 'button', 'text': 'Cancel', 'size': 'medium'}
    ]
    
    for theme in [UITheme.LIGHT, UITheme.DARK]:
        factory = provider.get_factory(theme)
        form = factory.create_form("Registration Form", form_config)
        
        print(f"   {theme.value.title()} theme form created:")
        print(f"     Title: {form.title}")
        print(f"     Components: {len(form.get_children())}")
        
        # Show form rendering preview
        form_html = form.render()
        print(f"     HTML preview: {form_html[:100]}...")
        print()
    
    # 5. Application Builder
    print("5. APPLICATION BUILDER:")
    
    app_builder = ApplicationBuilder(provider)
    
    # Create different application forms with different themes
    applications = {}
    
    for theme in themes_to_test:
        print(f"\n   Building application with {theme.value} theme:")
        
        try:
            login_form = app_builder.create_login_form(theme)
            settings_form = app_builder.create_settings_form(theme)
            dashboard = app_builder.create_dashboard(theme)
            
            applications[theme] = {
                'login': login_form,
                'settings': settings_form,
                'dashboard': dashboard
            }
            
            print(f"     ✓ Login form: {len(login_form.get_children())} components")
            print(f"     ✓ Settings form: {len(settings_form.get_children())} components")
            print(f"     ✓ Dashboard: {len(dashboard.get_children())} components")
            
        except ValueError as e:
            print(f"     ✗ Failed to create application: {e}")
    
    print()
    
    # 6. Theme Switching
    print("6. DYNAMIC THEME SWITCHING:")
    
    # Create a component and switch themes
    original_theme = provider.get_current_theme()
    
    for theme in themes_to_test:
        provider.set_current_theme(theme)
        factory = provider.get_current_factory()
        
        button = factory.create_button("Dynamic Button")
        style = button.get_default_style()
        
        print(f"   {theme.value} theme button:")
        print(f"     Background: {style.get('background_color')}")
        print(f"     Text Color: {style.get('text_color')}")
        print(f"     Border: {style.get('border_color')}")
    
    # Restore original theme
    provider.set_current_theme(original_theme)
    print()
    
    # 7. Batch Component Creation
    print("7. BATCH COMPONENT CREATION:")
    
    batch_config = [
        {'type': 'panel', 'title': 'Main Panel'},
        {'type': 'button', 'text': 'Action 1', 'size': 'large'},
        {'type': 'button', 'text': 'Action 2', 'size': 'medium'},
        {'type': 'text_field', 'placeholder': 'Search...', 'size': 'large'},
        {'type': 'checkbox', 'label': 'Option 1', 'size': 'small'},
        {'type': 'checkbox', 'label': 'Option 2', 'size': 'small'}
    ]
    
    for theme in [UITheme.LIGHT, UITheme.DARK]:
        components = provider.create_themed_components(theme, batch_config)
        print(f"   {theme.value} theme batch creation:")
        print(f"     Created {len(components)} components")
        
        for i, component in enumerate(components):
            component_type = component.__class__.__name__
            print(f"     {i+1}. {component_type}")
    
    print()
    
    # 8. Abstract Factory Benefits
    print("8. ABSTRACT FACTORY PATTERN BENEFITS:")
    print("   ✓ Consistency: All components in a family have consistent styling")
    print("   ✓ Flexibility: Easy to switch between entire product families")
    print("   ✓ Extensibility: New themes can be added without changing client code")
    print("   ✓ Isolation: Theme-specific code is isolated in concrete factories")
    print("   ✓ Polymorphism: Client code works with abstract interfaces")
    print("   ✓ Maintainability: Changes to a theme only affect one factory")
    print("   ✓ Testability: Each factory can be tested independently")
    print("   ✓ Scalability: Support for multiple product families and themes")
    print()
    
    print("=== ABSTRACT FACTORY PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_abstract_factory_pattern()
