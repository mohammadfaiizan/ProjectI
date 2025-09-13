"""
PROTOTYPE PATTERN - Creational Design Pattern
=============================================

Problem Statement:
Implement the Prototype pattern to create objects by cloning existing instances:
- Deep and shallow cloning mechanisms
- Prototype registry for managing prototypes
- Cloning complex object hierarchies
- Performance optimization through cloning
- Prototype-based object creation

Learning Objectives:
- Understand when to use Prototype pattern
- Implement proper cloning mechanisms
- Handle deep vs shallow copying
- Design prototype registries
- Optimize object creation performance
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import copy
import json
import time
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================

class CloneType(Enum):
    SHALLOW = "shallow"
    DEEP = "deep"
    CUSTOM = "custom"


class DocumentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    PDF = "pdf"


# ============================================================================
# ABSTRACT PROTOTYPE
# ============================================================================

class Prototype(ABC):
    """Abstract prototype interface."""
    
    @abstractmethod
    def clone(self) -> 'Prototype':
        """Create a clone of this object."""
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """Get the type identifier of this prototype."""
        pass


class DeepCloneable(ABC):
    """Interface for objects that support deep cloning."""
    
    @abstractmethod
    def deep_clone(self) -> 'DeepCloneable':
        """Create a deep clone of this object."""
        pass


class CustomCloneable(ABC):
    """Interface for objects with custom cloning logic."""
    
    @abstractmethod
    def custom_clone(self, **kwargs) -> 'CustomCloneable':
        """Create a custom clone with specific parameters."""
        pass


# ============================================================================
# DOCUMENT SYSTEM - CONCRETE PROTOTYPES
# ============================================================================

class DocumentMetadata:
    """Document metadata that can be cloned."""
    
    def __init__(self, title: str = "", author: str = "", created_date: datetime = None):
        self.title = title
        self.author = author
        self.created_date = created_date or datetime.now()
        self.modified_date = datetime.now()
        self.version = "1.0"
        self.tags: List[str] = []
        self.properties: Dict[str, Any] = {}
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to metadata."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_property(self, key: str, value: Any) -> None:
        """Set a custom property."""
        self.properties[key] = value
    
    def clone(self) -> 'DocumentMetadata':
        """Clone metadata with new timestamps."""
        cloned = DocumentMetadata(self.title, self.author)
        cloned.version = self.version
        cloned.tags = self.tags.copy()  # Shallow copy of tags
        cloned.properties = copy.deepcopy(self.properties)  # Deep copy of properties
        return cloned
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'author': self.author,
            'created_date': self.created_date.isoformat(),
            'modified_date': self.modified_date.isoformat(),
            'version': self.version,
            'tags': self.tags,
            'properties': self.properties
        }


class BaseDocument(Prototype, DeepCloneable, CustomCloneable):
    """Base document class implementing all cloning interfaces."""
    
    def __init__(self, doc_type: DocumentType, content: str = ""):
        self.doc_type = doc_type
        self.content = content
        self.metadata = DocumentMetadata()
        self.formatting: Dict[str, Any] = {}
        self.attachments: List[Dict[str, Any]] = []
        self.creation_time = time.time()
        self.clone_count = 0
    
    def set_metadata(self, title: str, author: str) -> None:
        """Set document metadata."""
        self.metadata.title = title
        self.metadata.author = author
        self.metadata.modified_date = datetime.now()
    
    def add_formatting(self, key: str, value: Any) -> None:
        """Add formatting option."""
        self.formatting[key] = value
        self.metadata.modified_date = datetime.now()
    
    def add_attachment(self, name: str, data: Any) -> None:
        """Add attachment to document."""
        attachment = {
            'name': name,
            'data': data,
            'added_date': datetime.now().isoformat()
        }
        self.attachments.append(attachment)
        self.metadata.modified_date = datetime.now()
    
    def get_type(self) -> str:
        """Get document type."""
        return self.doc_type.value
    
    def clone(self) -> 'BaseDocument':
        """Default clone implementation (shallow)."""
        return self.shallow_clone()
    
    def shallow_clone(self) -> 'BaseDocument':
        """Create a shallow clone."""
        cloned = copy.copy(self)
        cloned.clone_count = self.clone_count + 1
        cloned.creation_time = time.time()
        
        # Clone metadata but keep references to other objects
        cloned.metadata = self.metadata.clone()
        cloned.metadata.title = f"{self.metadata.title} (Copy {cloned.clone_count})"
        
        print(f"Shallow cloned {self.doc_type.value} document: {cloned.metadata.title}")
        return cloned
    
    def deep_clone(self) -> 'BaseDocument':
        """Create a deep clone."""
        cloned = copy.deepcopy(self)
        cloned.clone_count = self.clone_count + 1
        cloned.creation_time = time.time()
        cloned.metadata.title = f"{self.metadata.title} (Deep Copy {cloned.clone_count})"
        cloned.metadata.modified_date = datetime.now()
        
        print(f"Deep cloned {self.doc_type.value} document: {cloned.metadata.title}")
        return cloned
    
    def custom_clone(self, **kwargs) -> 'BaseDocument':
        """Create a custom clone with specific parameters."""
        # Start with deep clone
        cloned = self.deep_clone()
        
        # Apply custom parameters
        if 'title' in kwargs:
            cloned.metadata.title = kwargs['title']
        
        if 'author' in kwargs:
            cloned.metadata.author = kwargs['author']
        
        if 'content' in kwargs:
            cloned.content = kwargs['content']
        
        if 'clear_attachments' in kwargs and kwargs['clear_attachments']:
            cloned.attachments.clear()
        
        if 'reset_formatting' in kwargs and kwargs['reset_formatting']:
            cloned.formatting.clear()
        
        if 'add_tags' in kwargs:
            for tag in kwargs['add_tags']:
                cloned.metadata.add_tag(tag)
        
        print(f"Custom cloned {self.doc_type.value} document: {cloned.metadata.title}")
        return cloned
    
    def get_info(self) -> Dict[str, Any]:
        """Get document information."""
        return {
            'type': self.doc_type.value,
            'content_length': len(self.content),
            'metadata': self.metadata.to_dict(),
            'formatting_options': len(self.formatting),
            'attachments_count': len(self.attachments),
            'clone_count': self.clone_count,
            'creation_time': self.creation_time
        }


class TextDocument(BaseDocument):
    """Text document with specific text-related features."""
    
    def __init__(self, content: str = ""):
        super().__init__(DocumentType.TEXT, content)
        self.font_family = "Arial"
        self.font_size = 12
        self.line_spacing = 1.0
        self.word_wrap = True
    
    def set_font(self, family: str, size: int) -> None:
        """Set font properties."""
        self.font_family = family
        self.font_size = size
        self.metadata.modified_date = datetime.now()
    
    def get_word_count(self) -> int:
        """Get word count."""
        return len(self.content.split()) if self.content else 0
    
    def custom_clone(self, **kwargs) -> 'TextDocument':
        """Custom clone for text documents."""
        cloned = super().custom_clone(**kwargs)
        
        # Text-specific customizations
        if 'font_family' in kwargs:
            cloned.font_family = kwargs['font_family']
        
        if 'font_size' in kwargs:
            cloned.font_size = kwargs['font_size']
        
        if 'line_spacing' in kwargs:
            cloned.line_spacing = kwargs['line_spacing']
        
        return cloned


class ImageDocument(BaseDocument):
    """Image document with image-specific features."""
    
    def __init__(self, image_data: bytes = b"", image_format: str = "PNG"):
        super().__init__(DocumentType.IMAGE)
        self.image_data = image_data
        self.image_format = image_format
        self.width = 0
        self.height = 0
        self.dpi = 72
        self.color_mode = "RGB"
    
    def set_dimensions(self, width: int, height: int) -> None:
        """Set image dimensions."""
        self.width = width
        self.height = height
        self.metadata.modified_date = datetime.now()
    
    def set_quality_settings(self, dpi: int, color_mode: str) -> None:
        """Set image quality settings."""
        self.dpi = dpi
        self.color_mode = color_mode
        self.metadata.modified_date = datetime.now()
    
    def get_file_size(self) -> int:
        """Get approximate file size."""
        return len(self.image_data)
    
    def custom_clone(self, **kwargs) -> 'ImageDocument':
        """Custom clone for image documents."""
        cloned = super().custom_clone(**kwargs)
        
        # Image-specific customizations
        if 'resize' in kwargs:
            width, height = kwargs['resize']
            cloned.set_dimensions(width, height)
        
        if 'change_format' in kwargs:
            cloned.image_format = kwargs['change_format']
        
        if 'adjust_dpi' in kwargs:
            cloned.dpi = kwargs['adjust_dpi']
        
        if 'convert_color_mode' in kwargs:
            cloned.color_mode = kwargs['convert_color_mode']
        
        return cloned


class SpreadsheetDocument(BaseDocument):
    """Spreadsheet document with worksheet features."""
    
    def __init__(self):
        super().__init__(DocumentType.SPREADSHEET)
        self.worksheets: Dict[str, List[List[Any]]] = {}
        self.formulas: Dict[str, str] = {}
        self.charts: List[Dict[str, Any]] = []
    
    def add_worksheet(self, name: str, data: List[List[Any]]) -> None:
        """Add a worksheet."""
        self.worksheets[name] = data
        self.metadata.modified_date = datetime.now()
    
    def add_formula(self, cell: str, formula: str) -> None:
        """Add a formula to a cell."""
        self.formulas[cell] = formula
        self.metadata.modified_date = datetime.now()
    
    def add_chart(self, chart_type: str, data_range: str) -> None:
        """Add a chart."""
        chart = {
            'type': chart_type,
            'data_range': data_range,
            'created_date': datetime.now().isoformat()
        }
        self.charts.append(chart)
        self.metadata.modified_date = datetime.now()
    
    def get_cell_count(self) -> int:
        """Get total number of cells with data."""
        count = 0
        for worksheet in self.worksheets.values():
            for row in worksheet:
                count += len([cell for cell in row if cell is not None])
        return count
    
    def custom_clone(self, **kwargs) -> 'SpreadsheetDocument':
        """Custom clone for spreadsheet documents."""
        cloned = super().custom_clone(**kwargs)
        
        # Spreadsheet-specific customizations
        if 'worksheets_only' in kwargs:
            worksheet_names = kwargs['worksheets_only']
            cloned.worksheets = {name: data for name, data in cloned.worksheets.items() 
                               if name in worksheet_names}
        
        if 'clear_formulas' in kwargs and kwargs['clear_formulas']:
            cloned.formulas.clear()
        
        if 'clear_charts' in kwargs and kwargs['clear_charts']:
            cloned.charts.clear()
        
        return cloned


# ============================================================================
# PROTOTYPE REGISTRY
# ============================================================================

class PrototypeRegistry:
    """Registry for managing prototype instances."""
    
    def __init__(self):
        self._prototypes: Dict[str, Prototype] = {}
        self._clone_statistics: Dict[str, int] = {}
    
    def register_prototype(self, name: str, prototype: Prototype) -> None:
        """Register a prototype with a name."""
        self._prototypes[name] = prototype
        self._clone_statistics[name] = 0
        print(f"Registered prototype '{name}' of type {prototype.get_type()}")
    
    def unregister_prototype(self, name: str) -> bool:
        """Unregister a prototype."""
        if name in self._prototypes:
            del self._prototypes[name]
            del self._clone_statistics[name]
            print(f"Unregistered prototype '{name}'")
            return True
        return False
    
    def get_prototype(self, name: str) -> Optional[Prototype]:
        """Get a prototype by name."""
        return self._prototypes.get(name)
    
    def clone_prototype(self, name: str, clone_type: CloneType = CloneType.SHALLOW, 
                       **kwargs) -> Optional[Prototype]:
        """Clone a prototype by name."""
        prototype = self._prototypes.get(name)
        if not prototype:
            print(f"Prototype '{name}' not found")
            return None
        
        # Update statistics
        self._clone_statistics[name] += 1
        
        # Perform cloning based on type
        if clone_type == CloneType.SHALLOW:
            return prototype.clone()
        elif clone_type == CloneType.DEEP and isinstance(prototype, DeepCloneable):
            return prototype.deep_clone()
        elif clone_type == CloneType.CUSTOM and isinstance(prototype, CustomCloneable):
            return prototype.custom_clone(**kwargs)
        else:
            print(f"Clone type {clone_type.value} not supported for prototype '{name}'")
            return prototype.clone()  # Fallback to default clone
    
    def list_prototypes(self) -> List[str]:
        """List all registered prototype names."""
        return list(self._prototypes.keys())
    
    def get_prototype_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a prototype."""
        prototype = self._prototypes.get(name)
        if not prototype:
            return None
        
        return {
            'name': name,
            'type': prototype.get_type(),
            'class': prototype.__class__.__name__,
            'clone_count': self._clone_statistics[name],
            'supports_deep_clone': isinstance(prototype, DeepCloneable),
            'supports_custom_clone': isinstance(prototype, CustomCloneable)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cloning statistics."""
        total_clones = sum(self._clone_statistics.values())
        
        return {
            'total_prototypes': len(self._prototypes),
            'total_clones_created': total_clones,
            'clone_statistics': self._clone_statistics.copy(),
            'most_cloned': max(self._clone_statistics.items(), key=lambda x: x[1]) if self._clone_statistics else None
        }
    
    def clear_statistics(self) -> None:
        """Clear cloning statistics."""
        for name in self._clone_statistics:
            self._clone_statistics[name] = 0
        print("Cloning statistics cleared")


# ============================================================================
# PROTOTYPE FACTORY
# ============================================================================

class PrototypeFactory:
    """Factory that uses prototypes to create objects."""
    
    def __init__(self, registry: PrototypeRegistry):
        self.registry = registry
        self.creation_cache: Dict[str, List[Prototype]] = {}
    
    def create_document(self, prototype_name: str, clone_type: CloneType = CloneType.SHALLOW,
                       **customizations) -> Optional[BaseDocument]:
        """Create a document using a prototype."""
        cloned = self.registry.clone_prototype(prototype_name, clone_type, **customizations)
        
        if cloned and isinstance(cloned, BaseDocument):
            # Cache the created document
            if prototype_name not in self.creation_cache:
                self.creation_cache[prototype_name] = []
            self.creation_cache[prototype_name].append(cloned)
            
            return cloned
        
        return None
    
    def create_document_batch(self, prototype_name: str, count: int, 
                            clone_type: CloneType = CloneType.SHALLOW,
                            **customizations) -> List[BaseDocument]:
        """Create multiple documents from a prototype."""
        documents = []
        
        for i in range(count):
            # Add index to customizations for unique titles
            batch_customizations = customizations.copy()
            if 'title' in batch_customizations:
                batch_customizations['title'] = f"{batch_customizations['title']} #{i+1}"
            
            doc = self.create_document(prototype_name, clone_type, **batch_customizations)
            if doc:
                documents.append(doc)
        
        print(f"Created batch of {len(documents)} documents from prototype '{prototype_name}'")
        return documents
    
    def get_creation_history(self, prototype_name: str) -> List[Dict[str, Any]]:
        """Get creation history for a prototype."""
        if prototype_name not in self.creation_cache:
            return []
        
        history = []
        for doc in self.creation_cache[prototype_name]:
            history.append({
                'title': doc.metadata.title,
                'created_time': doc.creation_time,
                'clone_count': doc.clone_count,
                'type': doc.get_type()
            })
        
        return history
    
    def clear_cache(self) -> None:
        """Clear creation cache."""
        self.creation_cache.clear()
        print("Creation cache cleared")


# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================

class PerformanceAnalyzer:
    """Analyzer for comparing prototype vs direct instantiation performance."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}
    
    def measure_direct_creation(self, doc_type: DocumentType, count: int) -> float:
        """Measure time for direct object creation."""
        start_time = time.time()
        
        for i in range(count):
            if doc_type == DocumentType.TEXT:
                doc = TextDocument(f"Sample content {i}")
                doc.set_metadata(f"Document {i}", "Test Author")
                doc.set_font("Arial", 12)
            elif doc_type == DocumentType.IMAGE:
                doc = ImageDocument(b"fake_image_data", "PNG")
                doc.set_metadata(f"Image {i}", "Test Author")
                doc.set_dimensions(800, 600)
            elif doc_type == DocumentType.SPREADSHEET:
                doc = SpreadsheetDocument()
                doc.set_metadata(f"Spreadsheet {i}", "Test Author")
                doc.add_worksheet("Sheet1", [[1, 2, 3], [4, 5, 6]])
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Direct creation of {count} {doc_type.value} documents: {duration:.4f}s")
        return duration
    
    def measure_prototype_creation(self, registry: PrototypeRegistry, 
                                 prototype_name: str, count: int,
                                 clone_type: CloneType = CloneType.SHALLOW) -> float:
        """Measure time for prototype-based creation."""
        start_time = time.time()
        
        for i in range(count):
            cloned = registry.clone_prototype(
                prototype_name, 
                clone_type,
                title=f"Cloned Document {i}"
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Prototype creation of {count} documents from '{prototype_name}': {duration:.4f}s")
        return duration
    
    def compare_performance(self, doc_type: DocumentType, prototype_name: str,
                          registry: PrototypeRegistry, count: int = 1000) -> Dict[str, Any]:
        """Compare direct vs prototype creation performance."""
        print(f"\nPerformance comparison for {count} {doc_type.value} documents:")
        
        # Measure direct creation
        direct_time = self.measure_direct_creation(doc_type, count)
        
        # Measure shallow clone
        shallow_time = self.measure_prototype_creation(registry, prototype_name, count, CloneType.SHALLOW)
        
        # Measure deep clone
        deep_time = self.measure_prototype_creation(registry, prototype_name, count, CloneType.DEEP)
        
        # Calculate improvements
        shallow_improvement = ((direct_time - shallow_time) / direct_time) * 100 if direct_time > 0 else 0
        deep_improvement = ((direct_time - deep_time) / direct_time) * 100 if direct_time > 0 else 0
        
        results = {
            'document_type': doc_type.value,
            'count': count,
            'direct_creation_time': direct_time,
            'shallow_clone_time': shallow_time,
            'deep_clone_time': deep_time,
            'shallow_improvement_percent': shallow_improvement,
            'deep_improvement_percent': deep_improvement,
            'fastest_method': min([
                ('direct', direct_time),
                ('shallow', shallow_time),
                ('deep', deep_time)
            ], key=lambda x: x[1])[0]
        }
        
        self.results[doc_type.value] = results
        return results


def demonstrate_prototype_pattern():
    """
    Demonstrate Prototype pattern implementations.
    """
    print("=== PROTOTYPE PATTERN DEMONSTRATION ===\n")
    
    # 1. Basic Prototype Usage
    print("1. BASIC PROTOTYPE USAGE:")
    
    # Create original documents
    text_doc = TextDocument("This is a sample text document with some content.")
    text_doc.set_metadata("Original Text Document", "John Doe")
    text_doc.set_font("Times New Roman", 14)
    text_doc.add_formatting("bold", True)
    text_doc.add_attachment("notes.txt", "Additional notes")
    
    image_doc = ImageDocument(b"fake_image_data_bytes", "JPEG")
    image_doc.set_metadata("Original Image", "Jane Smith")
    image_doc.set_dimensions(1920, 1080)
    image_doc.set_quality_settings(300, "CMYK")
    
    spreadsheet_doc = SpreadsheetDocument()
    spreadsheet_doc.set_metadata("Original Spreadsheet", "Bob Johnson")
    spreadsheet_doc.add_worksheet("Data", [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    spreadsheet_doc.add_formula("D1", "=SUM(A1:C1)")
    spreadsheet_doc.add_chart("bar", "A1:C3")
    
    print(f"   Created original text document: {text_doc.metadata.title}")
    print(f"   Word count: {text_doc.get_word_count()}")
    print(f"   Created original image document: {image_doc.metadata.title}")
    print(f"   Image size: {image_doc.width}x{image_doc.height}")
    print(f"   Created original spreadsheet: {spreadsheet_doc.metadata.title}")
    print(f"   Cell count: {spreadsheet_doc.get_cell_count()}")
    print()
    
    # 2. Different Types of Cloning
    print("2. DIFFERENT TYPES OF CLONING:")
    
    # Shallow cloning
    print("   SHALLOW CLONING:")
    text_shallow = text_doc.shallow_clone()
    image_shallow = image_doc.shallow_clone()
    
    print(f"     Text clone: {text_shallow.metadata.title}")
    print(f"     Original attachments count: {len(text_doc.attachments)}")
    print(f"     Clone attachments count: {len(text_shallow.attachments)}")
    print(f"     Same attachments reference: {text_doc.attachments is text_shallow.attachments}")
    
    # Deep cloning
    print("\n   DEEP CLONING:")
    text_deep = text_doc.deep_clone()
    spreadsheet_deep = spreadsheet_doc.deep_clone()
    
    print(f"     Text deep clone: {text_deep.metadata.title}")
    print(f"     Same attachments reference: {text_doc.attachments is text_deep.attachments}")
    print(f"     Spreadsheet deep clone: {spreadsheet_deep.metadata.title}")
    print(f"     Same worksheets reference: {spreadsheet_doc.worksheets is spreadsheet_deep.worksheets}")
    
    # Custom cloning
    print("\n   CUSTOM CLONING:")
    text_custom = text_doc.custom_clone(
        title="Custom Text Document",
        author="Custom Author",
        font_family="Helvetica",
        font_size=16,
        clear_attachments=True,
        add_tags=["custom", "demo"]
    )
    
    image_custom = image_doc.custom_clone(
        title="Resized Image",
        resize=(800, 600),
        change_format="PNG",
        adjust_dpi=150
    )
    
    print(f"     Custom text clone: {text_custom.metadata.title}")
    print(f"     Font: {text_custom.font_family}, Size: {text_custom.font_size}")
    print(f"     Tags: {text_custom.metadata.tags}")
    print(f"     Attachments cleared: {len(text_custom.attachments) == 0}")
    
    print(f"     Custom image clone: {image_custom.metadata.title}")
    print(f"     New dimensions: {image_custom.width}x{image_custom.height}")
    print(f"     New format: {image_custom.image_format}")
    print(f"     New DPI: {image_custom.dpi}")
    print()
    
    # 3. Prototype Registry
    print("3. PROTOTYPE REGISTRY:")
    
    registry = PrototypeRegistry()
    
    # Register prototypes
    registry.register_prototype("standard_text", text_doc)
    registry.register_prototype("standard_image", image_doc)
    registry.register_prototype("standard_spreadsheet", spreadsheet_doc)
    
    # Create template documents
    template_text = TextDocument("Template content for business letters.")
    template_text.set_metadata("Business Letter Template", "Template Author")
    template_text.set_font("Arial", 11)
    template_text.add_formatting("letterhead", True)
    
    registry.register_prototype("business_letter", template_text)
    
    print(f"   Registered prototypes: {registry.list_prototypes()}")
    
    # Get prototype information
    for name in registry.list_prototypes():
        info = registry.get_prototype_info(name)
        print(f"     {name}: {info['type']} ({info['class']})")
        print(f"       Deep clone: {info['supports_deep_clone']}")
        print(f"       Custom clone: {info['supports_custom_clone']}")
    
    print()
    
    # 4. Cloning from Registry
    print("4. CLONING FROM REGISTRY:")
    
    # Clone different types
    cloned_text = registry.clone_prototype("standard_text", CloneType.SHALLOW)
    cloned_image = registry.clone_prototype("standard_image", CloneType.DEEP)
    cloned_letter = registry.clone_prototype(
        "business_letter", 
        CloneType.CUSTOM,
        title="Customer Response Letter",
        author="Customer Service",
        content="Dear Customer, Thank you for your inquiry..."
    )
    
    print(f"   Cloned text: {cloned_text.metadata.title if cloned_text else 'Failed'}")
    print(f"   Cloned image: {cloned_image.metadata.title if cloned_image else 'Failed'}")
    print(f"   Cloned letter: {cloned_letter.metadata.title if cloned_letter else 'Failed'}")
    
    # Show statistics
    stats = registry.get_statistics()
    print(f"\n   Registry statistics:")
    print(f"     Total prototypes: {stats['total_prototypes']}")
    print(f"     Total clones created: {stats['total_clones_created']}")
    print(f"     Most cloned: {stats['most_cloned'][0] if stats['most_cloned'] else 'None'}")
    print()
    
    # 5. Prototype Factory
    print("5. PROTOTYPE FACTORY:")
    
    factory = PrototypeFactory(registry)
    
    # Create single documents
    doc1 = factory.create_document("business_letter", CloneType.CUSTOM,
                                  title="Welcome Letter",
                                  author="HR Department")
    
    doc2 = factory.create_document("standard_spreadsheet", CloneType.DEEP,
                                  title="Monthly Report")
    
    print(f"   Factory created: {doc1.metadata.title if doc1 else 'Failed'}")
    print(f"   Factory created: {doc2.metadata.title if doc2 else 'Failed'}")
    
    # Create batch documents
    batch_docs = factory.create_document_batch(
        "business_letter", 
        5, 
        CloneType.CUSTOM,
        title="Batch Letter",
        author="Marketing Team"
    )
    
    print(f"   Batch created: {len(batch_docs)} documents")
    for doc in batch_docs[:3]:  # Show first 3
        print(f"     - {doc.metadata.title}")
    
    # Show creation history
    history = factory.get_creation_history("business_letter")
    print(f"\n   Creation history for 'business_letter': {len(history)} documents")
    print()
    
    # 6. Performance Analysis
    print("6. PERFORMANCE ANALYSIS:")
    
    analyzer = PerformanceAnalyzer()
    
    # Compare performance for different document types
    test_count = 100  # Reduced for demo
    
    text_results = analyzer.compare_performance(
        DocumentType.TEXT, 
        "standard_text", 
        registry, 
        test_count
    )
    
    print(f"\n   Text Document Results:")
    print(f"     Direct creation: {text_results['direct_creation_time']:.4f}s")
    print(f"     Shallow clone: {text_results['shallow_clone_time']:.4f}s")
    print(f"     Deep clone: {text_results['deep_clone_time']:.4f}s")
    print(f"     Shallow improvement: {text_results['shallow_improvement_percent']:.1f}%")
    print(f"     Fastest method: {text_results['fastest_method']}")
    
    image_results = analyzer.compare_performance(
        DocumentType.IMAGE,
        "standard_image",
        registry,
        test_count
    )
    
    print(f"\n   Image Document Results:")
    print(f"     Direct creation: {image_results['direct_creation_time']:.4f}s")
    print(f"     Shallow clone: {image_results['shallow_clone_time']:.4f}s")
    print(f"     Deep clone: {image_results['deep_clone_time']:.4f}s")
    print(f"     Shallow improvement: {image_results['shallow_improvement_percent']:.1f}%")
    print(f"     Fastest method: {image_results['fastest_method']}")
    print()
    
    # 7. Complex Cloning Scenarios
    print("7. COMPLEX CLONING SCENARIOS:")
    
    # Create a complex document with nested structures
    complex_doc = SpreadsheetDocument()
    complex_doc.set_metadata("Complex Financial Model", "Finance Team")
    
    # Add multiple worksheets
    complex_doc.add_worksheet("Income", [
        ["Revenue", 100000, 110000, 120000],
        ["Expenses", 80000, 85000, 90000],
        ["Profit", 20000, 25000, 30000]
    ])
    
    complex_doc.add_worksheet("Balance", [
        ["Assets", 500000, 550000, 600000],
        ["Liabilities", 300000, 320000, 340000],
        ["Equity", 200000, 230000, 260000]
    ])
    
    # Add formulas and charts
    complex_doc.add_formula("D2", "=B2-B3")
    complex_doc.add_formula("D3", "=C2-C3")
    complex_doc.add_chart("line", "B1:D3")
    complex_doc.add_chart("pie", "B4:D6")
    
    # Add to registry
    registry.register_prototype("financial_model", complex_doc)
    
    # Create customized versions
    q1_model = registry.clone_prototype(
        "financial_model",
        CloneType.CUSTOM,
        title="Q1 Financial Model",
        worksheets_only=["Income"],  # Only include Income worksheet
        clear_charts=True  # Remove charts
    )
    
    summary_model = registry.clone_prototype(
        "financial_model",
        CloneType.CUSTOM,
        title="Executive Summary Model",
        clear_formulas=True,  # Remove formulas
        clear_charts=True     # Remove charts
    )
    
    print(f"   Original model worksheets: {len(complex_doc.worksheets)}")
    print(f"   Original model formulas: {len(complex_doc.formulas)}")
    print(f"   Original model charts: {len(complex_doc.charts)}")
    
    if q1_model:
        print(f"   Q1 model worksheets: {len(q1_model.worksheets)}")
        print(f"   Q1 model charts: {len(q1_model.charts)}")
    
    if summary_model:
        print(f"   Summary model formulas: {len(summary_model.formulas)}")
        print(f"   Summary model charts: {len(summary_model.charts)}")
    
    print()
    
    # 8. Final Statistics
    print("8. FINAL STATISTICS:")
    
    final_stats = registry.get_statistics()
    print(f"   Total prototypes registered: {final_stats['total_prototypes']}")
    print(f"   Total clones created: {final_stats['total_clones_created']}")
    
    print("\n   Clone statistics by prototype:")
    for name, count in final_stats['clone_statistics'].items():
        print(f"     {name}: {count} clones")
    
    print()
    
    # 9. Prototype Pattern Benefits
    print("9. PROTOTYPE PATTERN BENEFITS:")
    print("   ✓ Performance: Faster object creation through cloning")
    print("   ✓ Flexibility: Different cloning strategies (shallow, deep, custom)")
    print("   ✓ Registry: Centralized prototype management")
    print("   ✓ Customization: Custom cloning with specific parameters")
    print("   ✓ Memory Efficiency: Shared references in shallow cloning")
    print("   ✓ Complex Objects: Easy duplication of complex object hierarchies")
    print("   ✓ Template System: Prototype-based template creation")
    print("   ✓ Runtime Configuration: Dynamic prototype registration")
    print()
    
    print("=== PROTOTYPE PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_prototype_pattern()
