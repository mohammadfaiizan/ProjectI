"""
FACTORY METHOD PATTERN - Creational Design Pattern
==================================================

Problem Statement:
Implement the Factory Method pattern to create objects without specifying
their exact classes:
- Abstract factory methods for object creation
- Concrete factories for specific product families
- Product hierarchies and polymorphism
- Factory registration and discovery
- Parameterized factories

Learning Objectives:
- Understand when to use Factory Method pattern
- Implement flexible object creation mechanisms
- Design extensible factory hierarchies
- Handle complex object initialization
- Integrate factories with dependency injection
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Callable
from enum import Enum
import json
import datetime


# ============================================================================
# PRODUCT HIERARCHY - DOCUMENT PROCESSING SYSTEM
# ============================================================================

class DocumentType(Enum):
    PDF = "pdf"
    WORD = "word"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    TEXT = "text"
    HTML = "html"


class Document(ABC):
    """Abstract base class for all documents."""
    
    def __init__(self, title: str, content: str = ""):
        self.title = title
        self.content = content
        self.created_at = datetime.datetime.now()
        self.metadata = {}
    
    @abstractmethod
    def save(self, filepath: str) -> bool:
        """Save document to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> bool:
        """Load document from file."""
        pass
    
    @abstractmethod
    def export(self, format_type: str) -> str:
        """Export document to different format."""
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Get file extension for this document type."""
        pass
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to document."""
        self.metadata[key] = value
    
    def get_info(self) -> Dict[str, Any]:
        """Get document information."""
        return {
            'title': self.title,
            'type': self.__class__.__name__,
            'created_at': self.created_at.isoformat(),
            'content_length': len(self.content),
            'metadata': self.metadata
        }


class PDFDocument(Document):
    """Concrete PDF document implementation."""
    
    def __init__(self, title: str, content: str = ""):
        super().__init__(title, content)
        self.pages = []
        self.bookmarks = []
        print(f"PDFDocument created: {title}")
    
    def save(self, filepath: str) -> bool:
        """Save PDF document."""
        print(f"Saving PDF document to {filepath}")
        # Simulate PDF saving logic
        return True
    
    def load(self, filepath: str) -> bool:
        """Load PDF document."""
        print(f"Loading PDF document from {filepath}")
        # Simulate PDF loading logic
        return True
    
    def export(self, format_type: str) -> str:
        """Export PDF to different format."""
        if format_type.lower() == "text":
            return f"PDF Text Export: {self.content}"
        elif format_type.lower() == "html":
            return f"<html><body><h1>{self.title}</h1><p>{self.content}</p></body></html>"
        return f"PDF export to {format_type}: {self.content}"
    
    def get_file_extension(self) -> str:
        return ".pdf"
    
    def add_page(self, page_content: str) -> None:
        """Add page to PDF."""
        self.pages.append(page_content)
    
    def add_bookmark(self, title: str, page: int) -> None:
        """Add bookmark to PDF."""
        self.bookmarks.append({'title': title, 'page': page})


class WordDocument(Document):
    """Concrete Word document implementation."""
    
    def __init__(self, title: str, content: str = ""):
        super().__init__(title, content)
        self.styles = {}
        self.headers_footers = {}
        print(f"WordDocument created: {title}")
    
    def save(self, filepath: str) -> bool:
        """Save Word document."""
        print(f"Saving Word document to {filepath}")
        # Simulate Word saving logic
        return True
    
    def load(self, filepath: str) -> bool:
        """Load Word document."""
        print(f"Loading Word document from {filepath}")
        # Simulate Word loading logic
        return True
    
    def export(self, format_type: str) -> str:
        """Export Word to different format."""
        if format_type.lower() == "pdf":
            return f"Word to PDF Export: {self.content}"
        elif format_type.lower() == "html":
            return f"<html><body><h1>{self.title}</h1><p>{self.content}</p></body></html>"
        return f"Word export to {format_type}: {self.content}"
    
    def get_file_extension(self) -> str:
        return ".docx"
    
    def apply_style(self, style_name: str, properties: Dict[str, Any]) -> None:
        """Apply style to document."""
        self.styles[style_name] = properties
    
    def set_header(self, header_text: str) -> None:
        """Set document header."""
        self.headers_footers['header'] = header_text
    
    def set_footer(self, footer_text: str) -> None:
        """Set document footer."""
        self.headers_footers['footer'] = footer_text


class ExcelDocument(Document):
    """Concrete Excel document implementation."""
    
    def __init__(self, title: str, content: str = ""):
        super().__init__(title, content)
        self.worksheets = {}
        self.formulas = []
        print(f"ExcelDocument created: {title}")
    
    def save(self, filepath: str) -> bool:
        """Save Excel document."""
        print(f"Saving Excel document to {filepath}")
        # Simulate Excel saving logic
        return True
    
    def load(self, filepath: str) -> bool:
        """Load Excel document."""
        print(f"Loading Excel document from {filepath}")
        # Simulate Excel loading logic
        return True
    
    def export(self, format_type: str) -> str:
        """Export Excel to different format."""
        if format_type.lower() == "csv":
            return f"Excel to CSV Export: {self.content}"
        elif format_type.lower() == "pdf":
            return f"Excel to PDF Export: {self.content}"
        return f"Excel export to {format_type}: {self.content}"
    
    def get_file_extension(self) -> str:
        return ".xlsx"
    
    def add_worksheet(self, name: str, data: List[List[Any]]) -> None:
        """Add worksheet to Excel."""
        self.worksheets[name] = data
    
    def add_formula(self, cell: str, formula: str) -> None:
        """Add formula to Excel."""
        self.formulas.append({'cell': cell, 'formula': formula})


class TextDocument(Document):
    """Concrete text document implementation."""
    
    def __init__(self, title: str, content: str = ""):
        super().__init__(title, content)
        self.encoding = "utf-8"
        print(f"TextDocument created: {title}")
    
    def save(self, filepath: str) -> bool:
        """Save text document."""
        print(f"Saving text document to {filepath}")
        # Simulate text saving logic
        return True
    
    def load(self, filepath: str) -> bool:
        """Load text document."""
        print(f"Loading text document from {filepath}")
        # Simulate text loading logic
        return True
    
    def export(self, format_type: str) -> str:
        """Export text to different format."""
        if format_type.lower() == "html":
            return f"<html><body><pre>{self.content}</pre></body></html>"
        elif format_type.lower() == "json":
            return json.dumps({'title': self.title, 'content': self.content})
        return f"Text export to {format_type}: {self.content}"
    
    def get_file_extension(self) -> str:
        return ".txt"
    
    def set_encoding(self, encoding: str) -> None:
        """Set text encoding."""
        self.encoding = encoding


# ============================================================================
# ABSTRACT FACTORY METHOD
# ============================================================================

class DocumentFactory(ABC):
    """Abstract factory for creating documents."""
    
    @abstractmethod
    def create_document(self, title: str, content: str = "", **kwargs) -> Document:
        """Factory method to create document."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        pass
    
    def create_document_with_template(self, title: str, template_name: str, **kwargs) -> Document:
        """Create document using template."""
        template_content = self.load_template(template_name)
        return self.create_document(title, template_content, **kwargs)
    
    def load_template(self, template_name: str) -> str:
        """Load template content (simulated)."""
        templates = {
            'business_letter': 'Dear [Recipient],\n\n[Content]\n\nSincerely,\n[Sender]',
            'report': 'REPORT: [Title]\n\nExecutive Summary:\n[Summary]\n\nDetails:\n[Details]',
            'invoice': 'INVOICE #[Number]\n\nBill To: [Customer]\nAmount: [Amount]\nDue Date: [Date]'
        }
        return templates.get(template_name, "")


# ============================================================================
# CONCRETE FACTORIES
# ============================================================================

class PDFDocumentFactory(DocumentFactory):
    """Factory for creating PDF documents."""
    
    def create_document(self, title: str, content: str = "", **kwargs) -> PDFDocument:
        """Create PDF document."""
        pdf = PDFDocument(title, content)
        
        # Handle PDF-specific parameters
        if 'pages' in kwargs:
            for page_content in kwargs['pages']:
                pdf.add_page(page_content)
        
        if 'bookmarks' in kwargs:
            for bookmark in kwargs['bookmarks']:
                pdf.add_bookmark(bookmark['title'], bookmark['page'])
        
        # Add metadata
        pdf.add_metadata('creator', 'PDF Factory')
        pdf.add_metadata('producer', 'Document Management System')
        
        return pdf
    
    def get_supported_formats(self) -> List[str]:
        """Get supported export formats for PDF."""
        return ['text', 'html', 'image']


class WordDocumentFactory(DocumentFactory):
    """Factory for creating Word documents."""
    
    def create_document(self, title: str, content: str = "", **kwargs) -> WordDocument:
        """Create Word document."""
        word = WordDocument(title, content)
        
        # Handle Word-specific parameters
        if 'styles' in kwargs:
            for style_name, properties in kwargs['styles'].items():
                word.apply_style(style_name, properties)
        
        if 'header' in kwargs:
            word.set_header(kwargs['header'])
        
        if 'footer' in kwargs:
            word.set_footer(kwargs['footer'])
        
        # Add metadata
        word.add_metadata('author', 'Word Factory')
        word.add_metadata('application', 'Document Management System')
        
        return word
    
    def get_supported_formats(self) -> List[str]:
        """Get supported export formats for Word."""
        return ['pdf', 'html', 'text', 'rtf']


class ExcelDocumentFactory(DocumentFactory):
    """Factory for creating Excel documents."""
    
    def create_document(self, title: str, content: str = "", **kwargs) -> ExcelDocument:
        """Create Excel document."""
        excel = ExcelDocument(title, content)
        
        # Handle Excel-specific parameters
        if 'worksheets' in kwargs:
            for name, data in kwargs['worksheets'].items():
                excel.add_worksheet(name, data)
        
        if 'formulas' in kwargs:
            for formula in kwargs['formulas']:
                excel.add_formula(formula['cell'], formula['formula'])
        
        # Add metadata
        excel.add_metadata('created_by', 'Excel Factory')
        excel.add_metadata('application', 'Document Management System')
        
        return excel
    
    def get_supported_formats(self) -> List[str]:
        """Get supported export formats for Excel."""
        return ['csv', 'pdf', 'html', 'xml']


class TextDocumentFactory(DocumentFactory):
    """Factory for creating text documents."""
    
    def create_document(self, title: str, content: str = "", **kwargs) -> TextDocument:
        """Create text document."""
        text = TextDocument(title, content)
        
        # Handle text-specific parameters
        if 'encoding' in kwargs:
            text.set_encoding(kwargs['encoding'])
        
        # Add metadata
        text.add_metadata('created_by', 'Text Factory')
        text.add_metadata('encoding', text.encoding)
        
        return text
    
    def get_supported_formats(self) -> List[str]:
        """Get supported export formats for text."""
        return ['html', 'json', 'xml', 'markdown']


# ============================================================================
# FACTORY REGISTRY AND MANAGER
# ============================================================================

class DocumentFactoryRegistry:
    """Registry for managing document factories."""
    
    def __init__(self):
        self._factories: Dict[DocumentType, DocumentFactory] = {}
        self._default_factory: Optional[DocumentFactory] = None
        self._register_default_factories()
    
    def _register_default_factories(self) -> None:
        """Register default document factories."""
        self.register_factory(DocumentType.PDF, PDFDocumentFactory())
        self.register_factory(DocumentType.WORD, WordDocumentFactory())
        self.register_factory(DocumentType.EXCEL, ExcelDocumentFactory())
        self.register_factory(DocumentType.TEXT, TextDocumentFactory())
        
        # Set text as default
        self._default_factory = self._factories[DocumentType.TEXT]
    
    def register_factory(self, doc_type: DocumentType, factory: DocumentFactory) -> None:
        """Register a factory for a document type."""
        self._factories[doc_type] = factory
        print(f"Registered factory for {doc_type.value}: {factory.__class__.__name__}")
    
    def unregister_factory(self, doc_type: DocumentType) -> bool:
        """Unregister a factory."""
        if doc_type in self._factories:
            del self._factories[doc_type]
            print(f"Unregistered factory for {doc_type.value}")
            return True
        return False
    
    def get_factory(self, doc_type: DocumentType) -> Optional[DocumentFactory]:
        """Get factory for document type."""
        return self._factories.get(doc_type)
    
    def create_document(self, doc_type: DocumentType, title: str, 
                       content: str = "", **kwargs) -> Optional[Document]:
        """Create document using registered factory."""
        factory = self.get_factory(doc_type)
        if factory:
            return factory.create_document(title, content, **kwargs)
        
        print(f"No factory registered for {doc_type.value}")
        return None
    
    def create_document_by_extension(self, filename: str, title: str, 
                                   content: str = "", **kwargs) -> Optional[Document]:
        """Create document based on file extension."""
        extension_map = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.WORD,
            '.doc': DocumentType.WORD,
            '.xlsx': DocumentType.EXCEL,
            '.xls': DocumentType.EXCEL,
            '.txt': DocumentType.TEXT
        }
        
        # Extract extension
        ext = '.' + filename.split('.')[-1].lower()
        doc_type = extension_map.get(ext)
        
        if doc_type:
            return self.create_document(doc_type, title, content, **kwargs)
        
        # Use default factory
        if self._default_factory:
            print(f"Using default factory for unknown extension: {ext}")
            return self._default_factory.create_document(title, content, **kwargs)
        
        return None
    
    def get_supported_types(self) -> List[DocumentType]:
        """Get list of supported document types."""
        return list(self._factories.keys())
    
    def get_all_supported_formats(self) -> Dict[DocumentType, List[str]]:
        """Get all supported export formats by document type."""
        formats = {}
        for doc_type, factory in self._factories.items():
            formats[doc_type] = factory.get_supported_formats()
        return formats


# ============================================================================
# PARAMETERIZED FACTORY
# ============================================================================

class ParameterizedDocumentFactory:
    """Factory that creates documents based on parameters."""
    
    def __init__(self, registry: DocumentFactoryRegistry):
        self.registry = registry
        self.creation_strategies = {
            'simple': self._create_simple_document,
            'template': self._create_template_document,
            'batch': self._create_batch_documents,
            'configured': self._create_configured_document
        }
    
    def create_document(self, strategy: str, **params) -> Any:
        """Create document using specified strategy."""
        if strategy not in self.creation_strategies:
            raise ValueError(f"Unknown creation strategy: {strategy}")
        
        return self.creation_strategies[strategy](**params)
    
    def _create_simple_document(self, doc_type: DocumentType, title: str, 
                              content: str = "", **kwargs) -> Optional[Document]:
        """Create simple document."""
        return self.registry.create_document(doc_type, title, content, **kwargs)
    
    def _create_template_document(self, doc_type: DocumentType, title: str,
                                template_name: str, **kwargs) -> Optional[Document]:
        """Create document from template."""
        factory = self.registry.get_factory(doc_type)
        if factory:
            return factory.create_document_with_template(title, template_name, **kwargs)
        return None
    
    def _create_batch_documents(self, documents_config: List[Dict[str, Any]]) -> List[Document]:
        """Create multiple documents from configuration."""
        documents = []
        
        for config in documents_config:
            doc_type = DocumentType(config['type'])
            title = config['title']
            content = config.get('content', '')
            kwargs = config.get('params', {})
            
            doc = self.registry.create_document(doc_type, title, content, **kwargs)
            if doc:
                documents.append(doc)
        
        return documents
    
    def _create_configured_document(self, config_file: str) -> Optional[Document]:
        """Create document from configuration file (simulated)."""
        # Simulate loading configuration
        config = {
            'type': 'pdf',
            'title': 'Configured Document',
            'content': 'This document was created from configuration',
            'params': {
                'pages': ['Page 1 content', 'Page 2 content'],
                'bookmarks': [{'title': 'Chapter 1', 'page': 1}]
            }
        }
        
        doc_type = DocumentType(config['type'])
        return self.registry.create_document(
            doc_type, 
            config['title'], 
            config['content'], 
            **config['params']
        )


# ============================================================================
# ADVANCED FACTORY PATTERNS
# ============================================================================

class DocumentFactoryBuilder:
    """Builder for creating customized document factories."""
    
    def __init__(self):
        self.factory_class = None
        self.default_params = {}
        self.validators = []
        self.post_processors = []
    
    def set_factory_class(self, factory_class: Type[DocumentFactory]) -> 'DocumentFactoryBuilder':
        """Set the factory class to build."""
        self.factory_class = factory_class
        return self
    
    def add_default_param(self, key: str, value: Any) -> 'DocumentFactoryBuilder':
        """Add default parameter."""
        self.default_params[key] = value
        return self
    
    def add_validator(self, validator: Callable[[Document], bool]) -> 'DocumentFactoryBuilder':
        """Add document validator."""
        self.validators.append(validator)
        return self
    
    def add_post_processor(self, processor: Callable[[Document], Document]) -> 'DocumentFactoryBuilder':
        """Add post-processor."""
        self.post_processors.append(processor)
        return self
    
    def build(self) -> 'CustomDocumentFactory':
        """Build the customized factory."""
        if not self.factory_class:
            raise ValueError("Factory class must be set")
        
        return CustomDocumentFactory(
            self.factory_class(),
            self.default_params,
            self.validators,
            self.post_processors
        )


class CustomDocumentFactory(DocumentFactory):
    """Customized document factory with validation and post-processing."""
    
    def __init__(self, base_factory: DocumentFactory, default_params: Dict[str, Any],
                 validators: List[Callable], post_processors: List[Callable]):
        self.base_factory = base_factory
        self.default_params = default_params
        self.validators = validators
        self.post_processors = post_processors
    
    def create_document(self, title: str, content: str = "", **kwargs) -> Document:
        """Create document with validation and post-processing."""
        # Merge default parameters
        merged_params = {**self.default_params, **kwargs}
        
        # Create document using base factory
        document = self.base_factory.create_document(title, content, **merged_params)
        
        # Validate document
        for validator in self.validators:
            if not validator(document):
                raise ValueError(f"Document validation failed: {validator.__name__}")
        
        # Apply post-processors
        for processor in self.post_processors:
            document = processor(document)
        
        return document
    
    def get_supported_formats(self) -> List[str]:
        """Get supported formats from base factory."""
        return self.base_factory.get_supported_formats()


def demonstrate_factory_method_pattern():
    """
    Demonstrate Factory Method pattern implementations.
    """
    print("=== FACTORY METHOD PATTERN DEMONSTRATION ===\n")
    
    # 1. Basic Factory Usage
    print("1. BASIC FACTORY USAGE:")
    
    pdf_factory = PDFDocumentFactory()
    word_factory = WordDocumentFactory()
    excel_factory = ExcelDocumentFactory()
    
    # Create documents using different factories
    pdf_doc = pdf_factory.create_document(
        "Technical Report", 
        "This is a technical report content",
        pages=["Introduction", "Analysis", "Conclusion"],
        bookmarks=[{"title": "Intro", "page": 1}, {"title": "Analysis", "page": 2}]
    )
    
    word_doc = word_factory.create_document(
        "Business Letter",
        "Dear Customer, Thank you for your business.",
        header="Company Letterhead",
        footer="© 2024 Company Name",
        styles={"heading": {"font": "Arial", "size": 14}}
    )
    
    excel_doc = excel_factory.create_document(
        "Sales Report",
        "Q4 Sales Data",
        worksheets={"Sales": [["Product", "Revenue"], ["Laptop", 50000], ["Mouse", 5000]]},
        formulas=[{"cell": "B3", "formula": "=SUM(B2:B3)"}]
    )
    
    print(f"   Created PDF: {pdf_doc.get_info()}")
    print(f"   Created Word: {word_doc.get_info()}")
    print(f"   Created Excel: {excel_doc.get_info()}")
    print()
    
    # 2. Factory Registry
    print("2. FACTORY REGISTRY:")
    
    registry = DocumentFactoryRegistry()
    
    # Create documents through registry
    pdf_from_registry = registry.create_document(
        DocumentType.PDF, 
        "Registry PDF", 
        "Created through registry"
    )
    
    word_from_registry = registry.create_document(
        DocumentType.WORD,
        "Registry Word",
        "Created through registry"
    )
    
    # Create by file extension
    doc_by_ext = registry.create_document_by_extension(
        "report.xlsx",
        "Excel Report",
        "Quarterly data"
    )
    
    print(f"   Supported types: {[t.value for t in registry.get_supported_types()]}")
    print(f"   PDF from registry: {pdf_from_registry.title if pdf_from_registry else 'None'}")
    print(f"   Word from registry: {word_from_registry.title if word_from_registry else 'None'}")
    print(f"   Doc by extension: {doc_by_ext.title if doc_by_ext else 'None'}")
    print()
    
    # 3. Supported Formats
    print("3. SUPPORTED EXPORT FORMATS:")
    
    all_formats = registry.get_all_supported_formats()
    for doc_type, formats in all_formats.items():
        print(f"   {doc_type.value}: {formats}")
    print()
    
    # 4. Template-based Creation
    print("4. TEMPLATE-BASED DOCUMENT CREATION:")
    
    business_letter = word_factory.create_document_with_template(
        "Customer Response",
        "business_letter"
    )
    
    report_doc = pdf_factory.create_document_with_template(
        "Monthly Report",
        "report"
    )
    
    print(f"   Business letter content preview: {business_letter.content[:50]}...")
    print(f"   Report content preview: {report_doc.content[:50]}...")
    print()
    
    # 5. Parameterized Factory
    print("5. PARAMETERIZED FACTORY:")
    
    param_factory = ParameterizedDocumentFactory(registry)
    
    # Simple creation
    simple_doc = param_factory.create_document(
        'simple',
        doc_type=DocumentType.TEXT,
        title="Simple Document",
        content="Simple content"
    )
    
    # Template creation
    template_doc = param_factory.create_document(
        'template',
        doc_type=DocumentType.WORD,
        title="Template Document",
        template_name="invoice"
    )
    
    # Batch creation
    batch_config = [
        {
            'type': 'pdf',
            'title': 'Batch PDF 1',
            'content': 'First PDF content'
        },
        {
            'type': 'word',
            'title': 'Batch Word 1',
            'content': 'First Word content'
        },
        {
            'type': 'excel',
            'title': 'Batch Excel 1',
            'content': 'First Excel content'
        }
    ]
    
    batch_docs = param_factory.create_document(
        'batch',
        documents_config=batch_config
    )
    
    print(f"   Simple document: {simple_doc.title if simple_doc else 'None'}")
    print(f"   Template document: {template_doc.title if template_doc else 'None'}")
    print(f"   Batch documents created: {len(batch_docs)}")
    
    for doc in batch_docs:
        print(f"     - {doc.title} ({doc.__class__.__name__})")
    print()
    
    # 6. Custom Factory with Builder
    print("6. CUSTOM FACTORY WITH BUILDER:")
    
    # Define validators
    def validate_title_length(doc: Document) -> bool:
        return len(doc.title) >= 3
    
    def validate_content_not_empty(doc: Document) -> bool:
        return len(doc.content.strip()) > 0
    
    # Define post-processors
    def add_creation_timestamp(doc: Document) -> Document:
        doc.add_metadata('processed_at', datetime.datetime.now().isoformat())
        return doc
    
    def add_word_count(doc: Document) -> Document:
        word_count = len(doc.content.split())
        doc.add_metadata('word_count', word_count)
        return doc
    
    # Build custom factory
    custom_factory = (DocumentFactoryBuilder()
                     .set_factory_class(WordDocumentFactory)
                     .add_default_param('header', 'Auto-Generated Header')
                     .add_default_param('footer', 'Auto-Generated Footer')
                     .add_validator(validate_title_length)
                     .add_validator(validate_content_not_empty)
                     .add_post_processor(add_creation_timestamp)
                     .add_post_processor(add_word_count)
                     .build())
    
    # Create document with custom factory
    try:
        custom_doc = custom_factory.create_document(
            "Custom Document",
            "This is content for the custom document with validation and processing."
        )
        print(f"   Custom document created: {custom_doc.title}")
        print(f"   Custom document metadata: {custom_doc.metadata}")
    except ValueError as e:
        print(f"   Custom document creation failed: {e}")
    
    # Try to create invalid document
    try:
        invalid_doc = custom_factory.create_document("AB", "")  # Too short title, empty content
        print(f"   Invalid document created: {invalid_doc.title}")
    except ValueError as e:
        print(f"   Invalid document rejected: {e}")
    
    print()
    
    # 7. Export Functionality
    print("7. DOCUMENT EXPORT FUNCTIONALITY:")
    
    test_doc = registry.create_document(
        DocumentType.PDF,
        "Export Test",
        "This document will be exported to different formats"
    )
    
    if test_doc:
        formats_to_test = ['text', 'html']
        for format_type in formats_to_test:
            exported = test_doc.export(format_type)
            print(f"   Exported to {format_type}: {exported[:50]}...")
    
    print()
    
    # 8. Factory Pattern Benefits
    print("8. FACTORY PATTERN BENEFITS DEMONSTRATED:")
    print("   ✓ Encapsulation: Object creation logic is encapsulated in factories")
    print("   ✓ Flexibility: Easy to add new document types without changing client code")
    print("   ✓ Consistency: All documents follow the same creation interface")
    print("   ✓ Extensibility: New factories can be added through registration")
    print("   ✓ Parameterization: Complex object creation with multiple parameters")
    print("   ✓ Template Support: Document creation from predefined templates")
    print("   ✓ Validation: Built-in validation and post-processing capabilities")
    print("   ✓ Registry Pattern: Centralized factory management and discovery")
    print()
    
    print("=== FACTORY METHOD PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_factory_method_pattern()
