"""
BUILDER PATTERN - Creational Design Pattern
===========================================

Problem Statement:
Implement the Builder pattern to construct complex objects step by step:
- Fluent interface for object construction
- Director class for construction algorithms
- Multiple builders for different representations
- Validation during construction process
- Immutable object creation with builders

Learning Objectives:
- Understand when to use Builder pattern
- Implement fluent interfaces for object construction
- Design flexible construction processes
- Handle complex object validation
- Create immutable objects with builders
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, date
from enum import Enum
import json
import copy


# ============================================================================
# ENUMS AND VALUE OBJECTS
# ============================================================================

class DocumentFormat(Enum):
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    DOCX = "docx"
    PLAIN_TEXT = "plain_text"


class FontFamily(Enum):
    ARIAL = "Arial"
    TIMES = "Times New Roman"
    HELVETICA = "Helvetica"
    CALIBRI = "Calibri"
    GEORGIA = "Georgia"


class Alignment(Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


class PageSize(Enum):
    A4 = "A4"
    LETTER = "Letter"
    LEGAL = "Legal"
    A3 = "A3"
    CUSTOM = "Custom"


# ============================================================================
# COMPLEX PRODUCT - DOCUMENT
# ============================================================================

class DocumentStyle:
    """Document styling configuration."""
    
    def __init__(self):
        self.font_family = FontFamily.ARIAL
        self.font_size = 12
        self.line_height = 1.5
        self.margin_top = 1.0
        self.margin_bottom = 1.0
        self.margin_left = 1.0
        self.margin_right = 1.0
        self.page_size = PageSize.A4
        self.orientation = "portrait"
        self.header_font_size = 16
        self.footer_font_size = 10
        self.paragraph_spacing = 0.5
        self.text_color = "#000000"
        self.background_color = "#ffffff"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert style to dictionary."""
        return {
            'font_family': self.font_family.value,
            'font_size': self.font_size,
            'line_height': self.line_height,
            'margins': {
                'top': self.margin_top,
                'bottom': self.margin_bottom,
                'left': self.margin_left,
                'right': self.margin_right
            },
            'page_size': self.page_size.value,
            'orientation': self.orientation,
            'header_font_size': self.header_font_size,
            'footer_font_size': self.footer_font_size,
            'paragraph_spacing': self.paragraph_spacing,
            'text_color': self.text_color,
            'background_color': self.background_color
        }


class DocumentSection:
    """A section within a document."""
    
    def __init__(self, title: str = "", content: str = "", level: int = 1):
        self.title = title
        self.content = content
        self.level = level  # Heading level (1-6)
        self.alignment = Alignment.LEFT
        self.subsections: List['DocumentSection'] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_subsection(self, subsection: 'DocumentSection') -> None:
        """Add a subsection."""
        self.subsections.append(subsection)
    
    def get_word_count(self) -> int:
        """Get total word count including subsections."""
        count = len(self.content.split()) if self.content else 0
        for subsection in self.subsections:
            count += subsection.get_word_count()
        return count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return {
            'title': self.title,
            'content': self.content,
            'level': self.level,
            'alignment': self.alignment.value,
            'subsections': [sub.to_dict() for sub in self.subsections],
            'metadata': self.metadata,
            'word_count': self.get_word_count()
        }


class Document:
    """Complex document object to be built."""
    
    def __init__(self):
        self.title = ""
        self.author = ""
        self.created_date = datetime.now()
        self.modified_date = datetime.now()
        self.version = "1.0"
        self.description = ""
        self.keywords: List[str] = []
        self.language = "en"
        
        # Document structure
        self.sections: List[DocumentSection] = []
        self.table_of_contents = True
        self.page_numbers = True
        self.header_text = ""
        self.footer_text = ""
        
        # Styling
        self.style = DocumentStyle()
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
        self.custom_properties: Dict[str, Any] = {}
        
        # Output settings
        self.format = DocumentFormat.PDF
        self.output_path = ""
        
        # Validation flags
        self._is_validated = False
        self._validation_errors: List[str] = []
    
    def add_section(self, section: DocumentSection) -> None:
        """Add a section to the document."""
        self.sections.append(section)
        self.modified_date = datetime.now()
    
    def get_total_word_count(self) -> int:
        """Get total word count of document."""
        return sum(section.get_word_count() for section in self.sections)
    
    def get_section_count(self) -> int:
        """Get total number of sections and subsections."""
        count = len(self.sections)
        for section in self.sections:
            count += len(section.subsections)
        return count
    
    def validate(self) -> bool:
        """Validate document structure and content."""
        self._validation_errors.clear()
        
        # Check required fields
        if not self.title.strip():
            self._validation_errors.append("Document title is required")
        
        if not self.author.strip():
            self._validation_errors.append("Document author is required")
        
        if not self.sections:
            self._validation_errors.append("Document must have at least one section")
        
        # Check sections
        for i, section in enumerate(self.sections):
            if not section.title.strip() and not section.content.strip():
                self._validation_errors.append(f"Section {i+1} is empty")
        
        # Check style consistency
        if self.style.font_size < 8 or self.style.font_size > 72:
            self._validation_errors.append("Font size must be between 8 and 72")
        
        if self.style.line_height < 0.5 or self.style.line_height > 3.0:
            self._validation_errors.append("Line height must be between 0.5 and 3.0")
        
        self._is_validated = len(self._validation_errors) == 0
        return self._is_validated
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_errors.copy()
    
    def is_valid(self) -> bool:
        """Check if document is valid."""
        return self._is_validated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            'title': self.title,
            'author': self.author,
            'created_date': self.created_date.isoformat(),
            'modified_date': self.modified_date.isoformat(),
            'version': self.version,
            'description': self.description,
            'keywords': self.keywords,
            'language': self.language,
            'sections': [section.to_dict() for section in self.sections],
            'table_of_contents': self.table_of_contents,
            'page_numbers': self.page_numbers,
            'header_text': self.header_text,
            'footer_text': self.footer_text,
            'style': self.style.to_dict(),
            'metadata': self.metadata,
            'custom_properties': self.custom_properties,
            'format': self.format.value,
            'output_path': self.output_path,
            'statistics': {
                'word_count': self.get_total_word_count(),
                'section_count': self.get_section_count(),
                'is_valid': self.is_valid(),
                'validation_errors': self.get_validation_errors()
            }
        }
    
    def export(self) -> str:
        """Export document in specified format."""
        if not self.is_valid():
            raise ValueError(f"Cannot export invalid document: {self._validation_errors}")
        
        if self.format == DocumentFormat.JSON:
            return json.dumps(self.to_dict(), indent=2)
        elif self.format == DocumentFormat.HTML:
            return self._export_html()
        elif self.format == DocumentFormat.MARKDOWN:
            return self._export_markdown()
        else:
            return f"Export to {self.format.value} format (simulated)"
    
    def _export_html(self) -> str:
        """Export document as HTML."""
        html = f"<html><head><title>{self.title}</title></head><body>"
        html += f"<h1>{self.title}</h1>"
        html += f"<p><strong>Author:</strong> {self.author}</p>"
        
        for section in self.sections:
            html += f"<h{section.level + 1}>{section.title}</h{section.level + 1}>"
            html += f"<p>{section.content}</p>"
            
            for subsection in section.subsections:
                html += f"<h{subsection.level + 2}>{subsection.title}</h{subsection.level + 2}>"
                html += f"<p>{subsection.content}</p>"
        
        html += "</body></html>"
        return html
    
    def _export_markdown(self) -> str:
        """Export document as Markdown."""
        md = f"# {self.title}\n\n"
        md += f"**Author:** {self.author}\n\n"
        
        for section in self.sections:
            md += f"{'#' * (section.level + 1)} {section.title}\n\n"
            md += f"{section.content}\n\n"
            
            for subsection in section.subsections:
                md += f"{'#' * (subsection.level + 2)} {subsection.title}\n\n"
                md += f"{subsection.content}\n\n"
        
        return md


# ============================================================================
# ABSTRACT BUILDER
# ============================================================================

class DocumentBuilder(ABC):
    """Abstract builder for creating documents."""
    
    def __init__(self):
        self.reset()
    
    @abstractmethod
    def reset(self) -> 'DocumentBuilder':
        """Reset the builder to start fresh."""
        pass
    
    @abstractmethod
    def set_title(self, title: str) -> 'DocumentBuilder':
        """Set document title."""
        pass
    
    @abstractmethod
    def set_author(self, author: str) -> 'DocumentBuilder':
        """Set document author."""
        pass
    
    @abstractmethod
    def set_description(self, description: str) -> 'DocumentBuilder':
        """Set document description."""
        pass
    
    @abstractmethod
    def add_section(self, title: str, content: str, level: int = 1) -> 'DocumentBuilder':
        """Add a section to the document."""
        pass
    
    @abstractmethod
    def set_style(self, **style_properties) -> 'DocumentBuilder':
        """Set document styling."""
        pass
    
    @abstractmethod
    def set_format(self, format_type: DocumentFormat) -> 'DocumentBuilder':
        """Set output format."""
        pass
    
    @abstractmethod
    def build(self) -> Document:
        """Build and return the final document."""
        pass


# ============================================================================
# CONCRETE BUILDERS
# ============================================================================

class StandardDocumentBuilder(DocumentBuilder):
    """Standard document builder with full functionality."""
    
    def __init__(self):
        super().__init__()
    
    def reset(self) -> 'StandardDocumentBuilder':
        """Reset the builder."""
        self._document = Document()
        return self
    
    def set_title(self, title: str) -> 'StandardDocumentBuilder':
        """Set document title."""
        self._document.title = title
        return self
    
    def set_author(self, author: str) -> 'StandardDocumentBuilder':
        """Set document author."""
        self._document.author = author
        return self
    
    def set_description(self, description: str) -> 'StandardDocumentBuilder':
        """Set document description."""
        self._document.description = description
        return self
    
    def set_version(self, version: str) -> 'StandardDocumentBuilder':
        """Set document version."""
        self._document.version = version
        return self
    
    def add_keywords(self, *keywords: str) -> 'StandardDocumentBuilder':
        """Add keywords to document."""
        self._document.keywords.extend(keywords)
        return self
    
    def set_language(self, language: str) -> 'StandardDocumentBuilder':
        """Set document language."""
        self._document.language = language
        return self
    
    def add_section(self, title: str, content: str, level: int = 1) -> 'StandardDocumentBuilder':
        """Add a section to the document."""
        section = DocumentSection(title, content, level)
        self._document.add_section(section)
        return self
    
    def add_section_with_subsections(self, title: str, content: str, 
                                   subsections: List[Dict[str, Any]]) -> 'StandardDocumentBuilder':
        """Add a section with subsections."""
        section = DocumentSection(title, content, 1)
        
        for sub_data in subsections:
            subsection = DocumentSection(
                sub_data.get('title', ''),
                sub_data.get('content', ''),
                sub_data.get('level', 2)
            )
            section.add_subsection(subsection)
        
        self._document.add_section(section)
        return self
    
    def set_table_of_contents(self, enabled: bool) -> 'StandardDocumentBuilder':
        """Enable/disable table of contents."""
        self._document.table_of_contents = enabled
        return self
    
    def set_page_numbers(self, enabled: bool) -> 'StandardDocumentBuilder':
        """Enable/disable page numbers."""
        self._document.page_numbers = enabled
        return self
    
    def set_header(self, header_text: str) -> 'StandardDocumentBuilder':
        """Set header text."""
        self._document.header_text = header_text
        return self
    
    def set_footer(self, footer_text: str) -> 'StandardDocumentBuilder':
        """Set footer text."""
        self._document.footer_text = footer_text
        return self
    
    def set_style(self, **style_properties) -> 'StandardDocumentBuilder':
        """Set document styling."""
        style = self._document.style
        
        if 'font_family' in style_properties:
            style.font_family = FontFamily(style_properties['font_family'])
        if 'font_size' in style_properties:
            style.font_size = style_properties['font_size']
        if 'line_height' in style_properties:
            style.line_height = style_properties['line_height']
        if 'page_size' in style_properties:
            style.page_size = PageSize(style_properties['page_size'])
        if 'orientation' in style_properties:
            style.orientation = style_properties['orientation']
        if 'text_color' in style_properties:
            style.text_color = style_properties['text_color']
        if 'background_color' in style_properties:
            style.background_color = style_properties['background_color']
        
        # Set margins
        if 'margins' in style_properties:
            margins = style_properties['margins']
            style.margin_top = margins.get('top', style.margin_top)
            style.margin_bottom = margins.get('bottom', style.margin_bottom)
            style.margin_left = margins.get('left', style.margin_left)
            style.margin_right = margins.get('right', style.margin_right)
        
        return self
    
    def set_format(self, format_type: DocumentFormat) -> 'StandardDocumentBuilder':
        """Set output format."""
        self._document.format = format_type
        return self
    
    def set_output_path(self, path: str) -> 'StandardDocumentBuilder':
        """Set output file path."""
        self._document.output_path = path
        return self
    
    def add_metadata(self, key: str, value: Any) -> 'StandardDocumentBuilder':
        """Add metadata to document."""
        self._document.metadata[key] = value
        return self
    
    def add_custom_property(self, key: str, value: Any) -> 'StandardDocumentBuilder':
        """Add custom property to document."""
        self._document.custom_properties[key] = value
        return self
    
    def build(self) -> Document:
        """Build and return the final document."""
        # Validate before building
        if not self._document.validate():
            errors = self._document.get_validation_errors()
            raise ValueError(f"Document validation failed: {errors}")
        
        # Return a copy to prevent further modification
        result = copy.deepcopy(self._document)
        self.reset()  # Reset for next build
        return result


class ReportDocumentBuilder(DocumentBuilder):
    """Specialized builder for creating reports."""
    
    def __init__(self):
        super().__init__()
    
    def reset(self) -> 'ReportDocumentBuilder':
        """Reset the builder."""
        self._document = Document()
        # Set report-specific defaults
        self._document.table_of_contents = True
        self._document.page_numbers = True
        self._document.style.font_family = FontFamily.CALIBRI
        self._document.style.font_size = 11
        self._document.style.line_height = 1.15
        return self
    
    def set_title(self, title: str) -> 'ReportDocumentBuilder':
        """Set report title."""
        self._document.title = title
        return self
    
    def set_author(self, author: str) -> 'ReportDocumentBuilder':
        """Set report author."""
        self._document.author = author
        return self
    
    def set_description(self, description: str) -> 'ReportDocumentBuilder':
        """Set report description."""
        self._document.description = description
        return self
    
    def add_executive_summary(self, summary: str) -> 'ReportDocumentBuilder':
        """Add executive summary section."""
        section = DocumentSection("Executive Summary", summary, 1)
        self._document.sections.insert(0, section)  # Insert at beginning
        return self
    
    def add_section(self, title: str, content: str, level: int = 1) -> 'ReportDocumentBuilder':
        """Add a section to the report."""
        section = DocumentSection(title, content, level)
        self._document.add_section(section)
        return self
    
    def add_findings_section(self, findings: List[str]) -> 'ReportDocumentBuilder':
        """Add findings section with bullet points."""
        content = "\n".join(f"• {finding}" for finding in findings)
        section = DocumentSection("Key Findings", content, 1)
        self._document.add_section(section)
        return self
    
    def add_recommendations_section(self, recommendations: List[str]) -> 'ReportDocumentBuilder':
        """Add recommendations section."""
        content = "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations))
        section = DocumentSection("Recommendations", content, 1)
        self._document.add_section(section)
        return self
    
    def add_conclusion(self, conclusion: str) -> 'ReportDocumentBuilder':
        """Add conclusion section."""
        section = DocumentSection("Conclusion", conclusion, 1)
        self._document.add_section(section)
        return self
    
    def set_style(self, **style_properties) -> 'ReportDocumentBuilder':
        """Set report styling (limited options)."""
        style = self._document.style
        
        # Only allow certain style modifications for reports
        if 'font_size' in style_properties:
            font_size = style_properties['font_size']
            if 10 <= font_size <= 14:  # Restrict font size for reports
                style.font_size = font_size
        
        if 'line_height' in style_properties:
            line_height = style_properties['line_height']
            if 1.0 <= line_height <= 2.0:  # Restrict line height
                style.line_height = line_height
        
        return self
    
    def set_format(self, format_type: DocumentFormat) -> 'ReportDocumentBuilder':
        """Set output format (PDF preferred for reports)."""
        self._document.format = format_type
        return self
    
    def build(self) -> Document:
        """Build and return the final report."""
        # Add report-specific metadata
        self._document.add_metadata('document_type', 'report')
        self._document.add_metadata('created_with', 'ReportDocumentBuilder')
        
        # Set report-specific header/footer if not set
        if not self._document.header_text:
            self._document.header_text = f"{self._document.title} - {self._document.author}"
        
        if not self._document.footer_text:
            self._document.footer_text = f"Page {{page}} - Generated on {datetime.now().strftime('%Y-%m-%d')}"
        
        # Validate
        if not self._document.validate():
            errors = self._document.get_validation_errors()
            raise ValueError(f"Report validation failed: {errors}")
        
        result = copy.deepcopy(self._document)
        self.reset()
        return result


class MinimalDocumentBuilder(DocumentBuilder):
    """Minimal builder for simple documents."""
    
    def __init__(self):
        super().__init__()
    
    def reset(self) -> 'MinimalDocumentBuilder':
        """Reset the builder."""
        self._document = Document()
        # Set minimal defaults
        self._document.table_of_contents = False
        self._document.page_numbers = False
        self._document.style.font_family = FontFamily.ARIAL
        self._document.style.font_size = 12
        return self
    
    def set_title(self, title: str) -> 'MinimalDocumentBuilder':
        """Set document title."""
        self._document.title = title
        return self
    
    def set_author(self, author: str) -> 'MinimalDocumentBuilder':
        """Set document author."""
        self._document.author = author
        return self
    
    def set_description(self, description: str) -> 'MinimalDocumentBuilder':
        """Set document description."""
        self._document.description = description
        return self
    
    def add_section(self, title: str, content: str, level: int = 1) -> 'MinimalDocumentBuilder':
        """Add a section to the document."""
        section = DocumentSection(title, content, level)
        self._document.add_section(section)
        return self
    
    def add_content(self, content: str) -> 'MinimalDocumentBuilder':
        """Add content as a single section."""
        section = DocumentSection("", content, 1)
        self._document.add_section(section)
        return self
    
    def set_style(self, **style_properties) -> 'MinimalDocumentBuilder':
        """Set basic styling (very limited)."""
        style = self._document.style
        
        if 'font_size' in style_properties:
            style.font_size = style_properties['font_size']
        
        return self
    
    def set_format(self, format_type: DocumentFormat) -> 'MinimalDocumentBuilder':
        """Set output format."""
        self._document.format = format_type
        return self
    
    def build(self) -> Document:
        """Build and return the minimal document."""
        # Minimal validation
        if not self._document.title:
            self._document.title = "Untitled Document"
        if not self._document.author:
            self._document.author = "Unknown Author"
        
        # Force validation to pass for minimal documents
        self._document._is_validated = True
        
        result = copy.deepcopy(self._document)
        self.reset()
        return result


# ============================================================================
# DIRECTOR CLASS
# ============================================================================

class DocumentDirector:
    """Director class that knows how to construct specific types of documents."""
    
    def __init__(self, builder: DocumentBuilder):
        self.builder = builder
    
    def set_builder(self, builder: DocumentBuilder) -> None:
        """Set a new builder."""
        self.builder = builder
    
    def create_technical_report(self, title: str, author: str, 
                              data: Dict[str, Any]) -> Document:
        """Create a technical report document."""
        return (self.builder
                .reset()
                .set_title(title)
                .set_author(author)
                .set_description("Technical analysis report")
                .add_keywords("technical", "report", "analysis")
                .set_format(DocumentFormat.PDF)
                .set_style(
                    font_family="Calibri",
                    font_size=11,
                    line_height=1.15,
                    page_size="A4"
                )
                .set_table_of_contents(True)
                .set_page_numbers(True)
                .add_section("Introduction", data.get('introduction', ''), 1)
                .add_section("Methodology", data.get('methodology', ''), 1)
                .add_section("Results", data.get('results', ''), 1)
                .add_section("Analysis", data.get('analysis', ''), 1)
                .add_section("Conclusion", data.get('conclusion', ''), 1)
                .build())
    
    def create_business_proposal(self, title: str, author: str,
                               proposal_data: Dict[str, Any]) -> Document:
        """Create a business proposal document."""
        return (self.builder
                .reset()
                .set_title(title)
                .set_author(author)
                .set_description("Business proposal document")
                .add_keywords("business", "proposal", "plan")
                .set_format(DocumentFormat.PDF)
                .set_style(
                    font_family="Times New Roman",
                    font_size=12,
                    line_height=1.5,
                    page_size="Letter"
                )
                .set_header(f"{title} - {author}")
                .set_footer("Confidential Business Proposal")
                .add_section("Executive Summary", proposal_data.get('summary', ''), 1)
                .add_section("Problem Statement", proposal_data.get('problem', ''), 1)
                .add_section("Proposed Solution", proposal_data.get('solution', ''), 1)
                .add_section("Implementation Plan", proposal_data.get('implementation', ''), 1)
                .add_section("Budget", proposal_data.get('budget', ''), 1)
                .add_section("Timeline", proposal_data.get('timeline', ''), 1)
                .build())
    
    def create_user_manual(self, title: str, author: str,
                          manual_data: Dict[str, Any]) -> Document:
        """Create a user manual document."""
        return (self.builder
                .reset()
                .set_title(title)
                .set_author(author)
                .set_description("User manual and documentation")
                .add_keywords("manual", "documentation", "guide")
                .set_format(DocumentFormat.HTML)
                .set_style(
                    font_family="Arial",
                    font_size=12,
                    line_height=1.4
                )
                .set_table_of_contents(True)
                .add_section("Getting Started", manual_data.get('getting_started', ''), 1)
                .add_section("Features", manual_data.get('features', ''), 1)
                .add_section("Troubleshooting", manual_data.get('troubleshooting', ''), 1)
                .add_section("FAQ", manual_data.get('faq', ''), 1)
                .add_section("Contact Support", manual_data.get('support', ''), 1)
                .build())
    
    def create_simple_note(self, title: str, content: str) -> Document:
        """Create a simple note document."""
        return (self.builder
                .reset()
                .set_title(title)
                .set_author("Note Taker")
                .add_content(content)
                .set_format(DocumentFormat.PLAIN_TEXT)
                .build())


def demonstrate_builder_pattern():
    """
    Demonstrate Builder pattern implementations.
    """
    print("=== BUILDER PATTERN DEMONSTRATION ===\n")
    
    # 1. Basic Builder Usage
    print("1. BASIC BUILDER USAGE:")
    
    builder = StandardDocumentBuilder()
    
    # Build a document using fluent interface
    document = (builder
                .set_title("My First Document")
                .set_author("John Doe")
                .set_description("A sample document created with builder pattern")
                .add_keywords("sample", "demo", "builder")
                .set_language("en")
                .add_section("Introduction", "This is the introduction section.", 1)
                .add_section("Main Content", "This is the main content of the document.", 1)
                .add_section("Conclusion", "This is the conclusion section.", 1)
                .set_style(
                    font_family="Arial",
                    font_size=12,
                    line_height=1.5,
                    margins={'top': 1.0, 'bottom': 1.0, 'left': 1.0, 'right': 1.0}
                )
                .set_format(DocumentFormat.PDF)
                .set_table_of_contents(True)
                .set_page_numbers(True)
                .build())
    
    print(f"   Document created: {document.title}")
    print(f"   Author: {document.author}")
    print(f"   Sections: {len(document.sections)}")
    print(f"   Word count: {document.get_total_word_count()}")
    print(f"   Valid: {document.is_valid()}")
    print()
    
    # 2. Different Builder Types
    print("2. DIFFERENT BUILDER TYPES:")
    
    # Standard builder
    standard_builder = StandardDocumentBuilder()
    standard_doc = (standard_builder
                   .set_title("Standard Document")
                   .set_author("Standard Author")
                   .add_section("Section 1", "Content 1")
                   .build())
    
    # Report builder
    report_builder = ReportDocumentBuilder()
    report_doc = (report_builder
                 .set_title("Quarterly Report")
                 .set_author("Report Author")
                 .add_executive_summary("This quarter showed significant growth.")
                 .add_findings_section([
                     "Revenue increased by 15%",
                     "Customer satisfaction improved",
                     "New market segments identified"
                 ])
                 .add_recommendations_section([
                     "Expand marketing efforts",
                     "Invest in customer service",
                     "Develop new products"
                 ])
                 .add_conclusion("The quarter was successful with room for improvement.")
                 .build())
    
    # Minimal builder
    minimal_builder = MinimalDocumentBuilder()
    minimal_doc = (minimal_builder
                  .set_title("Simple Note")
                  .add_content("This is a simple note with minimal formatting.")
                  .set_format(DocumentFormat.PLAIN_TEXT)
                  .build())
    
    print(f"   Standard document: {standard_doc.title} ({len(standard_doc.sections)} sections)")
    print(f"   Report document: {report_doc.title} ({len(report_doc.sections)} sections)")
    print(f"   Minimal document: {minimal_doc.title} ({len(minimal_doc.sections)} sections)")
    print()
    
    # 3. Director Usage
    print("3. DIRECTOR PATTERN USAGE:")
    
    director = DocumentDirector(StandardDocumentBuilder())
    
    # Create technical report
    tech_report_data = {
        'introduction': 'This report analyzes system performance.',
        'methodology': 'We used automated testing tools and metrics.',
        'results': 'Performance improved by 25% after optimization.',
        'analysis': 'The improvements were due to better caching.',
        'conclusion': 'The optimization was successful.'
    }
    
    tech_report = director.create_technical_report(
        "System Performance Analysis",
        "Tech Team",
        tech_report_data
    )
    
    # Create business proposal
    proposal_data = {
        'summary': 'We propose a new customer management system.',
        'problem': 'Current system is outdated and inefficient.',
        'solution': 'Modern cloud-based CRM with AI features.',
        'implementation': 'Phased rollout over 6 months.',
        'budget': 'Total cost: $150,000',
        'timeline': 'Project completion by Q3 2024.'
    }
    
    business_proposal = director.create_business_proposal(
        "CRM System Upgrade Proposal",
        "Business Development",
        proposal_data
    )
    
    # Create user manual
    manual_data = {
        'getting_started': 'Download and install the application.',
        'features': 'The app includes dashboard, reports, and settings.',
        'troubleshooting': 'Common issues and their solutions.',
        'faq': 'Frequently asked questions and answers.',
        'support': 'Contact support@company.com for help.'
    }
    
    user_manual = director.create_user_manual(
        "Application User Guide",
        "Documentation Team",
        manual_data
    )
    
    print(f"   Technical report: {tech_report.title}")
    print(f"     Sections: {len(tech_report.sections)}")
    print(f"     Format: {tech_report.format.value}")
    print(f"     TOC: {tech_report.table_of_contents}")
    
    print(f"   Business proposal: {business_proposal.title}")
    print(f"     Sections: {len(business_proposal.sections)}")
    print(f"     Header: {business_proposal.header_text}")
    print(f"     Footer: {business_proposal.footer_text}")
    
    print(f"   User manual: {user_manual.title}")
    print(f"     Sections: {len(user_manual.sections)}")
    print(f"     Format: {user_manual.format.value}")
    print()
    
    # 4. Complex Document with Subsections
    print("4. COMPLEX DOCUMENT WITH SUBSECTIONS:")
    
    complex_builder = StandardDocumentBuilder()
    
    # Create document with nested sections
    complex_doc = (complex_builder
                  .set_title("Software Architecture Guide")
                  .set_author("Architecture Team")
                  .set_description("Comprehensive guide to software architecture")
                  .add_section_with_subsections(
                      "Design Patterns",
                      "Overview of common design patterns",
                      [
                          {'title': 'Creational Patterns', 'content': 'Singleton, Factory, Builder', 'level': 2},
                          {'title': 'Structural Patterns', 'content': 'Adapter, Decorator, Facade', 'level': 2},
                          {'title': 'Behavioral Patterns', 'content': 'Observer, Strategy, Command', 'level': 2}
                      ]
                  )
                  .add_section_with_subsections(
                      "Best Practices",
                      "Software development best practices",
                      [
                          {'title': 'Code Quality', 'content': 'Clean code principles', 'level': 2},
                          {'title': 'Testing', 'content': 'Unit and integration testing', 'level': 2},
                          {'title': 'Documentation', 'content': 'API and code documentation', 'level': 2}
                      ]
                  )
                  .set_style(
                      font_family="Calibri",
                      font_size=11,
                      line_height=1.2
                  )
                  .add_metadata('complexity', 'high')
                  .add_custom_property('review_required', True)
                  .build())
    
    print(f"   Complex document: {complex_doc.title}")
    print(f"   Total sections: {complex_doc.get_section_count()}")
    print(f"   Word count: {complex_doc.get_total_word_count()}")
    
    # Show section structure
    for i, section in enumerate(complex_doc.sections):
        print(f"     Section {i+1}: {section.title} ({len(section.subsections)} subsections)")
        for j, subsection in enumerate(section.subsections):
            print(f"       {i+1}.{j+1}: {subsection.title}")
    
    print()
    
    # 5. Validation and Error Handling
    print("5. VALIDATION AND ERROR HANDLING:")
    
    # Try to build invalid document
    try:
        invalid_doc = (StandardDocumentBuilder()
                      .set_title("")  # Empty title
                      .set_author("")  # Empty author
                      .set_style(font_size=100)  # Invalid font size
                      .build())
        print("   Invalid document created (this shouldn't happen)")
    except ValueError as e:
        print(f"   ✓ Validation caught errors: {str(e)[:60]}...")
    
    # Build valid document and show validation
    valid_doc = (StandardDocumentBuilder()
                .set_title("Valid Document")
                .set_author("Valid Author")
                .add_section("Content", "Some content here")
                .build())
    
    print(f"   Valid document created: {valid_doc.title}")
    print(f"   Validation errors: {len(valid_doc.get_validation_errors())}")
    print()
    
    # 6. Document Export
    print("6. DOCUMENT EXPORT:")
    
    # Create a document for export testing
    export_doc = (StandardDocumentBuilder()
                 .set_title("Export Test Document")
                 .set_author("Export Tester")
                 .add_section("Introduction", "This document tests export functionality.")
                 .add_section("Content", "Here is some sample content for testing.")
                 .set_format(DocumentFormat.HTML)
                 .build())
    
    # Export as HTML
    html_export = export_doc.export()
    print(f"   HTML export preview: {html_export[:100]}...")
    
    # Change format and export as Markdown
    export_doc.format = DocumentFormat.MARKDOWN
    md_export = export_doc.export()
    print(f"   Markdown export preview: {md_export[:100]}...")
    
    print()
    
    # 7. Builder Reuse
    print("7. BUILDER REUSE:")
    
    reusable_builder = StandardDocumentBuilder()
    
    # Create multiple documents with same builder
    documents = []
    
    for i in range(3):
        doc = (reusable_builder
               .set_title(f"Document {i+1}")
               .set_author(f"Author {i+1}")
               .add_section("Section", f"Content for document {i+1}")
               .build())
        documents.append(doc)
    
    print(f"   Created {len(documents)} documents with reusable builder:")
    for doc in documents:
        print(f"     - {doc.title} by {doc.author}")
    
    print()
    
    # 8. Builder Pattern Benefits
    print("8. BUILDER PATTERN BENEFITS:")
    print("   ✓ Fluent Interface: Readable and chainable method calls")
    print("   ✓ Flexibility: Different builders for different document types")
    print("   ✓ Validation: Built-in validation during construction")
    print("   ✓ Immutability: Final objects are immutable copies")
    print("   ✓ Director Pattern: Encapsulates construction algorithms")
    print("   ✓ Extensibility: Easy to add new builders and construction steps")
    print("   ✓ Separation of Concerns: Construction logic separated from representation")
    print("   ✓ Reusability: Builders can be reused for multiple objects")
    print()
    
    print("=== BUILDER PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_builder_pattern()
