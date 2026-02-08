# Data Analysis Agent - Project Description

## Problem Statement

The Data Analysis Agent is an intelligent system designed to analyze CSV and tabular data files, generate insights, and create visualizations automatically. Traditional data analysis requires domain expertise, knowledge of data manipulation libraries (such as pandas), and visualization tools. This agent democratizes data analysis by allowing users to interact with their data using natural language queries, while the agent handles the technical complexity of data loading, schema analysis, query generation, code execution, and visualization creation.

The agent addresses several key challenges:
- **Accessibility**: Non-technical users can analyze data without writing code
- **Efficiency**: Automated schema analysis and insight generation save time
- **Safety**: Sandboxed code execution prevents malicious operations
- **Comprehensiveness**: Automatic discovery of patterns, correlations, and anomalies
- **Visualization**: Automatic generation of appropriate charts and graphs

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA ANALYSIS AGENT                              │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  CSV/Tabular │
│  Data Input  │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│  Data Loader    │  Load file, detect encoding, parse CSV
│  - Load CSV     │  Extract basic metadata
│  - Inspect      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Schema Analyzer │  Analyze column types, statistics
│  - Type Detect  │  Calculate correlations, distributions
│  - Statistics   │  Identify missing values, outliers
│  - Correlations │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Query Generator │  Convert natural language to pandas code
│  (LLM-based)    │  Generate safe, executable queries
│  - NL → Code    │  Validate syntax and operations
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Code Executor   │  Execute pandas code in sandbox
│  - Sandbox      │  Capture results and errors
│  - Execute      │  Return structured output
│  - Validate     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Visualization   │  Generate matplotlib/seaborn code
│  Generator      │  Create appropriate chart types
│  - Chart Type   │  Save visualizations to files
│  - Generate     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Insight         │  Summarize findings
│  Summarizer     │  Generate natural language reports
│  - Analyze      │  Highlight key patterns
│  - Report       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Final Report   │  Combined statistics, visualizations,
│  - Statistics   │  insights, and recommendations
│  - Charts       │
│  - Insights     │
└─────────────────┘
```

## Component Breakdown

### 1. Data_Loader
Responsible for ingesting data files and performing initial inspection. Handles various CSV formats, encoding detection, and provides basic metadata about the dataset.

**Key Responsibilities:**
- Load CSV files with automatic encoding detection
- Handle different delimiters (comma, tab, semicolon)
- Extract basic statistics (row count, column count, file size)
- Provide sample rows for preview
- Detect data types preliminarily

**Input:** File path (CSV or tabular data)
**Output:** Pandas DataFrame, basic metadata dictionary

### 2. Schema_Analyzer
Performs deep analysis of the dataset structure, data types, and statistical properties. Identifies relationships between columns and data quality issues.

**Key Responsibilities:**
- Analyze column data types (numeric, categorical, datetime, text)
- Calculate descriptive statistics (mean, median, std, min, max, quartiles)
- Detect missing values and their patterns
- Identify outliers using statistical methods
- Calculate correlations between numeric columns
- Analyze categorical value distributions
- Generate data quality metrics

**Input:** Pandas DataFrame
**Output:** Schema dictionary with types, statistics, correlations, quality metrics

### 3. Query_Generator
Converts natural language questions into executable pandas code using a Large Language Model. Ensures generated code is safe and follows best practices.

**Key Responsibilities:**
- Accept natural language queries from users
- Generate pandas code that answers the query
- Include schema context in prompt for accurate code generation
- Validate code syntax before execution
- Handle complex queries (filtering, grouping, aggregations, joins)
- Generate code with proper error handling

**Input:** Natural language query, schema information, sample data
**Output:** Executable pandas code string

### 4. Code_Executor
Safely executes generated pandas code in a controlled environment. Prevents malicious operations and captures results or errors.

**Key Responsibilities:**
- Execute pandas code in a sandboxed environment
- Restrict dangerous operations (file system access, network calls)
- Capture execution results (DataFrames, Series, scalars)
- Handle errors gracefully and provide meaningful error messages
- Validate output types and formats
- Timeout long-running operations

**Input:** Python code string, DataFrame context
**Output:** Execution results (DataFrame/Series/scalar) or error message

### 5. Visualization_Generator
Creates appropriate visualizations based on data characteristics and user queries. Generates matplotlib/seaborn code and executes it to produce charts.

**Key Responsibilities:**
- Determine appropriate chart types based on data (histogram, scatter, bar, line, heatmap)
- Generate matplotlib/seaborn visualization code
- Execute visualization code safely
- Save charts to files (PNG, SVG)
- Handle different data types appropriately
- Create publication-quality visualizations

**Input:** DataFrame, visualization requirements, chart type preferences
**Output:** Visualization file paths, chart metadata

### 6. Data_Analysis_Agent
Main orchestrator class that coordinates all components and provides high-level interfaces for data analysis tasks.

**Key Responsibilities:**
- Coordinate all sub-components
- Provide Load_Data() method for initial data ingestion and profiling
- Provide Ask_Question() method for natural language queries
- Provide Generate_Insights() method for automatic insight discovery
- Provide Create_Report() method for comprehensive analysis reports
- Manage state and context across operations
- Handle errors and retries

**Input:** Data file path, user queries
**Output:** Analysis results, visualizations, reports

## Data Flow

1. **Initialization**: User provides a data file path to the agent
2. **Data Loading**: Data_Loader reads the CSV file and creates a DataFrame
3. **Schema Analysis**: Schema_Analyzer processes the DataFrame to understand structure and statistics
4. **Query Processing**: User submits natural language questions
5. **Code Generation**: Query_Generator converts questions to pandas code using LLM
6. **Code Execution**: Code_Executor runs the generated code safely
7. **Visualization**: If needed, Visualization_Generator creates charts
8. **Insight Generation**: Agent analyzes results and generates natural language insights
9. **Report Creation**: All findings are compiled into a comprehensive report

## Design Decisions

### Why LLM-based Query Generation?
Natural language to code conversion is a complex task that benefits from the contextual understanding and code generation capabilities of Large Language Models. This approach allows users to ask questions in plain English without needing to learn pandas syntax.

### Sandboxed Code Execution
Security is paramount when executing dynamically generated code. The Code_Executor uses restricted execution environments to prevent malicious operations like file system access, network calls, or system modifications.

### Schema-Aware Code Generation
By providing schema information to the LLM, the generated code is more accurate and handles edge cases better. The agent includes column names, types, and sample values in the prompt.

### Automatic Insight Discovery
The Generate_Insights() method uses statistical analysis and pattern recognition to automatically discover interesting patterns in the data, reducing the need for users to know what questions to ask.

### Modular Component Design
Each component is designed as a separate class with clear interfaces, making the system testable, maintainable, and extensible. Components can be swapped or enhanced independently.

### Visualization Type Selection
The system automatically selects appropriate chart types based on data characteristics (numeric vs categorical, number of dimensions, data distribution), ensuring visualizations are meaningful and informative.

## Prerequisites

### Software Dependencies
- Python 3.8 or higher
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Basic plotting
- seaborn: Statistical visualizations
- openai: LLM API access for code generation
- python-dotenv: Environment variable management

### API Requirements
- OpenAI API key (for GPT models) or compatible LLM API
- Sufficient API credits for code generation requests

### Data Requirements
- CSV or tabular data files
- Reasonably structured data (consistent columns, recognizable types)
- File size limits based on available memory

### System Requirements
- Sufficient RAM for data loading (depends on dataset size)
- Disk space for generated visualizations
- Network connectivity for API calls

## Extensions

### Multi-Format Support
Extend Data_Loader to support additional formats:
- Excel files (.xlsx, .xls)
- JSON files
- Parquet files
- Database connections (SQL, PostgreSQL, MySQL)
- API data sources

### Advanced Visualizations
Enhance Visualization_Generator with:
- Interactive visualizations (Plotly, Bokeh)
- Custom chart templates
- Multi-panel dashboards
- Animated visualizations for time series

### Query Optimization
Improve Query_Generator with:
- Query caching for repeated questions
- Query optimization suggestions
- Multi-step query planning
- Query explanation and documentation

### Collaborative Features
Add collaboration capabilities:
- Share analysis results
- Comment on insights
- Version control for analyses
- Team workspaces

### Advanced Analytics
Extend analysis capabilities:
- Machine learning model training and evaluation
- Time series forecasting
- Statistical hypothesis testing
- A/B testing analysis
- Predictive modeling

### Performance Optimization
Improve system performance:
- Lazy loading for large datasets
- Parallel processing for multiple queries
- Incremental analysis updates
- Caching of intermediate results

### Integration Capabilities
Connect with external systems:
- Database connectors
- Cloud storage integration (S3, GCS, Azure)
- BI tool integration (Tableau, Power BI)
- Export to various formats (PDF reports, Excel, HTML)

### Enhanced Security
Strengthen security features:
- User authentication and authorization
- Audit logging of all operations
- Data encryption at rest and in transit
- Compliance with data privacy regulations (GDPR, HIPAA)
