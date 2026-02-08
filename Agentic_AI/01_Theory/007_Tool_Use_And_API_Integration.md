# Tool Use and API Integration

## Table of Contents

1. Why Agents Need Tools
2. Function Calling Fundamentals
3. Tool Types and Categories
4. Building Custom Tools
5. Tool Discovery and Selection
6. API Integration Patterns
7. Database Agents
8. Web Browsing Agents
9. Code Execution
10. Tool Composition and Chaining
11. Safety and Security

---

## 1. Why Agents Need Tools

### LLM Limitations

Large Language Models are powerful reasoning engines but have fundamental limitations that tools solve:

| Limitation | Description | Tool Solution |
|-----------|-------------|---------------|
| No real-time data | Training data has a cutoff date | Web search, API calls |
| No computation | Cannot reliably do math beyond basic | Calculator, code execution |
| No side effects | Cannot send emails, update databases | Action tools |
| No file access | Cannot read/write local files | File system tools |
| No verification | Cannot verify its own claims | Fact-checking tools |
| Hallucination risk | May generate false information | RAG, knowledge base tools |

### The Tool-Use Paradigm

```
WITHOUT TOOLS:
User: "What's the weather in London right now?"
LLM: "I don't have access to real-time weather data..." (or hallucinate)

WITH TOOLS:
User: "What's the weather in London right now?"
LLM: [Decides to use weather_tool]
     [Calls: get_weather(city="London")]
     [Receives: {"temp": 15, "condition": "cloudy"}]
LLM: "The current weather in London is 15C and cloudy."
```

### Tool-Use Flow

```
+--------+     +----------+     +------------+     +----------+
| User   | --> | LLM      | --> | Tool       | --> | LLM      |
| Query  |     | Decides  |     | Execution  |     | Uses     |
|        |     | which    |     | (external  |     | result   |
|        |     | tool     |     |  action)   |     | to       |
|        |     |          |     |            |     | respond  |
+--------+     +----------+     +------------+     +----------+
```

---

## 2. Function Calling Fundamentals

### How Function Calling Works

1. You define tools with JSON schemas (name, description, parameters)
2. Send tools along with the user message to the LLM
3. LLM analyzes the query and decides if a tool is needed
4. If yes, LLM outputs a structured function call (not the result)
5. Your code executes the function
6. You send the result back to the LLM
7. LLM generates the final response using the result

### OpenAI Function Calling

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "Get_Weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g., London, New York"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Search_Web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def Run_Agent(user_message):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            result = Execute_Function(function_name, arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        # Get final response with tool results
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        return final_response.choices[0].message.content

    return message.content

def Execute_Function(name, args):
    functions = {
        "Get_Weather": Get_Weather,
        "Search_Web": Search_Web,
    }
    func = functions.get(name)
    if func:
        return func(**args)
    return {"error": f"Unknown function: {name}"}
```

### Anthropic Tool Use

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "Get_Weather",
        "description": "Get the current weather for a given city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            "required": ["city"]
        }
    }
]

def Run_Claude_Agent(user_message):
    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a helpful assistant with tool access.",
        tools=tools,
        messages=messages
    )

    if response.stop_reason == "tool_use":
        tool_use_block = next(
            block for block in response.content if block.type == "tool_use"
        )

        tool_name = tool_use_block.name
        tool_input = tool_use_block.input
        tool_id = tool_use_block.id

        result = Execute_Function(tool_name, tool_input)

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps(result)
            }]
        })

        final = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="You are a helpful assistant with tool access.",
            tools=tools,
            messages=messages
        )
        return final.content[0].text

    return response.content[0].text
```

### Parallel Function Calling

Some models support calling multiple tools simultaneously:

```python
# OpenAI parallel tool calls
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "What's the weather in London AND the stock price of AAPL?"
    }],
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls=True  # Enable parallel calls
)

# response.choices[0].message.tool_calls may contain multiple calls:
# [
#   {"id": "call_1", "function": {"name": "Get_Weather", "arguments": '{"city":"London"}'}},
#   {"id": "call_2", "function": {"name": "Get_Stock_Price", "arguments": '{"symbol":"AAPL"}'}}
# ]

# Execute all in parallel
import asyncio

async def Execute_Parallel(tool_calls):
    tasks = []
    for call in tool_calls:
        name = call.function.name
        args = json.loads(call.function.arguments)
        tasks.append(Execute_Function_Async(name, args))

    results = await asyncio.gather(*tasks)
    return dict(zip([c.id for c in tool_calls], results))
```

---

## 3. Tool Types and Categories

### Information Retrieval Tools

```python
def Search_Web(query, num_results=5):
    """Search the web for current information."""
    # Integration with search APIs (Tavily, SerpAPI, Google)
    pass

def Query_Database(sql_query, database="main"):
    """Execute a SQL query against the database."""
    pass

def Read_Document(file_path, format="auto"):
    """Read and extract text from a document."""
    pass

def Fetch_URL(url, extract_text=True):
    """Fetch content from a URL."""
    pass
```

### Computation Tools

```python
def Calculate(expression):
    """Safely evaluate a mathematical expression."""
    import ast
    try:
        tree = ast.parse(expression, mode="eval")
        # Only allow safe operations
        result = eval(compile(tree, "<string>", "eval"))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

def Execute_Python(code, timeout=30):
    """Execute Python code in a sandboxed environment."""
    pass

def Analyze_Data(data, analysis_type):
    """Perform statistical analysis on data."""
    pass
```

### Action Tools

```python
def Send_Email(to, subject, body, cc=None):
    """Send an email to the specified recipient."""
    pass

def Create_File(path, content, format="text"):
    """Create a file with the given content."""
    pass

def Update_Database(table, data, condition):
    """Update records in a database table."""
    pass

def Create_Calendar_Event(title, start, end, attendees=None):
    """Create a calendar event."""
    pass

def Send_Slack_Message(channel, message, thread_ts=None):
    """Post a message to a Slack channel."""
    pass
```

### Tool Category Matrix

| Category | Examples | Risk Level | Requires Approval |
|----------|---------|------------|-------------------|
| Read-only queries | Search, database SELECT, file read | Low | No |
| Computation | Calculator, code execution (sandbox) | Low | No |
| External reads | API calls, web scraping | Medium | Sometimes |
| Write operations | File write, database UPDATE/INSERT | High | Yes |
| Communication | Email, Slack, notifications | High | Yes |
| Destructive | File delete, database DELETE/DROP | Critical | Always |
| Financial | Payments, transfers | Critical | Always |

---

## 4. Building Custom Tools

### Tool Interface Design

```python
from typing import Any
from pydantic import BaseModel, Field

class Tool_Input(BaseModel):
    """Base class for tool input schemas."""
    pass

class Tool_Output(BaseModel):
    """Base class for tool output."""
    success: bool
    data: Any = None
    error: str = None

class Base_Tool:
    name: str
    description: str
    input_schema: type  # Pydantic model

    def Execute(self, **kwargs) -> Tool_Output:
        raise NotImplementedError

    def To_Function_Schema(self):
        """Convert to OpenAI function calling format."""
        schema = self.input_schema.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema
            }
        }
```

### Practical Tool Example: Web Search

```python
import requests

class Web_Search_Input(Tool_Input):
    query: str = Field(description="The search query")
    num_results: int = Field(default=5, description="Number of results")

class Web_Search_Tool(Base_Tool):
    name = "Search_Web"
    description = "Search the web for current information on any topic"
    input_schema = Web_Search_Input

    def __init__(self, api_key):
        self.api_key = api_key

    def Execute(self, query, num_results=5):
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": num_results,
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            results = [
                {
                    "title": r["title"],
                    "url": r["url"],
                    "snippet": r["content"][:200],
                }
                for r in data.get("results", [])
            ]

            return Tool_Output(success=True, data=results)

        except requests.RequestException as e:
            return Tool_Output(success=False, error=str(e))
```

### Tool with Retry Logic

```python
import time

class Retry_Tool_Wrapper:
    def __init__(self, tool, max_retries=3, backoff_factor=2):
        self.tool = tool
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def Execute(self, **kwargs):
        last_error = None
        for attempt in range(self.max_retries):
            result = self.tool.Execute(**kwargs)
            if result.success:
                return result

            last_error = result.error
            if attempt < self.max_retries - 1:
                wait = self.backoff_factor ** attempt
                time.sleep(wait)

        return Tool_Output(
            success=False,
            error=f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        )
```

### LangChain-Style Tool Decorator

```python
from functools import wraps

def Agent_Tool(name=None, description=None):
    """Decorator to create agent tools from functions."""
    def decorator(func):
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return Tool_Output(success=True, data=result)
            except Exception as e:
                return Tool_Output(success=False, error=str(e))

        wrapper.tool_name = tool_name
        wrapper.tool_description = tool_description
        wrapper.is_tool = True
        return wrapper
    return decorator

# Usage
@Agent_Tool(name="Calculate_BMI", description="Calculate Body Mass Index")
def Calculate_BMI(weight_kg: float, height_m: float):
    """Calculate BMI from weight (kg) and height (m)."""
    bmi = weight_kg / (height_m ** 2)
    category = (
        "Underweight" if bmi < 18.5
        else "Normal" if bmi < 25
        else "Overweight" if bmi < 30
        else "Obese"
    )
    return {"bmi": round(bmi, 1), "category": category}
```

---

## 5. Tool Discovery and Selection

### Static Tool Registration

```python
class Tool_Registry:
    def __init__(self):
        self.tools = {}

    def Register(self, tool):
        self.tools[tool.name] = tool

    def Get(self, name):
        return self.tools.get(name)

    def List_All(self):
        return [
            {"name": t.name, "description": t.description}
            for t in self.tools.values()
        ]

    def Get_Schemas(self):
        return [t.To_Function_Schema() for t in self.tools.values()]

    def Execute(self, tool_name, **kwargs):
        tool = self.Get(tool_name)
        if not tool:
            return Tool_Output(success=False, error=f"Unknown tool: {tool_name}")
        return tool.Execute(**kwargs)
```

### Dynamic Tool Loading

```python
import importlib
import os

class Dynamic_Tool_Loader:
    def __init__(self, tool_directory="tools"):
        self.tool_dir = tool_directory
        self.loaded_tools = {}

    def Load_All(self):
        for filename in os.listdir(self.tool_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = filename[:-3]
                self.Load_Module(module_name)

    def Load_Module(self, module_name):
        try:
            module = importlib.import_module(f"{self.tool_dir}.{module_name}")

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, "is_tool") and attr.is_tool:
                    self.loaded_tools[attr.tool_name] = attr

        except ImportError as e:
            print(f"Failed to load tool module {module_name}: {e}")

    def Get_Tool(self, name):
        return self.loaded_tools.get(name)

    def Get_All_Schemas(self):
        schemas = []
        for name, tool in self.loaded_tools.items():
            schemas.append({
                "name": name,
                "description": tool.tool_description,
            })
        return schemas
```

### Managing Large Tool Sets (100+ tools)

When an agent has access to many tools, include only relevant ones in each LLM call.

```python
class Tool_Selector:
    def __init__(self, all_tools, embedding_model):
        self.tools = all_tools
        self.embedding_model = embedding_model
        self.tool_embeddings = self.Build_Tool_Embeddings()

    def Build_Tool_Embeddings(self):
        embeddings = {}
        for tool in self.tools:
            text = f"{tool.name}: {tool.description}"
            embeddings[tool.name] = self.embedding_model.embed(text)
        return embeddings

    def Select_Relevant_Tools(self, user_query, max_tools=10):
        query_embedding = self.embedding_model.embed(user_query)

        scored = []
        for name, tool_emb in self.tool_embeddings.items():
            similarity = self.Cosine_Similarity(query_embedding, tool_emb)
            scored.append((similarity, name))

        scored.sort(reverse=True)
        selected_names = [name for _, name in scored[:max_tools]]

        return [t for t in self.tools if t.name in selected_names]

    def Cosine_Similarity(self, a, b):
        import numpy as np
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

---

## 6. API Integration Patterns

### REST API Integration

```python
import requests
from typing import Optional

class REST_API_Tool:
    def __init__(self, base_url, api_key=None, headers=None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            self.session.headers.update(headers)

    def Get(self, endpoint, params=None, timeout=30):
        try:
            response = self.session.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=timeout
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def Post(self, endpoint, data=None, json_data=None, timeout=30):
        try:
            response = self.session.post(
                f"{self.base_url}/{endpoint}",
                data=data,
                json=json_data,
                timeout=timeout
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def Put(self, endpoint, json_data=None, timeout=30):
        try:
            response = self.session.put(
                f"{self.base_url}/{endpoint}",
                json=json_data,
                timeout=timeout
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def Delete(self, endpoint, timeout=30):
        try:
            response = self.session.delete(
                f"{self.base_url}/{endpoint}",
                timeout=timeout
            )
            response.raise_for_status()
            return {"success": True}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}
```

### Rate Limiting

```python
import time
from collections import deque

class Rate_Limiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests = deque()

    def Wait_If_Needed(self):
        now = time.time()

        # Remove expired timestamps
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()

        if len(self.requests) >= self.max_requests:
            wait_time = self.requests[0] + self.time_window - now
            if wait_time > 0:
                time.sleep(wait_time)

        self.requests.append(time.time())

class Rate_Limited_API(REST_API_Tool):
    def __init__(self, base_url, api_key=None, rate_limit=60, window=60):
        super().__init__(base_url, api_key)
        self.limiter = Rate_Limiter(rate_limit, window)

    def Get(self, endpoint, params=None, timeout=30):
        self.limiter.Wait_If_Needed()
        return super().Get(endpoint, params, timeout)

    def Post(self, endpoint, data=None, json_data=None, timeout=30):
        self.limiter.Wait_If_Needed()
        return super().Post(endpoint, data, json_data, timeout)
```

### Authentication Patterns

```python
class Auth_Manager:
    """Handles various authentication methods for API integrations."""

    @staticmethod
    def API_Key_Auth(api_key, header_name="X-API-Key"):
        return {header_name: api_key}

    @staticmethod
    def Bearer_Token_Auth(token):
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def OAuth2_Client_Credentials(client_id, client_secret, token_url):
        response = requests.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def Basic_Auth(username, password):
        import base64
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {credentials}"}
```

---

## 7. Database Agents

### Text-to-SQL Pattern

```python
class Database_Agent:
    def __init__(self, llm, db_connection, schema_info):
        self.llm = llm
        self.db = db_connection
        self.schema = schema_info

    def Query(self, natural_language_query):
        # Step 1: Generate SQL
        sql = self.Generate_SQL(natural_language_query)

        # Step 2: Validate SQL
        is_safe, reason = self.Validate_SQL(sql)
        if not is_safe:
            return {"error": f"Unsafe query: {reason}"}

        # Step 3: Execute
        try:
            results = self.db.execute(sql).fetchall()
            columns = [desc[0] for desc in self.db.description]
        except Exception as e:
            # Step 3b: Self-correct on error
            sql = self.Fix_SQL(sql, str(e))
            results = self.db.execute(sql).fetchall()
            columns = [desc[0] for desc in self.db.description]

        # Step 4: Format results
        formatted = self.Format_Results(columns, results)

        # Step 5: Generate natural language answer
        answer = self.llm.generate(f"""
        User question: {natural_language_query}
        SQL executed: {sql}
        Results: {formatted}

        Provide a natural language answer based on the results.
        """)

        return {"sql": sql, "results": formatted, "answer": answer}

    def Generate_SQL(self, query):
        return self.llm.generate(f"""
        Database schema:
        {self.schema}

        User question: {query}

        Generate a SQL query to answer this question.
        Return only the SQL query, nothing else.
        Use proper SQL syntax. Do not use SELECT *.
        """).strip()

    def Validate_SQL(self, sql):
        sql_upper = sql.upper().strip()

        # Block dangerous operations
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "EXEC"]
        for keyword in dangerous:
            if keyword in sql_upper:
                return False, f"Contains dangerous keyword: {keyword}"

        # Must be a SELECT query
        if not sql_upper.startswith("SELECT"):
            return False, "Only SELECT queries are allowed"

        # Block subqueries that modify data
        if "INTO" in sql_upper:
            return False, "SELECT INTO not allowed"

        return True, "OK"

    def Fix_SQL(self, original_sql, error_message):
        return self.llm.generate(f"""
        The following SQL query produced an error:
        SQL: {original_sql}
        Error: {error_message}
        Schema: {self.schema}

        Fix the SQL query. Return only the corrected SQL.
        """).strip()

    def Format_Results(self, columns, rows):
        if not rows:
            return "No results found."

        result = []
        for row in rows[:50]:  # Limit to 50 rows
            result.append(dict(zip(columns, row)))
        return json.dumps(result, indent=2, default=str)
```

### Schema Introspection

```python
class Schema_Inspector:
    def __init__(self, db_connection):
        self.db = db_connection

    def Get_Schema_Info(self):
        """Extract database schema for the LLM."""
        tables = self.Get_Tables()
        schema_parts = []

        for table in tables:
            columns = self.Get_Columns(table)
            sample = self.Get_Sample_Data(table, limit=3)

            schema_parts.append(f"""
Table: {table}
Columns:
{self.Format_Columns(columns)}
Sample data:
{sample}
""")

        return "\n".join(schema_parts)

    def Get_Tables(self):
        rows = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return [r[0] for r in rows]

    def Get_Columns(self, table):
        rows = self.db.execute(f"PRAGMA table_info({table})").fetchall()
        return [{"name": r[1], "type": r[2], "nullable": not r[3]} for r in rows]

    def Format_Columns(self, columns):
        return "\n".join(
            f"  - {c['name']} ({c['type']}, {'nullable' if c['nullable'] else 'not null'})"
            for c in columns
        )

    def Get_Sample_Data(self, table, limit=3):
        rows = self.db.execute(f"SELECT * FROM {table} LIMIT {limit}").fetchall()
        return str(rows)
```

---

## 8. Web Browsing Agents

### Web Scraping Tool

```python
import requests
from bs4 import BeautifulSoup

class Web_Scraper_Tool:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"
        })

    def Fetch_Page(self, url, extract_text=True):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            if extract_text:
                return self.Extract_Text(response.text)
            return response.text

        except requests.RequestException as e:
            return {"error": str(e)}

    def Extract_Text(self, html):
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def Extract_Links(self, url):
        try:
            response = self.session.get(url, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")

            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(strip=True)
                if href.startswith("http"):
                    links.append({"url": href, "text": text})

            return links

        except requests.RequestException as e:
            return {"error": str(e)}

    def Extract_Tables(self, url):
        try:
            response = self.session.get(url, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")

            tables = []
            for table in soup.find_all("table"):
                rows = []
                for tr in table.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    rows.append(cells)
                tables.append(rows)

            return tables

        except requests.RequestException as e:
            return {"error": str(e)}
```

### Browser Automation

```python
# Using Playwright for browser automation
from playwright.sync_api import sync_playwright

class Browser_Agent_Tool:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None

    def Start(self, headless=True):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page()

    def Navigate(self, url):
        self.page.goto(url, wait_until="networkidle")
        return {
            "title": self.page.title(),
            "url": self.page.url,
        }

    def Click(self, selector):
        self.page.click(selector)
        self.page.wait_for_load_state("networkidle")

    def Type_Text(self, selector, text):
        self.page.fill(selector, text)

    def Get_Text(self, selector=None):
        if selector:
            element = self.page.query_selector(selector)
            return element.text_content() if element else None
        return self.page.text_content("body")

    def Screenshot(self, path="screenshot.png"):
        self.page.screenshot(path=path)
        return path

    def Close(self):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
```

---

## 9. Code Execution

### Sandboxed Python Execution

```python
import subprocess
import tempfile
import os

class Code_Executor:
    def __init__(self, timeout=30, max_output_size=10000):
        self.timeout = timeout
        self.max_output_size = max_output_size

    def Execute_Python(self, code):
        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=self.Get_Safe_Env(),
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:self.max_output_size],
                "stderr": result.stderr[:self.max_output_size],
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timed out after {self.timeout} seconds",
            }

        finally:
            os.unlink(temp_path)

    def Get_Safe_Env(self):
        """Create a restricted environment."""
        safe_env = os.environ.copy()
        # Remove sensitive environment variables
        for key in list(safe_env.keys()):
            if any(s in key.upper() for s in ["SECRET", "KEY", "TOKEN", "PASSWORD"]):
                del safe_env[key]
        return safe_env

    def Validate_Code(self, code):
        """Basic safety checks before execution."""
        dangerous_patterns = [
            "os.system", "subprocess", "eval(", "exec(",
            "__import__", "open(", "shutil.rmtree",
            "os.remove", "os.unlink",
        ]
        for pattern in dangerous_patterns:
            if pattern in code:
                return False, f"Blocked pattern: {pattern}"
        return True, "OK"
```

### Docker-Based Execution

```python
import docker

class Docker_Code_Executor:
    def __init__(self, image="python:3.11-slim", timeout=30, memory_limit="256m"):
        self.client = docker.from_env()
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit

    def Execute(self, code, language="python"):
        try:
            container = self.client.containers.run(
                self.image,
                command=f"python -c '{code}'",
                detach=True,
                mem_limit=self.memory_limit,
                network_disabled=True,  # No network access
                read_only=True,         # Read-only filesystem
            )

            result = container.wait(timeout=self.timeout)
            logs = container.logs().decode("utf-8")
            container.remove()

            return {
                "success": result["StatusCode"] == 0,
                "output": logs,
                "exit_code": result["StatusCode"],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
```

---

## 10. Tool Composition and Chaining

### Sequential Tool Chaining

```python
class Tool_Chain:
    def __init__(self, steps):
        self.steps = steps  # List of (tool, transform_fn) tuples

    def Execute(self, initial_input):
        current_data = initial_input

        for i, (tool, transform) in enumerate(self.steps):
            # Transform input for this tool
            tool_input = transform(current_data) if transform else current_data

            # Execute tool
            result = tool.Execute(**tool_input)

            if not result.success:
                return Tool_Output(
                    success=False,
                    error=f"Step {i+1} ({tool.name}) failed: {result.error}"
                )

            current_data = result.data

        return Tool_Output(success=True, data=current_data)

# Usage
chain = Tool_Chain([
    (search_tool, lambda q: {"query": q}),
    (summarize_tool, lambda results: {"text": str(results)}),
    (translate_tool, lambda summary: {"text": summary, "target": "fr"}),
])

result = chain.Execute("latest AI research papers 2025")
```

### Parallel Tool Execution

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class Parallel_Tool_Executor:
    def __init__(self, max_workers=5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def Execute_Parallel(self, tool_calls):
        """Execute multiple tool calls in parallel."""
        futures = {}
        for call_id, (tool, kwargs) in tool_calls.items():
            future = self.executor.submit(tool.Execute, **kwargs)
            futures[call_id] = future

        results = {}
        for call_id, future in futures.items():
            try:
                results[call_id] = future.result(timeout=30)
            except Exception as e:
                results[call_id] = Tool_Output(success=False, error=str(e))

        return results

# Usage
executor = Parallel_Tool_Executor()

results = executor.Execute_Parallel({
    "weather": (weather_tool, {"city": "London"}),
    "news": (news_tool, {"topic": "technology"}),
    "stocks": (stock_tool, {"symbol": "AAPL"}),
})
```

### Conditional Tool Selection

```python
class Conditional_Tool_Router:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {t.name: t for t in tools}

    def Route_And_Execute(self, query):
        # Ask LLM which tool(s) to use
        tool_selection = self.llm.generate(f"""
        Query: {query}
        Available tools:
        {json.dumps([{"name": t.name, "description": t.description} for t in self.tools.values()])}

        Which tool(s) should be used? Return JSON:
        {{"tools": [{{"name": "...", "arguments": {{...}}}}]}}
        """)

        selections = json.loads(tool_selection)

        results = []
        for sel in selections["tools"]:
            tool = self.tools.get(sel["name"])
            if tool:
                result = tool.Execute(**sel["arguments"])
                results.append({"tool": sel["name"], "result": result})

        return results
```

---

## 11. Safety and Security

### Input Validation

```python
class Tool_Input_Validator:
    def __init__(self):
        self.rules = {}

    def Add_Rule(self, tool_name, param_name, validator_fn, error_msg):
        key = f"{tool_name}.{param_name}"
        self.rules[key] = {"validator": validator_fn, "error": error_msg}

    def Validate(self, tool_name, params):
        errors = []
        for param_name, value in params.items():
            key = f"{tool_name}.{param_name}"
            if key in self.rules:
                rule = self.rules[key]
                if not rule["validator"](value):
                    errors.append(f"{param_name}: {rule['error']}")

        return len(errors) == 0, errors

# Usage
validator = Tool_Input_Validator()
validator.Add_Rule(
    "Query_Database", "sql",
    lambda sql: "DROP" not in sql.upper(),
    "DROP statements are not allowed"
)
validator.Add_Rule(
    "Send_Email", "to",
    lambda email: "@" in email and "." in email.split("@")[1],
    "Invalid email format"
)
```

### Permission System

```python
class Tool_Permission_Manager:
    def __init__(self):
        self.permissions = {}
        self.approval_callbacks = {}

    def Set_Permission(self, tool_name, level):
        """
        Levels:
        - 'auto': Execute without approval
        - 'notify': Execute and notify
        - 'approve': Require human approval before execution
        - 'blocked': Never execute
        """
        self.permissions[tool_name] = level

    def Check_Permission(self, tool_name, params):
        level = self.permissions.get(tool_name, "approve")

        if level == "blocked":
            return False, "Tool is blocked"

        if level == "auto":
            return True, "Auto-approved"

        if level == "notify":
            self.Notify(tool_name, params)
            return True, "Executed with notification"

        if level == "approve":
            approved = self.Request_Approval(tool_name, params)
            return approved, "Approved" if approved else "Denied by user"

        return False, "Unknown permission level"

    def Request_Approval(self, tool_name, params):
        callback = self.approval_callbacks.get("default")
        if callback:
            return callback(tool_name, params)
        return False

    def Notify(self, tool_name, params):
        print(f"[NOTIFICATION] Tool executed: {tool_name}, params: {params}")

# Usage
permissions = Tool_Permission_Manager()
permissions.Set_Permission("Search_Web", "auto")
permissions.Set_Permission("Read_File", "auto")
permissions.Set_Permission("Send_Email", "approve")
permissions.Set_Permission("Delete_File", "blocked")
permissions.Set_Permission("Query_Database", "notify")
```

### Audit Logging

```python
import logging
from datetime import datetime

class Tool_Audit_Logger:
    def __init__(self, log_file="tool_audit.log"):
        self.logger = logging.getLogger("tool_audit")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def Log_Execution(self, tool_name, params, result, agent_id=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "params": self.Sanitize_Params(params),
            "success": result.success,
            "agent": agent_id,
        }

        if result.success:
            self.logger.info(json.dumps(entry))
        else:
            entry["error"] = result.error
            self.logger.warning(json.dumps(entry))

    def Sanitize_Params(self, params):
        """Remove sensitive values from logged parameters."""
        sanitized = {}
        sensitive_keys = {"password", "token", "api_key", "secret", "credit_card"}

        for key, value in params.items():
            if key.lower() in sensitive_keys:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value

        return sanitized
```

### Secure Tool Execution Wrapper

```python
class Secure_Tool_Executor:
    def __init__(self, registry, permissions, validator, audit_logger):
        self.registry = registry
        self.permissions = permissions
        self.validator = validator
        self.audit = audit_logger

    def Execute(self, tool_name, params, agent_id=None):
        # Step 1: Check permissions
        allowed, reason = self.permissions.Check_Permission(tool_name, params)
        if not allowed:
            self.audit.Log_Execution(
                tool_name, params,
                Tool_Output(success=False, error=f"Permission denied: {reason}"),
                agent_id
            )
            return Tool_Output(success=False, error=f"Permission denied: {reason}")

        # Step 2: Validate inputs
        valid, errors = self.validator.Validate(tool_name, params)
        if not valid:
            self.audit.Log_Execution(
                tool_name, params,
                Tool_Output(success=False, error=f"Validation failed: {errors}"),
                agent_id
            )
            return Tool_Output(success=False, error=f"Validation failed: {errors}")

        # Step 3: Execute
        tool = self.registry.Get(tool_name)
        result = tool.Execute(**params)

        # Step 4: Log
        self.audit.Log_Execution(tool_name, params, result, agent_id)

        return result
```

---

## Summary

Tool use transforms agents from passive text generators into active entities that can interact with the real world. Key principles:

1. **Design tools with clear schemas**: Well-defined inputs, outputs, and descriptions help the LLM use tools correctly
2. **Validate everything**: Check inputs before execution, validate outputs after
3. **Handle errors gracefully**: Retry logic, fallbacks, and informative error messages
4. **Secure by default**: Permission systems, audit logging, input sanitization
5. **Optimize tool selection**: For large tool sets, use semantic matching to select relevant tools
6. **Compose tools**: Chain tools sequentially, run in parallel, or route conditionally
7. **Rate limit external calls**: Protect against API quota exhaustion
8. **Sandbox code execution**: Never run untrusted code without isolation
9. **Log all tool interactions**: Essential for debugging, compliance, and improvement
10. **Start with read-only tools**: Add write capabilities gradually with proper safeguards
