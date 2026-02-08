"""
Autonomous Web Agent Implementation
A complete autonomous web agent that navigates websites, extracts data,
fills forms, and completes web tasks using OpenAI + requests + BeautifulSoup.
"""

import os
import json
import time
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
from openai import OpenAI


@dataclass
class Web_Page:
    """Represents a web page with its content."""
    url: str
    html_content: str
    text_content: str
    title: str
    links: List[Dict[str, str]] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Web_Action:
    """Represents a planned web action."""
    action_type: str  # navigate, extract, fill_form, click, search
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


class Web_Browser:
    """HTTP-based web browser for fetching and interacting with web pages."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.cookies = {}
        self.history: List[str] = []
    
    def get(self, url: str, timeout: int = 10) -> Tuple[str, int]:
        """Fetch a web page using HTTP GET."""
        try:
            response = self.session.get(url, timeout=timeout, cookies=self.cookies)
            response.raise_for_status()
            self.history.append(url)
            self.cookies.update(self.session.cookies.get_dict())
            return response.text, response.status_code
        except requests.RequestException as e:
            raise Exception(f"Error fetching {url}: {str(e)}")
    
    def post(self, url: str, data: Dict[str, Any], timeout: int = 10) -> Tuple[str, int]:
        """Submit a form using HTTP POST."""
        try:
            response = self.session.post(url, data=data, timeout=timeout, cookies=self.cookies)
            response.raise_for_status()
            self.history.append(url)
            self.cookies.update(self.session.cookies.get_dict())
            return response.text, response.status_code
        except requests.RequestException as e:
            raise Exception(f"Error posting to {url}: {str(e)}")
    
    def extract_text(self, html_content: str) -> str:
        """Extract readable text from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    
    def extract_links(self, html_content: str, base_url: str) -> List[Dict[str, str]]:
        """Extract all links from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        links = []
        for tag in soup.find_all('a', href=True):
            href = tag['href']
            absolute_url = urljoin(base_url, href)
            text = tag.get_text(strip=True)
            links.append({
                'url': absolute_url,
                'text': text,
                'href': href
            })
        return links
    
    def extract_forms(self, html_content: str) -> List[Dict[str, Any]]:
        """Extract form information from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        forms = []
        for form in soup.find_all('form'):
            form_data = {
                'action': form.get('action', ''),
                'method': form.get('method', 'GET').upper(),
                'fields': []
            }
            for input_tag in form.find_all(['input', 'textarea', 'select']):
                field = {
                    'name': input_tag.get('name', ''),
                    'type': input_tag.get('type', 'text'),
                    'value': input_tag.get('value', ''),
                    'required': input_tag.has_attr('required'),
                    'tag': input_tag.name
                }
                if input_tag.name == 'textarea':
                    field['value'] = input_tag.get_text()
                elif input_tag.name == 'select':
                    options = [opt.get('value', opt.get_text()) for opt in input_tag.find_all('option')]
                    field['options'] = options
                form_data['fields'].append(field)
            forms.append(form_data)
        return forms
    
    def handle_cookies(self, cookies: Dict[str, str]):
        """Update session cookies."""
        self.cookies.update(cookies)
        self.session.cookies.update(cookies)


class Content_Extractor:
    """Extracts structured data from web pages."""
    
    def __init__(self):
        pass
    
    def parse_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML content into BeautifulSoup object."""
        return BeautifulSoup(html_content, 'lxml')
    
    def extract_tables(self, html_content: str) -> List[Dict[str, Any]]:
        """Extract data from HTML tables."""
        soup = BeautifulSoup(html_content, 'lxml')
        tables = []
        for table in soup.find_all('table'):
            table_data = {
                'headers': [],
                'rows': []
            }
            # Extract headers
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                table_data['headers'] = headers
            
            # Extract rows
            for row in table.find_all('tr')[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                if cells:
                    table_data['rows'].append(cells)
            tables.append(table_data)
        return tables
    
    def extract_forms(self, html_content: str) -> List[Dict[str, Any]]:
        """Extract form elements and their structure."""
        soup = BeautifulSoup(html_content, 'lxml')
        forms = []
        for form in soup.find_all('form'):
            form_info = {
                'action': form.get('action', ''),
                'method': form.get('method', 'GET'),
                'fields': []
            }
            for field in form.find_all(['input', 'textarea', 'select']):
                field_info = {
                    'name': field.get('name', ''),
                    'type': field.get('type', 'text'),
                    'label': self._find_label(form, field),
                    'required': field.has_attr('required'),
                    'placeholder': field.get('placeholder', '')
                }
                form_info['fields'].append(field_info)
            forms.append(form_info)
        return forms
    
    def _find_label(self, form: BeautifulSoup, field: Any) -> str:
        """Find the label associated with a form field."""
        field_id = field.get('id', '')
        field_name = field.get('name', '')
        if field_id:
            label = form.find('label', {'for': field_id})
            if label:
                return label.get_text(strip=True)
        # Try to find preceding label
        prev = field.find_previous(['label', 'p', 'div'])
        if prev and prev.name == 'label':
            return prev.get_text(strip=True)
        return ''
    
    def find_elements(self, html_content: str, search_text: str) -> List[Dict[str, Any]]:
        """Find elements containing specific text."""
        soup = BeautifulSoup(html_content, 'lxml')
        elements = []
        for tag in soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']):
            text = tag.get_text(strip=True)
            if search_text.lower() in text.lower():
                elements.append({
                    'tag': tag.name,
                    'text': text,
                    'attributes': dict(tag.attrs)
                })
        return elements
    
    def extract_structured_data(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract structured data from a web page."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else ''
        
        meta_description = soup.find('meta', {'name': 'description'})
        description = meta_description.get('content', '') if meta_description else ''
        
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        main_text = main_content.get_text(strip=True) if main_content else ''
        
        # Extract headings
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    'level': i,
                    'text': heading.get_text(strip=True)
                })
        
        return {
            'url': url,
            'title': title_text,
            'description': description,
            'headings': headings,
            'main_text': main_text[:1000],  # Limit length
            'tables': self.extract_tables(html_content),
            'forms': self.extract_forms(html_content)
        }
    
    def identify_navigation(self, html_content: str) -> Dict[str, Any]:
        """Identify navigation elements on the page."""
        soup = BeautifulSoup(html_content, 'lxml')
        nav_elements = {
            'menus': [],
            'breadcrumbs': [],
            'pagination': []
        }
        
        # Find navigation menus
        nav_tags = soup.find_all('nav')
        for nav in nav_tags:
            menu_items = []
            for link in nav.find_all('a'):
                menu_items.append({
                    'text': link.get_text(strip=True),
                    'href': link.get('href', '')
                })
            nav_elements['menus'].append(menu_items)
        
        # Find breadcrumbs
        breadcrumb = soup.find(class_=re.compile('breadcrumb', re.I))
        if breadcrumb:
            for link in breadcrumb.find_all('a'):
                nav_elements['breadcrumbs'].append({
                    'text': link.get_text(strip=True),
                    'href': link.get('href', '')
                })
        
        return nav_elements


class Action_Planner:
    """Plans sequences of web actions using LLM."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def plan_actions(self, task_description: str, current_state: Optional[Dict[str, Any]] = None) -> List[Web_Action]:
        """Plan a sequence of actions to complete the task."""
        state_context = ""
        if current_state:
            state_context = f"\nCurrent State:\n- Visited pages: {len(current_state.get('visited_pages', []))}\n- Extracted data items: {len(current_state.get('extracted_data', []))}"
        
        planning_prompt = f"""You are planning web actions to complete a task. Analyze the task and create a sequence of actions.

Task: {task_description}
{state_context}

Plan a sequence of web actions. Each action should be one of:
- navigate: Go to a URL
- extract: Extract specific data from a page
- fill_form: Fill out a form
- click: Click on a link or button
- search: Perform a search

Return your plan as a JSON array of actions:
[
    {{
        "action_type": "navigate|extract|fill_form|click|search",
        "target": "URL or element description",
        "description": "what this action accomplishes",
        "parameters": {{}}
    }}
]"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at planning web automation tasks. Create clear, actionable plans."},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            actions_data = result.get('actions', [])
            
            actions = []
            for action_data in actions_data:
                actions.append(Web_Action(
                    action_type=action_data.get('action_type', 'navigate'),
                    target=action_data.get('target', ''),
                    parameters=action_data.get('parameters', {}),
                    description=action_data.get('description', '')
                ))
            return actions
        except Exception as e:
            # Fallback plan
            return [Web_Action(
                action_type="navigate",
                target="",
                description=f"Complete task: {task_description}"
            )]
    
    def determine_navigation(self, target_description: str, available_links: List[Dict[str, str]]) -> Optional[str]:
        """Determine which link to follow based on description."""
        if not available_links:
            return None
        
        link_descriptions = "\n".join([f"- {link['text']}: {link['url']}" for link in available_links[:20]])
        
        prompt = f"""Given the target: "{target_description}"

Available links:
{link_descriptions}

Which link URL should be followed? Return only the URL."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at understanding web navigation. Select the most relevant link."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            selected_url = response.choices[0].message.content.strip()
            # Verify URL is in available links
            for link in available_links:
                if selected_url in link['url'] or link['url'] in selected_url:
                    return link['url']
            return available_links[0]['url'] if available_links else None
        except Exception:
            return available_links[0]['url'] if available_links else None
    
    def plan_form_filling(self, form_description: str, form_fields: List[Dict[str, Any]]) -> Dict[str, str]:
        """Plan how to fill a form based on context."""
        fields_info = "\n".join([f"- {field['name']} ({field['type']}): {field.get('label', '')}" for field in form_fields])
        
        prompt = f"""Given this form context: "{form_description}"

Form fields:
{fields_info}

Determine appropriate values for each field. Return JSON:
{{
    "field_name": "value",
    ...
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at filling forms intelligently. Provide appropriate values."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {field['name']: 'test_value' for field in form_fields if field['name']}


class Action_Executor:
    """Executes planned web actions."""
    
    def __init__(self, browser: Web_Browser, content_extractor: Content_Extractor):
        self.browser = browser
        self.content_extractor = content_extractor
    
    def execute_navigation(self, url: str) -> Web_Page:
        """Navigate to a URL and return page object."""
        html_content, status_code = self.browser.get(url)
        text_content = self.browser.extract_text(html_content)
        links = self.browser.extract_links(html_content, url)
        forms = self.browser.extract_forms(html_content)
        tables = self.content_extractor.extract_tables(html_content)
        
        soup = BeautifulSoup(html_content, 'lxml')
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else ''
        
        return Web_Page(
            url=url,
            html_content=html_content,
            text_content=text_content,
            title=title,
            links=links,
            forms=forms,
            tables=tables
        )
    
    def execute_extraction(self, page: Web_Page, extraction_target: str) -> Dict[str, Any]:
        """Extract specified data from a page."""
        structured_data = self.content_extractor.extract_structured_data(page.html_content, page.url)
        
        # Use LLM to identify relevant content if needed
        if extraction_target:
            elements = self.content_extractor.find_elements(page.html_content, extraction_target)
            structured_data['matched_elements'] = elements
        
        return structured_data
    
    def execute_form_fill(self, page: Web_Page, form_index: int, form_values: Dict[str, str]) -> Tuple[str, int]:
        """Fill and submit a form."""
        if form_index >= len(page.forms):
            raise ValueError(f"Form index {form_index} out of range")
        
        form = page.forms[form_index]
        form_action = form.get('action', page.url)
        form_method = form.get('method', 'GET').upper()
        form_url = urljoin(page.url, form_action)
        
        # Prepare form data
        form_data = {}
        for field in form.get('fields', []):
            field_name = field.get('name', '')
            if field_name in form_values:
                form_data[field_name] = form_values[field_name]
            elif field.get('type') == 'hidden':
                form_data[field_name] = field.get('value', '')
        
        if form_method == 'POST':
            html_content, status_code = self.browser.post(form_url, form_data)
        else:
            html_content, status_code = self.browser.get(f"{form_url}?{requests.compat.urlencode(form_data)}")
        
        return html_content, status_code


class State_Tracker:
    """Tracks state across web interactions."""
    
    def __init__(self):
        self.visited_pages: List[str] = []
        self.extracted_data: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.session_state: Dict[str, Any] = {}
    
    def track_visited_page(self, url: str):
        """Record a visited page."""
        if url not in self.visited_pages:
            self.visited_pages.append(url)
    
    def store_extracted_data(self, data: Dict[str, Any]):
        """Store extracted data."""
        self.extracted_data.append(data)
    
    def log_action(self, action: Web_Action, result: Optional[Dict[str, Any]] = None):
        """Log an executed action."""
        self.action_history.append({
            'action': action.action_type,
            'target': action.target,
            'description': action.description,
            'timestamp': time.time(),
            'result': result
        })
    
    def detect_loops(self) -> bool:
        """Detect if agent is stuck in a navigation loop."""
        if len(self.visited_pages) < 3:
            return False
        recent_pages = self.visited_pages[-5:]
        return len(recent_pages) != len(set(recent_pages))
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state."""
        return {
            'visited_pages': len(self.visited_pages),
            'extracted_data_items': len(self.extracted_data),
            'actions_performed': len(self.action_history),
            'in_loop': self.detect_loops()
        }


class Web_Agent:
    """Main autonomous web agent class."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.browser = Web_Browser()
        self.content_extractor = Content_Extractor()
        self.action_planner = Action_Planner(client)
        self.action_executor = Action_Executor(self.browser, self.content_extractor)
        self.state_tracker = State_Tracker()
    
    def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a web task from start to finish."""
        print(f"\nExecuting task: {task_description}\n")
        
        # Plan actions
        actions = self.action_planner.plan_actions(task_description, self.state_tracker.get_state_summary())
        print(f"Planned {len(actions)} actions\n")
        
        results = []
        for i, action in enumerate(actions, 1):
            print(f"Action {i}/{len(actions)}: {action.action_type} - {action.description}")
            
            try:
                if action.action_type == "navigate":
                    page = self.action_executor.execute_navigation(action.target)
                    self.state_tracker.track_visited_page(page.url)
                    results.append({
                        'action': action.description,
                        'url': page.url,
                        'title': page.title,
                        'status': 'success'
                    })
                
                elif action.action_type == "extract":
                    if not hasattr(self, '_current_page'):
                        continue
                    extracted = self.action_executor.execute_extraction(self._current_page, action.target)
                    self.state_tracker.store_extracted_data(extracted)
                    results.append({
                        'action': action.description,
                        'extracted_data': extracted,
                        'status': 'success'
                    })
                
                elif action.action_type == "fill_form":
                    if not hasattr(self, '_current_page'):
                        continue
                    form_values = self.action_planner.plan_form_filling(
                        action.description,
                        self._current_page.forms[0].get('fields', []) if self._current_page.forms else []
                    )
                    html_content, status_code = self.action_executor.execute_form_fill(
                        self._current_page, 0, form_values
                    )
                    results.append({
                        'action': action.description,
                        'status_code': status_code,
                        'status': 'success' if status_code == 200 else 'partial'
                    })
                
                self.state_tracker.log_action(action, {'status': 'success'})
            
            except Exception as e:
                print(f"  Error: {str(e)}")
                self.state_tracker.log_action(action, {'status': 'error', 'error': str(e)})
                results.append({
                    'action': action.description,
                    'status': 'error',
                    'error': str(e)
                })
        
        return {
            'task': task_description,
            'results': results,
            'state_summary': self.state_tracker.get_state_summary()
        }
    
    def navigate_and_extract(self, url: str, extraction_target: str = "") -> Dict[str, Any]:
        """Navigate to a URL and extract relevant information."""
        print(f"Navigating to: {url}")
        page = self.action_executor.execute_navigation(url)
        self._current_page = page
        self.state_tracker.track_visited_page(url)
        
        print(f"Page title: {page.title}")
        print(f"Found {len(page.links)} links, {len(page.forms)} forms, {len(page.tables)} tables")
        
        extracted_data = self.action_executor.execute_extraction(page, extraction_target)
        self.state_tracker.store_extracted_data(extracted_data)
        
        return {
            'url': url,
            'title': page.title,
            'extracted_data': extracted_data,
            'links_count': len(page.links),
            'forms_count': len(page.forms)
        }
    
    def search_and_summarize(self, search_query: str, num_results: int = 5) -> Dict[str, Any]:
        """Perform a web search and summarize results."""
        # For demonstration, we'll simulate search results
        # In production, integrate with search APIs (Google, Bing, etc.)
        print(f"Searching for: {search_query}")
        
        summary_prompt = f"""You are summarizing web search results. Provide a comprehensive summary.

Search query: {search_query}
Number of results requested: {num_results}

Provide a summary that includes:
1. Key findings from the search
2. Important information relevant to the query
3. Main points and insights
4. Relevant sources and references"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing web search results."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            summary = response.choices[0].message.content
            
            return {
                'query': search_query,
                'summary': summary,
                'num_results': num_results
            }
        except Exception as e:
            return {
                'query': search_query,
                'error': str(e)
            }
    
    def multi_page_extraction(self, start_url: str, extraction_target: str, max_pages: int = 5) -> Dict[str, Any]:
        """Follow links and extract data across multiple pages."""
        print(f"Starting multi-page extraction from: {start_url}")
        
        # Navigate to start page
        page = self.action_executor.execute_navigation(start_url)
        self.state_tracker.track_visited_page(start_url)
        
        all_extracted_data = []
        pages_visited = [start_url]
        
        # Extract from start page
        extracted = self.action_executor.execute_extraction(page, extraction_target)
        all_extracted_data.append(extracted)
        
        # Follow links and extract
        for i, link in enumerate(page.links[:max_pages-1]):
            if link['url'] in self.state_tracker.visited_pages:
                continue
            
            try:
                next_page = self.action_executor.execute_navigation(link['url'])
                self.state_tracker.track_visited_page(link['url'])
                extracted = self.action_executor.execute_extraction(next_page, extraction_target)
                all_extracted_data.append(extracted)
                pages_visited.append(link['url'])
                
                if len(pages_visited) >= max_pages:
                    break
            except Exception as e:
                print(f"Error extracting from {link['url']}: {str(e)}")
                continue
        
        return {
            'start_url': start_url,
            'pages_visited': pages_visited,
            'extracted_data': all_extracted_data,
            'total_items': len(all_extracted_data)
        }


def main():
    """Main function demonstrating the autonomous web agent."""
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it using: export OPENAI_API_KEY='your-key-here'")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Create web agent
    agent = Web_Agent(client)
    
    # Example 1: Navigate and extract
    print("="*60)
    print("Example 1: Navigate and Extract")
    print("="*60)
    result1 = agent.navigate_and_extract(
        "https://example.com",
        "main content"
    )
    print(f"\nExtracted data keys: {list(result1['extracted_data'].keys())}")
    
    # Example 2: Search and summarize
    print("\n" + "="*60)
    print("Example 2: Search and Summarize")
    print("="*60)
    result2 = agent.search_and_summarize(
        "Python web frameworks comparison",
        num_results=5
    )
    print(f"\nSummary:\n{result2.get('summary', 'No summary')[:500]}...")
    
    # Example 3: Execute complex task
    print("\n" + "="*60)
    print("Example 3: Execute Complex Task")
    print("="*60)
    task_result = agent.execute_task(
        "Navigate to example.com, extract the main heading, and summarize the page content"
    )
    print(f"\nTask completed: {len(task_result['results'])} actions executed")
    print(f"State summary: {task_result['state_summary']}")


if __name__ == "__main__":
    main()
