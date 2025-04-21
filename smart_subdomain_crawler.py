
#!/usr/bin/env python3
"""
Smart Subdomain Web Crawler & Scraper
-------------------------------------
This script intelligently crawls website subdomains, using an LLM to automatically
generate selectors for data extraction, and outputs structured data in JSONL format
for LLM/RAG applications.
"""

import argparse
import json
import logging
import os
import random
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

# Import dotenv for environment variable support
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables from .env files won't be loaded.")
    def load_dotenv(*args, **kwargs):
        pass

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

try:
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("Warning: webdriver-manager not installed. You'll need to provide the path to chromedriver manually.")
    ChromeDriverManager = None

# Load environment variables from .env files
def load_environment():
    """Load environment variables from .env files"""
    # Try to find .env file in current directory or parent directories
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try parent directory
        parent_env = Path('..') / '.env'
        if parent_env.exists():
            load_dotenv(parent_env)
            
    # Also check for .env.local which overrides .env
    local_env = Path('.env.local')
    if local_env.exists():
        load_dotenv(local_env, override=True)

# Call this function early on
load_environment()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CrawlConfig:
    """Data class for crawler configuration"""
    start_url: str  # Starting URL for crawling
    max_pages: int = 100  # Maximum number of pages to crawl
    max_depth: int = 3  # Maximum link depth to crawl
    stay_on_subdomains: bool = True  # Stay on the same domain and subdomains
    respect_robots: bool = True  # Respect robots.txt
    delay_range: List[float] = field(default_factory=lambda: [1.0, 3.0])  # Delay between requests
    use_selenium: bool = False  # Use Selenium for JavaScript-heavy sites
    output_path: str = "crawl_output"  # Base path for output files
    user_agent: str = "SmartCrawler/1.0"  # User agent to use for requests
    headers: Dict[str, str] = field(default_factory=dict)  # HTTP headers
    exclude_patterns: List[str] = field(default_factory=list)  # URL patterns to exclude
    include_patterns: List[str] = field(default_factory=list)  # URL patterns to include
    auto_detect_content: bool = True  # Auto-detect content with LLM
    extract_all_pages: bool = False  # Extract content from all pages, not just leaf pages
    page_types_to_extract: List[str] = field(default_factory=lambda: ["article", "product", "detail"])  # Page types to extract
    save_html: bool = False  # Save raw HTML of pages
    threads: int = 1  # Number of threads for crawling (1 = single-threaded)
    llm_api_key: Optional[str] = None  # API key for LLM services


class LLMPageAnalyzer:
    """Uses an LLM to analyze webpages, generate selectors, and classify page types"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """Initialize the LLM page analyzer"""
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.selector_cache = {}  # Cache selectors by page type
        
    def _get_page_sample(self, html_content: str) -> dict:
        """Get a sample of the page for analysis"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, and other non-content elements
        for tag in soup(["script", "style", "meta", "noscript", "svg"]):
            tag.extract()
            
        # Get page text content (truncated)
        text = soup.get_text(separator='\n', strip=True)[:10000]
        
        # Extract key HTML structure patterns
        structure = []
        for tag in ['main', 'article', 'section', 'div', 'header', 'nav', 'footer']:
            elements = soup.find_all(tag, class_=True)
            for elem in elements[:5]:  # Limit to first 5 of each tag
                if elem.get('id'):
                    structure.append(f"{tag}#{elem.get('id')}")
                elif elem.get('class'):
                    structure.append(f"{tag}.{'.'.join(elem.get('class'))}")
        
        # Get links
        links = []
        for link in soup.find_all('a', href=True)[:20]:  # Limit to first 20 links
            href = link['href']
            text = link.get_text(strip=True)
            if text and href:
                links.append({"href": href, "text": text[:50]})
                
        # Get headings
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3'])[:10]:
            headings.append({"tag": h.name, "text": h.get_text(strip=True)[:100]})
            
        return {
            "title": soup.title.string if soup.title else "Untitled",
            "text_sample": text[:3000],
            "structure_sample": structure[:20],
            "links_sample": links,
            "headings": headings
        }
        
    def _call_llm(self, messages) -> dict:
        """Make a call to the LLM API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,  # Lower temperature for more deterministic outputs
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    
    def analyze_page(self, html_content: str, url: str) -> dict:
        """Analyze a webpage to determine its type and content"""
        page_sample = self._get_page_sample(html_content)
        
        # Modified prompt to avoid f-string formatting issues with pipe symbols
        system_content = """You are an expert web crawler and analyzer. 
        Your task is to analyze a webpage and determine its type, importance, and structure.
        Focus on identifying whether this is a content page (article, product, etc.) or a navigation page.
        Return a JSON object with your analysis."""
        
        user_content = f"""
        Analyze this webpage: {url}
        
        Title: {page_sample['title']}
        
        Headings:
        {json.dumps(page_sample['headings'], indent=2)}
        
        Sample of page structure:
        {json.dumps(page_sample['structure_sample'], indent=2)}
        
        Sample of links:
        {json.dumps(page_sample['links_sample'], indent=2)}
        
        Text sample:
        {page_sample['text_sample'][:1000]}...
        
        Analyze this information and provide:
        1. The page type (article, product, listing, category, home, search, profile, login, etc.)
        2. Whether this is a content page (has substantial unique content) or a navigation page
        3. Estimated importance (high, medium, low) based on content value
        4. If it contains a list of items or a single content item
        
        Return ONLY a JSON object with the following structure:
        {{
            "page_type": "article OR product OR listing OR category OR home OR search OR etc",
            "is_content_page": true OR false,
            "content_value": "high OR medium OR low",
            "has_item_list": true OR false,
            "estimated_items": 0
        }}
        
        Return ONLY the JSON object with no additional text."""
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Call the LLM
        response = self._call_llm(messages)
        
        try:
            # Extract JSON from response
            json_text = response['choices'][0]['message']['content'].strip()
            # Remove markdown code formatting if present
            if json_text.startswith('```json'):
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif json_text.startswith('```'):
                json_text = json_text.split('```')[1].strip()
                
            analysis = json.loads(json_text)
            logger.info(f"Page analysis for {url}: {json.dumps(analysis, indent=2)}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing LLM page analysis response: {str(e)}")
            # Fallback to basic analysis
            return {
                "page_type": "unknown",
                "is_content_page": False,
                "content_value": "low", 
                "has_item_list": False,
                "estimated_items": 0
            }
        
    def generate_selectors(self, html_content: str, url: str, page_type: str) -> Dict[str, str]:
        """Generate optimal CSS selectors for the webpage based on its type"""
        # Check if we already have selectors for this page type in cache
        if page_type in self.selector_cache:
            return self.selector_cache[page_type]
            
        page_sample = self._get_page_sample(html_content)
        
        system_content = f"""You are an expert web scraper. 
        Your task is to analyze a {page_type} webpage and generate optimal CSS selectors 
        to extract valuable content. Return only a JSON object with the selectors."""
        
        user_content = f"""
        I need to scrape data from this {page_type} page: {url}
        
        Here's information about the page:
        Title: {page_sample['title']}
        
        Headings:
        {json.dumps(page_sample['headings'], indent=2)}
        
        Page structure elements:
        {json.dumps(page_sample['structure_sample'], indent=2)}
        
        Page text preview:
        {page_sample['text_sample'][:1000]}...
        
        Analyze this information and provide me with optimal CSS selectors to extract
        the most valuable data from this {page_type} page. Return ONLY a JSON object with selector names
        as keys and CSS selectors as values.
        
        If this is a listing page with multiple similar items, use '_list' as a special key 
        for the container selector.
        
        Focus on extracting the most important content such as:
        - For articles: title, author, date, content, categories
        - For products: title, price, description, features, images
        - For listings: item containers, item titles, prices, etc.
        
        Return ONLY the JSON object with no additional text."""
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Call the LLM
        response = self._call_llm(messages)
        
        try:
            # Extract JSON from response
            json_text = response['choices'][0]['message']['content'].strip()
            # Remove markdown code formatting if present
            if json_text.startswith('```json'):
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif json_text.startswith('```'):
                json_text = json_text.split('```')[1].strip()
                
            selectors = json.loads(json_text)
            
            # Cache the selectors for this page type
            self.selector_cache[page_type] = selectors
            
            logger.info(f"Generated selectors for {page_type}: {json.dumps(selectors, indent=2)}")
            return selectors
            
        except Exception as e:
            logger.error(f"Error parsing LLM selectors response: {str(e)}")
            # Fallback to basic selectors
            basic_selectors = {
                "title": "h1",
                "content": "article, .content, main, #content, #main",
            }
            
            if page_type == "article":
                basic_selectors.update({
                    "author": ".author, .byline",
                    "date": ".date, time, .published-date",
                })
            elif page_type == "product":
                basic_selectors.update({
                    "price": ".price, .product-price",
                    "description": ".description, .product-description",
                })
            elif page_type in ["listing", "category"]:
                basic_selectors.update({
                    "_list": ".products li, .product-list .product, .items .item",
                    "item_title": ".title, h2, h3",
                })
                
            return basic_selectors


class UrlManager:
    """Manages URL operations, validation, and queue management"""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.queue = deque()  # Queue of URLs to crawl
        self.visited = set()  # Set of visited URLs
        self.discovered_urls = set()  # All discovered URLs
        self.base_domain = self._extract_domain(config.start_url)
        self.subdomains = set()  # Discovered subdomains
        self.robots_rules = {}  # Rules from robots.txt
        
        # Initialize with start URL
        self.queue.append((config.start_url, 0))  # (URL, depth)
        self.discovered_urls.add(config.start_url)
        
        # If respecting robots.txt, fetch it
        if config.respect_robots:
            self._fetch_robots_txt(config.start_url)
            
    def _extract_domain(self, url: str) -> str:
        """Extract the base domain from a URL"""
        parsed = urlparse(url)
        domain_parts = parsed.netloc.split('.')
        
        # Handle common TLDs with multiple parts (co.uk, com.au, etc.)
        if len(domain_parts) > 2 and domain_parts[-2] in ["co", "com", "org", "net", "ac", "gov"]:
            base_domain = '.'.join(domain_parts[-3:])
        else:
            # Get last two parts as base domain
            base_domain = '.'.join(domain_parts[-2:])
            
        return base_domain
    
    def _get_subdomain(self, url: str) -> str:
        """Extract the subdomain from a URL"""
        parsed = urlparse(url)
        domain_parts = parsed.netloc.split('.')
        base_domain = self._extract_domain(url)
        base_parts = len(base_domain.split('.'))
        
        if len(domain_parts) > base_parts:
            return '.'.join(domain_parts[:-base_parts])
        return ""
    
    def _is_same_domain_or_subdomain(self, url: str) -> bool:
        """Check if a URL belongs to the base domain or its subdomains"""
        parsed = urlparse(url)
        if not parsed.netloc:
            return True  # Relative URL
            
        return self._extract_domain(url) == self.base_domain
    
    def _fetch_robots_txt(self, url: str) -> None:
        """Fetch and parse robots.txt for the given domain"""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        try:
            session = requests.Session()
            response = session.get(robots_url, headers=self.config.headers)
            if response.status_code == 200:
                lines = response.text.splitlines()
                agent_rules = {}
                current_agent = None
                
                for line in lines:
                    line = line.strip().lower()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split(':', 1)
                    if len(parts) != 2:
                        continue
                        
                    key, value = parts[0].strip(), parts[1].strip()
                    
                    if key == 'user-agent':
                        current_agent = value
                        if current_agent not in agent_rules:
                            agent_rules[current_agent] = {'disallow': [], 'allow': []}
                    elif current_agent and key == 'disallow' and value:
                        agent_rules[current_agent]['disallow'].append(value)
                    elif current_agent and key == 'allow' and value:
                        agent_rules[current_agent]['allow'].append(value)
                
                # Store the rules for the domain
                domain = parsed.netloc
                self.robots_rules[domain] = agent_rules
                logger.info(f"Fetched robots.txt for {domain}")
        except Exception as e:
            logger.warning(f"Error fetching robots.txt: {str(e)}")
    
    def _is_allowed_by_robots(self, url: str) -> bool:
        """Check if a URL is allowed by robots.txt"""
        if not self.config.respect_robots:
            return True
            
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path or '/'
        
        if domain not in self.robots_rules:
            return True
            
        agent_rules = self.robots_rules[domain]
        
        # Check specific user agent rules
        user_agent = self.config.user_agent.lower()
        check_agents = [user_agent, '*']
        
        for agent in check_agents:
            if agent in agent_rules:
                rules = agent_rules[agent]
                
                # Check allow rules first
                for allow_path in rules.get('allow', []):
                    if path.startswith(allow_path):
                        return True
                
                # Then check disallow rules
                for disallow_path in rules.get('disallow', []):
                    if path.startswith(disallow_path):
                        return False
        
        return True
    
    def should_crawl_url(self, url: str) -> bool:
        """Determine if a URL should be crawled based on configured rules"""
        parsed = urlparse(url)
        
        # Check if it's a valid HTTP/HTTPS URL
        if parsed.scheme not in ['http', 'https']:
            return False
            
        # If we're staying on subdomains, check domain
        if self.config.stay_on_subdomains and not self._is_same_domain_or_subdomain(url):
            return False
            
        # Check against exclusion patterns
        for pattern in self.config.exclude_patterns:
            if re.search(pattern, url):
                return False
                
        # Check against inclusion patterns if specified
        if self.config.include_patterns:
            included = False
            for pattern in self.config.include_patterns:
                if re.search(pattern, url):
                    included = True
                    break
            if not included:
                return False
                
        # Check robots.txt
        if not self._is_allowed_by_robots(url):
            return False
            
        # Skip common non-content URLs
        skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.xml', '.json', '.zip']
        for ext in skip_extensions:
            if parsed.path.endswith(ext):
                return False
                
        skip_patterns = [
            r'/login', r'/signup', r'/register', r'/cart', r'/checkout',
            r'/account', r'/profile', r'/admin', r'/wp-admin'
        ]
        for pattern in skip_patterns:
            if re.search(pattern, parsed.path, re.IGNORECASE):
                return False
        
        return True
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and normalize links from a page"""
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip empty, javascript and anchor links
            if not href or href.startswith('javascript:') or href == '#':
                continue
                
            # Normalize URL
            full_url = urljoin(base_url, href)
            
            # Remove fragments
            parsed = urlparse(full_url)
            clean_url = parsed._replace(fragment='').geturl()
            
            links.append(clean_url)
            
        return links
    
    def add_url_to_queue(self, url: str, depth: int) -> None:
        """Add a URL to the crawl queue if it meets criteria"""
        if url not in self.visited and url not in self.discovered_urls:
            if self.should_crawl_url(url):
                self.queue.append((url, depth))
                self.discovered_urls.add(url)
    
    def get_next_url(self) -> Optional[Tuple[str, int]]:
        """Get the next URL to crawl from the queue"""
        return self.queue.popleft() if self.queue else None
    
    def mark_visited(self, url: str) -> None:
        """Mark a URL as visited and track its subdomain"""
        self.visited.add(url)
        subdomain = self._get_subdomain(url)
        if subdomain:
            self.subdomains.add(subdomain)


class PageFetcher:
    """Handles page fetching and Selenium operations"""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.session = requests.Session()
        self.driver = None
        
        # Set default headers
        if not self.config.headers:
            self.config.headers = {
                'User-Agent': self.config.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9',
            }
    
    def _init_selenium(self):
        """Initialize Selenium WebDriver if needed"""
        if self.driver:
            return
            
        logger.info("Initializing Selenium WebDriver")
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        if self.config.user_agent:
            chrome_options.add_argument(f"user-agent={self.config.user_agent}")
            
        try:
            if ChromeDriverManager:
                self.driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()),
                    options=chrome_options
                )
            else:
                # Fallback to system chromedriver
                self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            logger.error(f"Error initializing Selenium: {str(e)}")
            logger.warning("Falling back to non-Selenium mode")
            self.config.use_selenium = False
    
    def fetch_page(self, url: str) -> Tuple[Optional[str], Optional[BeautifulSoup]]:
        """Fetch a page and return the HTML content and soup"""
        try:
            if self.config.use_selenium:
                try:
                    self._init_selenium()
                    if not self.driver:
                        raise Exception("Failed to initialize Selenium driver")
                        
                    self.driver.get(url)
                    try:
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, "/html/body"))
                        )
                    except Exception:
                        pass  # Continue even if timeout
                        
                    html_content = self.driver.page_source
                    soup = BeautifulSoup(html_content, 'html.parser')
                    return html_content, soup
                except Exception as e:
                    logger.error(f"Selenium error for {url}: {str(e)}")
                    logger.info(f"Falling back to requests for {url}")
                    # Fall back to requests
                    
            response = self.session.get(url, headers=self.config.headers, timeout=10)
            response.raise_for_status()
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            return html_content, soup
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None, None
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()


class SmartCrawler:
    """Intelligent web crawler that discovers and scrapes content across a domain and its subdomains"""
    
    def __init__(self, config: CrawlConfig, llm_analyzer: Optional[LLMPageAnalyzer] = None):
        """Initialize the crawler with configuration"""
        self.config = config
        self._apply_env_overrides()
        self._initialize_components(llm_analyzer)
        
    def _apply_env_overrides(self):
        """Apply overrides from environment variables"""
        # Map environment variables to config attributes
        env_mapping = {
            "CRAWLER_MAX_PAGES": ("max_pages", int),
            "CRAWLER_MAX_DEPTH": ("max_depth", int),
            "CRAWLER_OUTPUT_PATH": ("output_path", str),
            "CRAWLER_USE_SELENIUM": ("use_selenium", lambda x: x.lower() == "true"),
            "CRAWLER_RESPECT_ROBOTS": ("respect_robots", lambda x: x.lower() == "true"),
            "CRAWLER_DELAY_MIN": (lambda config, val: setattr(config, "delay_range", [float(val), config.delay_range[1]])),
            "CRAWLER_DELAY_MAX": (lambda config, val: setattr(config, "delay_range", [config.delay_range[0], float(val)])),
        }
        
        # Apply mappings
        for env_var, config_attr in env_mapping.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                if callable(config_attr):
                    # Function to apply complex transformation
                    config_attr(self.config, env_value)
                else:
                    # Simple attribute mapping with type conversion
                    attr_name, converter = config_attr
                    setattr(self.config, attr_name, converter(env_value))
        
        # Special case for API key
        if not self.config.llm_api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.config.llm_api_key = api_key
                
    def _initialize_components(self, llm_analyzer: Optional[LLMPageAnalyzer]) -> None:
        """Initialize crawler components"""
        # Initialize core components
        self.url_manager = UrlManager(self.config)
        self.page_fetcher = PageFetcher(self.config)
        self.state_manager = StateManager(self.config)
        
        # Initialize LLM analyzer
        self.llm_analyzer = self._setup_llm_analyzer(llm_analyzer)
        
        # Initialize content extractor
        self.content_extractor = ContentExtractor(self.llm_analyzer)
        
    def _setup_llm_analyzer(self, llm_analyzer: Optional[LLMPageAnalyzer]) -> Optional[LLMPageAnalyzer]:
        """Set up LLM analyzer with appropriate configuration"""
        if llm_analyzer:
            return llm_analyzer
            
        if self.config.llm_api_key:
            return LLMPageAnalyzer(api_key=self.config.llm_api_key)
            
        # Try to get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return LLMPageAnalyzer(api_key=api_key)
            
        logger.warning("No OpenAI API key provided - automatic content detection disabled")
        self.config.auto_detect_content = False
        return None
        
    def _process_page(self, url: str, depth: int) -> None:
        """Process a single page during crawling"""
        logger.info(f"Processing page: {url} [depth: {depth}]")
        
        # Fetch and parse page
        html_content, soup = self.page_fetcher.fetch_page(url)
        if not html_content or not soup:
            return
            
        # Save HTML if configured
        if self.config.save_html:
            self._save_html_content(html_content)
        
        # Analyze and extract content
        page_analysis = self._analyze_page(html_content, url)
        content = self.content_extractor.extract_content(soup, html_content, url, page_analysis)
        if content:
            self.state_manager.save_content(content)
        
        # Process links if not at max depth
        if depth < self.config.max_depth:
            self._process_links(soup, url, depth)
    
    def _save_html_content(self, html_content: str) -> None:
        """Save raw HTML content if configured"""
        page_id = str(uuid.uuid4())[:8]
        html_path = os.path.join(self.config.output_path, "html")
        os.makedirs(html_path, exist_ok=True)
        with open(os.path.join(html_path, f"{page_id}.html"), 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _analyze_page(self, html_content: str, url: str) -> Optional[Dict]:
        """Analyze page content using LLM if configured"""
        if self.config.auto_detect_content and self.llm_analyzer:
            return self.llm_analyzer.analyze_page(html_content, url)
        return None
    
    def _process_links(self, soup: BeautifulSoup, url: str, depth: int) -> None:
        """Process and queue links found on the page"""
        links = self.url_manager.extract_links(soup, url)
        for link in links:
            self.url_manager.add_url_to_queue(link, depth + 1)
    
    def crawl(self) -> List[Dict]:
        """Execute the crawling process"""
        try:
            pages_visited = 0
            
            while pages_visited < self.config.max_pages:
                # Get next URL from queue
                url_info = self.url_manager.get_next_url()
                if not url_info:
                    break
                    
                url, depth = url_info
                self.url_manager.mark_visited(url)
                pages_visited += 1
                
                # Process the page
                self._process_page(url, depth)
                
                # Apply delay between requests
                if self.url_manager.queue:
                    delay = random.uniform(self.config.delay_range[0], self.config.delay_range[1])
                    time.sleep(delay)
                    
                # Periodically save state
                if pages_visited % 10 == 0:
                    self.state_manager.save_crawl_state(self.url_manager)
            
            # Save final state and content
            logger.info(f"Crawling completed: {pages_visited} pages visited, {len(self.state_manager.extracted_content)} content items extracted")
            self.state_manager.save_crawl_state(self.url_manager)
            self.state_manager.save_final_content()
            
            return self.state_manager.extracted_content
            
        finally:
            self.page_fetcher.cleanup()
    
    
class BaseContentExtractor:
    """Base class for content extraction with common utilities"""
    
    def _clean_content(self, element) -> str:
        """Clean extracted content by removing boilerplate elements"""
        # Remove unwanted elements
        for unwanted in element.select('script, style, nav, header, footer, .ad, .advertisement, .social-share'):
            unwanted.decompose()
            
        # Get text with preserved whitespace
        text = element.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
        
    def _extract_by_content_density(self, soup: BeautifulSoup) -> str:
        """Extract content by analyzing text density in different page sections"""
        # Remove obvious non-content areas
        for element in soup.select('header, footer, nav, aside'):
            element.decompose()
            
        # Score remaining blocks by text density
        blocks = []
        for block in soup.find_all(['div', 'section', 'article']):
            text = block.get_text(strip=True)
            if len(text) < 100:  # Skip very short blocks
                continue
                
            # Calculate text density score
            text_length = len(text)
            html_length = len(str(block))
            density = text_length / html_length
            
            blocks.append((block, density))
            
        if not blocks:
            return ""
            
        # Use block with highest density
        main_block = max(blocks, key=lambda x: x[1])[0]
        return self._clean_content(main_block)


class MetadataExtractor(BaseContentExtractor):
    """Handles extraction of page metadata"""
    
    def extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract key metadata from the page"""
        metadata = {}
        
        # Extract title
        if soup.title:
            metadata['title'] = soup.title.string.strip()
        else:
            h1 = soup.find('h1')
            if h1:
                metadata['title'] = h1.get_text(strip=True)
                
        # Extract meta description
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
            
        # Extract author
        author = soup.find('meta', {'name': ['author', 'article:author']})
        if author:
            metadata['author'] = author.get('content', '')
            
        # Extract date
        date = soup.find('meta', {'name': ['date', 'article:published_time']})
        if date:
            metadata['date'] = date.get('content', '')
            
        # Extract headings for structure
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3']):
            headings.append(h.get_text(strip=True))
        if headings:
            metadata['headings'] = headings
            
        return metadata


class NavigationExtractor(BaseContentExtractor):
    """Handles extraction of navigation elements"""
    
    def extract_navigation(self, soup: BeautifulSoup) -> Dict:
        """Extract navigation elements like menus and links"""
        navigation = {}
        
        # Extract main navigation
        nav_elements = soup.find_all(['nav', '[role="navigation"]'])
        if nav_elements:
            nav_links = []
            for nav in nav_elements:
                for link in nav.find_all('a'):
                    if link.get_text(strip=True):
                        nav_links.append(link.get_text(strip=True))
            if nav_links:
                navigation['nav_links'] = nav_links
                
        # Extract sidebar sections
        sidebars = soup.find_all(['aside', '.sidebar', '#sidebar'])
        if sidebars:
            sidebar_sections = []
            for sidebar in sidebars:
                sections = sidebar.find_all(['h2', 'h3', 'h4'])
                sidebar_sections.extend([s.get_text(strip=True) for s in sections])
            if sidebar_sections:
                navigation['sidebar_sections'] = sidebar_sections
                
        # Extract footer links
        footer = soup.find('footer')
        if footer:
            footer_text = footer.get_text(strip=True)
            if footer_text:
                navigation['footer'] = footer_text
                
        return navigation


class ContentExtractor(BaseContentExtractor):
    """Main content extraction coordinator"""
    
    def __init__(self, llm_analyzer=None):
        self.llm_analyzer = llm_analyzer
        self.metadata_extractor = MetadataExtractor()
        self.navigation_extractor = NavigationExtractor()
        
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content area of the page"""
        # Try common content container selectors
        content_selectors = [
            'article', 'main', '[role="main"]',
            '#content', '#main-content', '.content', '.main-content',
            '.post-content', '.article-content', '.entry-content'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                return self._clean_content(element)
                
        # Fallback to analyzing content density
        return self._extract_by_content_density(soup)
        
    def extract_content(self, soup: BeautifulSoup, html_content: str, url: str, page_analysis: dict = None) -> Optional[Dict]:
        """Extract content intelligently from the page"""
        # Extract core content areas
        content = {
            "url": url,
            "page_type": page_analysis.get('page_type', 'article') if page_analysis else 'article'
        }
        
        # Get metadata using dedicated extractor
        content.update(self.metadata_extractor.extract_metadata(soup))
        
        # Get main content
        main_content = self._extract_main_content(soup)
        if main_content:
            content['main_content'] = main_content
            
        # Get navigation elements using dedicated extractor
        navigation = self.navigation_extractor.extract_navigation(soup)
        content.update(navigation)
        
        # Use LLM analysis if available
        if self.llm_analyzer and page_analysis:
            selectors = self.llm_analyzer.generate_selectors(html_content, url, content['page_type'])
            
            # Extract any additional fields specified by LLM
            for field, selector in selectors.items():
                if field not in content:
                    try:
                        elements = soup.select(selector)
                        if elements:
                            if isinstance(selector, dict) and 'attribute' in selector:
                                content[field] = elements[0].get(selector['attribute'], '')
                            else:
                                content[field] = elements[0].get_text(strip=True)
                    except Exception:
                        pass
                        
        return content

class StateManager:
    """Manages crawler state and content storage"""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.extracted_content = []
        os.makedirs(self.config.output_path, exist_ok=True)
    
    def save_content(self, content: Dict) -> None:
        """Save extracted content to the output file"""
        self.extracted_content.append(content)
        
        # Save incrementally to avoid data loss
        output_file = os.path.join(self.config.output_path, "content.jsonl")
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(content, ensure_ascii=False) + '\n')
    
    def save_crawl_state(self, url_manager) -> None:
        """Save the current crawl state as JSON"""
        state = {
            "base_domain": url_manager.base_domain,
            "subdomains": list(url_manager.subdomains),
            "visited_count": len(url_manager.visited),
            "queue_count": len(url_manager.queue),
            "extracted_count": len(self.extracted_content)
        }
        
        with open(os.path.join(self.config.output_path, "crawl_state.json"), 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
            
        # Save URLs list
        with open(os.path.join(self.config.output_path, "visited_urls.txt"), 'w', encoding='utf-8') as f:
            for url in sorted(url_manager.visited):
                f.write(url + '\n')
                
        # Save subdomains list
        with open(os.path.join(self.config.output_path, "subdomains.txt"), 'w', encoding='utf-8') as f:
            for subdomain in sorted(url_manager.subdomains):
                f.write(subdomain + '\n')
    
    def save_final_content(self) -> None:
        """Save all extracted content as a single JSON file"""
        with open(os.path.join(self.config.output_path, "all_content.json"), 'w', encoding='utf-8') as f:
            json.dump(self.extracted_content, f, indent=2, ensure_ascii=False)


        


def main():
    """Main entry point for the crawler"""
    parser = argparse.ArgumentParser(description="Smart Subdomain Web Crawler & Scraper")
    parser.add_argument("--url", required=True, help="Starting URL for crawling")
    parser.add_argument("--output", default="crawl_output", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=100, help="Maximum pages to crawl")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum link depth")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--selenium", action="store_true", help="Use Selenium for JavaScript-heavy sites")
    parser.add_argument("--api-key", help="OpenAI API key for content detection")
    parser.add_argument("--save-html", action="store_true", help="Save raw HTML of pages")
    parser.add_argument("--include", nargs="*", help="URL patterns to include")
    parser.add_argument("--exclude", nargs="*", help="URL patterns to exclude")
    parser.add_argument("--all-pages", action="store_true", help="Extract content from all pages")
    parser.add_argument("--no-subdomains", action="store_true", help="Don't crawl subdomains")
    parser.add_argument("--no-robots", action="store_true", help="Don't respect robots.txt")
    
    args = parser.parse_args()
    
    # Create configuration
    config = CrawlConfig(
        start_url=args.url,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        stay_on_subdomains=not args.no_subdomains,
        respect_robots=not args.no_robots,
        delay_range=[args.delay, args.delay * 2],
        use_selenium=args.selenium,
        output_path=args.output,
        exclude_patterns=args.exclude or [],
        include_patterns=args.include or [],
        auto_detect_content=True,
        extract_all_pages=args.all_pages,
        save_html=args.save_html,
        llm_api_key=args.api_key
    )
    
    # Initialize and run crawler
    try:
        crawler = SmartCrawler(config)
        crawler.crawl()
    except KeyboardInterrupt:
        logger.info("Crawling interrupted by user")
    except Exception as e:
        logger.error(f"Error during crawling: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())