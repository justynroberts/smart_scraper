```
                 _____                      _    _____                    _           
                / ____|                    | |  / ____|                  | |          
               | (___  _ __ ___   __ _ _ __| |_| |     _ __ __ _ _    _| | ___ _ __ 
                \___ \| '_ ` _ \ / _` | '__| __| |    | '__/ _` | |__ | | |/ _ \ '__|
                ____) | | | | | | (_| | |  | |_| |____| | | (_| | |_| | |  __/ |   
               |_____/|_| |_| |_|\__,_|_|   \__|\_____| |  \__,_|\__,_|_|\___|_|   
                                                                               
```

# Smart Subdomain Crawler & Scraper 🕷️

An intelligent web crawler that discovers and scrapes content across domains and subdomains, using LLM-powered analysis for smart content extraction.

## 🌟 Features

### Core Capabilities
- 🔍 Intelligent content detection and extraction
- 🌐 Subdomain discovery and crawling
- 🤖 LLM-powered page analysis
- 📊 Smart content density analysis
- 🚀 JavaScript support via Selenium
- 🛡️ Respectful crawling with robots.txt support

### Smart Components
- 🧠 **LLM Page Analyzer**
  - Automatically classifies page types
  - Generates optimal CSS selectors
  - Analyzes content importance
  - Caches selectors for efficiency

- 🌍 **URL Manager**
  - Smart URL queue management
  - Domain/subdomain validation
  - robots.txt compliance
  - Pattern-based filtering

- 📥 **Page Fetcher**
  - Flexible fetching strategies
  - Selenium support for JS-heavy sites
  - Automatic fallback mechanisms
  - Configurable delays and retries

- 📝 **Content Extractor**
  - Content density analysis
  - Boilerplate removal
  - Metadata extraction
  - Navigation structure analysis

- 💾 **State Manager**
  - Incremental content saving
  - Crawl state persistence
  - Progress tracking
  - Multi-format output support

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/justynroberts/smart-scraper.git
cd smart-scraper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

The crawler is highly configurable through both command-line arguments and environment variables.

### Environment Variables
```bash
OPENAI_API_KEY=your-api-key
CRAWLER_MAX_PAGES=100
CRAWLER_MAX_DEPTH=3
CRAWLER_OUTPUT_PATH=crawl_output
CRAWLER_USE_SELENIUM=true
CRAWLER_RESPECT_ROBOTS=true
CRAWLER_DELAY_MIN=1.0
CRAWLER_DELAY_MAX=3.0
```

### Command Line Arguments
```bash
usage: smart_subdomain_crawler.py [-h] --url URL [--output OUTPUT]
                                [--max-pages MAX_PAGES] [--max-depth MAX_DEPTH]
                                [--delay DELAY] [--selenium] [--api-key API_KEY]
                                [--save-html] [--include [INCLUDE ...]]
                                [--exclude [EXCLUDE ...]] [--all-pages]
                                [--no-subdomains] [--no-robots]
```

## 🚀 Usage

### Basic Usage
```bash
# Basic crawling
python smart_subdomain_crawler.py --url https://example.com

# Advanced crawling with options
python smart_subdomain_crawler.py --url https://example.com \
    --max-pages 200 \
    --max-depth 5 \
    --delay 2.0 \
    --selenium \
    --save-html
```

### CLI Tool Usage 🛠️

The crawler comes with a powerful CLI tool that provides three main commands:

#### 1. Quick Start Command ⚡
```bash
# Start crawling quickly with minimal configuration
python smart_cli.py quick https://example.com
    --output crawl_output \
    --pages 50 \
    --selenium
```

#### 2. Configuration Wizard 🧙‍♂️
```bash
# Create a new configuration file interactively
python smart_cli.py config --output crawler_config.json
```

The wizard will guide you through configuring:
- Basic settings (URLs, limits)
- Advanced options (Selenium, subdomain handling)
- Content extraction settings
- URL patterns for inclusion/exclusion
- Delay configurations
- Page type specifications

#### 3. Run with Configuration 📝
```bash
# Run crawler with a configuration file
python smart_cli.py run --config crawler_config.json

# Run with config overrides
python smart_cli.py run \
    --config crawler_config.json \
    --url https://example.com \
    --output custom_output \
    --max-pages 200 \
    --api-key your-api-key \
    --selenium
```

### Advanced Usage Examples 🚀

```bash
# Create configuration and run
python smart_cli.py config --output my_config.json
python smart_cli.py run --config my_config.json --selenium

# Quick crawl with JavaScript support
python smart_cli.py quick https://example.com \
    --output js_content \
    --pages 100 \
    --selenium

# Run with pattern filtering
python smart_subdomain_crawler.py --url https://example.com \
    --include "*/blog/*" "*/news/*" \
    --exclude "*/author/*" "*/tag/*" \
    --all-pages

# Focused subdomain crawling
python smart_subdomain_crawler.py --url https://blog.example.com \
    --no-subdomains \
    --max-pages 50
```

### Configuration File Structure 📄

Example `crawler_config.json`:
```json
{
  "start_url": "https://example.com",
  "max_pages": 100,
  "max_depth": 3,
  "output_path": "crawl_output",
  "use_selenium": false,
  "stay_on_subdomains": true,
  "respect_robots": true,
  "auto_detect_content": true,
  "extract_all_pages": false,
  "save_html": false,
  "include_patterns": [
    "*/blog/*",
    "*/news/*"
  ],
  "exclude_patterns": [
    "*/author/*",
    "*/tag/*"
  ],
  "delay_range": [1.0, 2.0],
  "page_types_to_extract": [
    "article",
    "blog",
    "product"
  ]
}
```

## 📂 Output Structure

```
crawl_output/
├── content.jsonl       # Incremental content storage
├── all_content.json   # Combined content in JSON format
├── crawl_state.json   # Crawler state and statistics
├── visited_urls.txt   # List of crawled URLs
├── subdomains.txt     # Discovered subdomains
└── html/             # Raw HTML storage (if enabled)
    └── *.html
```

## 🎯 Use Cases

1. **Content Aggregation**
   - Automatically collect articles from news sites
   - Gather product information from e-commerce sites
   - Build knowledge bases from documentation sites

2. **SEO Analysis**
   - Discover and analyze site structure
   - Extract metadata and content patterns
   - Map internal linking relationships

3. **Data Mining**
   - Extract structured data from multiple sources
   - Build training datasets for ML models
   - Perform content analysis across domains

## 🧩 Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   SmartCrawler  │────▶│   UrlManager    │────▶│   PageFetcher   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  StateManager   │◀────│ContentExtractor │◀────│  LLMAnalyzer    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 🔧 Development

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

### Running Tests
```bash
python -m pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📫 Contact

For questions and support, please open an issue or contact the maintainers:
- GitHub Issues: [Create an issue](https://github.com/justynroberts/smart-scraper/issues)
- Email: justynroberts@gmail.com

---
Made with ❤️ by Justyn
