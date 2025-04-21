#!/usr/bin/env python3
"""
Smart Crawler CLI Tool
----------------------
A simple command line interface for running the smart subdomain crawler.
"""

import argparse
import os
import sys
import json
import logging
from smart_subdomain_crawler import SmartCrawler, CrawlConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_config_wizard():
    """Interactive wizard to create a crawler configuration"""
    print("\n=== Smart Crawler Configuration Wizard ===\n")
    
    config = {}
    
    # Basic configuration
    config["start_url"] = input("Enter the starting URL: ")
    if not config["start_url"]:
        print("Error: Starting URL is required")
        return None
        
    config["max_pages"] = int(input("Maximum pages to crawl [100]: ") or "100")
    config["max_depth"] = int(input("Maximum link depth [3]: ") or "3")
    
    # Advanced options
    print("\n=== Advanced Options (press Enter for defaults) ===")
    
    config["output_path"] = input("Output directory [crawl_output]: ") or "crawl_output"
    
    use_selenium = input("Use Selenium for JavaScript rendering? (y/n) [n]: ").lower()
    config["use_selenium"] = use_selenium == "y"
    
    stay_on_subdomains = input("Stay on domain and subdomains? (y/n) [y]: ").lower()
    config["stay_on_subdomains"] = stay_on_subdomains != "n"
    
    respect_robots = input("Respect robots.txt? (y/n) [y]: ").lower()
    config["respect_robots"] = respect_robots != "n"
    
    # Content extraction
    print("\n=== Content Extraction ===")
    
    auto_detect = input("Use LLM to auto-detect content? (y/n) [y]: ").lower()
    config["auto_detect_content"] = auto_detect != "n"
    
    if config["auto_detect_content"]:
        api_key = input("OpenAI API key [use environment variable if empty]: ")
        if api_key:
            config["llm_api_key"] = api_key
    
    extract_all = input("Extract content from all pages? (y/n) [n]: ").lower()
    config["extract_all_pages"] = extract_all == "y"
    
    save_html = input("Save raw HTML of pages? (y/n) [n]: ").lower()
    config["save_html"] = save_html == "y"
    
    # URL patterns
    print("\n=== URL Patterns ===")
    print("Enter URL patterns one per line. Empty line to finish.")
    
    print("\nInclude patterns (URLs that match these patterns will be crawled):")
    include_patterns = []
    while True:
        pattern = input("> ")
        if not pattern:
            break
        include_patterns.append(pattern)
    
    if include_patterns:
        config["include_patterns"] = include_patterns
    
    print("\nExclude patterns (URLs that match these patterns will be skipped):")
    exclude_patterns = []
    while True:
        pattern = input("> ")
        if not pattern:
            break
        exclude_patterns.append(pattern)
    
    if exclude_patterns:
        config["exclude_patterns"] = exclude_patterns
    
    # Delay configuration
    delay_min = float(input("\nMinimum delay between requests (seconds) [1.0]: ") or "1.0")
    delay_max = float(input("Maximum delay between requests (seconds) [2.0]: ") or "2.0")
    config["delay_range"] = [delay_min, delay_max]
    
    # Page types to extract
    print("\nPage types to extract (comma-separated, empty for all):")
    print("Options: article, product, blog, listing, category, detail")
    page_types = input("> ")
    if page_types:
        config["page_types_to_extract"] = [t.strip() for t in page_types.split(",")]
    
    return config

def save_config(config, filepath):
    """Save configuration to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filepath}")

def load_config(filepath):
    """Load configuration from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Smart Subdomain Web Crawler CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Create configuration")
    config_parser.add_argument("--output", default="crawler_config.json", help="Output file for configuration")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run crawler")
    run_parser.add_argument("--url", help="Starting URL (overrides config)")
    run_parser.add_argument("--config", help="Path to configuration file")
    run_parser.add_argument("--output", help="Output directory (overrides config)")
    run_parser.add_argument("--max-pages", type=int, help="Maximum pages (overrides config)")
    run_parser.add_argument("--api-key", help="OpenAI API key (overrides config)")
    run_parser.add_argument("--selenium", action="store_true", help="Use Selenium (overrides config)")
    
    # Quick command for starting a crawl with minimal configuration
    quick_parser = subparsers.add_parser("quick", help="Quick start crawler with minimal configuration")
    quick_parser.add_argument("url", help="Starting URL for crawling")
    quick_parser.add_argument("--output", default="crawl_output", help="Output directory")
    quick_parser.add_argument("--pages", type=int, default=50, help="Maximum pages to crawl")
    quick_parser.add_argument("--selenium", action="store_true", help="Use Selenium for JavaScript sites")
    
    args = parser.parse_args()
    
    # Handle config command
    if args.command == "config":
        config = create_config_wizard()
        if config:
            save_config(config, args.output)
        return 0
    
    # Handle run command
    elif args.command == "run":
        if not args.config and not args.url:
            print("Error: Either --config or --url must be specified")
            return 1
            
        if args.config:
            try:
                config_data = load_config(args.config)
            except Exception as e:
                print(f"Error loading configuration: {str(e)}")
                return 1
        else:
            # Minimal configuration if no config file
            config_data = {"start_url": args.url}
        
        # Override config with command line arguments
        if args.url:
            config_data["start_url"] = args.url
        if args.output:
            config_data["output_path"] = args.output
        if args.max_pages:
            config_data["max_pages"] = args.max_pages
        if args.api_key:
            config_data["llm_api_key"] = args.api_key
        if args.selenium:
            config_data["use_selenium"] = True
            
        # Run crawler
        try:
            config = CrawlConfig(**config_data)
            crawler = SmartCrawler(config)
            crawler.crawl()
            return 0
        except Exception as e:
            print(f"Error: {str(e)}")
            return 1
    
    # Handle quick command
    elif args.command == "quick":
        config = CrawlConfig(
            start_url=args.url,
            max_pages=args.pages,
            output_path=args.output,
            use_selenium=args.selenium,
            auto_detect_content=True,
            stay_on_subdomains=True
        )
        
        try:
            crawler = SmartCrawler(config)
            crawler.crawl()
            return 0
        except Exception as e:
            print(f"Error: {str(e)}")
            return 1
    
    else:
        # No command provided, show help
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())