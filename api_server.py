#!/usr/bin/env python3
"""
Startup script for the Historical Map Text Processing REST API server.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.api.main_api import HistoricalMapAPI


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(
        description='Historical Map Text Processing REST API Server'
    )
    
    parser.add_argument(
        '--host', 
        default='0.0.0.0',
        help='Host address to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000,
        help='Port number to bind to (default: 5000)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--upload-folder',
        default='uploads',
        help='Directory for uploaded files (default: uploads)'
    )
    
    parser.add_argument(
        '--results-folder',
        default='api_results',
        help='Directory for result files (default: api_results)'
    )
    
    parser.add_argument(
        '--max-file-size',
        type=int,
        default=16 * 1024 * 1024,
        help='Maximum file size in bytes (default: 16MB)'
    )
    
    args = parser.parse_args()
    
    # Create API configuration
    api_config = {
        'debug': args.debug,
        'upload_folder': args.upload_folder,
        'results_folder': args.results_folder,
        'max_file_size': args.max_file_size,
        'enable_cors': True
    }
    
    # Load additional configuration if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            api_config.update(file_config.get('api', {}))
    
    print("Starting Historical Map Text Processing API Server")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Upload folder: {api_config['upload_folder']}")
    print(f"Results folder: {api_config['results_folder']}")
    print(f"Max file size: {api_config['max_file_size']} bytes")
    print("-" * 50)
    
    try:
        # Create and run API
        api = HistoricalMapAPI(api_config)
        api.run(host=args.host, port=args.port, debug=args.debug)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())