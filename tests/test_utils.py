"""
Tests for basic utility functions.
"""
import os
from pathlib import Path


def test_config_directories():
    """Test that required config directories exist."""
    src_path = Path(__file__).parent.parent / "src"
    
    # Test config directory structure
    config_path = src_path / "config"
    assert os.path.exists(config_path), "config directory should exist"

def test_requirements_content():
    """Test that requirements.txt contains essential packages."""
    req_path = Path(__file__).parent.parent / "src" / "requirements.txt"
    
    with open(req_path, 'r') as f:
        content = f.read()
    
    # Check for critical packages
    essential_packages = [
        "streamlit",
        "openai",
        "pytest"
    ]
    
    for package in essential_packages:
        assert package in content, f"requirements.txt should include {package}"