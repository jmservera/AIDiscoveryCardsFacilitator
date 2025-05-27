"""
Sample test file to demonstrate pytest structure and functionality.
"""
import os
from pathlib import Path


def test_src_directory_exists():
    """Test that the src directory exists."""
    assert os.path.isdir(Path(__file__).parent.parent / "src")


def test_requirements_includes_pytest():
    """Test that pytest is included in requirements.txt."""
    requirements_path = Path(__file__).parent.parent / "src" / "requirements.txt"
    assert requirements_path.exists()
    
    with open(requirements_path, 'r') as f:
        requirements = f.read()
    
    assert "pytest" in requirements, "pytest should be included in requirements.txt"


def test_project_structure():
    """Test that key project directories exist."""
    base_dir = Path(__file__).parent.parent
    
    # Test if key directories exist
    assert (base_dir / "src").is_dir(), "src directory should exist"
    assert (base_dir / "src" / "agents").is_dir(), "agents directory should exist"
    assert (base_dir / "src" / "utils").is_dir(), "utils directory should exist"
    assert (base_dir / "tests").is_dir(), "tests directory should exist"