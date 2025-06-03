#!/usr/bin/env python3
"""
Test runner script for EasyMLOps
Provides convenient commands to run different types of tests
"""

import sys
import subprocess
import argparse
import shutil
from pathlib import Path


def is_poetry_project():
    """Check if this is a Poetry project"""
    return Path("pyproject.toml").exists() and Path("poetry.lock").exists()


def get_python_cmd():
    """Get the appropriate Python command to use"""
    if is_poetry_project() and shutil.which("poetry"):
        return ["poetry", "run", "python"]
    else:
        return ["python"]


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    if description:
        print(f"\n🧪 {description}")
        print("=" * 60)
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Success: {description}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Exit code: {e.returncode}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(
        description="EasyMLOps Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run only unit tests
  python run_tests.py --api              # Run only API tests
  python run_tests.py --coverage         # Run with detailed coverage
  python run_tests.py --fast             # Run fast tests only
  python run_tests.py --file test_config # Run specific test file
        """
    )
    
    # Test type options
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument("--service", action="store_true", help="Run service tests only")
    parser.add_argument("--model", action="store_true", help="Run model tests only")
    parser.add_argument("--database", action="store_true", help="Run database tests only")
    parser.add_argument("--monitoring", action="store_true", help="Run monitoring tests only")
    parser.add_argument("--deployment", action="store_true", help="Run deployment tests only")
    parser.add_argument("--config", action="store_true", help="Run config tests only")
    
    # Test execution options
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--slow", action="store_true", help="Run only slow tests")
    parser.add_argument("--parallel", "-j", type=int, help="Run tests in parallel (number of workers)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    
    # Coverage options
    parser.add_argument("--coverage", action="store_true", help="Run with detailed coverage report")
    parser.add_argument("--no-cov", action="store_true", help="Disable coverage reporting")
    
    # File selection
    parser.add_argument("--file", "-f", help="Run specific test file (without test_ prefix)")
    parser.add_argument("--function", help="Run specific test function")
    parser.add_argument("--class", dest="test_class", help="Run specific test class")
    
    # Other options
    parser.add_argument("--failfast", "-x", action="store_true", help="Stop on first failure")
    parser.add_argument("--lf", action="store_true", help="Run last failed tests")
    parser.add_argument("--pdb", action="store_true", help="Drop into debugger on failures")
    
    args = parser.parse_args()
    
    # Build pytest command with appropriate Python interpreter
    python_cmd = get_python_cmd()
    cmd = python_cmd + ["-m", "pytest"]
    
    # Add markers based on test type
    markers = []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.api:
        markers.append("api")
    if args.service:
        markers.append("service")
    if args.model:
        markers.append("model")
    if args.database:
        markers.append("database")
    if args.monitoring:
        markers.append("monitoring")
    if args.deployment:
        markers.append("deployment")
    if args.config:
        markers.append("config")
    
    if args.fast:
        markers.append("not slow")
    if args.slow:
        markers.append("slow")
    
    if markers:
        cmd.extend(["-m", " or ".join(markers)])
    
    # Add file selection
    if args.file:
        test_file = f"tests/test_{args.file}.py"
        if not Path(test_file).exists():
            print(f"❌ Test file not found: {test_file}")
            return 1
        cmd.append(test_file)
    
    if args.function:
        if not args.file:
            print("❌ --function requires --file to be specified")
            return 1
        cmd.append(f"::{args.function}")
    
    if args.test_class:
        if not args.file:
            print("❌ --class requires --file to be specified")
            return 1
        cmd.append(f"::{args.test_class}")
    
    # Add execution options
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    if args.verbose:
        cmd.append("-vvv")
    elif args.quiet:
        cmd.append("-q")
    
    # Coverage options
    if args.no_cov:
        cmd.extend(["--no-cov"])
    elif args.coverage:
        cmd.extend([
            "--cov=app",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml"
        ])
    
    # Other options
    if args.failfast:
        cmd.append("-x")
    if args.lf:
        cmd.append("--lf")
    if args.pdb:
        cmd.append("--pdb")
    
    # Run tests
    return run_command(cmd, "Running EasyMLOps Tests")


def run_quick_tests():
    """Run a quick test suite for development"""
    python_cmd = get_python_cmd()
    cmd = python_cmd + [
        "-m", "pytest",
        "-m", "not slow",
        "--tb=short",
        "-q"
    ]
    return run_command(cmd, "Quick Test Suite")


def run_ci_tests():
    """Run complete test suite for CI/CD"""
    python_cmd = get_python_cmd()
    cmd = python_cmd + [
        "-m", "pytest",
        "--cov=app",
        "--cov-report=xml",
        "--cov-fail-under=80",
        "--tb=short"
    ]
    return run_command(cmd, "CI/CD Test Suite")


def check_test_setup():
    """Check if test environment is properly set up"""
    print("🔍 Checking test environment setup...")
    
    # Check if pytest is installed in the appropriate environment
    python_cmd = get_python_cmd()
    try:
        result = subprocess.run(
            python_cmd + ["-c", "import pytest; print(pytest.__version__)"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        print(f"✅ pytest installed: {version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_type = "Poetry environment" if is_poetry_project() else "system Python"
        print(f"❌ pytest not installed in {env_type}. Run: poetry install" if is_poetry_project() else "pip install pytest")
        return False
    
    # Check test directory exists
    if not Path("tests").exists():
        print("❌ tests directory not found")
        return False
    print("✅ tests directory found")
    
    # Check for test files
    test_files = list(Path("tests").glob("test_*.py"))
    print(f"✅ Found {len(test_files)} test files")
    
    # Check conftest.py
    if Path("tests/conftest.py").exists():
        print("✅ conftest.py found")
    else:
        print("⚠️  conftest.py not found")
    
    # Check pytest.ini
    if Path("pytest.ini").exists():
        print("✅ pytest.ini found")
    else:
        print("⚠️  pytest.ini not found")
    
    return True


if __name__ == "__main__":
    # Special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            sys.exit(run_quick_tests())
        elif sys.argv[1] == "ci":
            sys.exit(run_ci_tests())
        elif sys.argv[1] == "check":
            success = check_test_setup()
            sys.exit(0 if success else 1)
    
    # Regular argument parsing
    sys.exit(main()) 