"""
Integration test runner with comprehensive reporting and benchmarking.
Provides automated execution of all integration tests with detailed reporting.

Requirements addressed:
- 6.4: Processing time and throughput metrics
- 6.5: Detailed comparison reports
- 7.1: Enhancement techniques validation
- 7.2: Normalization consistency validation
"""

import unittest
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# Import integration test classes
from tests.test_integration_pipeline import (
    TestEndToEndPipeline,
    TestPerformanceBenchmarking, 
    TestRegressionTesting,
    TestCompetitionFormatCompliance
)


class IntegrationTestRunner:
    """Comprehensive integration test runner with reporting and benchmarking."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the integration test runner.
        
        Args:
            output_dir: Directory to save test reports (default: tests/reports)
        """
        self.output_dir = output_dir or Path("tests/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        self.performance_metrics = {}
        self.regression_results = {}
        self.compliance_results = {}
        
        # Test suite configuration
        self.test_suites = {
            'end_to_end': TestEndToEndPipeline,
            'performance': TestPerformanceBenchmarking,
            'regression': TestRegressionTesting,
            'compliance': TestCompetitionFormatCompliance
        }
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run all integration test suites.
        
        Args:
            verbose: Whether to print detailed output
            
        Returns:
            Complete test results dictionary
        """
        start_time = time.time()
        
        if verbose:
            print("=" * 80)
            print("HISTORICAL MAP TEXT PROCESSING - INTEGRATION TEST SUITE")
            print("=" * 80)
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        overall_results = {
            'start_time': datetime.now().isoformat(),
            'test_suites': {},
            'summary': {},
            'performance_benchmarks': {},
            'regression_analysis': {},
            'compliance_validation': {}
        }
        
        # Run each test suite
        for suite_name, suite_class in self.test_suites.items():
            if verbose:
                print(f"Running {suite_name.replace('_', ' ').title()} Tests...")
                print("-" * 50)
            
            suite_results = self._run_test_suite(suite_class, verbose)
            overall_results['test_suites'][suite_name] = suite_results
            
            if verbose:
                self._print_suite_summary(suite_name, suite_results)
                print()
        
        # Generate comprehensive analysis
        overall_results['summary'] = self._generate_test_summary(overall_results['test_suites'])
        overall_results['performance_benchmarks'] = self._extract_performance_metrics(overall_results)
        overall_results['regression_analysis'] = self._extract_regression_analysis(overall_results)
        overall_results['compliance_validation'] = self._extract_compliance_results(overall_results)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        overall_results['total_execution_time'] = total_time
        overall_results['end_time'] = datetime.now().isoformat()
        
        if verbose:
            self._print_overall_summary(overall_results)
        
        # Save detailed report
        self._save_test_report(overall_results)
        
        return overall_results
    
    def run_specific_suite(self, suite_name: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run a specific test suite.
        
        Args:
            suite_name: Name of the test suite to run
            verbose: Whether to print detailed output
            
        Returns:
            Test results for the specific suite
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite_class = self.test_suites[suite_name]
        return self._run_test_suite(suite_class, verbose)
    
    def _run_test_suite(self, suite_class, verbose: bool = True) -> Dict[str, Any]:
        """Run a single test suite and collect results."""
        suite_start_time = time.time()
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(suite_class)
        
        # Custom test result collector
        result_collector = DetailedTestResult()
        
        # Run tests
        import os
        null_device = 'nul' if os.name == 'nt' else '/dev/null'
        runner = unittest.TextTestRunner(
            stream=sys.stdout if verbose else open(null_device, 'w'),
            verbosity=2 if verbose else 0,
            resultclass=lambda stream, descriptions, verbosity: result_collector
        )
        
        test_result = runner.run(suite)
        
        suite_execution_time = time.time() - suite_start_time
        
        # Collect detailed results
        suite_results = {
            'execution_time': suite_execution_time,
            'tests_run': test_result.testsRun,
            'failures': len(test_result.failures),
            'errors': len(test_result.errors),
            'skipped': len(getattr(test_result, 'skipped', [])),
            'success_rate': (test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / max(test_result.testsRun, 1),
            'detailed_results': result_collector.get_detailed_results(),
            'performance_data': result_collector.get_performance_data(),
            'failure_details': self._format_failures(test_result.failures),
            'error_details': self._format_errors(test_result.errors)
        }
        
        return suite_results
    
    def _generate_test_summary(self, test_suites: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall test summary."""
        total_tests = sum(suite['tests_run'] for suite in test_suites.values())
        total_failures = sum(suite['failures'] for suite in test_suites.values())
        total_errors = sum(suite['errors'] for suite in test_suites.values())
        total_skipped = sum(suite['skipped'] for suite in test_suites.values())
        
        overall_success_rate = (total_tests - total_failures - total_errors) / max(total_tests, 1)
        
        return {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'overall_success_rate': overall_success_rate,
            'suite_success_rates': {
                name: suite['success_rate'] 
                for name, suite in test_suites.items()
            }
        }
    
    def _extract_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance benchmarking metrics."""
        performance_data = {}
        
        if 'performance' in results['test_suites']:
            perf_suite = results['test_suites']['performance']
            performance_data = perf_suite.get('performance_data', {})
        
        # Add processing time metrics from end-to-end tests
        if 'end_to_end' in results['test_suites']:
            e2e_suite = results['test_suites']['end_to_end']
            e2e_performance = e2e_suite.get('performance_data', {})
            performance_data.update(e2e_performance)
        
        return performance_data
    
    def _extract_regression_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract regression testing analysis."""
        regression_data = {}
        
        if 'regression' in results['test_suites']:
            regression_suite = results['test_suites']['regression']
            
            # Analyze regression test results
            regression_data = {
                'baseline_comparison': {},
                'accuracy_trends': {},
                'performance_regression': {},
                'recommendations': []
            }
            
            # Extract specific regression metrics from detailed results
            detailed_results = regression_suite.get('detailed_results', {})
            for test_name, test_data in detailed_results.items():
                if 'regression' in test_name.lower():
                    regression_data['baseline_comparison'][test_name] = test_data
        
        return regression_data
    
    def _extract_compliance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract competition format compliance results."""
        compliance_data = {}
        
        if 'compliance' in results['test_suites']:
            compliance_suite = results['test_suites']['compliance']
            
            compliance_data = {
                'format_validation': {},
                'task_compliance': {},
                'edge_case_handling': {},
                'multilingual_support': {}
            }
            
            # Extract compliance metrics from detailed results
            detailed_results = compliance_suite.get('detailed_results', {})
            for test_name, test_data in detailed_results.items():
                if 'task' in test_name.lower():
                    task_id = self._extract_task_id(test_name)
                    if task_id:
                        compliance_data['task_compliance'][task_id] = test_data
                elif 'multilingual' in test_name.lower():
                    compliance_data['multilingual_support'][test_name] = test_data
                elif 'edge_case' in test_name.lower():
                    compliance_data['edge_case_handling'][test_name] = test_data
        
        return compliance_data
    
    def _extract_task_id(self, test_name: str) -> Optional[str]:
        """Extract task ID from test name."""
        for task_id in ['task1', 'task2', 'task3', 'task4']:
            if task_id in test_name.lower():
                return task_id
        return None
    
    def _format_failures(self, failures: List) -> List[Dict[str, str]]:
        """Format test failures for reporting."""
        formatted_failures = []
        for test, traceback_str in failures:
            formatted_failures.append({
                'test_name': str(test),
                'traceback': traceback_str,
                'error_type': 'AssertionError'
            })
        return formatted_failures
    
    def _format_errors(self, errors: List) -> List[Dict[str, str]]:
        """Format test errors for reporting."""
        formatted_errors = []
        for test, traceback_str in errors:
            formatted_errors.append({
                'test_name': str(test),
                'traceback': traceback_str,
                'error_type': 'Exception'
            })
        return formatted_errors
    
    def _print_suite_summary(self, suite_name: str, suite_results: Dict[str, Any]):
        """Print summary for a test suite."""
        print(f"Suite: {suite_name.replace('_', ' ').title()}")
        print(f"  Tests Run: {suite_results['tests_run']}")
        print(f"  Failures: {suite_results['failures']}")
        print(f"  Errors: {suite_results['errors']}")
        print(f"  Success Rate: {suite_results['success_rate']:.2%}")
        print(f"  Execution Time: {suite_results['execution_time']:.2f}s")
        
        if suite_results['failures'] > 0 or suite_results['errors'] > 0:
            print("  Issues Found:")
            for failure in suite_results['failure_details']:
                print(f"    FAIL: {failure['test_name']}")
            for error in suite_results['error_details']:
                print(f"    ERROR: {error['test_name']}")
    
    def _print_overall_summary(self, results: Dict[str, Any]):
        """Print overall test summary."""
        summary = results['summary']
        
        print("=" * 80)
        print("OVERALL TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Failures: {summary['total_failures']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Skipped: {summary['total_skipped']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.2%}")
        print(f"Total Execution Time: {results['total_execution_time']:.2f}s")
        print()
        
        # Performance summary
        if results['performance_benchmarks']:
            print("PERFORMANCE BENCHMARKS:")
            perf_data = results['performance_benchmarks']
            for metric_name, metric_value in perf_data.items():
                if isinstance(metric_value, (int, float)):
                    print(f"  {metric_name}: {metric_value:.3f}")
            print()
        
        # Compliance summary
        compliance = results['compliance_validation']
        if compliance.get('task_compliance'):
            print("COMPETITION COMPLIANCE:")
            for task_id, task_data in compliance['task_compliance'].items():
                status = "PASS" if task_data.get('passed', False) else "FAIL"
                print(f"  {task_id.upper()}: {status}")
            print()
        
        print(f"Detailed report saved to: {self.output_dir}")
        print("=" * 80)
    
    def _save_test_report(self, results: Dict[str, Any]):
        """Save comprehensive test report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_report_path = self.output_dir / f"integration_test_report_{timestamp}.json"
        with open(json_report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save human-readable report
        text_report_path = self.output_dir / f"integration_test_report_{timestamp}.txt"
        with open(text_report_path, 'w') as f:
            self._write_text_report(f, results)
        
        # Save performance report
        if results['performance_benchmarks']:
            perf_report_path = self.output_dir / f"performance_benchmarks_{timestamp}.json"
            with open(perf_report_path, 'w') as f:
                json.dump(results['performance_benchmarks'], f, indent=2, default=str)
    
    def _write_text_report(self, file, results: Dict[str, Any]):
        """Write human-readable text report."""
        file.write("HISTORICAL MAP TEXT PROCESSING - INTEGRATION TEST REPORT\n")
        file.write("=" * 80 + "\n\n")
        
        file.write(f"Report Generated: {results['end_time']}\n")
        file.write(f"Total Execution Time: {results['total_execution_time']:.2f} seconds\n\n")
        
        # Summary section
        summary = results['summary']
        file.write("EXECUTIVE SUMMARY\n")
        file.write("-" * 40 + "\n")
        file.write(f"Total Tests Executed: {summary['total_tests']}\n")
        file.write(f"Overall Success Rate: {summary['overall_success_rate']:.2%}\n")
        file.write(f"Failures: {summary['total_failures']}\n")
        file.write(f"Errors: {summary['total_errors']}\n\n")
        
        # Suite details
        file.write("TEST SUITE DETAILS\n")
        file.write("-" * 40 + "\n")
        for suite_name, suite_data in results['test_suites'].items():
            file.write(f"\n{suite_name.replace('_', ' ').title()} Suite:\n")
            file.write(f"  Tests: {suite_data['tests_run']}\n")
            file.write(f"  Success Rate: {suite_data['success_rate']:.2%}\n")
            file.write(f"  Execution Time: {suite_data['execution_time']:.2f}s\n")
            
            if suite_data['failures'] > 0 or suite_data['errors'] > 0:
                file.write("  Issues:\n")
                for failure in suite_data['failure_details']:
                    file.write(f"    FAIL: {failure['test_name']}\n")
                for error in suite_data['error_details']:
                    file.write(f"    ERROR: {error['test_name']}\n")
        
        # Performance benchmarks
        if results['performance_benchmarks']:
            file.write("\n\nPERFORMANCE BENCHMARKS\n")
            file.write("-" * 40 + "\n")
            for metric, value in results['performance_benchmarks'].items():
                if isinstance(value, (int, float)):
                    file.write(f"{metric}: {value:.3f}\n")
        
        # Compliance validation
        compliance = results['compliance_validation']
        if compliance:
            file.write("\n\nCOMPETITION COMPLIANCE VALIDATION\n")
            file.write("-" * 40 + "\n")
            
            if compliance.get('task_compliance'):
                file.write("Task Format Compliance:\n")
                for task_id, task_data in compliance['task_compliance'].items():
                    status = "PASS" if task_data.get('passed', False) else "FAIL"
                    file.write(f"  {task_id.upper()}: {status}\n")
            
            if compliance.get('multilingual_support'):
                file.write("\nMultilingual Support:\n")
                for test_name, test_data in compliance['multilingual_support'].items():
                    status = "PASS" if test_data.get('passed', False) else "FAIL"
                    file.write(f"  {test_name}: {status}\n")


class DetailedTestResult(unittest.TestResult):
    """Custom test result class that collects detailed information."""
    
    def __init__(self):
        super().__init__()
        self.detailed_results = {}
        self.performance_data = {}
        self.test_start_times = {}
    
    def startTest(self, test):
        """Record test start time."""
        super().startTest(test)
        self.test_start_times[test] = time.time()
    
    def stopTest(self, test):
        """Record test completion and collect data."""
        super().stopTest(test)
        
        test_name = str(test)
        execution_time = time.time() - self.test_start_times.get(test, time.time())
        
        # Store detailed results
        self.detailed_results[test_name] = {
            'execution_time': execution_time,
            'passed': test not in [f[0] for f in self.failures] and test not in [e[0] for e in self.errors],
            'test_method': test._testMethodName if hasattr(test, '_testMethodName') else 'unknown'
        }
        
        # Store performance data for performance tests
        if 'performance' in test_name.lower() or 'benchmark' in test_name.lower():
            self.performance_data[test_name] = {
                'execution_time': execution_time,
                'test_type': 'performance'
            }
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """Get detailed test results."""
        return self.detailed_results
    
    def get_performance_data(self) -> Dict[str, Any]:
        """Get performance-specific data."""
        return self.performance_data


def main():
    """Main entry point for integration test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Historical Map Text Processing Integration Tests')
    parser.add_argument('--suite', choices=['end_to_end', 'performance', 'regression', 'compliance'], 
                       help='Run specific test suite only')
    parser.add_argument('--output-dir', type=Path, help='Output directory for reports')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = IntegrationTestRunner(output_dir=args.output_dir)
    
    # Run tests
    if args.suite:
        results = runner.run_specific_suite(args.suite, verbose=not args.quiet)
    else:
        results = runner.run_all_tests(verbose=not args.quiet)
    
    # Exit with appropriate code
    if results.get('summary', {}).get('total_failures', 0) > 0 or results.get('summary', {}).get('total_errors', 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()