# Mathematical Utilities Module

This module contains utilities for solving and evaluating mathematical problems, including Python code execution and grading.

## Files

- `math_eval.py`: Core mathematical problem evaluation
- `math_utils.py`: Mathematical utility functions
- `python_executor.py`: Safe Python code execution
- `grader.py`: Problem grading utilities

## Usage

```python
from src.inference.math_utils import evaluate_math_problem, execute_python_code

# Evaluate a math problem
result = evaluate_math_problem(
    problem="What is 2+2?",
    solution="4",
    student_answer="The answer is 4."
)

# Execute Python code safely
result = execute_python_code(
    code="print(2+2)",
    timeout=5
)
```

## Functions

### evaluate_math_problem
Evaluates a mathematical problem solution.

### execute_python_code
Safely executes Python code with timeout and resource limits.

### grade_problem
Grades a mathematical problem solution.

### parse_math_expression
Parses and evaluates mathematical expressions. 