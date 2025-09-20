"""Expression-based generic checks for flexible custom validation.

This module provides powerful expression-based validation capabilities using
JSONPath queries and safe Python expression evaluation to enable custom
validation logic without requiring code changes.
"""

import ast
import json
import operator
import re
from typing import Any, Dict, List, Optional, Set, Union, Callable
from collections import defaultdict
from datetime import datetime, timedelta

from ...models.capture import RequestLog, CookieRecord, PageResult
from ...detectors.base import TagEvent
from ..indexing import PageIndex, AuditIndexes
from ..models import Severity
from .base import BaseCheck, CheckContext, CheckResult, register_check


class SafeExpressionError(Exception):
    """Exception raised for unsafe expression evaluation attempts."""
    pass


class JSONPathError(Exception):
    """Exception raised for JSONPath query errors."""
    pass


class SafeExpressionEvaluator:
    """Safe Python expression evaluator with restricted operations."""
    
    # Allowed operations for safe evaluation
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
        ast.Not: operator.not_,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    
    # Safe built-in functions
    SAFE_FUNCTIONS = {
        'abs': abs,
        'bool': bool,
        'int': int,
        'float': float,
        'str': str,
        'len': len,
        'min': min,
        'max': max,
        'sum': sum,
        'round': round,
        'sorted': sorted,
        'reversed': reversed,
        'any': any,
        'all': all,
        'isinstance': isinstance,
        'hasattr': hasattr,
        'getattr': getattr,
        're_search': re.search,
        're_match': re.match,
        're_findall': re.findall,
    }
    
    def __init__(self):
        self.context_vars: Dict[str, Any] = {}
    
    def set_context(self, variables: Dict[str, Any]) -> None:
        """Set context variables for expression evaluation."""
        self.context_vars = variables.copy()
        # Add safe functions to context
        self.context_vars.update(self.SAFE_FUNCTIONS)
    
    def evaluate(self, expression: str) -> Any:
        """Safely evaluate a Python expression.
        
        Args:
            expression: Python expression string to evaluate
            
        Returns:
            Result of expression evaluation
            
        Raises:
            SafeExpressionError: If expression contains unsafe operations
        """
        try:
            # Parse expression into AST
            tree = ast.parse(expression, mode='eval')
            
            # Validate AST for safety
            self._validate_ast_safety(tree)
            
            # Evaluate expression
            return self._eval_node(tree.body)
            
        except (ValueError, TypeError, SyntaxError, KeyError, AttributeError) as e:
            raise SafeExpressionError(f"Expression evaluation error: {e}")
    
    def _validate_ast_safety(self, node: ast.AST) -> None:
        """Validate that AST contains only safe operations."""
        for child in ast.walk(node):
            # Block dangerous node types
            if isinstance(child, (ast.Import, ast.ImportFrom, ast.Exec, 
                                 ast.FunctionDef, ast.ClassDef, ast.Lambda,
                                 ast.ListComp, ast.SetComp, ast.DictComp,
                                 ast.GeneratorExp, ast.Global, ast.Nonlocal)):
                raise SafeExpressionError(f"Unsafe operation: {type(child).__name__}")
            
            # Block attribute access to dangerous attributes
            if isinstance(child, ast.Attribute):
                if child.attr.startswith('_'):
                    raise SafeExpressionError(f"Access to private attribute: {child.attr}")
                
                # Block access to dangerous attributes
                dangerous_attrs = {
                    '__class__', '__bases__', '__subclasses__', '__mro__',
                    '__globals__', '__code__', '__closure__', '__dict__',
                    'func_globals', 'func_code', 'func_closure'
                }
                if child.attr in dangerous_attrs:
                    raise SafeExpressionError(f"Access to dangerous attribute: {child.attr}")
            
            # Block calls to non-whitelisted functions
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    func_name = child.func.id
                    if func_name not in self.SAFE_FUNCTIONS and func_name not in self.context_vars:
                        raise SafeExpressionError(f"Call to unsafe function: {func_name}")
    
    def _eval_node(self, node: ast.AST) -> Any:
        """Evaluate an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in self.context_vars:
                return self.context_vars[node.id]
            else:
                raise SafeExpressionError(f"Undefined variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise SafeExpressionError(f"Unsafe binary operator: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise SafeExpressionError(f"Unsafe unary operator: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            result = True
            
            for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                right = self._eval_node(comparator)
                op_func = self.SAFE_OPERATORS.get(type(op))
                if op_func is None:
                    raise SafeExpressionError(f"Unsafe comparison operator: {type(op).__name__}")
                
                result = result and op_func(left, right)
                if not result:
                    break
                left = right
            
            return result
        elif isinstance(node, ast.BoolOp):
            values = [self._eval_node(value) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
            else:
                raise SafeExpressionError(f"Unsafe boolean operator: {type(node.op).__name__}")
        elif isinstance(node, ast.Call):
            func = self._eval_node(node.func)
            args = [self._eval_node(arg) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value) for kw in node.keywords}
            return func(*args, **kwargs)
        elif isinstance(node, ast.Attribute):
            obj = self._eval_node(node.value)
            return getattr(obj, node.attr)
        elif isinstance(node, ast.Subscript):
            obj = self._eval_node(node.value)
            key = self._eval_node(node.slice)
            return obj[key]
        elif isinstance(node, ast.List):
            return [self._eval_node(element) for element in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(element) for element in node.elts)
        elif isinstance(node, ast.Dict):
            keys = [self._eval_node(key) for key in node.keys]
            values = [self._eval_node(value) for value in node.values]
            return dict(zip(keys, values))
        else:
            raise SafeExpressionError(f"Unsupported AST node type: {type(node).__name__}")


class SimpleJSONPath:
    """Simple JSONPath-like query implementation for audit data."""
    
    def __init__(self):
        pass
    
    def query(self, data: Any, path: str) -> List[Any]:
        """Execute JSONPath query on data.
        
        Args:
            data: Data to query
            path: JSONPath expression
            
        Returns:
            List of matching values
        """
        try:
            # Simple JSONPath implementation - supports basic paths
            if path == '$':
                return [data]
            
            if path.startswith('$.'):
                path = path[2:]
            elif path.startswith('$'):
                path = path[1:]
            
            return self._query_path(data, path.split('.'))
            
        except Exception as e:
            raise JSONPathError(f"JSONPath query error: {e}")
    
    def _query_path(self, data: Any, path_parts: List[str]) -> List[Any]:
        """Recursively query path parts."""
        if not path_parts:
            return [data]
        
        current_part = path_parts[0]
        remaining_parts = path_parts[1:]
        
        results = []
        
        # Handle array indexing
        if current_part.startswith('[') and current_part.endswith(']'):
            index_expr = current_part[1:-1]
            
            if index_expr == '*':
                # Wildcard - get all elements
                if isinstance(data, (list, tuple)):
                    for item in data:
                        results.extend(self._query_path(item, remaining_parts))
                elif isinstance(data, dict):
                    for value in data.values():
                        results.extend(self._query_path(value, remaining_parts))
            else:
                # Specific index
                try:
                    index = int(index_expr)
                    if isinstance(data, (list, tuple)) and 0 <= index < len(data):
                        results.extend(self._query_path(data[index], remaining_parts))
                except ValueError:
                    pass
        
        # Handle object property access
        elif isinstance(data, dict):
            if current_part == '*':
                # Wildcard - get all values
                for value in data.values():
                    results.extend(self._query_path(value, remaining_parts))
            elif current_part in data:
                results.extend(self._query_path(data[current_part], remaining_parts))
        
        # Handle object attribute access
        elif hasattr(data, current_part):
            attr_value = getattr(data, current_part)
            results.extend(self._query_path(attr_value, remaining_parts))
        
        # Handle list/array of objects
        elif isinstance(data, (list, tuple)):
            for item in data:
                results.extend(self._query_path(item, path_parts))
        
        return results


@register_check("expression")
class ExpressionCheck(BaseCheck):
    """Generic expression-based validation check."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
        self.evaluator = SafeExpressionEvaluator()
        self.jsonpath = SimpleJSONPath()
    
    @classmethod
    def get_supported_config_keys(cls) -> List[str]:
        return [
            'expression',
            'data_context',
            'jsonpath_queries',
            'expected_result',
            'result_type',
            'variables',
            'pre_expressions',
            'validation_mode'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute expression-based validation."""
        config = context.check_config
        
        expression = config.get('expression')
        if not expression:
            return self._create_result(
                context=context,
                passed=False,
                message="No expression provided for evaluation"
            )
        
        try:
            # Build evaluation context
            eval_context = self._build_evaluation_context(context, config)
            
            # Set context in evaluator
            self.evaluator.set_context(eval_context)
            
            # Execute pre-expressions if provided
            self._execute_pre_expressions(config)
            
            # Execute main expression
            result = self.evaluator.evaluate(expression)
            
            # Validate result
            validation_result = self._validate_result(result, config)
            
            return validation_result
            
        except (SafeExpressionError, JSONPathError) as e:
            return self._create_result(
                context=context,
                passed=False,
                message=f"Expression evaluation failed: {str(e)}",
                details=f"Expression: {expression}"
            )
    
    def _build_evaluation_context(
        self, 
        context: CheckContext, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build context variables for expression evaluation."""
        eval_context = {}
        
        # Add basic audit data
        eval_context.update({
            'indexes': context.indexes,
            'query': context.query,
            'pages': context.indexes.pages.pages,
            'summary': context.indexes.summary,
            'config': config,
            'rule_config': context.rule_config,
            'environment': context.environment,
            'target_urls': context.target_urls
        })
        
        # Add data context based on configuration
        data_context = config.get('data_context', 'full')
        
        if data_context == 'requests':
            eval_context['data'] = context.query.query().requests()
        elif data_context == 'cookies':
            eval_context['data'] = context.query.query().cookies()
        elif data_context == 'events':
            eval_context['data'] = context.query.query().events()
        elif data_context == 'pages':
            eval_context['data'] = context.indexes.pages.pages
        elif data_context == 'summary':
            eval_context['data'] = context.indexes.summary
        else:
            # Full context
            eval_context['data'] = {
                'requests': context.query.query().requests(),
                'cookies': context.query.query().cookies(),
                'events': context.query.query().events(),
                'pages': context.indexes.pages.pages,
                'summary': context.indexes.summary
            }
        
        # Execute JSONPath queries if provided
        jsonpath_queries = config.get('jsonpath_queries', {})
        for var_name, query in jsonpath_queries.items():
            try:
                query_results = self.jsonpath.query(eval_context['data'], query)
                eval_context[var_name] = query_results
            except JSONPathError:
                eval_context[var_name] = []
        
        # Add custom variables
        custom_vars = config.get('variables', {})
        eval_context.update(custom_vars)
        
        # Add utility functions
        eval_context.update({
            'count': len,
            'avg': lambda lst: sum(lst) / len(lst) if lst else 0,
            'unique': lambda lst: list(set(lst)),
            'flatten': lambda lst: [item for sublist in lst for item in sublist],
            'filter_by': self._filter_by,
            'group_by': self._group_by,
            'extract': self._extract_values,
            'datetime': datetime,
            'timedelta': timedelta,
            'now': datetime.now,
            'utcnow': datetime.utcnow,
        })
        
        return eval_context
    
    def _execute_pre_expressions(self, config: Dict[str, Any]) -> None:
        """Execute pre-expressions to set up additional context."""
        pre_expressions = config.get('pre_expressions', [])
        
        for expr in pre_expressions:
            try:
                self.evaluator.evaluate(expr)
            except SafeExpressionError:
                # Ignore pre-expression failures
                pass
    
    def _validate_result(
        self, 
        result: Any, 
        config: Dict[str, Any]
    ) -> CheckResult:
        """Validate expression result against expectations."""
        expected_result = config.get('expected_result')
        result_type = config.get('result_type', 'boolean')
        validation_mode = config.get('validation_mode', 'strict')
        
        # Convert result to expected type
        if result_type == 'boolean':
            actual_result = bool(result)
        elif result_type == 'number':
            try:
                actual_result = float(result)
            except (ValueError, TypeError):
                actual_result = 0
        elif result_type == 'string':
            actual_result = str(result)
        elif result_type == 'list':
            if isinstance(result, (list, tuple)):
                actual_result = list(result)
            else:
                actual_result = [result]
        else:
            actual_result = result
        
        # Determine if check passed
        if expected_result is not None:
            if validation_mode == 'strict':
                passed = actual_result == expected_result
            else:
                # Loose validation
                passed = bool(actual_result) == bool(expected_result)
        else:
            # No expected result - just check if result is truthy
            passed = bool(actual_result)
        
        # Create result message
        if expected_result is not None:
            message = f"Expression result: {actual_result} (expected: {expected_result})"
        else:
            message = f"Expression result: {actual_result}"
        
        return CheckResult(
            check_id=self.check_id,
            check_name=self.name,
            passed=passed,
            severity=self._determine_severity(
                CheckContext(
                    indexes=None,
                    query=None,
                    rule_id="",
                    rule_config=config,
                    check_config=config
                )
            ),
            message=message,
            found_count=1 if not passed else 0,
            expected_count=0,
            evidence=[{
                'expression_result': str(actual_result),
                'result_type': type(actual_result).__name__,
                'expected': str(expected_result) if expected_result is not None else None
            }]
        )
    
    def _filter_by(self, items: List[Any], condition: str) -> List[Any]:
        """Filter items by condition expression."""
        filtered = []
        for item in items:
            # Create temporary context with item
            temp_context = self.evaluator.context_vars.copy()
            temp_context['item'] = item
            
            temp_evaluator = SafeExpressionEvaluator()
            temp_evaluator.set_context(temp_context)
            
            try:
                if temp_evaluator.evaluate(condition):
                    filtered.append(item)
            except SafeExpressionError:
                continue
        
        return filtered
    
    def _group_by(self, items: List[Any], key_expr: str) -> Dict[Any, List[Any]]:
        """Group items by key expression."""
        groups = defaultdict(list)
        
        for item in items:
            # Create temporary context with item
            temp_context = self.evaluator.context_vars.copy()
            temp_context['item'] = item
            
            temp_evaluator = SafeExpressionEvaluator()
            temp_evaluator.set_context(temp_context)
            
            try:
                key = temp_evaluator.evaluate(key_expr)
                groups[key].append(item)
            except SafeExpressionError:
                groups['__error__'].append(item)
        
        return dict(groups)
    
    def _extract_values(self, items: List[Any], value_expr: str) -> List[Any]:
        """Extract values from items using expression."""
        values = []
        
        for item in items:
            # Create temporary context with item
            temp_context = self.evaluator.context_vars.copy()
            temp_context['item'] = item
            
            temp_evaluator = SafeExpressionEvaluator()
            temp_evaluator.set_context(temp_context)
            
            try:
                value = temp_evaluator.evaluate(value_expr)
                values.append(value)
            except SafeExpressionError:
                continue
        
        return values


@register_check("jsonpath")
class JSONPathCheck(BaseCheck):
    """JSONPath-based validation check."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
        self.jsonpath = SimpleJSONPath()
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'jsonpath',
            'expected_count',
            'min_count',
            'max_count',
            'expected_values',
            'data_context',
            'value_validation'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute JSONPath-based validation."""
        config = context.check_config
        
        jsonpath_expr = config.get('jsonpath')
        if not jsonpath_expr:
            return self._create_result(
                context=context,
                passed=False,
                message="No JSONPath expression provided"
            )
        
        try:
            # Get data context
            data = self._get_data_context(context, config)
            
            # Execute JSONPath query
            results = self.jsonpath.query(data, jsonpath_expr)
            
            # Validate results
            validation_result = self._validate_jsonpath_results(results, config, context)
            
            return validation_result
            
        except JSONPathError as e:
            return self._create_result(
                context=context,
                passed=False,
                message=f"JSONPath query failed: {str(e)}",
                details=f"JSONPath: {jsonpath_expr}"
            )
    
    def _get_data_context(self, context: CheckContext, config: Dict[str, Any]) -> Any:
        """Get data context for JSONPath query."""
        data_context = config.get('data_context', 'full')
        
        if data_context == 'requests':
            # Convert to serializable format
            return [self._serialize_object(req) for req in context.query.query().requests()]
        elif data_context == 'cookies':
            return [self._serialize_object(cookie) for cookie in context.query.query().cookies()]
        elif data_context == 'events':
            return [self._serialize_object(event) for event in context.query.query().events()]
        elif data_context == 'pages':
            return [self._serialize_object(page) for page in context.indexes.pages.pages]
        elif data_context == 'summary':
            return self._serialize_object(context.indexes.summary)
        else:
            # Full context
            return {
                'requests': [self._serialize_object(req) for req in context.query.query().requests()],
                'cookies': [self._serialize_object(cookie) for cookie in context.query.query().cookies()],
                'events': [self._serialize_object(event) for event in context.query.query().events()],
                'pages': [self._serialize_object(page) for page in context.indexes.pages.pages],
                'summary': self._serialize_object(context.indexes.summary)
            }
    
    def _serialize_object(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if hasattr(obj, 'dict'):
            # Pydantic model
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            # Regular object
            return {k: self._serialize_value(v) for k, v in obj.__dict__.items()}
        else:
            return self._serialize_value(obj)
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, datetime):
            return value.isoformat()
        else:
            return str(value)
    
    def _validate_jsonpath_results(
        self, 
        results: List[Any], 
        config: Dict[str, Any], 
        context: CheckContext
    ) -> CheckResult:
        """Validate JSONPath query results."""
        found_count = len(results)
        
        # Check count expectations
        expected_count = config.get('expected_count')
        min_count = config.get('min_count')
        max_count = config.get('max_count')
        
        count_passed = True
        count_message_parts = []
        
        if expected_count is not None:
            count_passed = found_count == expected_count
            count_message_parts.append(f"expected exactly {expected_count}")
        else:
            if min_count is not None:
                count_passed = count_passed and found_count >= min_count
                count_message_parts.append(f"min {min_count}")
            
            if max_count is not None:
                count_passed = count_passed and found_count <= max_count
                count_message_parts.append(f"max {max_count}")
        
        # Check value expectations
        expected_values = config.get('expected_values')
        value_validation = config.get('value_validation')
        value_passed = True
        
        if expected_values is not None:
            if isinstance(expected_values, list):
                value_passed = set(results) == set(expected_values)
            else:
                value_passed = expected_values in results
        
        if value_validation:
            # Additional value validation logic could be added here
            pass
        
        overall_passed = count_passed and value_passed
        
        # Build message
        count_desc = " and ".join(count_message_parts) if count_message_parts else "any count"
        message = f"JSONPath query returned {found_count} results ({count_desc})"
        
        return self._create_result(
            context=context,
            passed=overall_passed,
            message=message,
            found_count=found_count,
            expected_count=expected_count,
            evidence=results[:10]  # Limit evidence to first 10 results
        )