# Exit Code Reference for CI/CD Integration

Tag Sentinel CLI uses standardized exit codes to integrate seamlessly with CI/CD pipelines and automated quality gates.

## Exit Code Mapping

| Exit Code | Name | Description | When It Occurs |
|-----------|------|-------------|----------------|
| `0` | SUCCESS | All operations completed successfully | All pages captured successfully, all rules passed |
| `1` | RULE_FAILURES | Some rules failed or pages failed to capture | Non-critical rule failures, page capture failures |
| `2` | CRITICAL_FAILURES | Critical rule failures detected | Critical severity rule failures found |
| `3` | CONFIG_ERROR | Configuration or setup error | Invalid config files, missing files, invalid parameters |
| `4` | RUNTIME_ERROR | Runtime error during execution | Unexpected errors, system issues |
| `5` | TIMEOUT_ERROR | Operation timed out | Operations exceeded configured timeout limits |

## CI/CD Integration Examples

### Basic Success/Failure Check
```bash
# Run audit and check if it passed
if openaudit run --env production --rules config/rules.yaml https://example.com; then
    echo "✅ Audit passed"
else
    echo "❌ Audit failed with exit code $?"
    exit 1
fi
```

### Detailed Exit Code Handling
```bash
openaudit run --env production --rules config/rules.yaml https://example.com
exit_code=$?

case $exit_code in
    0)
        echo "✅ All checks passed"
        ;;
    1)
        echo "⚠️  Rule failures or page capture issues detected"
        # Continue deployment with warnings
        ;;
    2)
        echo "🚨 Critical failures detected"
        echo "❌ Blocking deployment"
        exit 2
        ;;
    3)
        echo "❌ Configuration error"
        exit 3
        ;;
    *)
        echo "❌ Unexpected error (exit code: $exit_code)"
        exit $exit_code
        ;;
esac
```

### GitHub Actions Integration
```yaml
name: Web Analytics Audit
on: [push, pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install Tag Sentinel
        run: pip install tag-sentinel

      - name: Run Analytics Audit
        run: |
          openaudit run \
            --env ${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }} \
            --rules config/audit-rules.yaml \
            --json \
            --out audit-results/ \
            https://example.com

      - name: Upload Audit Results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: audit-results
          path: audit-results/
```

### Jenkins Pipeline Integration
```groovy
pipeline {
    agent any

    stages {
        stage('Analytics Audit') {
            steps {
                script {
                    def exitCode = sh(
                        script: '''
                            openaudit run \
                                --env ${BRANCH_NAME == 'main' ? 'production' : 'staging'} \
                                --rules config/rules.yaml \
                                --json \
                                --out audit-results/ \
                                ${TARGET_URL}
                        ''',
                        returnStatus: true
                    )

                    switch(exitCode) {
                        case 0:
                            echo "✅ Audit passed"
                            break
                        case 1:
                            echo "⚠️  Non-critical issues found"
                            // Continue with warnings
                            break
                        case 2:
                            error("🚨 Critical audit failures - blocking deployment")
                            break
                        default:
                            error("❌ Audit failed with exit code ${exitCode}")
                    }
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'audit-results/**', allowEmptyArchive: true
                }
            }
        }
    }
}
```

## Configuration Options for Exit Codes

Control exit code behavior with these CLI flags:

- `--exit-on-warnings`: Exit with code 1 for warning-level rule failures (default: false)
- `--exit-on-critical`: Exit with code 2 for critical-level rule failures (default: true)

## JSON Output for Programmatic Access

Use `--json` flag to get machine-readable output including exit code:

```json
{
  "summary": {
    "start_time": "2025-01-15T10:30:00.000Z",
    "duration_seconds": 12.5,
    "environment": "production"
  },
  "capture": {
    "total_pages": 5,
    "successful": 5,
    "failed": 0,
    "success_rate": 100.0
  },
  "rules": {
    "evaluated": true,
    "total": 25,
    "passed": 23,
    "failed": 2,
    "critical": 0,
    "warning": 2
  },
  "exit_code": 1
}
```

This allows CI/CD systems to parse results programmatically and make decisions based on specific failure types and counts.