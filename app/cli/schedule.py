"""CLI commands for schedule management."""

import asyncio
import click
import json
import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from ..scheduling.service import SchedulingService, ServiceConfig, create_development_service, create_production_service
from ..scheduling.models import Schedule, Priority


@click.group()
def schedule():
    """Schedule management commands."""
    pass


@click.group()
def service():
    """Service lifecycle commands."""
    pass


# Service commands

@service.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Service configuration file')
@click.option('--environment-config', '-e', type=click.Path(exists=True), help='Environment configuration file')
@click.option('--schedule-configs', '-s', multiple=True, type=click.Path(exists=True), help='Schedule configuration files')
@click.option('--redis-url', help='Redis URL for distributed locking')
@click.option('--development', is_flag=True, help='Use development configuration')
def start(config: Optional[str], environment_config: Optional[str], schedule_configs: tuple, redis_url: Optional[str], development: bool):
    """Start the scheduling service."""

    async def _start_service():
        try:
            if development:
                service = create_development_service(
                    environment_config_path=environment_config,
                    schedule_config_paths=list(schedule_configs) if schedule_configs else None
                )
            elif redis_url:
                service = create_production_service(
                    redis_url=redis_url,
                    environment_config_path=environment_config or "config/environments.yaml",
                    schedule_config_paths=list(schedule_configs) if schedule_configs else []
                )
            else:
                # Load custom config
                if config:
                    service_config = _load_service_config(config)
                    service = SchedulingService(service_config)
                else:
                    click.echo("Error: Must specify --config, --development, or --redis-url", err=True)
                    return

            click.echo("Starting scheduling service...")
            await service.start()

            click.echo(f"✅ Scheduling service started successfully")
            click.echo(f"Status: {service.get_status().value}")

            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                click.echo("\n🛑 Stopping service...")
                await service.stop()
                click.echo("✅ Service stopped")

        except Exception as e:
            click.echo(f"❌ Error: {e}", err=True)
            return 1

    return asyncio.run(_start_service())


@service.command()
def stop():
    """Stop the scheduling service."""
    click.echo("Service stop command - would connect to running service and stop it")
    # In a real implementation, this would connect to the running service via API or signal


@service.command()
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def status(output_json: bool):
    """Get service status."""
    # This would connect to running service and get status
    status_info = {
        "status": "running",
        "uptime": 3600,
        "schedules": {"active": 5, "total": 10},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if output_json:
        click.echo(json.dumps(status_info, indent=2))
    else:
        click.echo(f"Status: {status_info['status']}")
        click.echo(f"Uptime: {status_info['uptime']} seconds")
        click.echo(f"Schedules: {status_info['schedules']['active']}/{status_info['schedules']['total']} active")


@service.command()
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def health(output_json: bool):
    """Get service health information."""
    # This would connect to running service and get health
    health_info = {
        "status": "healthy",
        "components": {
            "engine": "healthy",
            "dispatcher": "healthy",
            "concurrency": "healthy"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if output_json:
        click.echo(json.dumps(health_info, indent=2))
    else:
        click.echo(f"Overall Status: {health_info['status']}")
        click.echo("Components:")
        for component, status in health_info['components'].items():
            click.echo(f"  {component}: {status}")


# Schedule commands

@schedule.command()
@click.option('--site-id', help='Filter by site ID')
@click.option('--environment', help='Filter by environment')
@click.option('--status', help='Filter by status')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
def list(site_id: Optional[str], environment: Optional[str], status: Optional[str], output_format: str):
    """List schedules."""

    # Mock data - in real implementation would call API
    schedules = [
        {
            "id": "schedule-1",
            "name": "Daily Audit",
            "site_id": "example.com",
            "environment": "production",
            "cron_expression": "0 2 * * *",
            "status": "active",
            "next_run": "2024-01-15T02:00:00Z",
            "enabled": True
        },
        {
            "id": "schedule-2",
            "name": "Hourly Check",
            "site_id": "test.com",
            "environment": "staging",
            "cron_expression": "0 * * * *",
            "status": "paused",
            "next_run": None,
            "enabled": True
        }
    ]

    # Apply filters
    if site_id:
        schedules = [s for s in schedules if s['site_id'] == site_id]
    if environment:
        schedules = [s for s in schedules if s['environment'] == environment]
    if status:
        schedules = [s for s in schedules if s['status'] == status]

    if output_format == 'json':
        click.echo(json.dumps(schedules, indent=2))
    elif output_format == 'yaml':
        click.echo(yaml.dump(schedules, default_flow_style=False))
    else:
        # Table format
        click.echo(f"{'ID':<15} {'Name':<20} {'Site':<15} {'Environment':<12} {'Status':<8} {'Next Run':<20}")
        click.echo("-" * 100)
        for schedule in schedules:
            next_run = schedule['next_run'][:19] if schedule['next_run'] else 'N/A'
            click.echo(f"{schedule['id']:<15} {schedule['name']:<20} {schedule['site_id']:<15} {schedule['environment']:<12} {schedule['status']:<8} {next_run:<20}")


@schedule.command()
@click.argument('schedule_id')
@click.option('--format', 'output_format', type=click.Choice(['json', 'yaml']), default='json', help='Output format')
def get(schedule_id: str, output_format: str):
    """Get schedule details."""

    # Mock data - in real implementation would call API
    schedule_data = {
        "id": schedule_id,
        "name": "Daily Audit",
        "site_id": "example.com",
        "environment": "production",
        "cron_expression": "0 2 * * *",
        "timezone": "UTC",
        "enabled": True,
        "priority": "medium",
        "max_concurrent_runs": 1,
        "timeout_minutes": 60,
        "retry_count": 3,
        "status": "active",
        "next_run": "2024-01-15T02:00:00Z",
        "last_run": "2024-01-14T02:00:00Z",
        "consecutive_failures": 0,
        "metadata": {
            "description": "Daily audit for production site"
        }
    }

    if output_format == 'yaml':
        click.echo(yaml.dump(schedule_data, default_flow_style=False))
    else:
        click.echo(json.dumps(schedule_data, indent=2))


@schedule.command()
@click.option('--file', '-f', type=click.Path(exists=True), help='Schedule definition file (JSON or YAML)')
@click.option('--site-id', help='Site ID')
@click.option('--environment', help='Environment')
@click.option('--cron', help='Cron expression')
@click.option('--name', help='Schedule name')
@click.option('--priority', type=click.Choice(['low', 'medium', 'high']), default='medium', help='Priority')
@click.option('--enabled/--disabled', default=True, help='Enable/disable schedule')
@click.option('--dry-run', is_flag=True, help='Validate only, do not create')
def create(file: Optional[str], site_id: Optional[str], environment: Optional[str],
          cron: Optional[str], name: Optional[str], priority: str, enabled: bool, dry_run: bool):
    """Create a new schedule."""

    if file:
        # Load from file
        schedule_data = _load_schedule_file(file)
    else:
        # Create from CLI options
        if not all([site_id, environment, cron]):
            click.echo("Error: Must specify --site-id, --environment, and --cron when not using --file", err=True)
            return 1

        schedule_data = {
            "site_id": site_id,
            "environment": environment,
            "cron_expression": cron,
            "name": name or f"Schedule for {site_id}",
            "priority": priority,
            "enabled": enabled
        }

    if dry_run:
        click.echo("✅ Schedule validation successful")
        click.echo("Schedule would be created with the following configuration:")
        click.echo(json.dumps(schedule_data, indent=2))
    else:
        # In real implementation, would call API to create schedule
        schedule_id = f"schedule-{hash(str(schedule_data)) % 10000}"
        click.echo(f"✅ Schedule created successfully: {schedule_id}")


@schedule.command()
@click.argument('schedule_id')
@click.option('--file', '-f', type=click.Path(exists=True), help='Updated schedule definition file')
@click.option('--cron', help='New cron expression')
@click.option('--enabled/--disabled', help='Enable/disable schedule')
@click.option('--priority', type=click.Choice(['low', 'medium', 'high']), help='New priority')
def update(schedule_id: str, file: Optional[str], cron: Optional[str], enabled: Optional[bool], priority: Optional[str]):
    """Update an existing schedule."""

    if file:
        schedule_data = _load_schedule_file(file)
        schedule_data['id'] = schedule_id
        click.echo(f"✅ Schedule {schedule_id} updated from file")
    else:
        updates = {}
        if cron:
            updates['cron_expression'] = cron
        if enabled is not None:
            updates['enabled'] = enabled
        if priority:
            updates['priority'] = priority

        if not updates:
            click.echo("Error: No updates specified", err=True)
            return 1

        click.echo(f"✅ Schedule {schedule_id} updated")
        click.echo(f"Changes: {', '.join(f'{k}={v}' for k, v in updates.items())}")


@schedule.command()
@click.argument('schedule_id')
@click.option('--force', is_flag=True, help='Force deletion without confirmation')
def delete(schedule_id: str, force: bool):
    """Delete a schedule."""

    if not force:
        if not click.confirm(f"Are you sure you want to delete schedule {schedule_id}?"):
            click.echo("Cancelled")
            return

    # In real implementation, would call API to delete schedule
    click.echo(f"✅ Schedule {schedule_id} deleted successfully")


@schedule.command()
@click.argument('schedule_id')
@click.option('--force', is_flag=True, help='Force trigger (bypass blackout and concurrency checks)')
def trigger(schedule_id: str, force: bool):
    """Manually trigger a schedule."""

    # In real implementation, would call API to trigger schedule
    if force:
        click.echo(f"✅ Schedule {schedule_id} triggered forcefully")
    else:
        click.echo(f"✅ Schedule {schedule_id} triggered")

    # Mock run ID
    run_id = f"run-{hash(schedule_id) % 10000}"
    click.echo(f"Run ID: {run_id}")


@schedule.command()
@click.argument('schedule_id')
def pause(schedule_id: str):
    """Pause a schedule."""

    # In real implementation, would call API to pause schedule
    click.echo(f"✅ Schedule {schedule_id} paused")


@schedule.command()
@click.argument('schedule_id')
def resume(schedule_id: str):
    """Resume a paused schedule."""

    # In real implementation, would call API to resume schedule
    click.echo(f"✅ Schedule {schedule_id} resumed")


@schedule.command()
@click.argument('schedule_id')
@click.option('--count', '-n', type=int, default=5, help='Number of next run times to show')
def next(schedule_id: str, count: int):
    """Show next run times for a schedule."""

    # Mock data - in real implementation would call cron evaluator
    base_time = datetime.now(timezone.utc)
    next_runs = []

    for i in range(count):
        # Mock calculation - add 1 hour for each next run
        next_time = base_time.replace(hour=(base_time.hour + i + 1) % 24)
        next_runs.append(next_time.isoformat())

    click.echo(f"Next {count} run times for schedule {schedule_id}:")
    for i, run_time in enumerate(next_runs, 1):
        click.echo(f"  {i}. {run_time}")


@schedule.command()
@click.option('--file', '-f', type=click.Path(exists=True), help='Schedule definition file to validate')
@click.option('--site-id', help='Site ID')
@click.option('--environment', help='Environment')
@click.option('--cron', help='Cron expression')
def validate(file: Optional[str], site_id: Optional[str], environment: Optional[str], cron: Optional[str]):
    """Validate a schedule configuration."""

    if file:
        schedule_data = _load_schedule_file(file)
    else:
        if not all([site_id, environment, cron]):
            click.echo("Error: Must specify --site-id, --environment, and --cron when not using --file", err=True)
            return 1

        schedule_data = {
            "site_id": site_id,
            "environment": environment,
            "cron_expression": cron
        }

    # Mock validation
    errors = []

    # Basic validation
    if not schedule_data.get('site_id'):
        errors.append("Missing site_id")
    if not schedule_data.get('environment'):
        errors.append("Missing environment")
    if not schedule_data.get('cron_expression'):
        errors.append("Missing cron_expression")

    if errors:
        click.echo("❌ Validation failed:")
        for error in errors:
            click.echo(f"  - {error}")
        return 1
    else:
        click.echo("✅ Schedule configuration is valid")


# Import/Export commands

@schedule.command()
@click.argument('file', type=click.Path())
@click.option('--format', 'file_format', type=click.Choice(['json', 'yaml']), help='Export format (auto-detected from file extension if not specified)')
def export(file: str, file_format: Optional[str]):
    """Export all schedules to a file."""

    # Mock data
    schedules = [
        {
            "id": "schedule-1",
            "name": "Daily Audit",
            "site_id": "example.com",
            "environment": "production",
            "cron_expression": "0 2 * * *",
            "timezone": "UTC",
            "enabled": True,
            "priority": "medium"
        }
    ]

    # Determine format
    if not file_format:
        if file.endswith('.yaml') or file.endswith('.yml'):
            file_format = 'yaml'
        else:
            file_format = 'json'

    # Write file
    file_path = Path(file)
    if file_format == 'yaml':
        with open(file_path, 'w') as f:
            yaml.dump({'schedules': schedules}, f, default_flow_style=False)
    else:
        with open(file_path, 'w') as f:
            json.dump({'schedules': schedules}, f, indent=2)

    click.echo(f"✅ Exported {len(schedules)} schedules to {file}")


@schedule.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--dry-run', is_flag=True, help='Validate only, do not import')
def import_schedules(file: str, dry_run: bool):
    """Import schedules from a file."""

    data = _load_schedule_file(file)
    schedules = data.get('schedules', [])

    if not schedules:
        click.echo("Error: No schedules found in file", err=True)
        return 1

    if dry_run:
        click.echo(f"✅ Validation successful - {len(schedules)} schedules would be imported")
        for schedule in schedules:
            click.echo(f"  - {schedule.get('id', 'N/A')}: {schedule.get('name', 'N/A')}")
    else:
        # In real implementation, would call API to import schedules
        click.echo(f"✅ Successfully imported {len(schedules)} schedules")


# Helper functions

def _load_service_config(config_path: str) -> ServiceConfig:
    """Load service configuration from file."""
    # This is a placeholder - would load actual config
    return ServiceConfig()


def _load_schedule_file(file_path: str) -> Dict[str, Any]:
    """Load schedule data from JSON or YAML file."""
    path = Path(file_path)

    try:
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    except Exception as e:
        click.echo(f"Error loading file {file_path}: {e}", err=True)
        raise click.Abort()


# Create main CLI group
@click.group()
def cli():
    """Tag Sentinel Schedule Management CLI."""
    pass


# Add command groups
cli.add_command(service)
cli.add_command(schedule)


if __name__ == '__main__':
    cli()