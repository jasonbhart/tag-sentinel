"""REST API endpoints for schedule management."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from aiohttp import web, web_request
from aiohttp.web_response import Response
import json

from ..scheduling.service import SchedulingService, ServiceStatus, ServiceHealth
from ..scheduling.models import Schedule, Priority, CatchUpPolicy, BlackoutWindow
from ..scheduling.scheduler import EngineStats, ScheduleState

logger = logging.getLogger(__name__)


class ScheduleAPI:
    """REST API handler for schedule management."""

    def __init__(self, scheduling_service: SchedulingService):
        """Initialize the schedule API.

        Args:
            scheduling_service: The scheduling service instance
        """
        self.service = scheduling_service

    def setup_routes(self, app: web.Application) -> None:
        """Set up API routes."""
        # Service management
        app.router.add_get('/api/v1/service/status', self.get_service_status)
        app.router.add_get('/api/v1/service/health', self.get_service_health)
        app.router.add_get('/api/v1/service/stats', self.get_service_stats)
        app.router.add_post('/api/v1/service/start', self.start_service)
        app.router.add_post('/api/v1/service/stop', self.stop_service)
        app.router.add_post('/api/v1/service/pause', self.pause_service)
        app.router.add_post('/api/v1/service/resume', self.resume_service)
        app.router.add_post('/api/v1/service/reload', self.reload_service)

        # Schedule CRUD
        app.router.add_get('/api/v1/schedules', self.list_schedules)
        app.router.add_get('/api/v1/schedules/{schedule_id}', self.get_schedule)
        app.router.add_post('/api/v1/schedules', self.create_schedule)
        app.router.add_put('/api/v1/schedules/{schedule_id}', self.update_schedule)
        app.router.add_delete('/api/v1/schedules/{schedule_id}', self.delete_schedule)

        # Schedule operations
        app.router.add_post('/api/v1/schedules/{schedule_id}/trigger', self.trigger_schedule)
        app.router.add_post('/api/v1/schedules/{schedule_id}/pause', self.pause_schedule)
        app.router.add_post('/api/v1/schedules/{schedule_id}/resume', self.resume_schedule)

        # Schedule validation
        app.router.add_post('/api/v1/schedules/validate', self.validate_schedule)

        # Bulk operations
        app.router.add_post('/api/v1/schedules/bulk/import', self.bulk_import_schedules)
        app.router.add_get('/api/v1/schedules/bulk/export', self.bulk_export_schedules)

    # Service management endpoints

    async def get_service_status(self, request: web_request.Request) -> Response:
        """Get service status."""
        try:
            status = self.service.get_status()
            return web.json_response({
                'status': status.value,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return self._error_response(500, "Internal server error")

    async def get_service_health(self, request: web_request.Request) -> Response:
        """Get service health information."""
        try:
            health = self.service.get_health()
            return web.json_response({
                'status': health.status.value,
                'uptime_seconds': health.uptime_seconds,
                'schedules': {
                    'active': health.schedules_active,
                    'total': health.schedules_total
                },
                'runs': {
                    'queued': health.runs_queued,
                    'running': health.runs_running
                },
                'components': health.component_status,
                'last_error': health.last_error,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting service health: {e}")
            return self._error_response(500, "Internal server error")

    async def get_service_stats(self, request: web_request.Request) -> Response:
        """Get service statistics."""
        try:
            stats = self.service.get_engine_stats()
            if not stats:
                return self._error_response(503, "Service not initialized")

            return web.json_response({
                'schedules': {
                    'total': stats.total_schedules,
                    'active': stats.active_schedules,
                    'paused': stats.paused_schedules,
                    'disabled': stats.disabled_schedules
                },
                'runs': {
                    'total_scheduled': stats.total_runs_scheduled,
                    'total_completed': stats.total_runs_completed,
                    'total_failed': stats.total_runs_failed,
                    'catch_up_runs': stats.catch_up_runs
                },
                'performance': {
                    'blackout_blocks': stats.blackout_blocks,
                    'lock_conflicts': stats.lock_conflicts,
                    'last_tick_time': stats.last_tick_time.isoformat() if stats.last_tick_time else None,
                    'average_tick_duration_ms': stats.average_tick_duration_ms
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return self._error_response(500, "Internal server error")

    async def start_service(self, request: web_request.Request) -> Response:
        """Start the scheduling service."""
        try:
            await self.service.start()
            return web.json_response({
                'message': 'Service started successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error starting service: {e}")
            return self._error_response(500, f"Failed to start service: {str(e)}")

    async def stop_service(self, request: web_request.Request) -> Response:
        """Stop the scheduling service."""
        try:
            await self.service.stop()
            return web.json_response({
                'message': 'Service stopped successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error stopping service: {e}")
            return self._error_response(500, f"Failed to stop service: {str(e)}")

    async def pause_service(self, request: web_request.Request) -> Response:
        """Pause the scheduling service."""
        try:
            await self.service.pause()
            return web.json_response({
                'message': 'Service paused successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error pausing service: {e}")
            return self._error_response(500, f"Failed to pause service: {str(e)}")

    async def resume_service(self, request: web_request.Request) -> Response:
        """Resume the scheduling service."""
        try:
            await self.service.resume()
            return web.json_response({
                'message': 'Service resumed successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error resuming service: {e}")
            return self._error_response(500, f"Failed to resume service: {str(e)}")

    async def reload_service(self, request: web_request.Request) -> Response:
        """Reload service configuration."""
        try:
            await self.service.reload_config()
            return web.json_response({
                'message': 'Configuration reloaded successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error reloading config: {e}")
            return self._error_response(500, f"Failed to reload config: {str(e)}")

    # Schedule CRUD endpoints

    async def list_schedules(self, request: web_request.Request) -> Response:
        """List all schedules."""
        try:
            # Parse query parameters
            limit = int(request.query.get('limit', 100))
            offset = int(request.query.get('offset', 0))
            status_filter = request.query.get('status')
            site_filter = request.query.get('site_id')
            env_filter = request.query.get('environment')

            schedules = self.service.list_schedules()

            # Apply filters
            if status_filter:
                schedules = [s for s in schedules if s['status'] == status_filter]
            if site_filter:
                schedules = [s for s in schedules if s['site_id'] == site_filter]
            if env_filter:
                schedules = [s for s in schedules if s['environment'] == env_filter]

            # Apply pagination
            total = len(schedules)
            schedules = schedules[offset:offset + limit]

            return web.json_response({
                'schedules': schedules,
                'pagination': {
                    'total': total,
                    'limit': limit,
                    'offset': offset
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Error listing schedules: {e}")
            return self._error_response(500, "Internal server error")

    async def get_schedule(self, request: web_request.Request) -> Response:
        """Get a specific schedule."""
        try:
            schedule_id = request.match_info['schedule_id']

            if not self.service.schedule_engine:
                return self._error_response(503, "Service not initialized")

            schedule_state = self.service.schedule_engine.get_schedule(schedule_id)
            if not schedule_state:
                return self._error_response(404, f"Schedule {schedule_id} not found")

            schedule_data = self._serialize_schedule_state(schedule_state)
            return web.json_response({
                'schedule': schedule_data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Error getting schedule: {e}")
            return self._error_response(500, "Internal server error")

    async def create_schedule(self, request: web_request.Request) -> Response:
        """Create a new schedule."""
        try:
            data = await request.json()

            # Validate and create schedule
            schedule = self._deserialize_schedule(data)

            # Add to service
            await self.service.add_schedule(schedule)

            schedule_data = self._serialize_schedule(schedule)
            return web.json_response({
                'schedule': schedule_data,
                'message': f'Schedule {schedule.id} created successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, status=201)

        except ValueError as e:
            return self._error_response(400, f"Invalid schedule data: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating schedule: {e}")
            return self._error_response(500, "Internal server error")

    async def update_schedule(self, request: web_request.Request) -> Response:
        """Update an existing schedule."""
        try:
            schedule_id = request.match_info['schedule_id']
            data = await request.json()

            # Ensure ID matches
            data['id'] = schedule_id

            # Validate and create schedule
            schedule = self._deserialize_schedule(data)

            # Remove old schedule and add new one
            await self.service.remove_schedule(schedule_id)
            await self.service.add_schedule(schedule)

            schedule_data = self._serialize_schedule(schedule)
            return web.json_response({
                'schedule': schedule_data,
                'message': f'Schedule {schedule.id} updated successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        except ValueError as e:
            return self._error_response(400, f"Invalid schedule data: {str(e)}")
        except Exception as e:
            logger.error(f"Error updating schedule: {e}")
            return self._error_response(500, "Internal server error")

    async def delete_schedule(self, request: web_request.Request) -> Response:
        """Delete a schedule."""
        try:
            schedule_id = request.match_info['schedule_id']

            success = await self.service.remove_schedule(schedule_id)
            if not success:
                return self._error_response(404, f"Schedule {schedule_id} not found")

            return web.json_response({
                'message': f'Schedule {schedule_id} deleted successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Error deleting schedule: {e}")
            return self._error_response(500, "Internal server error")

    # Schedule operations

    async def trigger_schedule(self, request: web_request.Request) -> Response:
        """Manually trigger a schedule."""
        try:
            schedule_id = request.match_info['schedule_id']

            # Parse request body for options
            try:
                data = await request.json()
            except:
                data = {}

            force = data.get('force', False)

            run_id = await self.service.trigger_schedule(schedule_id, force=force)
            if not run_id:
                return self._error_response(400, "Failed to trigger schedule (blackout or concurrency limit)")

            return web.json_response({
                'run_id': run_id,
                'message': f'Schedule {schedule_id} triggered successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Error triggering schedule: {e}")
            return self._error_response(500, "Internal server error")

    async def pause_schedule(self, request: web_request.Request) -> Response:
        """Pause a schedule."""
        try:
            schedule_id = request.match_info['schedule_id']

            if not self.service.schedule_engine:
                return self._error_response(503, "Service not initialized")

            success = self.service.schedule_engine.pause_schedule(schedule_id)
            if not success:
                return self._error_response(404, f"Schedule {schedule_id} not found")

            return web.json_response({
                'message': f'Schedule {schedule_id} paused successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Error pausing schedule: {e}")
            return self._error_response(500, "Internal server error")

    async def resume_schedule(self, request: web_request.Request) -> Response:
        """Resume a paused schedule."""
        try:
            schedule_id = request.match_info['schedule_id']

            if not self.service.schedule_engine:
                return self._error_response(503, "Service not initialized")

            success = self.service.schedule_engine.resume_schedule(schedule_id)
            if not success:
                return self._error_response(404, f"Schedule {schedule_id} not found or not paused")

            return web.json_response({
                'message': f'Schedule {schedule_id} resumed successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Error resuming schedule: {e}")
            return self._error_response(500, "Internal server error")

    # Schedule validation

    async def validate_schedule(self, request: web_request.Request) -> Response:
        """Validate a schedule configuration."""
        try:
            data = await request.json()

            # Try to deserialize the schedule
            schedule = self._deserialize_schedule(data)

            # Additional validations
            validation_errors = []

            # Validate cron expression
            if self.service.cron_evaluator:
                try:
                    next_run = self.service.cron_evaluator.get_next_run(
                        schedule.cron,
                        schedule.timezone
                    )
                except Exception as e:
                    validation_errors.append(f"Invalid cron expression: {str(e)}")

            # Validate environment if environment manager is available
            if self.service.environment_manager and schedule.environment:
                try:
                    env_config = await self.service.environment_manager.get_environment_config(
                        schedule.site_id,
                        schedule.environment
                    )
                    if not env_config:
                        validation_errors.append(f"Environment {schedule.environment} not found for site {schedule.site_id}")
                except Exception as e:
                    validation_errors.append(f"Environment validation error: {str(e)}")

            is_valid = len(validation_errors) == 0

            response_data = {
                'valid': is_valid,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            if validation_errors:
                response_data['errors'] = validation_errors

            if is_valid:
                response_data['schedule'] = self._serialize_schedule(schedule)

            status_code = 200 if is_valid else 400
            return web.json_response(response_data, status=status_code)

        except ValueError as e:
            return self._error_response(400, f"Invalid schedule data: {str(e)}")
        except Exception as e:
            logger.error(f"Error validating schedule: {e}")
            return self._error_response(500, "Internal server error")

    # Bulk operations

    async def bulk_import_schedules(self, request: web_request.Request) -> Response:
        """Import multiple schedules."""
        try:
            data = await request.json()
            schedules_data = data.get('schedules', [])

            if not isinstance(schedules_data, list):
                return self._error_response(400, "Expected 'schedules' to be a list")

            results = []
            success_count = 0
            error_count = 0

            for i, schedule_data in enumerate(schedules_data):
                try:
                    schedule = self._deserialize_schedule(schedule_data)
                    await self.service.add_schedule(schedule)

                    results.append({
                        'index': i,
                        'schedule_id': schedule.id,
                        'status': 'success'
                    })
                    success_count += 1

                except Exception as e:
                    results.append({
                        'index': i,
                        'schedule_id': schedule_data.get('id', 'unknown'),
                        'status': 'error',
                        'error': str(e)
                    })
                    error_count += 1

            return web.json_response({
                'summary': {
                    'total': len(schedules_data),
                    'success': success_count,
                    'errors': error_count
                },
                'results': results,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Error importing schedules: {e}")
            return self._error_response(500, "Internal server error")

    async def bulk_export_schedules(self, request: web_request.Request) -> Response:
        """Export all schedules."""
        try:
            schedules = self.service.list_schedules()

            # Convert to full schedule format
            export_data = []
            for schedule_data in schedules:
                # Get full schedule state if available
                if self.service.schedule_engine:
                    schedule_state = self.service.schedule_engine.get_schedule(schedule_data['id'])
                    if schedule_state:
                        export_data.append(self._serialize_schedule(schedule_state.schedule))

            return web.json_response({
                'schedules': export_data,
                'export_info': {
                    'total_schedules': len(export_data),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'version': '1.0'
                }
            })

        except Exception as e:
            logger.error(f"Error exporting schedules: {e}")
            return self._error_response(500, "Internal server error")

    # Helper methods

    def _serialize_schedule(self, schedule: Schedule) -> Dict[str, Any]:
        """Serialize a Schedule object to a dictionary."""
        return {
            'id': schedule.id,
            'name': schedule.name,
            'site_id': schedule.site_id,
            'environment': schedule.environment,
            'cron': schedule.cron,
            'timezone': schedule.timezone,
            'enabled': schedule.enabled,
            'priority': schedule.priority.value,
            'max_concurrent_runs': schedule.max_concurrent_runs,
            'timeout_minutes': schedule.timeout_minutes,
            'retry_count': schedule.retry_count,
            'blackout_windows': [self._serialize_blackout_window(bw) for bw in schedule.blackout_windows],
            'catch_up_policy': self._serialize_catch_up_policy(schedule.catch_up_policy) if schedule.catch_up_policy else None,
            'metadata': schedule.metadata
        }

    def _serialize_schedule_state(self, schedule_state: ScheduleState) -> Dict[str, Any]:
        """Serialize a ScheduleState object to a dictionary."""
        schedule_data = self._serialize_schedule(schedule_state.schedule)
        schedule_data.update({
            'status': schedule_state.status.value,
            'next_run': schedule_state.next_run.isoformat() if schedule_state.next_run else None,
            'last_run': schedule_state.last_run.isoformat() if schedule_state.last_run else None,
            'consecutive_failures': schedule_state.consecutive_failures,
            'last_error': schedule_state.last_error,
            'metadata': schedule_state.metadata
        })
        return schedule_data

    def _serialize_blackout_window(self, blackout: BlackoutWindow) -> Dict[str, Any]:
        """Serialize a BlackoutWindow object."""
        return {
            'type': blackout.type.value,
            'start_time': blackout.start_time.isoformat() if blackout.start_time else None,
            'end_time': blackout.end_time.isoformat() if blackout.end_time else None,
            'timezone': blackout.timezone,
            'recurrence_rule': blackout.recurrence_rule,
            'description': blackout.description
        }

    def _serialize_catch_up_policy(self, policy: CatchUpPolicy) -> Dict[str, Any]:
        """Serialize a CatchUpPolicy object."""
        return {
            'enabled': policy.enabled,
            'strategy': policy.strategy,
            'max_catch_up_runs': policy.max_catch_up_runs,
            'catch_up_window_hours': policy.catch_up_window_hours
        }

    def _deserialize_schedule(self, data: Dict[str, Any]) -> Schedule:
        """Deserialize a dictionary to a Schedule object."""
        # Generate ID if not provided
        schedule_id = data.get('id') or str(uuid4())

        # Parse priority
        priority_str = data.get('priority', 'medium')
        try:
            priority = Priority(priority_str)
        except ValueError:
            raise ValueError(f"Invalid priority: {priority_str}")

        # Parse blackout windows
        blackout_windows = []
        for bw_data in data.get('blackout_windows', []):
            blackout_windows.append(self._deserialize_blackout_window(bw_data))

        # Parse catch-up policy
        catch_up_policy = None
        if 'catch_up_policy' in data and data['catch_up_policy']:
            catch_up_policy = self._deserialize_catch_up_policy(data['catch_up_policy'])

        return Schedule(
            id=schedule_id,
            name=data.get('name', f"Schedule {schedule_id}"),
            site_id=data['site_id'],
            environment=data['environment'],
            cron=data['cron'],
            timezone=data.get('timezone', 'UTC'),
            enabled=data.get('enabled', True),
            priority=priority,
            max_concurrent_runs=data.get('max_concurrent_runs', 1),
            timeout_minutes=data.get('timeout_minutes', 60),
            retry_count=data.get('retry_count', 0),
            blackout_windows=blackout_windows,
            catch_up_policy=catch_up_policy,
            metadata=data.get('metadata', {})
        )

    def _deserialize_blackout_window(self, data: Dict[str, Any]) -> BlackoutWindow:
        """Deserialize a BlackoutWindow from dictionary."""
        # This is a simplified version - real implementation would parse all fields
        return BlackoutWindow(
            type=data.get('type', 'absolute'),
            start_time=datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            timezone=data.get('timezone', 'UTC'),
            description=data.get('description')
        )

    def _deserialize_catch_up_policy(self, data: Dict[str, Any]) -> CatchUpPolicy:
        """Deserialize a CatchUpPolicy from dictionary."""
        return CatchUpPolicy(
            enabled=data.get('enabled', False),
            strategy=data.get('strategy', 'skip'),
            max_catch_up_runs=data.get('max_catch_up_runs'),
            catch_up_window_hours=data.get('catch_up_window_hours')
        )

    def _error_response(self, status: int, message: str) -> Response:
        """Create an error response."""
        return web.json_response({
            'error': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, status=status)


def create_app(scheduling_service: SchedulingService) -> web.Application:
    """Create and configure the aiohttp application."""
    app = web.Application()

    # Set up API routes
    api = ScheduleAPI(scheduling_service)
    api.setup_routes(app)

    # Add CORS headers middleware
    async def cors_middleware(request: web_request.Request, handler):
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    app.middlewares.append(cors_middleware)

    return app