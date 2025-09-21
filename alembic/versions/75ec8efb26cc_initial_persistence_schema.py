"""Initial persistence schema

Revision ID: 75ec8efb26cc
Revises: 
Create Date: 2025-09-21 07:03:35.968348

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '75ec8efb26cc'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial persistence schema."""
    # Create runs table
    op.create_table(
        'runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('site_id', sa.String(length=255), nullable=False),
        sa.Column('environment', sa.String(length=50), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('config_json', sa.JSON(), nullable=True),
        sa.Column('summary_json', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_runs_site_env', 'runs', ['site_id', 'environment'])
    op.create_index('ix_runs_status_started', 'runs', ['status', 'started_at'])
    op.create_index(op.f('ix_runs_environment'), 'runs', ['environment'])
    op.create_index(op.f('ix_runs_site_id'), 'runs', ['site_id'])
    op.create_index(op.f('ix_runs_status'), 'runs', ['status'])

    # Create page_results table
    op.create_table(
        'page_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(length=2048), nullable=False),
        sa.Column('final_url', sa.String(length=2048), nullable=True),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('capture_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('load_time_ms', sa.Float(), nullable=True),
        sa.Column('timings_json', sa.JSON(), nullable=True),
        sa.Column('errors_json', sa.JSON(), nullable=True),
        sa.Column('capture_error', sa.Text(), nullable=True),
        sa.Column('metrics_json', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_page_results_run_status', 'page_results', ['run_id', 'status'])
    op.create_index('ix_page_results_url', 'page_results', ['url'])
    op.create_index(op.f('ix_page_results_run_id'), 'page_results', ['run_id'])
    op.create_index(op.f('ix_page_results_status'), 'page_results', ['status'])

    # Create request_logs table
    op.create_table(
        'request_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('page_result_id', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(length=2048), nullable=False),
        sa.Column('method', sa.String(length=10), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=True),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('status_text', sa.String(length=255), nullable=True),
        sa.Column('request_headers_json', sa.JSON(), nullable=True),
        sa.Column('response_headers_json', sa.JSON(), nullable=True),
        sa.Column('timings_json', sa.JSON(), nullable=True),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('sizes_json', sa.JSON(), nullable=True),
        sa.Column('vendor_tags_json', sa.JSON(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('error_text', sa.Text(), nullable=True),
        sa.Column('protocol', sa.String(length=20), nullable=True),
        sa.Column('remote_address', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['page_result_id'], ['page_results.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_request_logs_page_url', 'request_logs', ['page_result_id', 'url'])
    op.create_index('ix_request_logs_status_code', 'request_logs', ['status_code'])
    op.create_index('ix_request_logs_start_time', 'request_logs', ['start_time'])
    op.create_index(op.f('ix_request_logs_page_result_id'), 'request_logs', ['page_result_id'])
    op.create_index(op.f('ix_request_logs_url'), 'request_logs', ['url'])

    # Create cookies table
    op.create_table(
        'cookies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('page_result_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('domain', sa.String(length=255), nullable=False),
        sa.Column('path', sa.String(length=1024), nullable=False),
        sa.Column('expires', sa.DateTime(timezone=True), nullable=True),
        sa.Column('max_age', sa.Integer(), nullable=True),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('secure', sa.Boolean(), nullable=False),
        sa.Column('http_only', sa.Boolean(), nullable=False),
        sa.Column('same_site', sa.String(length=20), nullable=True),
        sa.Column('first_party', sa.Boolean(), nullable=False),
        sa.Column('essential', sa.Boolean(), nullable=True),
        sa.Column('is_session', sa.Boolean(), nullable=False),
        sa.Column('value_redacted', sa.Boolean(), nullable=False),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('set_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('modified_time', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['page_result_id'], ['page_results.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_cookies_page_name_domain', 'cookies', ['page_result_id', 'name', 'domain'])
    op.create_index('ix_cookies_domain_first_party', 'cookies', ['domain', 'first_party'])
    op.create_index('ix_cookies_essential', 'cookies', ['essential'])
    op.create_index(op.f('ix_cookies_domain'), 'cookies', ['domain'])
    op.create_index(op.f('ix_cookies_name'), 'cookies', ['name'])
    op.create_index(op.f('ix_cookies_page_result_id'), 'cookies', ['page_result_id'])

    # Create datalayer_snapshots table
    op.create_table(
        'datalayer_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('page_result_id', sa.Integer(), nullable=False),
        sa.Column('exists', sa.Boolean(), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('truncated', sa.Boolean(), nullable=False),
        sa.Column('sample_json', sa.JSON(), nullable=True),
        sa.Column('schema_valid', sa.Boolean(), nullable=True),
        sa.Column('validation_errors_json', sa.JSON(), nullable=True),
        sa.Column('capture_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['page_result_id'], ['page_results.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_datalayer_page_exists', 'datalayer_snapshots', ['page_result_id', 'exists'])
    op.create_index('ix_datalayer_size', 'datalayer_snapshots', ['size_bytes'])
    op.create_index(op.f('ix_datalayer_snapshots_page_result_id'), 'datalayer_snapshots', ['page_result_id'])

    # Create rule_failures table
    op.create_table(
        'rule_failures',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('rule_id', sa.String(length=255), nullable=False),
        sa.Column('rule_name', sa.String(length=255), nullable=True),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('page_url', sa.String(length=2048), nullable=True),
        sa.Column('details_json', sa.JSON(), nullable=True),
        sa.Column('detected_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_rule_failures_run_rule', 'rule_failures', ['run_id', 'rule_id'])
    op.create_index('ix_rule_failures_severity_detected', 'rule_failures', ['severity', 'detected_at'])
    op.create_index('ix_rule_failures_page_url', 'rule_failures', ['page_url'])
    op.create_index(op.f('ix_rule_failures_rule_id'), 'rule_failures', ['rule_id'])
    op.create_index(op.f('ix_rule_failures_run_id'), 'rule_failures', ['run_id'])
    op.create_index(op.f('ix_rule_failures_severity'), 'rule_failures', ['severity'])

    # Create artifacts table
    op.create_table(
        'artifacts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('kind', sa.String(length=50), nullable=False),
        sa.Column('path', sa.String(length=1024), nullable=False),
        sa.Column('checksum', sa.String(length=64), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('content_type', sa.String(length=255), nullable=True),
        sa.Column('page_url', sa.String(length=2048), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('storage_backend', sa.String(length=50), nullable=False),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.CheckConstraint('size_bytes >= 0', name='ck_artifacts_size_positive'),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('path')
    )
    op.create_index('ix_artifacts_run_kind', 'artifacts', ['run_id', 'kind'])
    op.create_index('ix_artifacts_path', 'artifacts', ['path'])
    op.create_index('ix_artifacts_checksum', 'artifacts', ['checksum'])
    op.create_index(op.f('ix_artifacts_created_at'), 'artifacts', ['created_at'])
    op.create_index(op.f('ix_artifacts_kind'), 'artifacts', ['kind'])
    op.create_index(op.f('ix_artifacts_run_id'), 'artifacts', ['run_id'])


def downgrade() -> None:
    """Drop all persistence tables."""
    op.drop_table('artifacts')
    op.drop_table('rule_failures')
    op.drop_table('datalayer_snapshots')
    op.drop_table('cookies')
    op.drop_table('request_logs')
    op.drop_table('page_results')
    op.drop_table('runs')
