# EMS for AI – Physical Data Model (PDM)

This document summarizes the current and proposed database tables for the EMS demo. Existing tables are already defined in `backend/models.py`; additional tables extend lifecycle coverage (procurement → operation → predictive maintenance).

## Legend
- **PK**: Primary key
- **FK**: Foreign key
- **UQ**: Unique constraint
- **IDX**: Suggested index

## Current tables (implemented)

### `device_models`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | INTEGER | PK | Auto-increment | 
| `name` | VARCHAR(100) | UQ, NOT NULL | Model name |
| `manufacturer` | VARCHAR(100) | NULL | Vendor/manufacturer |
| `description` | VARCHAR(255) | NULL | Short description |
| `created_at` | DATETIME | NOT NULL | Defaults to UTC now |

- Relationships: One-to-many → `devices` (`DeviceModel.devices`).
- Indexes: PK (`id`), UQ on `name`.

### `devices`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | INTEGER | PK | Auto-increment |
| `serial_number` | VARCHAR(100) | UQ, NOT NULL | Asset serial |
| `model_id` | INTEGER | FK → `device_models.id`, NOT NULL | Links to model |
| `location` | VARCHAR(100) | NULL | Site/room/rack |
| `status` | VARCHAR(50) | NOT NULL | Defaults to `active` |
| `created_at` | DATETIME | NOT NULL | Defaults to UTC now |
| `last_service_date` | DATETIME | NULL | Last maintenance timestamp |

- Relationships: Many-to-one → `device_models` (`Device.model`).
- Indexes: PK (`id`), UQ on `serial_number`.

## Proposed lifecycle extensions (to add)
These tables provide coverage for procurement, deployment, operations, and predictive maintenance.

### `locations`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | INTEGER | PK | |
| `name` | VARCHAR(100) | UQ, NOT NULL | Site/room name |
| `parent_id` | INTEGER | FK → `locations.id`, NULL | Hierarchical sites |
| `metadata` | JSON | NULL | Arbitrary tags (e.g., climate zone) |

- IDX: (`parent_id`).

### `vendors`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | INTEGER | PK | |
| `name` | VARCHAR(150) | UQ, NOT NULL | Supplier/maintainer |
| `contact` | VARCHAR(150) | NULL | Primary contact |
| `phone` | VARCHAR(50) | NULL | |
| `email` | VARCHAR(150) | NULL | |

### `device_warranty`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | INTEGER | PK | |
| `device_id` | INTEGER | FK → `devices.id`, UQ, NOT NULL | One-to-one warranty card |
| `vendor_id` | INTEGER | FK → `vendors.id`, NULL | Warranty provider |
| `start_date` | DATE | NOT NULL | |
| `end_date` | DATE | NOT NULL | |
| `terms` | TEXT | NULL | Coverage summary |

### `device_status_history`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | INTEGER | PK | |
| `device_id` | INTEGER | FK → `devices.id`, NOT NULL | |
| `status` | VARCHAR(50) | NOT NULL | e.g., active, standby, retired |
| `changed_at` | DATETIME | NOT NULL | Event timestamp |
| `reason` | VARCHAR(255) | NULL | Free-text justification |

- IDX: (`device_id`, `changed_at` DESC).

### `work_orders`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | INTEGER | PK | |
| `device_id` | INTEGER | FK → `devices.id`, NOT NULL | Target asset |
| `type` | VARCHAR(50) | NOT NULL | maintenance, inspection, repair |
| `priority` | VARCHAR(20) | NOT NULL | low/medium/high |
| `status` | VARCHAR(20) | NOT NULL | open, in_progress, done, canceled |
| `scheduled_at` | DATETIME | NULL | Planned time |
| `completed_at` | DATETIME | NULL | Actual completion |
| `assignee` | VARCHAR(100) | NULL | Technician or team |
| `notes` | TEXT | NULL | Free text |

- IDX: (`device_id`, `status`), (`scheduled_at`).

### `maintenance_tasks`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | INTEGER | PK | |
| `device_id` | INTEGER | FK → `devices.id`, NOT NULL | |
| `work_order_id` | INTEGER | FK → `work_orders.id`, NULL | Optional link |
| `task_type` | VARCHAR(50) | NOT NULL | Preventive, predictive |
| `due_at` | DATETIME | NOT NULL | Planned execution time |
| `completed_at` | DATETIME | NULL | |
| `result` | VARCHAR(50) | NULL | success, partial, failed |
| `remarks` | TEXT | NULL | |

- IDX: (`device_id`, `due_at`).

### `telemetry`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | BIGINT | PK | |
| `device_id` | INTEGER | FK → `devices.id`, NOT NULL | |
| `metric` | VARCHAR(50) | NOT NULL | e.g., temperature, vibration |
| `value` | DOUBLE | NOT NULL | |
| `unit` | VARCHAR(20) | NULL | |
| `recorded_at` | DATETIME | NOT NULL | Timestamp |
| `ingested_at` | DATETIME | NOT NULL | Default now |

- IDX: (`device_id`, `metric`, `recorded_at` DESC).

### `anomalies`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | BIGINT | PK | |
| `device_id` | INTEGER | FK → `devices.id`, NOT NULL | |
| `source` | VARCHAR(50) | NOT NULL | rule, model, operator |
| `metric` | VARCHAR(50) | NULL | Related metric |
| `score` | DOUBLE | NOT NULL | Anomaly score |
| `threshold` | DOUBLE | NULL | Trigger threshold |
| `detected_at` | DATETIME | NOT NULL | Event time |
| `context` | JSON | NULL | Additional payload |

- IDX: (`device_id`, `detected_at` DESC).

### `health_scores`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | BIGINT | PK | |
| `device_id` | INTEGER | FK → `devices.id`, NOT NULL | |
| `score` | DOUBLE | NOT NULL | 0–1 or 0–100 health metric |
| `model_version` | VARCHAR(50) | NULL | AI model identifier |
| `computed_at` | DATETIME | NOT NULL | |
| `metadata` | JSON | NULL | Feature summary |

- IDX: (`device_id`, `computed_at` DESC).

### `maintenance_recommendations`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | BIGINT | PK | |
| `device_id` | INTEGER | FK → `devices.id`, NOT NULL | |
| `recommendation` | TEXT | NOT NULL | Human-readable action |
| `priority` | VARCHAR(20) | NOT NULL | low/medium/high/critical |
| `source` | VARCHAR(50) | NOT NULL | ai_model, rule, operator |
| `confidence` | DOUBLE | NULL | 0–1 |
| `created_at` | DATETIME | NOT NULL | |
| `expires_at` | DATETIME | NULL | Optional expiry |

- IDX: (`device_id`, `priority`), (`created_at` DESC).

### `audit_events`
| Column | Type | Attributes | Notes |
| --- | --- | --- | --- |
| `id` | BIGINT | PK | |
| `actor` | VARCHAR(100) | NOT NULL | User/service name |
| `action` | VARCHAR(100) | NOT NULL | e.g., CREATE_DEVICE |
| `entity` | VARCHAR(100) | NOT NULL | Target table/entity |
| `entity_id` | VARCHAR(100) | NULL | Target id |
| `payload` | JSON | NULL | Before/after snapshot |
| `created_at` | DATETIME | NOT NULL | |

- IDX: (`entity`, `entity_id`), (`created_at` DESC).

## Implementation notes
- The current FastAPI models map to `device_models` and `devices`; new tables can follow the same SQLAlchemy style in `backend/models.py` with migrations managed via Alembic.
- Prefer UTC timestamps and `TIMESTAMP WITH TIME ZONE` when using PostgreSQL.
- For high-volume telemetry, partitioning by time and device (e.g., monthly) is recommended in PostgreSQL/ClickHouse.
- Add foreign key cascades thoughtfully (e.g., restrict deletes on telemetry if historical data is required).
