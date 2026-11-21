CREATE SCHEMA IF NOT EXISTS aiops_go1;
SET LOCAL search_path TO aiops_go1, pg_catalog;

CREATE TABLE IF NOT EXISTS catalog_versions (
    version integer PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS scene_sets (
    id uuid PRIMARY KEY,
    title text NOT NULL,
    created_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS scene_set_members (
    set_id uuid NOT NULL REFERENCES scene_sets(id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal > 0),
    scene_id uuid NOT NULL,
    PRIMARY KEY (set_id, ordinal),
    UNIQUE (set_id, scene_id)
);
CREATE TABLE IF NOT EXISTS test_case_sets (
    id uuid PRIMARY KEY,
    title text NOT NULL,
    created_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS training_tasks (
    id uuid PRIMARY KEY,
    title text NOT NULL,
    device text NOT NULL,
    state text NOT NULL,
    started_at timestamptz NOT NULL,
    finished_at timestamptz,
    scene_id uuid,
    base_model_id uuid,
    logs text NOT NULL,
    error text NOT NULL,
    snapshot_key text NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    id uuid PRIMARY KEY,
    title text NOT NULL,
    format text NOT NULL,
    policy_sha256 text NOT NULL,
    training_id uuid UNIQUE REFERENCES training_tasks(id) DEFERRABLE INITIALLY DEFERRED
);
DO $$ BEGIN
    ALTER TABLE training_tasks ADD CONSTRAINT training_base_model_fk
        FOREIGN KEY (base_model_id) REFERENCES models(id) DEFERRABLE INITIALLY DEFERRED;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
CREATE TABLE IF NOT EXISTS training_parameters (
    training_id uuid NOT NULL REFERENCES training_tasks(id) ON DELETE CASCADE,
    name text NOT NULL,
    value double precision NOT NULL,
    PRIMARY KEY (training_id, name)
);
CREATE TABLE IF NOT EXISTS training_metrics (
    training_id uuid NOT NULL REFERENCES training_tasks(id) ON DELETE CASCADE,
    iteration integer NOT NULL,
    name text NOT NULL,
    value double precision NOT NULL,
    PRIMARY KEY (training_id, iteration, name)
);
CREATE TABLE IF NOT EXISTS simulation_tasks (
    id uuid PRIMARY KEY,
    title text NOT NULL,
    state text NOT NULL,
    started_at timestamptz NOT NULL,
    finished_at timestamptz,
    model_id uuid NOT NULL REFERENCES models(id) DEFERRABLE INITIALLY DEFERRED,
    scene_id uuid NOT NULL,
    scene_set_id uuid REFERENCES scene_sets(id) DEFERRABLE INITIALLY DEFERRED,
    test_case_set_id uuid REFERENCES test_case_sets(id) DEFERRABLE INITIALLY DEFERRED,
    recording_id uuid REFERENCES simulation_tasks(id) DEFERRABLE INITIALLY DEFERRED,
    seed integer NOT NULL,
    steps integer NOT NULL CHECK (steps > 0),
    progress integer NOT NULL CHECK (progress >= 0),
    recorded boolean NOT NULL,
    vision_injected boolean NOT NULL,
    outcome text,
    passed boolean,
    logs text NOT NULL,
    error text NOT NULL,
    snapshot_key text NOT NULL
);
CREATE TABLE IF NOT EXISTS test_case_set_members (
    set_id uuid NOT NULL REFERENCES test_case_sets(id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal > 0),
    simulation_id uuid NOT NULL REFERENCES simulation_tasks(id) DEFERRABLE INITIALLY DEFERRED,
    PRIMARY KEY (set_id, ordinal),
    UNIQUE (set_id, simulation_id)
);
CREATE TABLE IF NOT EXISTS simulation_cases (
    simulation_id uuid NOT NULL REFERENCES simulation_tasks(id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal > 0),
    scene_id uuid NOT NULL,
    seed integer NOT NULL,
    steps integer NOT NULL CHECK (steps > 0),
    PRIMARY KEY (simulation_id, ordinal)
);
CREATE TABLE IF NOT EXISTS artifacts (
    path text PRIMARY KEY,
    object_key text UNIQUE NOT NULL,
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    model_id uuid REFERENCES models(id) ON DELETE CASCADE,
    simulation_id uuid REFERENCES simulation_tasks(id) ON DELETE CASCADE,
    CHECK (num_nonnulls(model_id, simulation_id) = 1)
);

-- Scenes now live in code. Keep historical rows/snapshots, but remove catalog
-- foreign keys so new tasks and sets can refer to fixed scenes without inserts.
ALTER TABLE scene_set_members DROP CONSTRAINT IF EXISTS scene_set_members_scene_id_fkey;
ALTER TABLE training_tasks DROP CONSTRAINT IF EXISTS training_tasks_scene_id_fkey;
ALTER TABLE simulation_tasks DROP CONSTRAINT IF EXISTS simulation_tasks_scene_id_fkey;
ALTER TABLE simulation_cases DROP CONSTRAINT IF EXISTS simulation_cases_scene_id_fkey;
