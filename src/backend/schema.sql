-- INTEGRA backend — Supabase (PostgreSQL) schema
-- Run this once in the Supabase project's SQL Editor.
-- Replaces the old storage/jobs/{job_id}/status.json file-based tracking.

create table if not exists jobs (
    job_id text primary key,
    filename text,
    status text not null default 'uploaded',
    module1 text default 'waiting',
    module2 text default 'waiting',
    module3 text default 'waiting',
    module4 text default 'waiting',
    final_video text,
    final_json text,
    error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- "Automatically expose new tables" was left OFF (recommended), so the
-- service_role key needs these grants explicitly - RLS is still in effect,
-- but service_role bypasses RLS policies once it has the base privilege.
grant select, insert, update on public.jobs to service_role;
