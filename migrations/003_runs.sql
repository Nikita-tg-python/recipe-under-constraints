-- Applied on every API start: must stay idempotent.

-- Placeholder of the first version (never written to, its migration removed): replaced by recipe_runs.
DROP TABLE IF EXISTS recipe_requests;

-- Every POST /recipe call, whatever the outcome: the audit trail of what was asked, how it was
-- understood and what was answered.
CREATE TABLE IF NOT EXISTS recipe_runs (
    id                 BIGSERIAL   PRIMARY KEY,
    request_text       TEXT        NOT NULL,
    parsed_constraints JSONB,                -- null when parsing itself failed
    matched_template   TEXT,
    status             TEXT        NOT NULL, -- ok | infeasible | unsupported_product | error
    result             JSONB       NOT NULL, -- the response body (or {"error": ...})
    duration_ms        INTEGER     NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS recipe_runs_created_at_idx ON recipe_runs (created_at DESC);
