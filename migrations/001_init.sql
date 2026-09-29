-- Applied on every API start: must stay idempotent.

-- Log of POST /recipe calls (filled in KAN-47): request text, outcome, full response for audit.
CREATE TABLE IF NOT EXISTS recipe_requests (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    query       TEXT        NOT NULL,
    status      TEXT        NOT NULL,  -- solved | infeasible | error
    response    JSONB,
    error_code  TEXT,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS recipe_requests_created_at_idx ON recipe_requests (created_at DESC);
