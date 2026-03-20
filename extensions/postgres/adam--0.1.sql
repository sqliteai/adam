-- Adam PostgreSQL Extension — SQL function definitions

CREATE FUNCTION adam_config(key text, value text)
RETURNS text
AS 'MODULE_PATHNAME', 'pg_adam_config'
LANGUAGE C STRICT;

CREATE FUNCTION adam(message text)
RETURNS text
AS 'MODULE_PATHNAME', 'pg_adam'
LANGUAGE C STRICT;

CREATE FUNCTION adam_ask(message text)
RETURNS text
AS 'MODULE_PATHNAME', 'pg_adam_ask'
LANGUAGE C STRICT;

CREATE FUNCTION adam_sql(question text)
RETURNS text
AS 'MODULE_PATHNAME', 'pg_adam_sql'
LANGUAGE C STRICT;

CREATE FUNCTION adam_create_session()
RETURNS text
AS 'MODULE_PATHNAME', 'pg_adam_create_session'
LANGUAGE C STRICT;

CREATE FUNCTION adam_get_session()
RETURNS text
AS 'MODULE_PATHNAME', 'pg_adam_get_session'
LANGUAGE C;

CREATE FUNCTION adam_clear_session()
RETURNS boolean
AS 'MODULE_PATHNAME', 'pg_adam_clear_session'
LANGUAGE C STRICT;
