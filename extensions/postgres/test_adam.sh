#!/bin/bash
set -e
API_KEY=$(grep ANTHROPIC_API_KEY /tmp/.env | cut -d= -f2)

# Initialize and start PostgreSQL
su - postgres -c "/usr/lib/postgresql/16/bin/initdb -D /var/lib/postgresql/16/main" 2>/dev/null || true
su - postgres -c "/usr/lib/postgresql/16/bin/pg_ctl -D /var/lib/postgresql/16/main -l /tmp/pg.log start"
sleep 2

# Run tests
su - postgres -c "psql -c 'CREATE EXTENSION adam;'"
echo "=== adam_config ==="
su - postgres -c "psql -c \"SELECT adam_config('provider', 'anthropic');\""
su - postgres -c "psql -c \"SELECT adam_config('api_key', '$API_KEY');\""

echo "=== adam() ==="
su - postgres -c "psql -c \"SELECT adam('Say hello in 3 words');\""

echo "=== adam_ask() ==="
su - postgres -c "psql -c 'CREATE TABLE IF NOT EXISTS users (id serial, name text, age int);'"
su - postgres -c "psql -c \"INSERT INTO users VALUES (1,'Alice',30),(2,'Bob',25),(3,'Carol',35) ON CONFLICT DO NOTHING;\""
su - postgres -c "psql -c \"SELECT adam_ask('How many users and what are their names?');\""

echo "=== DONE ==="
