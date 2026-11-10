# Database Architecture

## Overview

The Intel Inference Toolkit uses a **shared CloudNativePG (CNPG) PostgreSQL cluster**. Each consumer (LiteLLM, Langfuse, Keycloak, and the other services Toolkit Manager installs) gets its own isolated database and user within the same cluster.

```
┌─────────────────────────────────────────────────────────────────┐
│  PostgreSQL Cluster (CNPG)                                      │
│  Pod: postgresql-1                                              │
│  Service: postgresql-rw.postgresql.svc.cluster.local:5432       │
│                                                                 │
│  ┌──────────────┬──────────────┬──────────────┐                │
│  │ DB: keycloak │ DB: litellm  │ DB: langfuse │                │
│  │ User:keycloak│ User:litellm │ User:langfuse│                │
│  └──────────────┴──────────────┴──────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Role | Description |
|-----------|------|-------------|
| CNPG Operator | `cnpg-system` namespace | Manages PostgreSQL cluster lifecycle, failover, backups |
| PostgreSQL Cluster | `postgresql` namespace | Single shared PostgreSQL 17 instance (configurable replicas) |
| Database CRDs | `postgresql` namespace | Declarative database creation via CNPG `Database` custom resource |
| Consumer Secrets | Per-consumer namespace | Credentials copied from postgresql namespace to each consumer |

## Configuration

Databases are declared in the inventory/global config:

```yaml
postgresql_databases:
  - name: keycloak            # first entry = bootstrap DB (created at cluster init)
    user: keycloak
    target_namespace: keycloak
  - name: litellm
    user: litellm
    target_namespace: litellm
  - name: langfuse
    user: langfuse
    target_namespace: langfuse
```

## Provisioning Flow

### Phase 1: PostgreSQL role (runs first in playbook)

```
Step 1 ─ Install CNPG Operator (Helm)
  └─→ Deploys operator in cnpg-system namespace

Step 2 ─ Create credential Secrets (check-then-create, never overwritten)
  └─→ postgresql-keycloak-secret   (namespace: postgresql)
  └─→ postgresql-litellm-secret    (namespace: postgresql)
  └─→ postgresql-langfuse-secret   (namespace: postgresql)
       Each contains: {username, password (auto-generated)}

Step 3 ─ Create SQL init Secrets (CREATE USER statements)
  └─→ postgresql-init-litellm      (namespace: postgresql)
  └─→ postgresql-init-langfuse     (namespace: postgresql)
       Each contains SQL: CREATE USER <name> WITH PASSWORD '<pass>'

Step 4 ─ Deploy CNPG Cluster CR
  └─→ bootstrap.initdb creates first DB (keycloak)
  └─→ postInitSQLRefs runs SQL from init Secrets (creates users)
  └─→ Result: postgresql-1 Pod running PostgreSQL 17

Step 5 ─ Create additional databases via CNPG Database CRD
  └─→ Database CR "litellm"  → operator creates DB, owner=litellm
  └─→ Database CR "langfuse" → operator creates DB, owner=langfuse

Step 6 ─ Copy secrets to consumer namespaces
  └─→ litellm-db-secret   (namespace: litellm)
  └─→ langfuse-db-secret  (namespace: langfuse)
       Each contains:
         username: <user>
         password: <password>
         database: <db_name>
         host: postgresql-rw.postgresql.svc.cluster.local
         port: 5432
```

### Phase 2: Consumer roles (run later in playbook)

Each consumer role follows the same pattern:

```
Step 1 ─ Wait for "<name>-db-secret" in its namespace
  └─→ Retries with timeout — waits for postgresql role to copy secret

Step 2 ─ Wait for Database CRD status.applied = true
  └─→ Confirms the database actually exists in PostgreSQL

Step 3 ─ Extract connection details from secret
  └─→ host, port, database, username, password

Step 4 ─ Pass credentials to Helm chart
  └─→ LiteLLM:  db.useExisting=true, db.secret.name=litellm-db-secret
  └─→ Langfuse: postgresql.deploy=false, postgresql.host=<host>
```

## Example: LiteLLM connecting to PostgreSQL

```
LiteLLM Pod
  │
  │ reads litellm-db-secret from namespace "litellm"
  │   username: litellm
  │   password: R4nd0mG3n3r4t3d
  │   host: postgresql-rw.postgresql.svc.cluster.local
  │   port: 5432
  │   database: litellm
  │
  ▼
postgresql-rw.postgresql.svc.cluster.local:5432
  │
  ▼
postgresql-1 Pod (CNPG managed)
  └─→ database "litellm", owned by user "litellm"
      └─→ Tables: virtual keys, spend logs, team/org data (managed by Prisma)
```

## Example: Langfuse connecting to PostgreSQL

```
Langfuse Web/Worker Pods
  │
  │ Helm values inject credentials directly:
  │   postgresql.host: postgresql-rw.postgresql.svc.cluster.local
  │   postgresql.auth.username: langfuse
  │   postgresql.auth.password: <from langfuse-db-secret>
  │   postgresql.auth.database: langfuse
  │
  ▼
postgresql-rw.postgresql.svc.cluster.local:5432
  │
  ▼
postgresql-1 Pod (CNPG managed)
  └─→ database "langfuse", owned by user "langfuse"
      └─→ Tables: traces, scores, projects, users (managed by Prisma)
```

## Design Decisions

### Why shared PostgreSQL (not per-service)?

| Concern | Shared CNPG | Per-service Bitnami standalone |
|---------|-------------|-------------------------------|
| High availability | CNPG handles replication + failover | Single-node, no HA |
| Backups | WAL archiving, PITR | Manual, no built-in |
| Resource efficiency | 1 pod serves all services | N pods (one per service) |
| Operational complexity | Single cluster to manage | Multiple independent instances |
| Isolation | Separate databases + users (can't access each other) | Full process isolation |

### Why passwords are stable across re-runs

The postgresql role uses a **check-then-create** pattern:
1. Check if `postgresql-<name>-secret` exists
2. Only create it if it doesn't exist
3. Never overwrite existing secrets

This ensures re-running the playbook won't rotate passwords and break running services.

### Why secrets are copied to consumer namespaces

- CNPG creates credentials in the `postgresql` namespace
- Pods in other namespaces (litellm, langfuse) can't read cross-namespace secrets
- The postgresql role copies each consumer's credentials into their target namespace
- Consumer roles just wait for their local secret to appear

## Secrets Reference

| Secret name | Namespace | Created by | Contents |
|-------------|-----------|------------|----------|
| `postgresql-<name>-secret` | postgresql | postgresql role | username, password |
| `postgresql-init-<name>` | postgresql | postgresql role | SQL CREATE USER statement |
| `litellm-db-secret` | litellm | postgresql role (copied) | username, password, database, host, port |
| `langfuse-db-secret` | langfuse | postgresql role (copied) | username, password, database, host, port |

## Adding a New Database Consumer

To add a new service that needs PostgreSQL:

1. Add an entry to `postgresql_databases`:
   ```yaml
   postgresql_databases:
     - name: myservice
       user: myservice
       target_namespace: myservice
   ```

2. Re-run the playbook — the postgresql role will:
   - Generate credentials
   - Create the user + database
   - Copy the secret to the target namespace

3. In your service's Helm values, reference the secret:
   ```yaml
   db:
     host: postgresql-rw.postgresql.svc.cluster.local
     port: 5432
     secret:
       name: myservice-db-secret
       usernameKey: username
       passwordKey: password
   ```

## Troubleshooting

### Check if database exists
```bash
kubectl get database -n postgresql
```

### Check database readiness
```bash
kubectl get database <name> -n postgresql -o jsonpath='{.status.applied}'
# Should return: true
```

### Check consumer secret
```bash
kubectl get secret <name>-db-secret -n <namespace> -o jsonpath='{.data.host}' | base64 -d
```

### Connect to PostgreSQL directly
```bash
kubectl exec -it postgresql-1 -n postgresql -c postgres -- psql -U postgres
\l          # list databases
\du         # list users
```

### Check CNPG cluster health
```bash
kubectl get cluster -n postgresql
kubectl describe cluster postgresql -n postgresql
```
