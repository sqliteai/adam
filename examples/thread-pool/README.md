# Thread Pool

Demonstrates concurrent agent execution using Adam's thread pool. A pool of 4 worker threads is created, and 10 independent jobs are submitted, each asking for a fun fact about a different number. Each job gets its own settings and history (these are not shared across threads). Results are printed as they arrive via the `on_done` callback, which is called from the worker thread that completed the job.

## Build & Run

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
make
./thread-pool
```

## What It Demonstrates

- Creating a thread pool with `adam_pool_create`
- Submitting jobs with `adam_pool_submit` using per-job settings and history
- Using `adam_job_done_fn` to receive results as they complete
- Waiting for all jobs with `adam_pool_destroy`
