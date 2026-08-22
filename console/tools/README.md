# Console tools

`console.tools` contains standalone operators for world audits, remote worker
execution, browser receipts, and device checks. They call
Console/Arena interfaces but are not imported by the service request path.

Use [`remote_run.py`](remote_run.py) for the controlled worker boundary,
[`audit_worldgen.py`](audit_worldgen.py) and
[`audit_worldgen_families.py`](audit_worldgen_families.py) for world inspection,
and [`replay_policy_checkpoint.py`](replay_policy_checkpoint.py) for replay
inspection. Training itself remains a `POST /api/train` operation.
