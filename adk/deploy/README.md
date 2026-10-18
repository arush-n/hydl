# Policy deployment

`adk.deploy` turns a validated checkpoint into a native policy bundle and
checks that the bundle matches the observation, action, role, and bridge
contracts expected by the server.

It depends on ADK policy identity and the HytaleRL native bridge. Deployment is
separate from training: Arena or an ADK algorithm produces a checkpoint, while
this package verifies and exports the artifact that a native session can load.

## Entry points

- [`bundle.py`](bundle.py) exports and verifies bundles.
- [`compatibility.py`](compatibility.py) checks producer/consumer identities.
- [`tensor_identity.py`](tensor_identity.py) binds parameter structure and
  hashes.
- [`isolated.py`](isolated.py) runs a bundle in an isolated native process.

`isolated.py --policy-trace` enables the standalone policy mod's bounded
`hytale_policy_native_policy_trace_v1` JSONL artifact. It contains the native
structured observation/mask, Java head factors, selected action, and paired
next observation. This is a diagnostic input/action trace, not a complete
JAX-vs-server transition certificate.

For a longer capture, pass `--policy-trace-max-transitions N`; the default
nonterminal-row budget is 512. The cap is recorded in both the trace header and
run report. One terminal row may be retained even after the nonterminal budget
is used, and no later callbacks can append rows after that terminal event. This
only changes evidence capture: it does not change the task's tick limit, and a
run that exceeds that limit remains a certification failure.

To keep the isolated server running beyond the normal runtime gate until the
native actor trace ends in a validated terminal row, pass both
`--policy-trace` and `--require-complete-policy-trace`. The option is opt-in;
without it, the runner still stops as soon as the existing runtime gate passes.
Choose a transition budget large enough for the episode: the budget limits
nonterminal rows, with only one additional native terminal row allowed. The
runner fails with the trace receipt/reason if the server times out or exits
before a complete trace, the actor terminates before the runtime gate, the
trace is malformed or discontinuous, or the row budget makes a complete trace
impossible. It never converts a `trace_end` row into a terminal row.

For an episode expected to fit within 2,048 nonterminal transitions, append
`--policy-trace --require-complete-policy-trace --policy-trace-max-transitions 2048 --timeout-seconds 900`
to the launch command above, increasing the row budget or timeout if needed.

A successful complete trace can be replayed against the trained policy to
check action-boundary agreement across the native episode. That evidence does
not establish JAX/Hytale transition or reward parity, nor does it establish
pursuit success.

The Java-side consumer is documented in
[`HytaleRL/hytale-plugin/README.md`](../../HytaleRL/hytale-plugin/README.md).

## Bundle compatibility and isolated launch

The Python exporter owns the exact policy contract. It places the observation
and action contract SHA-256 values plus `policy_projection_abi` in `agent.json`
and `contract.txt`. Exact hashes identify the exported policy semantics; the
stable projection ABI identifies the observation layout and action meanings
implemented by the Java port. Compatible policy hashes may change without a
Java edit. A Java semantic change requires a new ABI and a Java port that
explicitly supports it. There is no launch-time expected-hash override.

Java checks the ABI, canonical matching observation/action identities, tensor
identity, and (for live profiles) the profile's exact policy hash and ABI
before it loads weights or claims an actor. An unsupported ABI reports
`policy_projection_abi` and the supported value; missing or changed exact
identities and altered tensors also fail closed. Bundle schema v6 carries the
training-time per-head action scope and a digest bound to the exact action
contract. Java intersects that scope with live server legality before inference.
Bundles from older schemas must be re-exported; no expected policy hash is
entered at launch.

New pursuit training runs write their action scope into checkpoint metadata. A
supported older pursuit run (`arena-recurrent-pursuit-run-v16`) can be exported
with `--training-report <matching-report.json>`; the adapter verifies checkpoint
lineage, selected update/step count, and stage contract before reconstructing
the scope. Other missing-scope bundles fail closed.

All deployment artifacts are launch inputs; their hashes are written into the
run report. The bundle hash covers each relative file name and file content.
The runner also lets you select the server bind IP and a source world/universe
without editing Java or the bundle:

```powershell
py -3 -m adk.deploy.isolated <policy-bundle> <new-runtime-directory> `
  --policy-jar <policy-agent.jar> --bridge-jar <bridge.jar> `
  --server-jar <HytaleServer.jar> --assets <Assets.zip> --java <java.exe> `
  --bind-address 127.0.0.1 --world-or-universe <world-or-universe-directory>
```

`--bind-address` accepts an IPv4 or IPv6 address and defaults to loopback; a
public/wildcard address exposes the server to that network. A supplied
universe (containing `worlds/`) or world (containing `config.json`) is copied
into the new private runtime, so the source is not used as the server's writable
universe. A world directory is staged as `worlds/default`. `--flat-world` and
`--world-or-universe` cannot be combined.

The isolated runtime gate certifies that the selected policy loads, produces
policy ticks and legal actions, and causes movement attributed to the policy
under the declared fixture. It does not certify visual perception, exact
simulator transition fidelity, catching an opponent, or winning a pursuit.
Those need separate evidence and are not part of this launch gate.

### Hytale 0.5.9 episode evidence

A headless run of the actual trained pursuer passed the runtime gate in a flat
test world with pursuit context enabled. It faced an explicit `Trork_Brawler`
target from 12 blocks away; jump was disabled in the policy's action scope. The
native episode ended after 593 policy ticks with the genuine
`evidence_agent_dead` terminal: the pursuer died without catching the target.
The target was observed on 593/593 ticks, there were no illegal-action
rejections, and the pursuer traveled 55.4016 horizontal blocks. Its closest
target distance was 1.6402 blocks; its final distance was 1.9294 blocks.

The native trace was complete: 593 rows, including one genuine terminal row.
Replaying its observations in JAX matched Java's selected action outputs on all
593 decisions (zero mismatches); 587 rows also recorded next observations.
This confirms action-output agreement for one episode only, not matching next
states or rewards. The live run ended in pursuer death, so it proves neither a
catch nor performance across multiple seeds.

The run's selected-policy evaluation recorded 118 catches in 284 episodes, with
`jump_request_fraction` and `jump_takeoff_fraction` both 0.0. This selection
result is not live sim-to-server proof; the live evidence remains one episode
ending in pursuer death.

The action scope excluded jump and the trace contained zero jump-held rows. The
held-jump fidelity mismatch was therefore not exercised and is not a reason to
patch mechanics for this policy. Revisit it if a policy enables jump. Focused
isolated-runner tests pass **42** (baseline **35**); the Java-agent suite passes
**46/46**.
