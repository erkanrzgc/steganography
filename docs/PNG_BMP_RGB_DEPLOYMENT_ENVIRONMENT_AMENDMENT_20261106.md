# RGB pilot deployment environment amendment — 2026-11-06

This prospective administrative amendment binds the existing SSH session to its
already-running user manager. The initial source-only deployment verifies all
248 source bindings and the exact CUDA environment, but its read-only manager
check fails because SSH has neither XDG_RUNTIME_DIR nor DBUS_SESSION_BUS_ADDRESS.
Preserve that failure and the original e5c2ea05f7078f10656adfda595c4c0cd6705c7cfecb3f878ea8b3d05f35fc81
protocol/package. No launch or attempt marker exists and no GPU job was started;
this is not a retry of a consumed physical experiment.

The controller now checks the existing owned `/run/user/<uid>` directory and
its bus socket, then sets those two process-local environment variables for
its SSH commands. It creates no user manager and changes no shell, package,
driver, key, server configuration or login file. A separate read-only check
successfully reaches the existing manager and finds the new profile unit absent.

The candidate source-only package is staged in a fresh private directory; the
original package remains intact. The deployment record binds this amendment's
SHA-256, and the controller verifies it both locally and in staged commands.
The original application/test files, cache, proposal, grouping, payloads,
optimizer recipe, learning rule, seeds, budgets and historical ledgers stay
unchanged. All 464 verified application/test files remain identical. The
controller and transport fixture receive a fresh generated, zero-SSH dry-run
before the amended source/environment freeze. The new frozen protocol is
`benchmarks/png-bmp-rgb-pilot-environment-amendment-20261106.json`.

The same single fresh physical profile is allowed: 240 seconds including
parent closeout, 8-GiB RAM, zero swap, 4-GiB allocator, 256-MiB output, zero
native updates and six generated updates. A failure consumes that attempt and
never authorizes retry. Later native fitting still requires the accepted final
profile and a separate explicit user approval. No reserved pixels, new media,
old experiment restart or native optimizer step is permitted by this amendment.
