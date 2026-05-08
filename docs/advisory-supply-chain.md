# Advisory supply-chain tracker (`cargo deny`)

[`deny.toml`](../deny.toml) lists **ignored** [RustSec](https://rustsec.org) advisories that affect **transitive** dependencies (Burn, `tokenizers`, gitoxide). Each row should be reviewed **at least quarterly**; when an upstream release removes the vulnerable crate version, drop the matching `ignore` entry and re-run `cargo deny check`.

| ID | Crate (transitive path) | Revisit when |
|----|-------------------------|---------------|
| RUSTSEC-2025-0141 | `bincode` 2.0.0-rc.x via `burn-core` | Burn / `bincode` story changes; workspace may drop pin `=2.0.0-rc.3` |
| RUSTSEC-2025-0021 | `gix-features` via `burn-dataset` → `gix-tempfile` | Burn bumps gitoxide stack or replaces dataset temp handling |
| RUSTSEC-2024-0350 | `gix-fs` (same path as above) | Same as RUSTSEC-2025-0021 |
| RUSTSEC-2025-0119 | `number_prefix` via `tokenizers` → `indicatif` | `tokenizers` / `indicatif` upgrade removes crate |
| RUSTSEC-2024-0436 | `paste` via `tokenizers` | `tokenizers` drops `paste` / migrates macros |

Optional: open a **single** tracking issue (e.g. “Supply-chain: transitive advisory ignores”) and link it from the table when your org uses GitHub Issues.
