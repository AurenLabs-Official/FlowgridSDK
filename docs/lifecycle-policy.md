# Lifecycle Policy (API, Runtime, Dependencies)

This policy anchors Phase D operational maturity: predictable upgrades and deprecations.

## Public SDK (`flowgrid` crate)

- Semver is enforced via CI semver-checks and [CONTRIBUTING.md](../CONTRIBUTING.md).
- Breaking changes require migration notes and a changelog entry under the correct version.

## Preview LLM / ML crates

- Breaking changes are allowed but must be called out in [CHANGELOG.md](../CHANGELOG.md) with migration hints.
- Checkpoint manifest evolution: prefer additive fields; bump manifest semantics only with tooling updates.

## Runtime profiles

- Profile defaults (`FLOWGRID_DEPLOYMENT_PROFILE`) may change between minor releases when documented.
- After dependency upgrades affecting Burn/tokenizers/network stacks, re-run profile KPI smokes.

## Deprecation process

1. Document deprecation in code/rustdoc and `CHANGELOG.md`.
2. Keep deprecated behavior for at least one planned release cycle unless security-critical.
3. Provide replacement commands/paths in [migration.md](migration.md).

## Upgrade checklist

- [ ] `cargo test --workspace` (or scoped crates for preview areas)
- [ ] `just check-ml-core` / gate bundle as appropriate
- [ ] KPI smoke for active deployment profiles when runtime changed
