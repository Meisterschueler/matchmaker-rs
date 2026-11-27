# Feature Specification: Synchronize pymatchmaker 0.5.2

**Feature Branch**: `007-pymatchmaker-052-sync`

**Created**: 2026-11-27

**Status**: Draft

**Input**: User description: "es gibt bei pymatchmaker eine neue Version: 0.5.2. aktualisiere matchmaker-rs dahingehend. Überprüfe ansonsten alle Dateien auf Konsistenz bezüglich pymatchmaker. MIr ist wichtig, dass man änderungen in pymatchmaker leicht nachziehen kann in matchmaker-rs."

This feature moves the port's reference from pymatchmaker 0.5.1 to 0.5.2, checks all current-state references to pymatchmaker across the repository, and makes the relationship between upstream changes and the corresponding port decisions easy to follow. Historical records and attribution for assets from older releases remain historically accurate rather than being rewritten as current pins.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use the 0.5.2 reference (Priority: P1)

A maintainer updates matchmaker-rs to the newly released pymatchmaker version and can verify that the port's supported behavior agrees with that release.

**Why this priority**: The port's reference parity depends on using the requested released version.

**Independent Test**: Regenerate reference outputs using the pinned release and run the existing parity tests.

**Acceptance Scenarios**:

1. **Given** a clean, locked reference environment, **When** a maintainer checks its pymatchmaker version, **Then** it is 0.5.2.
2. **Given** reference outputs regenerated from 0.5.2, **When** parity tests run, **Then** supported behavior matches within each test's documented tolerance.
3. **Given** an upstream change that alters an existing result, **When** the regenerated outputs are reviewed, **Then** the difference is traceable to a specific 0.5.2 change.

### User Story 2 - Find and reconcile every current reference (Priority: P1)

A contributor can find every current-state mention or dependency on pymatchmaker and determine whether it agrees with 0.5.2, without mistaking historical release records or unrelated package versions for the active pin.

**Why this priority**: Conflicting versions, defaults, or behavioral descriptions make the port difficult to trust and maintain.

**Independent Test**: Search the repository for pymatchmaker references and review each result as current, historical, or attribution-only; confirm current references agree.

**Acceptance Scenarios**:

1. **Given** a repository-wide audit of pymatchmaker references, **When** each result is classified, **Then** every current pin and current behavior description agrees with the 0.5.2 reference.
2. **Given** an older version mentioned in a historical specification or asset attribution, **When** the audit is reviewed, **Then** it remains clearly historical and is not incorrectly changed to 0.5.2.
3. **Given** a changed upstream API, default, or behavior, **When** it affects supported user-facing surfaces, **Then** the related library, command-line, Python, documentation, and tests are consistent.

### User Story 3 - Carry future upstream changes forward efficiently (Priority: P2)

A contributor handling a later pymatchmaker release can compare it with the currently pinned release and quickly identify the Rust counterpart, validation evidence, or explicit reason a change is not ported.

**Why this priority**: An auditable, complete change map reduces missed changes and makes future release syncs less costly.

**Independent Test**: Review the release-change record and trace each changed upstream source file and behavior to its port counterpart, test evidence, or an explicit not-ported decision.

**Acceptance Scenarios**:

1. **Given** the complete 0.5.1-to-0.5.2 upstream change set, **When** a contributor consults the change record, **Then** each changed source file and relevant change has a Rust counterpart or a reasoned not-ported decision.
2. **Given** a Rust implementation corresponding to an upstream behavior, **When** a contributor follows its source reference, **Then** the upstream file and symbol can be identified without relying on undocumented knowledge.
3. **Given** a future release sync, **When** an upstream file not previously represented in the port changes, **Then** the process requires it to be assessed and recorded rather than silently excluded.

### Edge Cases

- An upstream source file changes without a change to an already mapped symbol; it still receives an explicit assessment.
- A release changes a default or accepted configuration key without changing the main algorithm; all supported entry points must agree.
- A change is intentionally not ported; its reason and user-visible limitation remain explicit.
- A historical specification, vendored asset attribution, or unrelated dependency version mentions an older release; it must not be mistaken for an active reference pin.
- A regenerated golden value changes for reasons unrelated to the 0.5.2 diff; the mismatch must be investigated rather than accepted as an unexplained update.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The active reference environment MUST pin pymatchmaker 0.5.2, and reference-output generation MUST verify that it is using that release.
- **FR-002**: The release comparison MUST assess the complete set of upstream files changed between 0.5.1 and 0.5.2, including files not already represented in the port's module map.
- **FR-003**: Every relevant upstream change MUST be recorded as implemented, already equivalent in the port, or intentionally not ported with a stated reason. The record MUST identify the upstream file and affected symbol or behavior where applicable.
- **FR-004**: The repository-wide consistency review MUST cover current pymatchmaker version declarations, source mappings, behavior and configuration descriptions, reference generators, tests, and supported user-facing surfaces. Current-state descriptions MUST agree with the 0.5.2 release.
- **FR-005**: References that are historical, attribution-only, or for a different dependency MUST remain accurate and MUST NOT be changed merely to match the active pymatchmaker pin.
- **FR-006**: Changes to supported behavior, defaults, or configuration MUST be reflected consistently in the library, command-line interface, Python package, documentation, and relevant tests.
- **FR-007**: Reference outputs MUST be regenerated from the locked 0.5.2 environment, and each changed expected result MUST be explainable by an assessed upstream change or an explicitly documented port difference.
- **FR-008**: Existing supported behavior not changed by pymatchmaker 0.5.2 MUST retain its established results and documented limits.
- **FR-009**: The contributor-facing release-sync guidance MUST describe how to compare the complete upstream release change set, trace changes to their port counterparts, validate results, and record exclusions. It MUST NOT limit future reviews to only the files already listed in the module map.

### Key Entities

- **Pinned reference release**: The pymatchmaker release used to define expected behavior and generate reference outputs.
- **Upstream change record**: The assessed set of release changes, each associated with its source location and a port, equivalence, or exclusion decision.
- **Current reference surface**: A repository declaration, implementation mapping, test, or user-facing description that describes the active pymatchmaker behavior.
- **Historical reference**: A release mention retained to describe past work or the provenance of an asset rather than the active behavior baseline.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The active reference environment and all current-state release declarations identify pymatchmaker 0.5.2, with no contradictory active version remaining.
- **SC-002**: 100% of relevant files in the 0.5.1-to-0.5.2 upstream change set have a recorded implementation, equivalence, or reasoned exclusion decision.
- **SC-003**: 100% of changed golden expectations are traceable to an assessed upstream change or a documented port difference, and the supported parity tests pass within their stated tolerances.
- **SC-004**: Every current pymatchmaker reference found in the repository audit is either consistent with 0.5.2 or explicitly classified as historical, attribution-only, or unrelated to the active pin.
- **SC-005**: A contributor can trace each implemented upstream change from its upstream source location to the corresponding port location and validation evidence using the repository's maintained documentation.

## Assumptions

- The requested target is the released pymatchmaker 0.5.2, with the currently recorded baseline at 0.5.1.
- The sync includes upstream changes relevant to the existing scope of matchmaker-rs; unsupported upstream capabilities may remain unported when the exclusion and user-visible limitation are recorded.
- Existing project parity tolerances and validation practices remain in effect unless the 0.5.2 changes require an explicitly justified update.
- Historical specifications and asset provenance are evidence of past decisions and should remain unchanged unless they contain a factual error independent of the active version update.
