# Task Plan: STEP385 remote isolated OPC build controller

## Goal
Add a disabled-by-default, offline-testable STEP385 controller and tests without remote access or changes to existing implementation files.

## Current Phase
Complete

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify constraints
- [x] Document in findings.md
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Define approach
- [x] Create project structure
- **Status:** complete

### Phase 3: Implementation
- [x] Execute the plan
- [x] Write to files before executing
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Verify requirements met
- [x] Document test results
- **Status:** complete

### Phase 5: Delivery
- [x] Review outputs
- [x] Deliver to user
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| New STEP385 controller/test only | Preserve existing files and dirty worktree |
| Exact single-token remote adapter enablement | Remote copy is build-capable while local adapter remains disabled |
| Gate main/execute before helper/backend construction | Fail closed and make dry-run transport-free |
| Unique non-latest attempt name | Failed attempts cannot be reused or redirected through latest aliases |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| Python 3.9 lacks `hashlib.file_digest` | Replaced with portable chunked SHA-256 and scheduled full rerun |
