# Codestra-Face-Liveness — Architecture Charts

> Repository: `appolon1908/Codestra-Face-Liveness`  
> Baseline branch: `main`  
> Repository-local visual architecture. Update these charts whenever ownership, interfaces, persistence, or deployment changes.

## 1. System context
```mermaid
flowchart LR
 A["FACE-ID / clients"] --> B["Liveness API"]
 B --> R["Codestra-Face-Liveness<br/>Face liveness verification service"]
 R --> S["ephemeral analysis state"]
 R --> D["recognition / decision consumers"]
```

## 2. Internal architecture
```mermaid
flowchart TB
 I["Entrypoint / API / CLI"] --> P["Auth, policy, validation"]
 P --> C["Core domain / orchestration"]
 C --> S["State / configuration / persistence"]
 C --> A["Adapters / integrations"]
 A --> X["Approved external dependencies"]
 C --> O["Metrics, logs, traces, audit"]
```

## 3. Critical flow
```mermaid
sequenceDiagram
 participant U as Caller
 participant B as Codestra-Face-Liveness
 participant P as Policy
 participant C as Core
 participant S as State
 participant X as Dependency
 U->>B: Request / event / command
 B->>P: Authenticate + validate
 P-->>B: Decision
 B->>C: Capture, quality checks, liveness score and decision
 C->>S: Read / persist state
 C->>X: Bounded integration
 X-->>C: Result / readback
 C-->>U: Normalized response
```

## 4. Deployment and promotion
```mermaid
flowchart LR
 F["Feature branch"] --> T["Focused tests"]
 T --> PR["Pull request + review"]
 PR --> CI["Required CI green"]
 CI --> ST["Staging / isolated runtime"]
 ST --> V["Exact-SHA verification"]
 V --> G{"Production approval?"}
 G -- No --> ST
 G -- Yes --> P["Production promotion"]
 P --> H["Health/readiness + rollback check"]
```

## 5. Observability and recovery
```mermaid
flowchart LR
 R["Codestra-Face-Liveness"] --> M["Metrics"]
 R --> L["Logs / audit"]
 R --> T["Traces / correlation"]
 M --> O["Observability"]
 L --> O
 T --> O
 O --> A["Dashboards / alerts"]
 R --> B["Backup / config snapshot"]
 B --> RR["Restore / rollback rehearsal"]
```

## Ownership notes
- **Role:** Face liveness verification service
- **Primary boundary:** Liveness API
- **State/config:** ephemeral analysis state
- **Dependencies/consumers:** recognition / decision consumers
- Cross-repository effects must use reviewed contracts; production effects remain separately gated.
