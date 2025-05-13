┌──────────────┐   (1) employee reply   ┌──────────────┐
│  Employee UI │ ─────────────────────► │  Coach LLM   │
└──────────────┘                        │ (“grader”)   │
                                        └──────────────┘
                                             │ (2) coach‑feedback JSON
                                             ▼
                                    ┌────────────────┐
                                    │  Referee LLM   │
                                    │ (“auditor”)    │
                                    └────────────────┘
                                        │ pass/fail
                ┌───────────────┐       │
                │ Orchestrator  │ ◄─────┘ (3) coaching_grade
                └───────────────┘
                      │
            ┌─────────┴─────────┐
            │  If pass ▶ next   │
            │  If fail ▶ ask    │
            │  Coach to retry   │
            └───────────────────┘


Customer Agent  ◀────  (only responds after coach passes)
      ▲
      │
      ▼
Employee replies  ─▶  Coach grades  ─▶  Referee audits  ─▶  Orchestrator
