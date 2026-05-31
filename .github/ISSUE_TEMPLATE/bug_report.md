---
name: Bug report
about: Something broke or produced wrong output
labels: bug
---

**Goal you ran**
<!-- Paste the exact goal string you used -->

**What happened**
<!-- What did AgentForge do? Paste the session log output. -->

**What you expected**
<!-- What should have happened? -->

**Environment**
- OS:
- Python version:
- Ollama model (`agentforge models list`):
- AgentForge version (`git log --oneline -1`):

**Reproduction steps**
```bash
agentforge run "your goal here" --working-dir /tmp/repro --auto-approve
```
