# skills/

Agent-readable procedures for working on the kits in this repository: plain markdown with a short front-matter header (name,
description), usable by any coding agent. Point an agent at a skill's SKILL.md from your AGENTS.md or prompt; agents that load
skill folders can link the directory into their skills path.

| Skill | What it covers |
|---|---|
| [loom-strix-optimise](loom-strix-optimise/SKILL.md) | speed work on a Loom kit on Strix Halo: served scoreboard against hardware floors, diagnosis, candidates, A/B verdicts |

Inside a skill: `rules.md` (device-neutral rules, each with its evidence), `devices/<gfx>.md` (facts measured on one GPU target;
add a file per target rather than generalizing), `gotchas.md`, and SDK notes per Loom version. Model-specific knowledge lives
with its kit (`kits/<kit>/NOTES.md`, `lanes.json`).
