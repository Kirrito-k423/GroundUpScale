# Issue tracker: GitHub

Issues and PRDs for this repository live in GitHub Issues:
https://github.com/Kirrito-k423/GroundUpScale/issues

Use the `gh` CLI for issue operations. Infer the repository from `origin`.

- Create: `gh issue create`
- Read: `gh issue view <number> --comments`
- List: `gh issue list`
- Comment: `gh issue comment <number>`
- Label: `gh issue edit <number> --add-label "..."`
- Close: `gh issue close <number> --comment "..."`

PRs as a request surface: no.

When a skill says “publish to the issue tracker”, create a GitHub issue.
When a skill says “fetch the relevant ticket”, read the corresponding GitHub issue.

Wayfinder maps and child tickets are represented by GitHub issues.
Prefer native sub-issues and issue dependencies when available; otherwise use
task lists and explicit `Blocked by: #<number>` declarations.
