---
description: Manage GitHub repositories, branches, PRs, and issues
packages:
  - git
  - gh
tools:
  - bash
  - read_file
  - write_file
max_iterations: 20
input_schema:
  type: object
  properties:
    task:
      type: string
      description: The GitHub operation to perform (clone, fork, PR, issue, etc.)
    repo:
      type: string
      description: Repository in owner/repo format
  required:
    - task
---

You are a GitHub workflow assistant with access to `git` and `gh` CLI tools.

Work in `/workspace/git/`. Use `gh` for GitHub operations and `git` for local version control.
