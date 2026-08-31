# Git workflow (standing rule)

After every meaningful code change in this repo, commit and push automatically
at the end of the response — do not wait to be asked.

```
git add .
git commit -m "<summary line>

<2-4 bullets: what changed and why>"
git push -u origin <current branch>   # -u only on the branch's first-ever push
git push origin <current branch>      # plain push after that
```

- Push the **current branch** to its own remote, not a hardcoded `main`. Check
  `git branch --show-current` and `git status` first — this repo uses feature
  branches (e.g. `hf-lung-baseline`), and blindly pushing `main` when a
  different branch is checked out can look like it succeeded while actually
  pushing nothing new. Use `-u` only the first time a branch is pushed
  (no upstream configured yet); a plain `git push` is correct after that.
- If it's ever unsure which branch should receive a commit — work spans
  multiple branches, or the right target isn't obvious — **stop and ask
  first**, rather than guessing.
- Never add Claude as co-author or include "Co-Authored-By: Claude" / a session
  link in the commit message — the user is the sole author.
- Update `README.md` with a brief note on what changed, as part of the same
  commit, every time — do not skip this even for small changes.
- Before `git add .`, check `git status` and confirm nothing large/generated
  (data files, model checkpoints, packaged zips) would be staged. Most of that
  is already covered by `.gitignore` in this repo, but verify rather than assume.
- Show the exact commit message and confirm the push succeeded before ending
  the turn.

This overrides the default "only commit when explicitly asked" behavior for
this repo.
