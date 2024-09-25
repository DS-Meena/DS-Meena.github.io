---
layout: post
title:  "Advancing in Git"
date:   2023-09-23 15:08:10 +0530
categories: Web
---

## Introduction

This blog post discusses intermediate techniques for using Git, a version control system commonly used in software development. Topics covered include branch management, interactive staging, cherry-picking commits, creating and applying diff patches, and rebasing. The post also explains the use of pre-receive hooks and the Git Rerere feature.

## Branch Management

### Git reset
    
If you're using Git and you need to undo changes you made to files, you can use the `git reset` command. Basically, it resets your working directory and staging area to a previous commit. So, let's say you made some changes to files that you haven't committed yet, you can undo those changes with `git reset`.

| Command | Description |
| --- | --- |
| git reset | reset your working directory and staging area to a previous commit |
| git reset --soft HEAD^ | reset your working directory to the previous commit but keep your changes in the staging area |
| git reset --hard HEAD^ | reset both your working directory and staging area to the previous commit |
| git reset <file> | unstage a file that you accidentally added to the staging area |
| git clean -f  | remove untracked files |
| git rm --cached testfile.js | no longer track the file |
| git restore --staged file.txt | unstage the changes made to file |
| git restore file.txt | restore the file to its previous state |

Just keep in mind that `git reset` can be a bit risky since it can permanently discard changes and commits. So be careful when using it, alright?
    
### Fetch
    
`git fetch` is a command in Git that downloads new changes from a remote repository without merging them into the local branch. It updates the remote-tracking branch, which is a local branch that tracks changes in the remote repository.

After running `git fetch`, you can use `git merge` or `git rebase` to integrate the changes from the remote repository into your local branch. Alternatively, you can use `git checkout` to switch to the remote-tracking branch and inspect the changes without merging them.

`git fetch` is a useful command to use when collaborating with others on a project. It allows you to keep your local repository up-to-date with changes made by others, without affecting your working directory.

| Command | Description |
| --- | --- |
| git fetch | Downloads new changes from a remote repository without merging them into the local branch. |
| git fetch --tags | Fetches all tags from the remote repository that are not already present in the local repository. |
| git fetch --prune | Deletes any remote-tracking branches that are no longer on the remote repository. |
| git fetch -p | Shortcut for `git fetch --prune`. |

```bash
# command to overwrite of your local files with the master branch
git fetch --all
git reset --hard origin/master
```

The git amend command allows you to modify the most recent commit on your branch. You can use it to add changes you forgot to include in the commit, or to modify the commit message. When you run `git commit --amend`, Git will open up your default text editor and allow you to edit the commit message. Once you save and close the editor, the commit message on your most recent commit will be updated. 

If you have staged changes that were not included in the previous commit, running `git commit --amend` will add those changes to the previous commit. You can also use the `-m` flag to modify the commit message without opening the text editor. 

To remove a file from the previous commit, you can use `git rm --cached <file>` and then run `git commit --amend`.

```bash
# change the previous commit
git add
git commit --amend # will add the staged changes to previous commit

git commit --amend -m "New commit message"  # modify commit message

# remove file from staging 
git rm --cached testfile.js    # no longer tracked
```
    

### Force push to a remote

Reasons to use force push:

- Local version is preferable to the remote version
- The remote version went wrong and needs repair
- Versions have diverged and merging is undesirable
- Force push replaces the remote branch with your local branch
- Use with caution
- Commits may disappear
- It can be disruptive for others using the remote branch
- It's an easy way to frustrate your development team

| Command | Description |
| --- | --- |
| `git push -f` or `git push --force` | Force push the changes to the remote repository, replacing the commits that are already there and not in the local repository. |
| `git push --force-with-lease` | Allow force push if no one else has pushed changes to that branch since you pulled. |
| `git commit -am "Add salsa to shopping list"` | Automatically add changed files to the staging area and add the message. |
| `git log` | Show the commits in the local repository. |
| `git log origin/main` | Show the commits in the `origin/main` branch. |
| `git show origin/main` | Show the changes done in commits in the `origin/main` branch. |
| `git reset --hard origin/main` | Replace the local repository with the remote repository. Collaborators can use this command to replace their local repository with the remote repository. |

### Identify Merged branches

Here's a cool trick for Git! You can use the command `git branch --merged` to see which branches have been merged into your current branch. This is really useful if you want to keep track of what features have been incorporated or if you need to do some cleanup after merging a bunch of features. By default, the command uses your current branch, but you can specify other branch names or commits too. Basically, it shows you all the branches whose tips are reachable from the specified commit (or HEAD if you don't specify anything). So, if you're working on a project with multiple branches, give this command a try!

| Command | Description |
| --- | --- |
| git branch --merged | List branches that have been merged to the current branch |
| git branch --no-merged | List branches that haven't been merged to the current branch |
| git branch -r --merged | Show results for remote branches that have been merged |
| git merge main | Merge the main branch into the current branch |
| git branch --merged july_release | Show what branches are merged into the specified branch |
| git branch --merged origin/july_release | Show what branches are merged into the specified remote branch |
| git branch --merged b325a7c49 | Show what branches have this commit |

### Prune Stale Branches

To keep your Git repository organized, it's important to delete stale branches. A stale branch is a remote-tracking branch that no longer tracks anything because the actual branch in the remote repository has been deleted. To delete a remote branch, you must also delete your remote-tracking branch. However, if another collaborator deletes a remote branch, your remote-tracking branch remains. Fetching does not automatically delete remote-tracking branches, so you must manually prune them.

| Command | Description |
| --- | --- |
| `git branch -d bugfix` | Delete local branch |
| `git push -d origin bugfix` | Delete remote branch |
| `git remote prune origin` | Delete stale remote-tracking branches |
| `git remote prune origin --dry-run` | Demo which branch would be pruned or removed |
| `git branch -r` | Show remote-tracking branches |
| `git fetch --prune` or `git fetch -p` | Shortcut to prune, then fetch |
| `git config --global fetch.prune true` | Always prune before fetch |
| `git prune` | Prune all unreachable objects |
| `git gc` | Part of garbage collection |
| `git prune --expire <time>` | Prune unreachable objects older than specified time |

## Taging

### Create Tags

Tags in Git are like bookmarks, marking important points in the history of a repository. They can be used to mark software versions or to highlight key features or changes. You can also use tags to mark points for discussion with collaborators, like bugs or issues. So, if you're working on a project in Git, don't forget to use tags to help keep track of important points in your repository's history!

| Command | Description |
| --- | --- |
| `git tag issue_136 655da716e7` | Add lightweight tag (using hash or branch name) |
| `git tag -am "Version 1.0" v1.0 dd5c49428a0` | Add annotated tag (most common) |
| `git tag -d v1.0` or `git tag --delete v1.0` | Delete a tag |

### List Tags

| Command | Description |
| --- | --- |
| git tag
git tag --list
git tag -l | List tags alphabetically |
| git tag -l "v2*" | List tags beginning with "v2" |
| git tag -n | List tags with first line of each annotation |
| git tag -n5 | List tags with five lines of each annotation |
| git show v1.1 | Show changes made in the commits tagged with v1.1 |
| git diff v1.0..v1.1 | Show all differences from v1.0 to v1.1 |
| git switch v1.0 | Switch to the commit or branch labeled as v1.0 |
| git switch -c branch_v1 v1.0 | Create a new branch from a tag |

### Push Tags to a Remote

Like branches, tags are local unless shared to a remote. Git push does not transfer tags, so they must be explicitly transferred. However, git fetch automatically retrieves shared tags. So, if you're collaborating with others on a project, make sure to share your tags to keep everyone in the loop!

| Command | Description |
| --- | --- |
| `git push origin v1.0` | Push a tag to a remote repository |
| `git push origin --tags` | Push all tags to a remote repository |
| `git fetch` | Fetch commits and tags |
| `git fetch --tags` | Fetch only tags (with necessary commits) (rarely used) |
| `git push -d origin v1.0` | Delete remote tags like remote branches |

## Interactive Staging

### Interactive Mode

Interactive staging is a cool feature in Git that lets you pick and choose which changes you want to stage. This means you can make smaller, focused commits and avoid committing changes you're not sure about. It's also a feature of many Git GUI tools. So, next time you're using Git, give interactive staging a try!

To enter into interactive mode, use: 

```bash
git add -i
git add --interactive
```

![Untitled](/assets/2024/September/interactive%20mode.png)

In interactive mode, you can stage changes, unstage changes, and add untracked files. You can choose options either by clicking on the corresponding number or the first letter of the option:

- s: Status of the repository
- u: Add files to the staging area
- r: Remove files from the staging area
- a: Add untracked files
- d: Differences in file
- q: Quit interactive mode
- h: Help

### Patch mode

In Git, you can pick and choose which changes you want to stage using interactive staging. This means you can make smaller, focused commits and avoid committing changes you're not sure about. You can stage each hunk (chunk of changes) separately. It's really useful!

To enter patch mode, go to interactive mode and enter "p", followed by the file number. 

![Untitled](/assets/2024/September/patch%20mode.png)

Other ways to use patch mode

| Command | Description |
| --- | --- |
| git add --patch or git add -p | Interactively choose which changes you want to add to the staging area. |
| git stash -p | Interactively choose which changes you want to stash.  |
| git reset -p | Interactively choose which changes you want to unstage. |
| git restore -p | Interactively choose which changes you want to discard from your working directory. |
| git commit -p | Interactively choose which changes you want to include in your commit. |

### Split a Hunk

When using Git's interactive staging feature, you can split a hunk further by using the "s" option in patch mode. This is useful when a hunk contains multiple changes and requires one or more unchanged lines between them.

### Edit a Hunk

When editing a hunk in Git, you can do it manually if needed. This is especially useful when a hunk cannot be split automatically. However, make sure to pay attention to the prefixes (+, -, space) while editing, or the hunk might not be staged correctly. So, take your time and give them the respect they deserve!

## Share Select Changes

### Cherry-Picking Commits

Cherry-picking commits is like copying and pasting code from one branch to another. Each commit becomes a new commit on the current branch, and they'll have different SHA codes. You can cherry-pick commits from any branch, but you can't do it with merge commits. You can use the --edit or -e flag to edit the commit message if you need to. However, conflicts can arise that you'll need to resolve. It's a useful feature to have in your Git toolkit!

![Untitled](/assets/2024/September/cherry%20pick.png)

```bash
git cherry-pick d4e8411d09
git cherry-pick d438411d09..57d290ec44
```

Resolve cherry-picking conflicts is similar to resolving merge conflicts. Just edit in editor and try again!

### Diff Patches

If you want to share changes with collaborators but the changes aren't ready for a public branch or your collaborators don't share a remote, you can use diff patches to share the changes via files. It's useful for discussing bugs or issues with collaborators or for sharing changes that need further testing before merging into the main branch.

```bash
git diff from-commit to-commit > output.diff
```

Use following common to apply changes in a diff patch file to the working directory. But remember, apply diff patches does not transfer commit history.

```bash
git apply output.diff 
```

### Formatted Patches

In Git, you can export each commit in Unix mailbox format using formatted patches. It's a great way to distribute changes via email and includes commit messages. You can apply formatted patches using the `git am` command. It's similar to cherry-picking, which copies and pastes code from one branch to another. However, formatted patches are better for sharing changes that aren't ready for a public branch or when collaborators don't share a remote. Keep in mind that applying formatted patches transfers the commit history.

| Command | Description |
| --- | --- |
| `git format-patch 2e33d..655da` | Creates patch files for all commits in the range |
| `git format-patch main` or `git format-patch main..HEAD` | Creates patch files for all commits on the current branch that are not in the `main` branch |
| `git format-path -1 655da` | Creates a patch file for a single commit with hash `655da` |
| `git format-patch 2e33d..655da -o ~/feature_patches` | Creates patch files for all commits in the range and puts them into a directory named `feature_patches` |
| `git format-patch 2e33d..655da --stdout > feature.patch` | Outputs patch files as a single file named `feature.patch` |
| `git am feature/0001-some-name.patch` | Applies a single patch |
| `git am feature/*.patch` | Applies all patches in a directory |

In the command `git am feature/*.patch`, `am` stands for "apply mailbox". This command applies a mailbox-style patch to the current branch.

## Rebasing

### Rebase Commits

Rebasing is a way to move commits from one branch to another. It's useful when you want to integrate recent commits without merging and to maintain a clearer, more linear project history. It also ensures that topic branch commits apply cleanly. So, if you're working on a project and want to keep your commits organized, give rebasing a try!

![Untitled](/assets/2024/September/rebase.svg)

*Fig: Rebasing Feature branch [Credit](www.atlassian.com)*

| Command | Description |
| --- | --- |
| git rebase main | Rebase current branch on tip of main (from feature branch) |
| git rebase main new_feature | Rebase new_feature to tip of main (from main) |
| git rebase --onto newbase upstream branch | Rebase branch onto newbase |
| git rebase --onto target main new_feature | Rebase new_feature commits on target branch that are not on main (merged) |

#### Handle Rebase Conflicts

When you rebase commits, it can cause conflicts with existing code. Git will pause the rebase before each conflicting commit, and you'll need to resolve the conflicts. This process is similar to resolving merge conflicts. It's important to be patient and take your time to ensure that the conflicts are resolved correctly.

| Command | Description |
| --- | --- |
| git rebase --continue | Continue the rebase after resolving conflicts |
| git rebase --skip | Skip the current commit during the rebase process |
| git log --graph --all --decorate --oneline | Visualize the branch history as a graph |
| git merge-base main new_feature | Return the commit SHA where the topic branch diverges from main |
| git rebase -i | Open an interactive rebase prompt to choose which commits to move |

![Fig: The branch test diverges from main](/assets/2024/September/diverges.png)

*Fig: The branch test diverges from main*

### Merging vs. Rebasing

- Two ways to incorporate changes from one branch into another branch
- Similar ends but the means are different
- Side effects are important to understand

![Edureak](https://www.edureka.co/blog/wp-content/uploads/2022/01/fig13.png)
*Fig: Git Merge vs Git Rebase [Credit: Edureka]()*

| Merging | Rebasing |
| --- | --- |
| Adds a merge commit | No additional merge commit |
| Nondestructive | Destructive: SHA changes, commits are rewritten |
| Complete record of what happened and when | No longer a complete record of what happened and when |
| Easy to undo | Tricky to undo |
| Logs can become cluttered and nonlinear | Logs are cleaner and more linear |

**The Golden Rule of Rebasing**

- Thou Shalt not rebase a public branch
- Rebase abandons existing, shared commits and creates new, similar commits instead
- Collaborators would see project history vanish
- Getting all collaborators back in sync can be hassle

**How to Choose**

- Merge to allow commits to stand out or to be clearly grouped
- Merge to bring large topic branches back into main
- Rebase to add minor commits in main to a topic branch
- Rebase to move commits from one branch to another
- Merge anytime the topic branch is already public and being used by others (The Golden Rule of Rebasing).

### Interactive Rebasing

Interactive rebasing is a feature in Git that allows you to modify commits as they're being replayed. When you run `git rebase -i`, Git will open up the git-rebase-todo file for editing. In this file, you can reorder, skip, or edit commits. The options available to you in interactive rebasing include:

- `pick`: include the commit
- `drop`: remove the commit
- `reword`: edit the commit message
- `edit`: pause the rebasing process to allow you to make changes to the commit
- `squash`: combine the commit with the one immediately before it
- `fixup`: combine the commit with the one immediately before it, but discard its commit message

Interactive rebasing is useful when you want to modify the history of a branch before sharing it with others. It can also be helpful for cleaning up your commit history by grouping related changes or removing unnecessary commits. However, be careful when using interactive rebasing, as it can be risky if not done properly. It's always a good idea to make a backup copy of your branch before rebasing it.

```bash
# Interactive rebase
git rebase -i main new_feature

# Rebase last three commits onto the same branch
# but with the opportunity to modify them
git rebase -i HEAD~3
```

![Untitled](/assets/2024/September/rebase.png)

### Squash Commits

Squash commits is a way to combine multiple commits into one. It's useful for when you have several small commits that are related to each other and you want to make them into a single, cohesive commit. This can help to keep your commits organized and make it easier to understand the history of your code.

When squashing commits, you'll take the changes from each commit and combine them into a single commit. The commit message for the new commit will be a combination of the commit messages from the original commits. You can use the `git rebase -i` command to interactively rebase your branch and squash commits.

To squash commits, follow these steps:

1. Use `git log` to find the SHA IDs of the commits you want to squash.
2. Use `git rebase -i HEAD~<number of commits>` to start an interactive rebase.
3. In the interactive rebase, change `pick` to `squash` for the commits you want to squash. You can also edit the commit messages if needed.
4. Save and close the file to complete the rebase.

After squashing the commits, you'll have a single commit that contains all the changes from the original commits. This can be helpful for keeping your commit history clean and organized, especially when collaborating with others on a project.

```bash
# Rebase last four commits onto the same branch
# but with the opportunity to modify them
git rebase -i HEAD~4

pick 81a73ff Redesign
squash b2baf90 Change image sizes
fixup c0261b3 Bug fix to the design
squash 0f7760e Adjust styles
```

### Pull Rebase

Pull rebase is a way to fetch changes from a remote repository and then rebase them onto the local branch instead of merging them. This helps to keep the commit history cleaner by reducing the number of merge commits. However, it should only be used for local commits that are not shared to a remote branch.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200415234509/Rebasing-in-git.png)

*Fig: Rebasing [Credit: GFG](geeksforgeeks.org)*

| Command | Description |
| --- | --- |
| `git pull -r` | Pull with rebase |
| `git pull --rebase` | Pull with rebase |
| `git pull --rebase=preserve` | Pull with rebase and preserve merge commits |
| `git pull --rebase=interactive` | Pull with interactive rebase |

## Track Down Problems

### Log Options

`git log` is a command in Git that displays the commit history for a repository. It shows the SHA-1 hash, author, date, and commit message for each commit in reverse chronological order. By default, it shows the entire commit history for the current branch. 

However, there are many options to customize the output, such as sorting, filtering, and formatting. Some common options are `--oneline` to show each commit on one line, `--graph` to display the commit history as a graph, and `--since` or `--until` to filter the commit history by date. You can also use `git log` to show the commit history for a specific file or directory, or to show the changes made by a specific commit.

| Command | Description |
| --- | --- |
| git log filename.txt | List commits that changed filename.txt |
| git log -p or git log --patch | List commits as patches (with diffs) |
| git log -L 100,150:filename.txt or git log -L 100,+50:filename.txt | List changes (as patches) to lines 100-150 in filename.txt |
| git log -S "MaxConnections" | List all commits that add or change the string |
| git log --pretty=format:"%h %cn %cd %s %an" | Show commit hash, committer name, commit date, commit message, author name |
| git reflog | Used to recover lost commits and branches |
| git log --since=yesterday | Show commits since yesterday (midnight) |
| git log --since="May 1, 2021" | Show commits since a specific date |
| git log --since="May 1, 2021 14:23:45" | Show commits since a specific date and time |
| git log --since="3 days ago" | Show commits since a certain number of days ago |
| git log --since="2 hours ago" | Show commits since a certain number of hours ago |

**Git diff**

Git uses standard UNIX less program do show the git diff.

![Untitled](/assets/2024/September/git%20diff.png)

**Git pager** (used with diff)

The `core.pager` setting determines the pager used when Git pages output. This setting can be configured globally or per-repository.

To set the pager globally, use the following command:

```
git config --global core.pager <pager>
```

Replace `<pager>` with the name of the pager you want to use, such as `less` or `more`.

To set the pager per-repository, use the same command without the `--global` option, inside the repository directory.

Git comes with a default pager, which is `less`. If you haven't set a pager explicitly, `less` will be used as the default.

### Blame

`git blame` is a command in Git that allows you to see who made changes to a file, which lines were changed, and when the changes were made. It can be helpful for understanding the history of a file, tracking down the source of a bug, or determining who to contact with questions about a particular piece of code. The output of `git blame` includes the commit SHA, author name, date, and the specific line of code that was changed. By default, `git blame` shows the annotations for the entire file, but you can also specify a specific range of lines or a specific revision to show annotations for.

| Command | Description |
| --- | --- |
| git blame filename.txt | Annotate file with commit details |
| git blame -w filename.txt | Annotate file with commit details, ignoring whitespace |
| git blame -L 100,150 filaname.txt | Annotate lines 100-150 |
| git blame -L 100,+50 filename.txt | Annotate lines 100-150 |
| git blame d9dba0 filename.txt | Annotate file at revision d9dba0 |
| git blame d9dba0 -- filename.txt | Same as previous command |
| git config --global alias.praise blame | Add a global alias for "praise" (if blame sounds negative) |
| git annotate filename.txt | Annotate file with commit details, different output format |

### Bisect

If you're trying to find a bug in your Git project, binary search is your friend! Here's how it works: first, find the commit that introduced the bug or regression. Then, mark the last good revision and the first bad revision. Reset your code to the midpoint between these two revisions and test it out. If the code is still broken, mark the midpoint as bad. If it's fixed, mark it as good. Keep repeating this process, dividing the revisions in half each time, until you find the exact commit that introduced the bug. It may take a few tries, but it's worth it to squash that bug!

![Untitled](/assets/2024/September/bisect.png)

```bash
git bisect start
git bisect good <treeish>
git bisect bad <treeish>
git bisect reset
```

That's all for today's learning. 🥰

## References

[LinkedIn Learning](
https://github.com/LinkedInLearning/git-intermediate-techniques-3082618
)