---
layout: post
title:  "Understanding git"
date:   2023-04-09 15:08:10 +0530
categories: Web
---

## Introduction

Git is a distributed version control system that enables developers to track changes in their code and collaborate on projects with others. Created by Linus Torvalds in 2005, Git has since become one of the most popular version control systems in use today.

GitHub is a web-based platform that uses Git for version control and collaboration. It provides hosting for software development and a range of features for managing projects, such as bug tracking, task management, and wikis.

GitHub was created by Tom Preston-Werner, Chris Wanstrath, and PJ Hyett in 2008. You can remember the basic difference by Dates.

## Features of Git

Git has several features, including:

- **Version control:** Git allows developers to track changes in their code and easily revert to previous versions if necessary.
- **Collaboration:** Git enables developers to work on projects with others by sharing their code and collaborating on changes.
- **Branching:** Git allows developers to create separate lines of development, called branches, which can be used to work on new features or bug fixes without affecting the main codebase.
- **Command line interface:** Git can be used through a series of commands, which can be accessed through the command line interface.

## How does Git Works?

Git works by creating a repository, which is a directory where all of the project's files and history are stored. Whenever a change is made to a file in the repository, Git records that change and stores it as a new version of the file. Developers can then easily view and revert to previous versions of the code if necessary.

Git uses a series of commands that developers can use to interact with the repository.

Now, let’s read about different commands used in different tasks: -

### Starting new Project in Git

To start a new project in Git, you need to create a new repository. This can be done using the `git init` command. First, navigate to the directory where you want to create the repository using the `cd` command. Then, use the `git init` command to create a new repository in that directory.

```
cd /path/to/your/directory
git init
```

Once the repository has been created, you can start adding files to it using the `git add` command, and then commit those changes using the `git commit` command.

You can also clone an existing project into your directory, using HTTPS, SSH or GitHub CLI.

![Fig: Ways to Clone an existing Project](../_site/web/2023/04/09/Untitled-500x533.png)

Fig: Ways to Clone an existing Project

- HTTPS → `git clone [HTTPS_link]`
- How to generate SSH (Secure Shell Protocol) Key
    
    To access a repository using SSH, you need to generate a public SSH key and add it to your GitHub account.
    
    1. Generate a new SSH key by running the following command in your terminal: `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`
    2. Follow the prompts to create a passphrase and save the key. 
        
        ![Fig: Generating SSH Key](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12178030-b1a2-4779-8c40-5f24f44ed687/Untitled.png)
        
        Fig: Generating SSH Key
        
        Note → [Do not enter file name, we need files inside .ssh folder of User.]
        
    3. Add the SSH key to your GitHub account by going to your account settings and selecting "SSH and GPG keys". Click "New SSH key" and paste the contents of your public key into the text box.
    4. Test your SSH connection by running the command `ssh -T git@github.com`. You should receive a message indicating that you've successfully authenticated.
        
        ![Fig: Authenticating SSH Key](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/32f74387-253b-46ac-b4a6-91b0c038f5a0/Untitled.png)
        
        Fig: Authenticating SSH Key
        
        Note → [It shows shell access not provided because it is not necessary for most users. You can still use CMD.]
        
    
    You can learn more about setting up SSH keys in GitHub by reading the [official documentation](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) and [Youtube](https://www.youtube.com/watch?v=k805rTsX_iI).
    
- SSH → `git clone ssh:[SSH_link]`
- GitHub CLI → `gh repo clone [user_name]/[repo_name]`

### Taking Snapshots in Git

A **snapshot** in Git refers to a saved version of the code at a specific point in time. Every time a change is made to a file in the repository, Git records a new snapshot of that file. Developers can then use Git to view and revert to previous snapshots if necessary. Snapshots are stored in the repository as commits, which are identified by a unique SHA-1 hash.

Snapshotting is a two-step process. The two-step process in Git consists of staging and committing changes. 

1. **Staging changes**

The staging area, also known as the index, is a feature in Git that allows developers to prepare changes for committing. Before changes can be committed to the repository, they must first be added to the staging area using the `git add` command. 

| Command | Description |
| --- | --- |
| git add [file-name] | Add a file to the staging area |
| git add -A | Add all changed files to the staging area |
| git rm -r [file_name] | Remove a file from staging area |
| git add -p | Opens the patch mode |

1. **Committing changes**

Once changes have been added to the staging area, they can be committed to the repository using the `git commit` command. This allows developers to selectively choose which changes to commit, rather than committing all changes at once.

| Command | Description |
| --- | --- |
| git status | Shows the status of changes as untracked, modified, or staged. |
| git commit m “[commit message]” | Saves the snapshot to the project-history and completes change-tracking process. |

### Branching and Merging

Branching in Git allows developers to create separate lines of development, called branches, which can be used to work on new features or bug fixes without affecting the main codebase. This allows developers to experiment with new ideas and features without impacting the stability of the main codebase.

Here, you need to understand the difference between local branch and remote branch. A **local branch** is a branch that exists only on your local machine, while a **remote branch** is a branch that exists on a remote repository, such as on GitHub. Local branches can be used to work on new features or bug fixes without affecting the main codebase, and can later be merged into the main branch using Git. Remote branches can be used to collaborate with others on a project, and changes made to the remote branch can be pulled into your local branch using Git.

You can use following commands to deal with branches:

| Command | Description |
| --- | --- |
| git branch | Lists all the branches for the current repository |
| git branch -a | Lists all branches, including local and remote |
| git branch [branch_name] | Creates a new branch |
| git branch -d [branch_name] | Deletes a branch |
| git branch -D  | Deletes local branch regardless of push and merge status |
| git push origin --delete [branch_name] | Deletes a remote branch |
| git checkout -b [branch_name] | Creates a new branch and switches to it |
| git checkout -b [branch_name] origin/[branch_name] | Clones a remote branch and switches to it |
| git branch -m [old_branch_name] [new_branch_name] | Renames a local branch |
| git checkout [branch_name] | Switches to an existing branch |
| git checkout - | Switches to the last checked out branch |
| git checkout . | discards all changes made to the current directory and returns it to the last committed state. |
| git branch --merged  | list the branches that have been merged into the currently checked-out branch |

Once you've made changes to the new branch, you can merge those changes back into the main branch using the `git merge [branch-name]` command. This combines the changes from the specified branch into the current branch.

When merging branches, Git will attempt to automatically merge the changes. However, if there are conflicts between the two branches, you may need to resolve those conflicts manually.

You can use following commands to do merge:

```bash
git merge [branch-name]  # merge given branch into current branch
git merge [source-branch] [target-branch]
git checkout --[file-name]  # discard the changes done in this file

# when merge conflict arise - stop merging and return to pre-merge state
git merge --abort  
```

In Git, you can use the `git stash` command to temporarily save changes that are not ready to be committed. This can be useful if you need to switch to a different branch or work on a different part of the project, but don't want to commit your changes yet.

To stash changes, use the `git stash` command. To apply the stashed changes later, use the `git stash apply` command.

| Command | Description |
| --- | --- |
| git stash | Stashes changes |
| git stash apply | Applies top most stashed changes but leaves in stash |
| git stash pop | Applies top most stash changes and pop them |

You can also use the `git stash list` command to view a list of all stashed changes.

| Command | Description |
| --- | --- |
| git stash list | Lists all stashed changes |
| git stash clear | Clear all stashed changes |
| git stash branch <branch_name> | Restore previously stashed work to a new branch |
| git stash show -p | show the changes done |

Hey, remember branch and fork are not the same thing. A **branch** is a separate line of development with the same repository while A **fork** is a copy of the whole repository that allows you to make changes to the codebase without affecting the original repository. Forking is often used in open-source development, where developers can fork a repository, make their changes, and then submit a pull request to the original repository owner to incorporate their changes.

### Updating Projects

To share your Git project with others, you can use the `git push` command to push your changes to a remote repository, such as on GitHub. This will allow others to access the latest version of your code and collaborate with you on the project.

To update your Git project with changes from a remote repository, you can use the `git pull` command. This will pull the latest changes from the remote repository and merge them with your local repository.

You can also use the `git fetch` command to fetch changes from a remote repository without merging them into your local repository. This can be useful if you want to review the changes before merging them.

**To push changes to a remote repository:**

1. First, add and commit your changes using the `git add` and `git commit` commands.
2. Then, use the `git push` command to push your changes to the remote repository.

| Command  | Description |
| --- | --- |
| git push origin [branch_name] | Push the changes to branch [branch_name] of remote repository [origin]. |
| git push [-u | —set-upstream] origin [branch_name] | Push changes to remote repository (and remember the branch) |
| git push | Push changes to the upstream branch [If set] |
| git push origin [-d | —delete] [branch_name] | Delete a remote branch |

Remember in Git, the **upstream branch** refers to the branch on a remote repository that your local branch is tracking. Upstream branch works with both push and pull. 

So when pushing changes to a remote repository, the `-u` flag or the `--set-upstream` flag sets the upstream branch for the current branch. This means that in the future, you can simply use the `git push` command without specifying the remote branch name, since Git will assume that you want to push to the upstream branch.

In summary, using the `-u` flag simplifies the process of pushing changes to the remote repository in the future, but is not necessary for pushing changes initially.

After pushing changes into remote repository, we can **create a Pull request**. A pull request in GitHub is a way for developers to propose changes to a repository hosted on GitHub. When a developer creates a pull request, they are essentially asking that their changes be reviewed and potentially merged into the main codebase. Other developers can review the changes and provide feedback before the pull request is merged. Pull requests are a key feature of the collaborative nature of GitHub, allowing developers to work together on open source projects and contribute to each other's code. You can learn more about pull requests from the [official GitHub documentation](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).

!https://docs.github.com/assets/cb-23923/mw-1000/images/help/repository/branching.webp

**To pull changes from a remote repository:**

1. Use the `git pull` command to pull the latest changes from the remote repository and merge them with your local repository.

Similar to `git push`, when you use the `git pull` command, Git automatically pulls changes from the upstream branch if you have set upstream branch.

| Command | Description |
| --- | --- |
| git pull | Pull latest changes |
| git pull origin [branch_name] | Pull changes from given remote branch  |

**To fetch changes from a remote repository:**

1. Use the `git fetch` command to fetch the latest changes from the remote repository.

The `git fetch` command allows you to fetch the latest changes from the remote repository without merging them into your local repository. This can be useful if you want to review the changes before merging them. 

In summary, the `pull` command is a combination of the `fetch` and `merge` commands, while the `fetch` command only downloads changes from the remote repository without merging them.

| Command | Description |
| --- | --- |
| git fetch | Download objects and refs from remote repository. |

### Remote & Remote Repository

In Git, a remote is a version of your repository that is hosted on a remote server, such as on GitHub. A remote repository is the repository that is hosted on the remote server. Remote repositories can be used to collaborate with others on a project, and changes made to the remote repository can be pulled into your local repository using Git. 

The `git remote` command is used to manage remote repositories. It allows you to view the remote repositories that are currently associated with your local repository, as well as add or remove remote repositories.

Here are some commonly used `git remote` commands:

- `git remote`: Lists all the remote repositories that are currently associated with your local repository.
- `git remote -v`: Lists all the remote repositories along with their corresponding URLs.
- `git remote add <name> <url>`: Adds a new remote repository with the given name and URL to your local repository. By default, origin is the name of your remote repository [Main one].
Example →
    - git remote add origin <https://github.com/user/repo.git>
    
    In this example, `origin` is the name of the remote repository, and `https://github.com/user/repo.git` is the URL of the remote repository. Once you've added the remote repository, you can push your changes to it using the `git push` command.
        
        
- `git remote remove <name>`: Removes the remote repository with the given name from your local repository.

Overall, this allows you to work with more than one remote repository from single local repository. You can learn more about `git remote` command and its options from the [official Git documentation](https://git-scm.com/docs/git-remote).

### Inspection and Comparison

Git provides several commands that allow developers to inspect and compare different versions of their code. These commands can be used to view the history of changes, identify differences between versions, and troubleshoot issues that may arise during development.

Here are some commonly used inspection and comparison commands:

| Commands  | Description |
| --- | --- |
| git log | Displays the commit history of the repository. |
| git log —summary | Displays detailed history |
| git log —oneline | Displays brief history |
| git diff | Shows the differences between the working directory and the most recent commit. |
| git diff [source_branch] [target_branch] | Shows difference between source and target branch. |
| git blame | Shows which commit and author last modified each line of a file. |
| git blame -L 10, 20 [filename.txt] | Will show information about line 10 to 20 only. |
| git show | Displays the details of a particular commit, including the changes that were made and the commit message. |
| git show Head | Displays details of latest commit |
| git show [commit hash] | Displays details about given commit. |
| git show [commit hash]:[file_path] | Displays details about the given file for this commit. |
| git show <commit> --stat | Displays histogram |
| git show-ref --head | Find head of current branch |

These commands can be used in combination with one another to gain a better understanding of the history of the code and identify any issues that may need to be addressed.

You can learn more about Git commands from [GitHub Docs](https://git-scm.com/docs).

## Conclusion

In conclusion, Git is an essential tool for developers who want to keep track of changes in their code and collaborate with others. It allows developers to easily view the history of their code, revert to previous versions, and work on separate branches. By mastering the basics of Git, developers can become more efficient and effective in their work.

That’s all for this Blog. Hope this was worth your time. ❤️❤️