## basic 
To initialize a Git repository, use the **git init** command.

Add files to the Git repository in two steps:

Use the command **git add file**, note that it can be used repeatedly to add multiple files;
Use the command **git commit -m message** to complete.
To keep track of the status of the workspace, use the **git status** command.

If git status tells you that some files have been modified, use **git diff** to view the modified content.

The version pointed to by HEAD is the current version. Therefore, Git allows us to shuttle between version history, using the command **git reset --hard commit_id**.

Before shuttle, use **git log** to view the commit history to determine which version to roll back to.

To return to the future, use **git reflog** to view the command history to determine which version to return to in the future.

When you mess up the contents of a file in the workspace and want to discard the changes in the workspace directly, use the command **git checkout - file**

When you not only change the content of a file in the workspace, but also add it to the temporary storage area, you want to discard the changes. There are two steps. The first step is to use the command **git reset HEAD <file>**, and you return to scenario 1, and second Step by step operation according to scenario 1

## branch
View branch: **git branch**
Create a branch: **git branch < name >**
Switch branch: **git checkout < name >** or **git switch < name >**
Create + switch branches: **git checkout -b < name >** or **git switch -c < name >**
Merge a branch to the current branch: **git merge < name >**
Delete branch: **git branch -d < name >**

When Git cannot automatically merge branches, the conflict must be resolved first. After resolving the conflict, submit it again and the merge is complete.

To resolve the conflict is to manually edit the file that failed to merge in Git to the content we want, and then submit it.

Use the **git log --graph** command to see the branch merge graph.
## Bug branch
When fixing a bug, we will fix it by creating a new bug branch, then merge, and finally delete;

When the work at hand is not completed, first git stash the work site, then fix the bug, after the repair, then git stash pop and return to the work site;

If you want to merge the bugs fixed on the master branch to the current dev branch, you can use the **git cherry-pick < commit > **command to "copy" the changes submitted by the bug to the current branch to avoid duplication of work.
## Feature branch
To develop a new feature, it is best to create a new branch;

If you want to discard a branch that has not been merged, you can use **git branch -D < name >** to delete it forcefully.
## Multi-person collaboration
The working mode of multi-person collaboration is usually like this:

First, you can try to push your own changes with git push origin <branch-name>;

If the push fails, because the remote branch is newer than your local, you need to use git pull to try to merge;

If there is a conflict in the merge, resolve the conflict and submit it locally;

After there is no conflict or the conflict is resolved, use git push origin <branch-name> to push successfully