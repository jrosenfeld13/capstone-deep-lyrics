# Git survival guide

Here is a practical 101 on how to use git for sunny day scenarios. We can add more advanced scenarios as we need them.

## Concepts

Git can be simplified as having a server (or group or servers) where the main repo is and group of clients, which are our computers with our clones. This means there are server branches (master for example, in git jargon called ```origin master``` or ```remote master```) and client branches, which are local branches we create. We can push local branches to the server to create pull requests, to merge from our remote branches to master for example. 

Below is the usual process for a work item.

## Update master

First, you want to get your local copy of the remote master branch updated. This basically downloads the server master branch to your local.

```
git checkout master
git pull
```

Note that generally you should not have any changes in master. If you do have changes, you cannot pull. 

## Create local branch for your feature

Next, create a local branch off master where you'll work on a specific feature:

```
git checkout -b <your_name>/<your feature>
```

For example:

```
git checkout ccastro/fix-api-serialization
```

## Make changes, commit and push

Make your changes regularly, you can stage changes to be commited by running ```git add <file|folder>```. Then commit your changes using ```git commit -m "message"```. When we commit, we are only saving the commit in our local branch, if our hard drive is destroyed, the work will be lost. To push our changes to the server, we need to create a remote branch and push to it. This is achieved with a single command:

```
git push --set-upstream origin <your_name>/<your feature>
```

After this command, you can see your server branch in github. Then you can send a pull request any time.

## Multiple branches in parallel

Note that you can work on multiple features in parallel, as long as you commit the changes on a branch before moving to the other branch. If your changes are not ready to commit, you can store a stash on the current branch by running ```git stash``` and then work in your other branch. When you go back to the original branch, run ```git stash list``` and you'll see a list of your stashes. Apply a specific stash by running ```git stash apply -l <stash>```.