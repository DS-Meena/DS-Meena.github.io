---
layout: post
title:  "Improving command line skills"
date:   2023-06-25 10:00:10 +0530
categories: Linux
---

Welcome to our comprehensive guide on improving your Command Line skills, where we primarily focus on the Linux-based command line. This emphasises Linux because most commands are identical across Linux and Windows, with just a few exceptions. This blog will explore various commands related to the Linux file system, their applications, and how to use them effectively to enhance your navigation and manipulation of the Linux operating system.

## Linux File System

This section focuses on commands used in Linux for working with files and directories – we will discuss some less-discussed commands. I hope that you are already aware of the primary file system commands like ‘pwd’ for printing the current directory, ‘cd’ for changing directories, ‘ls’ for listing directory contents, and shortcuts for navigation ‘.’ (current folder), ‘..’ (parent folder), ‘~’ (current user’s home directory) and wildcard symbols for matching patterns (*, ?), creating new files and directories using ‘touch’ and ‘mkdir’.

The Linux File System follows a tree-like structure starting at a base (or root) directory, indicated by the slash (/). Locations on the file system are shown using file paths. Paths that start at the base directory (/) are known as absolute paths. Paths that start from the shell’s current working directory are known as relative paths.
Absolute: /home/sarah/Documents/file1.txt
Relative: Documents/file1.txt

### Editing Files📝

#### Nano

Nano is a simple, user-friendly text editor for the terminal. It is ideal for creating and editing text files and scripts. Here are some basic commands:

- To open or create a file, use `nano <filename>`. If the file already exists, it will open for editing. If not, a new file will be created.
- To save changes, press `Ctrl+O`. This will prompt you for a filename for the file. Press `Enter` to save.
- To exit Nano, press `Ctrl+X`. If there are unsaved changes, Nano will ask if you want to save changes before exiting.
- To cut and paste text, use `Ctrl+K` to cut a line and `Ctrl+U` to paste it back.

![Untitled](/assets/2024/September/nano.png)

It comes with a built-in toolbar of various commands and all one needs to know in order to use these options are the meaning of 2 special symbols.

| Character | Meaning |
| --- | --- |
| ^ | This is the CTRL key on your keyboard. For example, ^O is CTRL + O. |
| M- | This is the “meta” key on your keyboard. Depending on your keyboard layout this may be the ALT, ESC, CMD key. Try it out 😄 Assuming M- is the ALT key, then M-X is ALT + X |

If want to learn more about nano, then read this page [nano.](https://www.nano-editor.org/docs.php)

#### Vim

Vim is a highly configurable text editor for efficiently creating and changing any kind of text. It is especially useful for editing programs. Here are some basics:

- To start Vim and open a file, use `vim <filename>`. If the file does not exist, it will be created.
- Vim operates in different modes, primarily command mode and insert mode.
- To switch to insert mode, press `i`. You can now enter and edit text.
- To switch back to command mode, press `Esc`.
- In command mode:
    - `:w` will save the file without exiting.
    - `:q` will quit Vim. If there are unsaved changes, Vim will not exit and will warn about unsaved changes.
    - `:wq` will save changes and then exit.
    - `:q!` will exit without saving changes.
- To cut and paste text in Vim, use `dd` to delete a line in command mode and `p` to paste it back.

Here are some common commands used in Vim:

- `i` - enter insert mode
- `:q` - quit
- `esc + :wq` - save and quit
- `esc + :q!` - quit without saving
- `dd` - delete a line
- `#dd` - delete # lines
- `U` - undo
- `Ctrl+r` - redo
- `:/search` - search for a word
- `n,N` - move to the next/previous search result
- `:%s/word/to_word/gc` - replace all occurrences of "word" with "to_word" in the entire file, prompting for confirmation
    - Use g only if don't want to prompt for confirmation.
- `:set number` - display line numbers
- `:$` - go to last line
- `tail -f filename` - output the last 10 lines of a file and continue to output as new lines are added

### The Locate Command

The locate command searches a database on your file system for the files that match the text (or regular expression) that you provide it as a command line argument.

If results are found, the locate command will return the absolute path to all matching files.

For example:

```bash
locate *.txt
```

will find all files with filenames ending in .txt that are registered in the database.

The locate command is fast, but because it relies on a database it can be error prone if the database isn’t kept up to date.

Below are some commands to update the database and some reassuring procedures in case one cannot access administrator privileges.

| Command | Description |
| --- | --- |
| locate -S or —statistics | Print information about the database file. |
| sudo updatedb | Update the database. As the updatedb command is an administrator command, the sudo command is used to run updatedb as the root user (the administrator) |
| Locate —existing or -e | Check whether a result actually exists before returning it. (Inspite of being present in database, cross check) |
| locate —limit 5 or -l | Limit the output to only show 5 results |

Examples →

![locate command with limit and existing options and -S option](/assets/2024/September/locate.png)

locate command with limit and existing options and -S option

### The Find Command 🔎

The find command can be used for more sophisticated search tasks than the locate command.

This is made possible due to the many powerful options that the find command has.

The first thing to note is that the find command will list both files and directories, below the

point the file tree that it is told to start at.

For example: `find .` will list all files and directories below the current working directory (which is denoted by the .)

![Untitled](/assets/2024/September/find.png)

While `find /` will list all files and directories below the base directory (/); thereby listing everything on the entire file system!

By default, the find command will list everything on the file system below its starting point, to an infinite depth.

The search depth can however be limited using the –maxdepth option.

For example

```bash
find / -maxdepth 4
```

Will list everything on the file system below the base directory, provided that it is within 4 levels of the base directory.

There are many other options for the find command. Some of the most useful are tabulated below:

| Command | Description |
| --- | --- |
| -type | Only list items of a certain type. `–type f` restricts the search to file and `–type d` restricts the search to directories. |
| -name “*.txt” | Search for items matching a certain name. This name may contain a regular expression and should be enclosed in double quotes as shown. In this example, the find command will return all items with names ending in .txt. |
| -iname | Same as  `–name` but uppercase and lowercase do not matter. |
| -size | Find files based on their size. e.g `–size +100k` finds files over 100 KiB in size `–size -5M` finds files less than 5MiB in size. Other units include G for GiB and c for bytes**. |

**Note:** 1 Kibibyte (KiB) = 1024 bytes. 1 Mebibyte (MiB) = 1024 KiB. 1 Gibibyte = 1024 MiB.

#### Exec & ok

A supremely useful feature of the find command is the ability to execute another command on each of the results.

For example

```bash
find /etc –exec cp {} ~/Desktop \;
```

will copy every item below the /etc folder on the file system to the ~/Desktop directory. 

The argument to the `–exec` option is the command you want to execute on each item found by the find command.

Commands should be written as they would normally, with **{} used as a placeholder for the results of the find command.**

Be sure to terminate the `–exec` option using \; (a backslash then a semicolon).

The `–ok` option can also be used, to prompt the user for permission before each action.

This can be tedious for a large number of files, but provides an extra layer of security of a small number of files; especially when doing destructive processes such as deletion.

An example may be:

```bash
find /etc –ok cp {} ~/Desktop \;
```

![Find command within current directory with name option and remove operation for each result ](/assets/2024/September/find%20and%20ok.png)

Find command within current directory with name option and remove operation for each result 

exec → just execute the command 

ok → ask at each step 🌇

### Viewing File Content 🪟

There exist commands to open files and print their contents to standard output. One such example is the cat command. Let’s say we have a file called hello.txt on the Desktop.

By performing: `cat ~/Desktop/hello.txt`

This will print out the contents of hello.txt to standard output where it can be viewed or piped to other commands if required.

One such command to pipe to would be the less command. The less command is known as a “pager” program and excels at allowing a user to page through large amounts of output in a more user-friendly manner than just using the terminal.

An example may be:

```bash
cat ~/Desktop/hello.txt | less
```

Or more simply:

```bash
less ~/Desktop/hello.txt
```

By pressing the q key, the less command can be terminated and control regained over the shell.

Here are some other ways to view file contents:

|Command | Description |
| --- | --- |
| tac <path/to/file> | Print a file’s contents to standard output, reversed vertically. |
| rev <path/to/file> | Print a file’s content to standard output, reversed horizontally (along rows). |
| head -n 15 <path/to/file> | Read the first 15 lines from a file (10 by default if -n option not provided.) |
| tail -n 15 <path/to/file> | Read the bottom 15 lines from a file (10 by default if -n option not provided.) |

![Usage of cat, tac and rev command.](/assets/2024/September/cat,rev.png)

*Fig: Usage of cat, tac and rev command.*

### Searching File Contents

The ability to search for and filter out what you want from a file or standard output makes working with the command line a much more efficient process.

The command for this is called the grep command.

The grep command will return all lines that match the particular piece of text (or regular expression) provided as a search term.

For example: `grep hello myfile.txt` will return all lines containing the word “hello” in myfile.txt and `ls /etc | grep *.conf` will return all lines with anything ending in “.conf” in data piped from the ls command.

Some common options when working with the grep command include:

|Command|Description|
| --- | --- |
| grep -i | Search in a case insensitive manner. (upper case and lowercase don’t matter). |
| grep -v | Invert the search. i.e return all lines that DON’T contain a certain search term. |
| grep -c | Return the number of lines (count) that match a search term rather than the lines themselves. |
| grep -n  | return the line no’s also |

## File Archiving and Compression 🗃️

In this section, we will delve into how we can archive and compress files. This process is crucial to not only save space on your system but also make file transfer more efficient. Stay tuned as we explore various commands and techniques for this task.

### The Overall Process

Archiving and compressing files in Linux is a two-step process.

1) Create a Tarball

    First, you will create what is known as a tar file or “tarball”. A tarball is a way of bundling together the files that you want to archive.

2) Compress the tarball with a compression algorithm

    Secondly, you will then compress that tarball with one of a variety of compression algorithms; leaving you with a compressed archive.

### 1. Creating a Tarball

Tarballs are created using the tar command
`tar –cvf <name of tarball> <file>... `

`The –c option`: “create”. This allows us to create a tarball. [required]

`The –v option`: “verbose”. This makes tar give us feedback on its progress. [optional]

`The –f option`: Tells tar that the next argument is the name of the tarball. [required]

`<name of tarball>`: The absolute or relative file path to where you want the tarball to be placed;

e.g. ~/Desktop/myarchive.tar. It is recommended that you add .tar to your proposed filename for clarity.

`<file>`: The absolute or relative file paths to files that you want to insert into the tarball. You can have as many as you like and wildcards are accepted.

### 1.1 Checking a Tarball’s Contents
    
Once the tarball has been created, you can check what is inside it using the tar command.

`tar –tf <name of tarball>`

`The –t option`: “test-label”. This allows us to check the contents of a tarball. [required]

`The –f option`: Tells tar that the next argument is the name of the tarball. [required]

`<name of tarball>`: The absolute or relative file path to where you want the tarball to be placed;

e.g. ~/Desktop/myarchive.tar
    

### 1.2 Extracting From a Tar ball
    
Let’s say that you download a tar file from the internet and you want to extract its contents using the command line. How can you do that?

For this you would again use the tar command `tar –xvf <name of tarball>`

`The –x option`: “extract”. This allows us to extract a tarball’s contents. [required]

`The –v option`: “verbose”. This makes tar give us feedback on its progress. [optional]

`The –f option:` Tells tar that the next argument is the name of the tarball. [required]

`<name of tarball>`: The absolute or relative file path to where the tarball is located; 

e.g. ~/Desktop/myarchive.tar

cool 😎 → Extracting a tarball does not empty the tarball. You can extract from a tarball as many times as you want without affecting the tarball’s contents.

### 2. Compressing Tarballs

Tarballs are just containers for files. They don’t by themselves do any compression, but the can be compressed using a variety of compression algorithms.

The main types of compression algorithms are gzip and bzip2.

The gzip compression algorithm tends to be faster than bzip2 but, as a trade-off, gzip usually offers less compression.

You can find a comparison of various compression algorithms using this excellent blog post.

### 2.1 Compressing and Decompressing with gzip

|Comand | Description|
| --- | --- |
| Compressing with gzip | gzip <name of tarball> |
| Decompressing with gzip | gunzip <name of tarball> |

When compressing with gzip, the file extension .gz is automatically added to the .tar archive.

Therefore, the gzip compressed tar archive would, by convention, have the file extension .tar.gz

    
### 2.2 Compressing and Decompressing with bzip2
    
|Command|Description|
| --- | --- |    
| Compressing with bzip2 | bzip2 <name of tarball> |
| Decompressing with bzip2 | bunzip2 <name of tarball> |

When compressing with bzip2, the file extension .bz2 is automatically added to the .tar archive.

Therefore, the bzip2 compressed tar archive would, by convention, have the file extension .tar.bz2

### 3. Doing it all in one step

Because compressing tar archives is such a common function, it is possible to create a tar archive and compress it all in one step using the tar command. It is also possible to decompress and extract a compressed archive in one step using the tar command too.

To perform compression/decompression using gzip compression algorithm in the tar command, you provide the z option in addition to the other options required. [**Just add z to the basic commands**]

|Command|Description|
| --- | --- |
| Creating a tarball and compressing via gzip | tar –cvzf <name of tarball> <file>... |
| Decompressing a tarball and extracting via xzip  | tar –xvzf <name of tarball> |

To perform compression/decompression using bzip2 compression algorithm in the tar command, you provide the j option to the other options required. **[Just add j to the basic commands]**

|Command|Description|
| --- | --- |
| Creating a tarball and compressing via bzip2 | tar –cvjf <name of tarball> <file>… |
| Decompressing a tarball and extracting via bzip2 | tar –xvjf <name of tarball> |

To perform compression/decompression using the xzip compression algorithm in the tar command, you provide the J option to the other options required.

|Command|Description|
| --- | --- |
| Creating a tarball and compressing via xzip | tar –cvJf <name of tarball> <file>... |
| Decompressing a tarball and extracting via xzip |  tar –xvJf <name of tarball>  |

![Creating tarballs using different compression algorithms](/assets/2024/September/zip-unzip.png)

*Fig: Creating tarballs using different compression algorithms*

### 4. Creating .zip files

Although .tar.gz and .tar.bz2 archives are the archives of choice on Linux, .zip archives are common on other operating systems such as Windows and Mac OSX.

In order to create such archives, you can use the following commands.

|Command|Description|
| --- | --- |
| Creating a .zip archive  | zip <name of zipfile> <file>... |
| Extracting a .zip archive |  unzip <name of zipfile> |

`<name of zipfile>`: The absolute or relative file path to the .zip file e.g. ~/ myarchive.zip

`<file>`: The absolute or relative file paths to files that you want to insert into the .zip file. You can have as many as you like and wildcards are accep

![Zipping and unzipping](/assets/2024/September/unzip.png)

*Fig: Zipping and unzipping*

That’s all for this blog, I hope you found this helpful. ❤️❤️

## Coming soon

Systemctl commands - services
