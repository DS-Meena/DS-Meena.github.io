# How to Add large Files On Github

## Download Git LFS

You can download the git extension from this [link](https://git-lfs.github.com/).

After downloading install it.   

## Activate git LFS
    
From anylocation in command prompt run this command to activate and initialize the git LFS.

```git lfs install```

On successful exeuction it will show LFS initialized.

## Track required file

Open your reposity in your local machine. If not already initialized as git repo then use

```git init```
    
Now, add files to git LFS using the following command.
    
 ```git lfs track "*.file_type"```

File type is the format of your file it can .pkl , .wav or .mp4. This will tell the lfs that which required to be track.

## Add .gitattributes

Need to store .gitattributes file in your repository this contains information about which files will be handled by git lfs.

Inside your local repositroy run this command.

 ```git add .gitattributes```

this will create a .gitattributes file inside your repository.    

## Push files 
    
After this commands when you will push your files it will be handled by git lfs.

next steps would be like this:-

```git add filename.type```

commit your changes

```git commit -m "your commit message"```

create a brach [branch]
    
```git branch -M [branch name]```

add remote link,
here branch name is the branch you have created above or otherwise it will we "main" (by default)(in newer versions).
You can give any name to your remote version. Usually it is "origin"

```git remote add [remote name] [remote link.git]```

push the changes

```git push -u [remote name] [branch name]```

Now it is done, you can see the changes on your github account.

Hope, this will be helpfull.