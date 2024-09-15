# The Object

This is my personal blog. You can see [here](https://ds-meena.github.io/)

Here I post some one my experiences and what i learnt in my journey.


# Learnings

## How to install Jekyll 🎃

To install Jekyll on Ubuntu, you'll need to follow these steps:

1. First, ensure you have Ruby  installed. If not, you can install it using:

    ```bash
    sudo apt-get install ruby-full build-essential zlib1g-dev
    ```

    Run following commands to avoid installing gem as root user:
    ```
    echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
    echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
    echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```

2. Then, install Jekyll and Bundler:

    ```bash
    gem install bundler jekyll
    ```

## Starting a website 🌐

1. Create a Jekyll site.
    ```bash
    jekyll new my-blog
    cd my-blog
    ```

    Or if you already have folder, then try:
    ```
    jekyll new . --force

    ```
    The above command will create a new Jekyll site in the current directory.

2. Install dependencies:
    ```
    bundle install
    ```

3. Start the server.
    ```
    bundle exec jekyll serve
    ```

Your site should now be accessible at http://localhost:4000. Any changes you make to your files will be automatically reflected in the local version of your site, allowing you to preview your blog before pushing changes to GitHub.

