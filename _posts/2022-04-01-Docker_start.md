---
layout: post
title:  "Getting started with Docker"
date:   2022-04-01 10:00:10 +0530
categories: Cloud
---

Docker is a tool that lets developers easily package up applications and all their dependencies into a single container. These containers can then be shared and run on any machine with Docker installed. In this blog, we'll cover some basic commands for working with Docker, as well as show you how to create your own custom Docker images and work with Docker networking and storage. So let's dive in and get started!

## Terminology

#### Container

A container is like a portable package that contains everything needed to run an application. It includes the code, libraries, tools, and settings required for the application to work. Containers are created using Docker, which is a technology that allows developers to pack all the necessary components of an application into a single container.

#### Image

An image is like a ready-to-use package that contains all the necessary files and settings to run a specific application within a container. You can think of it as a blueprint for creating containers. Images can be obtained from a registry, such as Docker Hub, or built by following instructions in a Dockerfile.

A Docker image is made up of multiple layers stacked on top of each other. Each layer represents a change to the filesystem and is stored separately. When a container is created from an image, a new layer is added on top of the image's filesystem, allowing the container to make changes without affecting the original image.

![Fig: Creating container using Image](/assets/2024/September/Untitled.png)

*Fig: Creating container using Image*

## Docker Basic Commands

First we will see use of some basic docker commands including creation and removal of images and containers.

Here are some basic Docker commands:

| Command | Description |
| --- | --- |
| `docker images` | Lists all available images |
| `docker ps` | Lists all running containers |
| `docker ps -a` | List all containers |
| `docker stop [container id\| container name]` | Stops a running container |
| `docker rm [container id \| container name]` | Removes a container [After stopping it] |
| `docker rmi [image name]` | Removes an image [After stopping all dependent containers] |
| `docker pull [image name]` | Downloads an image without creating an instance |
| `docker -v` or `docker --version` | Checks the version of Docker |
| `docker inspect [container name\| container id]` | Inspects a container and gets information associated with it |
| `docker logs [container name\| container id]` | Checks the logs associated with a container |
| `docker run [image name]` | Runs an image instance‍ |
| `Docker build -t [Name of Image]:[tag] [Path to Source Code]` | Build an Image from source code  |

## Docker Run Commands

Know this: a Docker image instance refers to a Docker container. When you run an image using the `docker run` command, it creates a container instance of that image.

Example:

![Fig: Creating an Image Instance](/assets/2024/September/6203cf53a8d82b4c2b84a851_mysql.png)

*Fig: Creating an Image Instance*

Now, let’s read about different customizations of run command.

| Command | Description |
| --- | --- |
| `docker run [Image name]` | Creates a new Image instance or container with a random name |
| `docker run [Image name]:tag` | Define the version of the image |
| `docker run [-i | -it] [Container name]` | Run an existing container |
| `docker run -p [host port number]:[container port number] [image name]` | Create a new Image instance with port connectivity to host |
| `docker run -v /path/on/host:/path/in/container [image name]` | Create a new Image instance with volume mapping to host machine |
| `docker run -e [variable name]=[value] [image name]` | Create a new Image instance while setting environment variables |
| `docker run --name [Name of Instance] [Image name]` | Creates a new container instance with a specified name |
| `docker run --name [Name of Instance] --link [Name of Container]:[Alias] [Image Name]` | Create a new container and connect it with another container |
| `docker run -d [Image name]` | Runs a docker container in the background as a daemon. |

Now, lets read and understand few topics:

### Port Mapping -p

In order to map a port in a container to a port on the host machine, use the `-p` flag when running a container. 

For example, the command `docker run -p 80:8080 mywebapp:latest` would create a new instance of the `mywebapp` image with port connectivity. 

The left side port number shows which port the host is using and right side number shows through which port the container is connected to the host port. There can be multiple ports available simultaneously on the host but a single host port can be accessed by only a single container at a time.

In below Image you can see a container with port connectivity.

![Fig: Example of container with port connectivity to host](/assets/2024/September/6203b51bb2599c944c9ad3f4_port%20desc.png)

*Fig: Example of container with port connectivity to host*

![Fig: Creating instance of Image with port connectivity](/assets/2024/September/6203b701e7b6e474ffbc0a01_port%20p.png)

*Fig: Creating instance of Image with port connectivity*

### Volume Mapping -v

In order to map a volume in a container to a directory on the host machine, use the -v flag when running a container.

For example, the command `docker run -v /path/on/host:/path/in/container myimage` would create a new instance of the myimage image with a volume mapped to the /path/on/host directory on the host machine. Any changes made to the volume within the container will be reflected in the corresponding directory on the host machine, and vice versa.

Note that the first path specified after the -v flag corresponds to the directory on the host machine, while the second path corresponds to the directory in the container.

You can also create a named volume using the docker volume create command, and then use the --mount flag to specify the volume when running a container.

![Fig: Create an Image instance with Volume mapping to host](/assets/2024/September/6203cd315f8a38167a404a5c_postgres.png)

*Fig: Create an Image instance with Volume mapping to host*

### Environment Variables -e

To set environment variables for a Docker container when creating an instance of an image, you can use the `-e` flag followed by the variable name and value. The syntax is as follows:

`docker run -e [variable name]=[value] [image name]`

For example, to set a password variable to "secret" for a Redis container, you can use:

```
docker run -e REDIS_PASSWORD=secret redis
```

![Fig: Create an Image instance with given variables](/assets/2024/September/6203d4ee16ae6432ab358ba2_user%20redis.png)

*Fig: Create an Image instance with given variables*

### Link the Instance –link

The `--link` flag in the `docker run` command is used to connect containers together over a network. It allows one container to access the other container's environment variables and network ports without exposing them to the host machine. The syntax for the command is as follows:

`docker run --name [Name of Instance] --link [Name of Container]:[Alias] [Image Name]`

Here, `[Name of Container]` refers to the name of the container that you want to link to, and `[Alias]` refers to the name that you want to use to access that container from within the linked container.

For example, the following command links a Redis container to a Python container, and sets the alias "redis" for the Redis container:

`docker run --name mypythonapp --link myredis:redis python`

## Docker Custom Images

We can create custom images to containerize our application.

It includes the following steps:

1. Creating a container with a defined OS.
2. Updating the OS.
3. Installing OS dependencies associated.
4. Installing Python dependencies.
5. Declaring the path to copy source code.
6. Running the web server.

We define all these steps in a Dockerfile. First, we have to create a Dockerfile associated with the application.

### Dockerfile:

```Dockerfile
FROM python:3.6
RUN apt-get update && apt-get install -y \\
    libffi-dev \\
    libssl-dev \\
    libxml2-dev \\
    libxslt-dev \\
    libjpeg-dev \\
    libopenjpeg-dev \\
    libfreetype6-dev \\
    zlib1g-dev \\
    libmysqlclient-dev \\
    libpq-dev \\
    postgresql-client \\
    git
COPY . /opt/myapp
WORKDIR /opt/myapp
RUN pip install -r requirements.txt
EXPOSE 8000
ENTRYPOINT ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

- **From ->** It defines the working environment. It can be a base operating system or another image, such as Ubuntu, Redis, Python, etc.
- **Run ->** The command written after run is used to install the dependencies of the environment. It can contain pip install or apt-get update commands depending on the environment.‍
- **Copy ->** This shows where the data is written while creating its image. For examply → My source code from the current folder will be copied to /opt/ folder in the Image on docker.‍
- **Workdir ->** shows where the source code is present.‍
- **Entrypoint ->** This command is used to run the application.‍
- **Expose ->** It defines which port of the container is used to connect with the host port.

Docker uses a layered architecture for the images. This means that with each command in the Dockerfile, a new layer is added to the image.

For example, the base layer is of python:3.6 image and with each command including run, copy, expose, workdir, entrypoint the new layers are added to the below layer. The advantage of this is that when you change the source code of the image and try to create a new image, it will use the old layers.

**Layers can be as follows:**

1. Base Python:3.6 Image Layer
2. Changes in Pip Packages
3. Source Code
4. Exposed port
5. Working directory change
6. Update Entry point

### Build Image

After creating the Dockerfile, we build the image with the following command: `Docker build -t [Name of Image]:[tag] [Path to Source Code]`

For example, to create an image called "mywebapp" from the current directory, the command would be: `docker build -t mywebapp:latest .`

![Fig: Building a custom Image](/assets/2024/September/6204cf48fef7f1b285f489f5_my%20image.png)

*Fig: Building a custom Image*

### Run Image

After creating the image successfully, we can create its instances in the same way as we do for the official images.

For example, to create a container instance from the "mywebapp" image and map port 8000 on the host to port 8000 in the container, the command would be:

`docker run -p 8000:8000 mywebapp:latest`

This will start a new container instance from the "mywebapp" image and map port 8000 on the host to port 8000 in the container.

### CMD and ENTRYPOINT

#### CMD

The `CMD` instruction in a Dockerfile specifies what command to run by default when a container is started from the image. If a command is specified when the container is started, it overrides the `CMD` instruction.

#### Entrypoint

The `ENTRYPOINT` instruction in a Dockerfile specifies the command to run when a container is started from the image. Any arguments passed to `docker run` will be appended to the `ENTRYPOINT` command.

For example, if the `ENTRYPOINT` is set to `["python", "app.py"]` and the command `docker run myimage arg1 arg2` is used, the container will run the command `python app.py arg1 arg2`.

Both `CMD` and `ENTRYPOINT` are optional instructions in a Dockerfile, but at least one of them should be present.

## Docker Networking

Docker provides three default network types:

- **Bridge**: This is the default network type and allows containers to communicate with each other using internal IP addresses.
- **Host**: With this network type, containers share the network namespace with the host, meaning they can directly access the host's network interfaces.
- **None**: Containers with this network type have no network access.

These network types help manage how containers are connected and communicate with each other and the host machine.

We use following commands while dealing with networks in docker:

| Command | Description |
| --- | --- |
| `docker network list` | shows the list of available networks [We have created] |
| `docker run --name [container name] --network=[name of network] [image name]` | Create a container with given network |
| `docker network create --driver [Network driver name] --subnet [subnet id] --gateway [Gateway id] [Network name]` | Create a custom network  |
| `docker network inspect [Network name]` | Inspect any network’s configuration |

Example of custom network:

![Fig: Create custom network](/assets/2024/September/6205009a6eff383631641438_custom%20nw.png)

*Fig: Create custom network*

## Docker Storage

Docker stores data at **/var/lib/docker** on Linux-based systems. If you are using Docker Desktop on a Windows WSL machine, the data is stored at **\\wsl$\docker-desktop-data\version-pack-data\community\docker\**.

You can directly search this address in windows explorer and it will show you the storage folder. Once you get there it contains different folders including volumes and images. As shows in image:

![Fig: Checking docker storage](/assets/2024/September/6206208b121a9a56fc4e0675_docker%20storage%20folder.png)

*Fig: Checking docker storage*

In the Docker storage architecture, image layers are read-only, and the container layer can be updated. Image layers are defined in the Dockerfile, while the container layer is created when we create a new instance of an image. The container uses image layers but can have custom changes in the container layer.

When a container is deleted, the data stored in the container layer is also deleted. To persist data, we can use volumes created by the user. 

| Command | Description |
| --- | --- |
| `docker volume create [volume name]` | Create a volume |
| `docker run -v [volume name | folder path]:[default folder used] [image name]` | Maps the folder used by the container to store data |
| `docker run --mount type=bind, source=[source folder path], target=[target folder path] [image name]` | Maps the folder used by the container to store data. |

#### Example to Create custom Volume

![Fig: Create custom volume](/assets/2024/September/6206259677915b152ac79122_my%20volumne.png)

*Fig: Create custom volume*

![Fig: Modified volumes folder](/assets/2024/September/620625d4c28c683945be4251_folder%20volumne.png)

*Fig: Modified volumes folder*

#### Example of Mounting folder

![Fig: Mounting folder](https://uploads-ssl.webflow.com/61fa86c5f8078e10e03570d4/6206361e5032e625cf349073_attach%20folder.png)

*Fig: Mounting folder*

## Docker Compose

Docker Compose is a tool that allows developers to define and create multi-container applications. It uses a YAML file to define all the services/containers and their configurations. 

The process involves defining the application's environment in a Dockerfile, defining the application's services in a docker-compose.yml file, and starting the application using the `docker-compose up` command.

Here is an example of a docker-compose.yml file:

```yml
version: '3.9'
services:
  web:
    build: .
    ports:
      - "5000:5000"
  redis:
    image: "redis:alpine"

```

In this example, we have two services defined: `web` and `redis`.

The `web` service is built using the Dockerfile in the current directory (`.`), and maps port 5000 on the host to port 5000 in the container.

The `redis` service uses the official Redis image from Docker Hub (`redis:alpine`).

To start the application, navigate to the directory containing the docker-compose.yml file and run the `docker-compose up` command. This will start both services and output their logs to the console.

```
$ docker-compose up
```

Note that you can also use the `docker-compose` command to manage the lifecycle of the application's services, such as starting, stopping, and restarting them.

```
$ docker-compose start
$ docker-compose stop
$ docker-compose restart
```

That’s enough to know about docker if you are just starting. Hope it will help someone.❤❤