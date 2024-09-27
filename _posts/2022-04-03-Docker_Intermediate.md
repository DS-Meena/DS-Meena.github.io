---
layout: post
title:  "Advancing in Docker"
date:   2022-04-03 10:00:10 +0530
categories: Cloud
---

In this blog, we'll explore the world of Docker, from the basics to advanced topics. Whether you're a Docker pro or just starting out, this document has you covered!

We'll cover everything you need to know about Docker, including its core principles and real-world applications. We'll discuss Docker registries, public and private, and how they store and distribute Docker images. You'll learn how to access private repositories and publish your own images.

But that's not all! We'll also dive into Docker Engine, which powers Docker containerization. We'll explore its main components and how they create, manage, and run Docker containers.

# Docker Registry

Docker registry is a server-side application that allows users to store and distribute Docker images. It is highly scalable and can be used to create both public and private registries. Users can store and distribute Docker images through the registry, making it easy to share images between different environments and teams. 

## Public Registry

A public registry is a Docker registry that is open to the public and can be accessed by anyone. 

The Docker Hub is an example of a public registry. 

#### Naming

When naming a Docker image in a public registry, the format is typically: **Registry / User Account / Image Repository**.

For example, the Docker Hub (which is a public registry) with a user account of Redis could have an image repository of Redis.

So the image name would be: `docker.io/Redis/Redis`

## Private Registry

A private registry, on the other hand, is a Docker registry that is only accessible by authorized users or machines. 

Users can create their own private registry using the registry container image provided by Docker. This allows them to store and distribute Docker images internally, rather than relying on a public registry.

#### Naming

When naming a Docker image in a private registry, the format is typically: **Registry / Image Repository / Image Name**. 

For example, if the registry is located at `myregistry.example.com` and the image repository is named `myimages`, the name of a custom image might be `myregistry.example.com/myimages/myimage:tag`.

### Creating Private Repository

We can use the following commands related to Docker Registry:

| Command | Description |
| --- | --- |
| `docker run -d -p 5000:5000 --name registry registry:2` | Creates a Docker registry Image container |
| `docker tag [Image name\|Image Id] [Tag]` | Adds a tag to a custom image that will be pushed to the registry |
| `docker push [Image Tag]` | Pushes an image to a private registry by referring to the tag |
| `docker pull [Private Registry name.io]/User Account/Image Repository` | Pulls an image from a private registry |

The command `docker run -d -p 5000:5000 --name registry registry:2` creates a Docker registry. By default, it creates a local private registry that can be accessed on port 5000 using the URL localhost:5000 from the local host. When accessing the registry from the same machine, both the registry and user account can be referred to using this URL.

![Fig: Accessing Docker Registry](/assets/2024/September/62078c4babdb2e5620711a2b_registy%202.png)

*Fig: Accessing Docker Registry*

### Accessing Private Repository

If you have your private repository hosted on server then you can access that with following commands:

To login: **docker login [Private Registry name.io]**

To pull image: **docker pull [Private Registry name.io]/User Account/Image Repository**

![Fig: Pull image from Private registry](/assets/2024/September/6208a2acccb873a85c53c1da_pull%20own.png)

*Fig: Pull image from Private registry*

#### Publish your Image to Private registry

1. Add a tag to the custom Image. e.g. `docker tag ccee localhost:5000/my_image_repo`
2. Push the image to the private registry using tag.

![Fig: Push image to Private registry](/assets/2024/September/6208a210d0c851da9c03afc0_push.png)

*Fig: Push image to Private registry*

# Docker Engine

Docker Engine is an open source technology that helps us containerize our applications. It's got three important parts that work together to make it all happen.

1. Docker CLI, which is like our go-to command line tool for talking to the Docker daemon. We can use it to do all sorts of cool stuff with our containers, images, and volumes.
2. Rest API that acts as a middleman. It helps the CLI and the daemon communicate with each other in a standardized way. So, when we want to automate things or make our lives easier, we can use the Rest API.
3. Docker Daemon, this is the real powerhouse behind Docker Engine. It takes care of creating and managing our images, containers, and volumes. It's like the engine that drives everything.

In our systems, we rely on the Docker CLI to interact with the Docker host. It's how we manage and control our containerized applications. It makes things super smooth and efficient.

![alt text](/assets/2024/September/image.png)

You can read more about Docker Architecture from here [See Docker Architecture](https://docs.docker.com/get-started/overview/#docker-architecture).‍

## Container Orchestration

Container orchestration is the management of multiple containers across multiple hosts. It provides a way to scale and manage containerized applications, making it easier to deploy, manage, and monitor them. 

Container orchestration tools ensure that containers are automatically deployed, scaled, and managed based on the application's requirements. This helps to ensure that the application is always available and running smoothly. Some popular container orchestration tools include Docker Swarm, Kubernetes, and Mesos.

Docker Desktop provides both of Docker swarm and Kubernetes tools for Container Orchestration.

### Docker Swarm

Docker Swarm makes it easy to manage and deploy containerized applications at a big scale. It's all about this cluster thing with manager nodes and worker nodes.

So, the manager nodes are like the boss of the swarm. They control everything and make sure the cluster is doing what it's supposed to do. They handle stuff like deploying containers, scheduling services, and managing resources. And you can totally interact with the swarm through these manager nodes using the Docker CLI or API.

Now, the worker nodes are the ones actually running the containers. They take instructions from the manager nodes and spin up container instances. They spread the workload around the cluster to make sure containers are deployed and scaled as needed. And you can totally add or remove worker nodes as you need to.

![Fig: Docker Swarm, Manager and worker nodes](/assets/2024/September/image-1.png)

*Fig: Docker Swarm, Manager and worker nodes (docker.com)*

To enable Docker Swarm, follow these steps:

1. Create a swarm manager using the command `docker swarm init`. This node will manage all other nodes or hosts.

    ![Fig: Initializing a Docker Swarm](/assets/2024/September/6208d9b3816bc0c7dffdaddf_swarm%20init.png)
    
    *Fig: Initializing a Docker Swarm*
    
    If you encounter an IP address conflict, you can try using a different IP address.
    
2. Join other nodes to the swarm manager using the command `docker swarm join --token [token] [IP address of swarm manager]:2377`. Replace `[token]` with the token provided in the swarm init output, and replace `[IP address of swarm manager]` with the IP address of the swarm manager. 
For example: `docker swarm join --token SWMTKN-1-1v68tj6e5bes37vz18tsrktt2km4lk55k4h5u1xqagj2fx2g0x-325dprf4zgdbzjvm6j7ab04sn 10.17.106.9:2377`.

3. Create services using the swarm manager using the command `docker service create --replicas=[number of replicas] [image name]` in a Docker swarm.

    ![Fig: Create services in a Docker swarm](/assets/2024/September/62091070460a086ba14504df_services.png)
    
    *Fig: Create services in a Docker swarm*
    

You can also create a swarm using the Docker Desktop GUI.

When you run `docker swarm init`, the token displayed is for joining nodes as worker nodes. If you want to add nodes as manager nodes, you need to use a different token. You can check the manager token by using `docker swarm join-token manager`.

Similarly, if you forget the worker node token, you can use `docker swarm join-token worker`.

# Kubernetes

Kubernetes is an open-source container orchestration tool that is used for managing and deploying containerized applications. It has a different CLI known as **kubectl**. 

It provides multiple  functionality such as creating multiple instances of the container in a single command, updating instances by using rolling update command, and rolling back the updates using the rolling back command. 

Kubernetes has a master instance that controls all other instances in the Kubernetes cluster. The components of Kubernetes include the API server, etcd, kubelet, container runtime, controller, and scheduler.

![Fig: Comparing Deployment Traditional vs Kubernetes (docker.com)](/assets/2024/September/Docker-Kubernetes-together.webp)

*Fig: Comparing Deployment Traditional vs Kubernetes (docker.com)*

| Feature | Docker Swarm | Kubernetes |
| --- | --- | --- |
| Architecture | Less complex architecture with multiple manager and worker nodes. | More complex architecture with multiple master and worker nodes. |
| Ease of Use | Generally considered to be easier to use and set up than Kubernetes. | More difficult to set up and use, but has more features. |
| Features | Has a simpler feature set than Kubernetes, but includes basic container orchestration capabilities. | More extensive feature set than Docker Swarm, including automatic rollouts and rollbacks, service discovery and load balancing, and support for multiple storage options. |
| Scalability | Both tools can scale horizontally, but Kubernetes is generally considered to be more scalable than Docker Swarm. | Both tools can scale horizontally, but Kubernetes is generally considered to be more scalable than Docker Swarm. |
| Community | Smaller and less active community than Kubernetes. | Larger and more active community than Docker Swarm, which means more support and resources available for users. |

Overall, Docker Swarm is a good choice for smaller-scale deployments, while Kubernetes is better suited for larger and more complex deployments that require more advanced features and scalability.

### Kubernetes Commands

You can enable Kubernetes from the Docker Desktop settings. Go to Settings -> Kubernetes -> Enable Kubernetes.

Here are some useful kubectl commands:

| Command | Description |
| --- | --- |
| `kubectl run [Application]` | Starts the application. |
| `kubectl cluster-info` | Gets information about the cluster. |
| `kubectl get nodes` | Gets all the nodes in the cluster. |
| `kubectl run [Application name] --image=[Image name] --replicas=[Instances count]` | Creates a new Kubernetes application. |
| `*kubectl run –replicas=[Count of Instances] [Image name]‍‍‍‍*` | Creating multiple instances of the container in a single command. |
| `kubectl scale --replicas=[Count of Instances] [Image name]` | Can be set to automatically create new instances on requirement. |
| `kubectl rolling-update [Image name] --image=[New Image]` | Updates the instances using rolling update command. |
| `kubectl rolling-update [Image name] --rollback` | Rolls back the updates using the rolling back command. |

That’s it for this Blog Post, hope this was worth your time ❤️❤️.