---
layout: post
title:  "Start a Webserver with aws"
date:   2023-10-22 15:08:10 +0530
categories: cloud
---

## Introduction

In this blog, we are gonna try to create our own webserver in aws. We will follow operation Excellence principle by designing it failure by using multi-AZ architecture or extending it to more than 1 availability zones. We will use load balancer to uniformly distribute the requests coming to our webserver to different EC2 instances. and If you have a domain we will learn how to create DNS records so that your domain points to this newly created webserver. While doing all of this we will not forget the security principle of well-architected framework.

### Create an EC2 Instance

Let’s first create an EC2 instance to run our webserver. Amazon Elastic compute cloud (EC2) is a web service that provides resizable computing capacity in the cloud.

EC2 offers a wide range of [instance types](https://aws.amazon.com/ec2/instance-types/) to cater to different workload requirements. Here are some commonly used EC2 instance types:

- General Purpose Instances
- Compute-Optimized Instances
- Memory-Optimized Instances
- Storage-Optimized Instances

For creating a new EC2 instance follow the steps

1. Go to EC2 service, then go instances and click Launch an Instance.
2. Give a name to your instance and choose an ubuntu AMI (Amazon Machine Image), its similar to docker images. 
3. Choose an Instance type, there are different instance families including General Purpose, compute-optimized, memory-optimized and storage-optimized. For now just choose t2.micro.
4. Amazon recommends to use a key pair to securely connect to your instance. If you already have one use that otherwise you can easily create a new one.
5. Leave other settings to default.
6. Then click Launch Instance.

![Fig: Creating a new EC2 Instance](/assets/2024/September/start%20ec2%20instance.png)

*Fig: Creating a new EC2 Instance*

### Set up a web server

For this connect to your EC2 instance, by choosing your newly created instance and then click on connect. Make sure it’s in running state.

![Fig: Connecting to EC2 Instance](/assets/2024/September/connect%20to%20ec2%20instance.png)

*Fig: Connecting to EC2 Instance*

After connecting, run the following commands:

```bash
sudo apt update
sudo apt upgrade

# This will instance apache web server
sudo apt install apache2
```

If you see something like this after running upgrade, then you need to reboot your system. The different between rebooting and Start/stop is that on start/stop the hardware running your instance is changed but on rebooting the hardware is the same.

![Untitled](/assets/2024/September/start%20ec2%20instance2.png)

In the next screen, select via tab then click enter.

![Fig: Rebooting Instance](/assets/2024/September/reboot%20instance.png)

*Fig: Rebooting Instance*

Again connect to your instance and install the Apache webserver. If at this time you try to connect to your apache webserver via your EC2 pubic IP, it will be timed out.  because currently your EC2 instance does not allow http traffic.  For that we need to modify security group.

### Using security group as firewall

Security groups in AWS are virtual firewalls that control inbound and outbound traffic for EC2 instances. They act as a protective layer around instances, allowing you to define rules that specify the protocols, ports, and IP ranges that are allowed to access the instances.

We will create a custom security group for our EC2 instance.  Inside EC2 service go to Network & security panel and click on Create security group. 

Add following inbound rules to the security group.

![Fig: Inbound rules to allow http traffic](/assets/2024/September/inbound%20rules.png)

*Fig: Inbound rules to allow http traffic*

To change add this newly created security group to your EC2 instance following the steps:

Instances → select instance → Action → security → change security group

Add the newly created security group and remove the old security group, then save it.

![Fig: Adding security group to EC2 instance](/assets/2024/September/change%20security%20group.png)

*Fig: Adding security group to EC2 instance*

Now, you will be able to send request to your webserver over http connection. To confirm, type the IPv4 address of your EC2 instance in browser and you will see the default Apache web server page.

Hooray, but that’s not it we are gonna do more than this to improve the architecture of our web server.

### Create backups with AMI snapshot

You can create the backup of your current webserver by creating an Image that contains all the information required to launch more EC2 instances with same configuration.

To create an image of your webserver follow the steps:

Instances → select Instance → Action → Image and templates → Create Image

Give some appropriate name to your image and keep the rest settings default. Click on create Image.

It will take some time to create the Image or we can say Amazon Machine Image (AMI). You will be able to check this AMI under AMI section of EC2 instance.

![Fig: Amazon Machine Images](/assets/2024/September/create%20ami%20snapshot.png)

*Fig: Amazon Machine Images*

After the status is changed to Available, click on Launch Instance from AMI to create your duplicate web servers. Prefer to select same key-pair, security group and under network settings choose a subnet (Availability zone) other than that your current EC2 instance is using to design for failures.

![Fig: Creating new instance from AMI](/assets/2024/September/create%20new%20instance%20from%20ami.png)

*Fig: Creating new instance from AMI*

After this you will have 2 EC2 instances running your webserver in two different availability zones. You can create as many as you want but for now let’s keep it two and try some more things.

If you try to access the IPv4 of your new instance, you will see the default Apache page.

### Scaling with Elastic Load Balancer (ELB)

We will create an Elastic Load balancer (ELB) to uniformly route the incoming requests to healthy instances. 

There are three types of load balancers that AWS provides: 

1. **Application Load Balancer (ALB)**: This load balancer operates at the application layer and is best suited for applications that require advanced routing capabilities. It can distribute traffic based on URL paths or HTTP headers, making it ideal for routing requests to different microservices or handling multiple domains. ALBs are commonly used for web applications, API gateways, and container-based architectures.
2. **Network Load Balancer (NLB)**: This load balancer operates at the transport layer (Layer 4) and is designed for applications that require ultra-high performance and low latency. It can handle extremely high volumes of traffic and is suitable for TCP/UDP-based applications, gaming, and real-time streaming.
3. **Gateway Load Balancer (GLB)**: This load balancer is designed to handle traffic at the edge of the AWS network and is used for scenarios where you need to distribute traffic across multiple virtual appliances, such as firewalls, intrusion detection systems, or deep packet inspection systems. 

For our task application load balancer is best choice. To create ALB go to EC2 dashboard then to Load Balancing section. 

1. Click on Create new load balancer.
2. Choose Application load balancer.
3. Add security group that allows inbound traffic over internet.
4. Create a target group and add the EC2 instances → Include as pending below → Register pending targets.

    ![Untitled](/assets/2024/September/register%20target%20group.png)

5. Click Create Load balancer.

    ![Fig: Application load balancer](/assets/2024/September//create%20load%20balancer.png)

    *Fig: Application load balancer*

If you go to the DNS name of the elb, you will be routed to one of the healthy EC2 instances.

But this domain doesn’t look good, if you already have a domain or want to buy a domain in that case you use Route 53.

### Route 53

Amazon Route 53 is a highly available and scalable Domain Name System (DNS) web service.

You can buy a domain from here also in around $12. 

Pricing:

- $0.50 per hosted zone / month for the first 25 hosted zones
- $0.10 per hosted zone / month for additional hosted zones

If hosted zone deleted within 12 hours of creation then no charge.

### Create a Hosted zone

In Amazon Route 53, a hosted zone is a container for a collection of DNS records that define how domain names are resolved. It is a way to manage the DNS records for a domain. Each hosted zone represents a domain and contains the DNS records that specify how traffic for that domain is routed.

Create a hosted zone by going to Route 53 → Hosted zones. In the domain add you own domain like [abc.com](http://abc.com) Keep the rest default and click create Hosted zone.

![Fig: Hosted zone for my website](/assets/2024/September/create%20hosted%20zone.png)

*Fig: Hosted zone for my website*

Nameservers are part of the Domain Name System (DNS) infrastructure and are responsible for translating domain names into IP addresses

Replace your domain name server urls with hosted zone urls. 

![Fig: Added Route 53 Name servers](/assets/2024/September/hostinger.png)

*Fig: Added Route 53 Name servers*

It might take few hours until the changes reflect for your domain. Till then lets add the record for our ALB.

1st record: It will point to the application load balancer.

![Untitled](/assets/2024/September/create%20record.png)

Create another record with subdomain www, this will point to the our first record.

![Fig: Creating record with www. and routing to dsm-blogs.in](/assets/2024/September/create%20another%20record.png)

*Fig: Creating record with www. and routing to dsm-blogs.in*

Now, if you try to access your domain you will see the default Apache web server page.

![Fig: Now, [dsm-blogs.in](http://dsm-blogs.in) is point to lba which directs to one of EC2 instances](/assets/2024/September/apache%20running.png)

Fig: Now, [dsm-blogs.in](http://dsm-blogs.in) is point to lba which directs to one of EC2 instances

Hurray, we come really far starting from just EC2 instance to our own domain.

That’s enough for this blog post. We will continue developing our webserver in future blog posts, by adding storage services like EBS, EFS and S3 and distributing content with CloudFront. Our aim is better understand the usage of services and when to use which service for our application.

Hope you learnt something useful from this blog. See ya in next blog posts ❤️❤️.  
