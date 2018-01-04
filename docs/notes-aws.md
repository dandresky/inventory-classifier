My random notes on Amazon EC2 offerings for reference purposes.

### Storage Options

Amazon offers S3, EFS, and EBS as storage Options
- S3 is suitable for persistent long-term storage of large amounts of data
- EFS is designed to provide scalable storage for EC2 instances
- EBS was also created for EC2 - virtual disk storage

S3
- Stores data as objects in a flat environment (without a hierarchy)
- Data objects (images, json, mp4, etc.) contain a header with a unique identifier (key), so access to them can be obtained through web requests
- Slowest access of the the three storage types
- [Cost](https://aws.amazon.com/s3/pricing/) is $0.023/month for standard and $0.0125/month for Standard-Infrequent Access
- Not good for frequent access of data, better suited for long-term storage or archiving.

EBS
- Virtual drive for virtual EC2 instances
- Fastest access of the three storage types
- Stores data in blocks and organizes through a folder hierarchy similar to a traditional file system
- Only exists in combination with one EC2 instance
- Can't be scaled, if you need more memory you will need to build and configure another volume (or create with image of existing volume)
- Must be either attached to an EC2 instance or can be put in standby
- General Purpose Volumes are backed with Solid State Drive (SSD)
    * good fit for applications that need a lot of read and write operations, like PostgreSQL, MS SQL or Oracle databases
- Provisioned IOPS Volumes allow to buy read/write operations on demand
    * designed for heavy workloads
- Magnetic Volumes are a low-cost volume that don’t require a lot of read/write operations
    * designed to be used with testing and development on EC2.
- [Cost](https://aws.amazon.com/ebs/pricing/) is approximately $0.025 to $0.10/GB-month depending on type

EFS - Elastic File System
- Scalable storage for EC2 with relatively fast output
- Can attach to multiple EC2 instances
- Web and file system interface
- Faster than S3 but slower than EBS
- To access file system, mount on an EC2 Linux-based instance using the standard Linux mount command and the file system’s DNS name
    * Once mounted, you can work with the files and directories just like a local file system
- [Cost](https://aws.amazon.com/efs/pricing/) is $0.30/GB-month

### Compute Options

Some EC2 types come with a local store that lives only as long as the instance lives. Once stopped or terminated, the data is lost. This local store takes 5x to 10x longer to boot.

### Compute Terminology
ECU stands for Elastic Compute Unit.
- EC2 instances are VM's that allow Amazon to abstract compute capability from the underlying hardware so that product offerings are consistent, even when hardware changes.
- To create a standardized terminology and account for Moore’s Law, AWS created a logical computation unit known as an Elastic Compute Unit (ECU)
- Provides each instance with a consistent and predictable amount of CPU capacity
- [See this blog for more detail](https://www.datadoghq.com/blog/are-all-aws-ecu-created-equal/)

vCPU is a virtual CPU and can be read as the number of CPU cores assigned to an EC2 instance.  
