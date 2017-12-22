### Request access to GPU instances on EC2 (if you haven't already)

References  
[EC2 instance types](https://aws.amazon.com/ec2/instance-types/)  
[EC2 instance pricing](https://aws.amazon.com/ec2/pricing/on-demand/)

Log in to your AWS console, then click EC2.  Under the 'EC2 Dashboard' on the  
left you'll see a 'Limits' menu item.  Click it.  

Under the heading 'Instance Limits' you'll see Name, Current Limit, and
Action.  Scroll down to the 'p2' instances for GPU enhanced Accelerated
Computing. For a description of the available instances and their cost
see the references above.  This guide recommends requesting two 'p2.xlarge'
instances with an on-demand rate of $0.90/hr each. It's a
place to start - you can request more and larger and more expensive
instances later.  With two instances, you could be training two models in parallel.

Under in the 'p2.xlarge' row under 'Action' select 'Request limit increase'

Region: N. Virginia  
Instance type: p2.xlarge  
Limit: Instance limit  
New limit value: 2  
Add a use case description: Student learning about neural networks  

Contact method:  
Phone

If you use 'Phone' they will call you immediately and they say it takes up to
two days, but often it is completed much faster and you will receive an email
notifying you if access has been granted.


