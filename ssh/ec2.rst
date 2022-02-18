
EC2
===

1. Create a google account
2. Sign up for amazon ec2, i.e., create a new account using the above gmail.
3. Wait for several minutes to activate the account.
4. Go to EC2 dashboard, instances, launch instances, create a new one,
5. Choose an instance that is available for free tier
6. Use default storage 8 GB 
7. Configure security group, enable tcp port 41951 and ssh port 22
8. Create a new key pair, download it and rename it to ec2-2022-02-06.pem
9. Caution: The key can be only downloaded once
10. Change it permission to readonly for owner, i.e., -r--------

Enter the following to `~/.ssh/config` locally:

.. code-block::

  # aws ec2, EU Frankfurt
  HOST ec2 35.157.124.169
   HostName 35.157.124.169
   User ubuntu
   PreferredAuthentications publickey
   IdentityFile ~/.ssh/ec2-2022-02-06.pem
   #LocalForward 8888 localhost:8888
   #ForwardX11 yes
   ServerAliveInterval 120
   TCPKeepAlive no
