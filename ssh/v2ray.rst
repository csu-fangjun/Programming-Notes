
v2ray
=====

It requires only 2GB for t2.micro to install Ubuntu 18.04.

.. code-block::

  wget https://git.io/v2ray.sh
  sudo ./v2ray.sh

Use default settings:

.. code-block::

    ---------- V2Ray 配置信息 -------------

    地址 (Address) = 3.133.115.97

    端口 (Port) = 8081

    用户ID (User ID / UUID) = 5de121f4-80df-46fc-8eaf-6e540f9f8827

    额外ID (Alter Id) = 233

    传输协议 (Network) = tcp

    伪装类型 (header type) = none


.. code-block::

  service v2ray status

Config file is ``/etc/v2ray/config.json``.

References
----------

- V2Ray一键安装脚本

    `<https://github.com/233boy/v2ray/wiki/V2Ray%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85%E8%84%9A%E6%9C%AC>`_



.. code-block::

  $ curl ifconfig.me
  3.133.115.97

  $ curl myip.ipip.net
  当前 IP：3.133.115.97  来自于：美国 俄亥俄州 都柏林  amazon.com

  $ curl ipinfo.io
  {
    "ip": "3.133.115.97",
    "hostname": "ec2-3-133-115-97.us-east-2.compute.amazonaws.com",
    "city": "Columbus",
    "region": "Ohio",
    "country": "US",
    "loc": "40.1357,-83.0076",
    "org": "AS16509 Amazon.com, Inc.",
    "postal": "43236",
    "timezone": "America/New_York",
    "readme": "https://ipinfo.io/missingauth"
  }
