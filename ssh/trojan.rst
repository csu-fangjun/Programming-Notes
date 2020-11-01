
trojan
======

Go to godaddy to buy a domain, e.g., ``meinemail.xyz``.

DNS -> Manage area: 
- type: A, name @, value: your ip

.. code-block::

  wget https://bootstrap.pypa.io/get-pip.py
  sudo python3 ./get-pip.py

  wget https://raw.githubusercontent.com/hijkpw/scripts/master/trojan-go.sh
  # enable port 443 and 80 on EC2 !!!
  chmod +x trojan-go.sh
  sudo ./trojan-go.sh
  # use default settings, use only 1 password

.. code-block::

   BBR模块已启用

     trojan-go配置信息：

     当前状态：已安装 正在运行
     IP：3.121.218.134
     域名/主机名(host)：meinemail.xyz
     端口(port)：443
     密码(password)：lNzJfBTSKgBbD2HR


References
----------

- `<https://v2raytech.com/shadowrocket-config-trojan-tutorial/>`_
- `<https://v2raytech.com/trojan-one-click-scrip/>`_

