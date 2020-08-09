
HTTP
====

content type: `<https://tool.oschina.net/commons>`_

URL
---

.. code-block::

  <scheme>://<user>:<password>@<host>:<port>/<path>;<params>?<query>#<frag>

Example::

  http://foo.com/test;type=d

  http://foo.com/test;sale=false/index.html;graphics=true

  ftp://anonymous:my_password@foo.com/pub/gnu

  http://test.com/foo.cgi?item=1234&color=blue

Special encoding:

- space, ``%20``
- ``~``, ``%7e``
- ``%``, ``%25``

Format of the reqeust message::

  <method> <request-URL> <version>
  <headers>

  <entity-body>

Format of the response message::

  <version> <status> <reason-phrase>
  <headers>

  <entity-body>

Format of the headers::

  <key><:>[optional_space]<value><CRLF>

Meaning of the status code:
- 1xx, informational
- 2xx, success
- 3xx, redirection
- 4xx, client error
- 5xx, server error

curl
----

Line ending is ``0x130x10`` (carriage return 0x13, line feed 0x10, CRLF)

.. code-block::

  curl -v -I www.baidu.com

  * Rebuilt URL to: www.baidu.com/
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                   Dload  Upload   Total   Spent    Left  Speed
    0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0*   Trying 103.235.46.39...
  * Connected to www.baidu.com (103.235.46.39) port 80 (#0)
  > HEAD / HTTP/1.1
  > Host: www.baidu.com
  > User-Agent: curl/7.47.0
  > Accept: */*
  >
  < HTTP/1.1 200 OK
  < Accept-Ranges: bytes
  < Cache-Control: private, no-cache, no-store, proxy-revalidate, no-transform
  < Connection: keep-alive
  < Content-Length: 277
  < Content-Type: text/html
  < Date: Thu, 30 Jul 2020 03:25:16 GMT
  < Etag: "575e1f6f-115"
  < Last-Modified: Mon, 13 Jun 2016 02:50:23 GMT
  < Pragma: no-cache
  < Server: bfe/1.0.8.18
  <
    0   277    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  * Connection #0 to host www.baidu.com left intact
  HTTP/1.1 200 OK
  Accept-Ranges: bytes
  Cache-Control: private, no-cache, no-store, proxy-revalidate, no-transform
  Connection: keep-alive
  Content-Length: 277
  Content-Type: text/html
  Date: Thu, 30 Jul 2020 03:25:16 GMT
  Etag: "575e1f6f-115"
  Last-Modified: Mon, 13 Jun 2016 02:50:23 GMT
  Pragma: no-cache
  Server: bfe/1.0.8.18

.. code-block::


  curl -v www.baidu.com -o abc

  * Rebuilt URL to: www.baidu.com/
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                   Dload  Upload   Total   Spent    Left  Speed
    0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0*   Trying 103.235.46.39...
  * Connected to www.baidu.com (103.235.46.39) port 80 (#0)
  > GET / HTTP/1.1
  > Host: www.baidu.com
  > User-Agent: curl/7.47.0
  > Accept: */*
  >
  < HTTP/1.1 200 OK
  < Accept-Ranges: bytes
  < Cache-Control: private, no-cache, no-store, proxy-revalidate, no-transform
  < Connection: keep-alive
  < Content-Length: 2381
  < Content-Type: text/html
  < Date: Thu, 30 Jul 2020 03:27:01 GMT
  < Etag: "588604eb-94d"
  < Last-Modified: Mon, 23 Jan 2017 13:28:11 GMT
  < Pragma: no-cache
  < Server: bfe/1.0.8.18
  < Set-Cookie: BDORZ=27315; max-age=86400; domain=.baidu.com; path=/
  <
  { [2381 bytes data]
  100  2381  100  2381    0     0  23943      0 --:--:-- --:--:-- --:--:-- 24050
  * Connection #0 to host www.baidu.com left intact



