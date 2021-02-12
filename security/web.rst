Web
===

Check what server a website is using:

.. code-block::

    curl -I k2-fsa.org

    HTTP/1.1 200 OK
    Date: Wed, 10 Feb 2021 14:39:21 GMT
    Server: Apache/2.4.38 (Debian)
    Last-Modified: Mon, 08 Feb 2021 12:54:10 GMT
    ETag: "2a7-5bad2aa864f6f"
    Accept-Ranges: bytes
    Content-Length: 679
    Vary: Accept-Encoding
    Content-Type: text/html

acme
----

Install::

  csukuangfj@openslr-server:~$ curl https://get.acme.sh | sh -s
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                   Dload  Upload   Total   Spent    Left  Speed
  100   937    0   937    0     0  11289      0 --:--:-- --:--:-- --:--:-- 11289
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                   Dload  Upload   Total   Spent    Left  Speed
  100  202k  100  202k    0     0  8430k      0 --:--:-- --:--:-- --:--:-- 8430k
  [Wed Feb 10 14:51:17 UTC 2021] Installing from online archive.
  [Wed Feb 10 14:51:17 UTC 2021] Downloading https://github.com/acmesh-official/acme.sh/archive/master.tar.gz
  [Wed Feb 10 14:51:17 UTC 2021] Extracting master.tar.gz
  [Wed Feb 10 14:51:17 UTC 2021] Installing to /home/csukuangfj/.acme.sh
  [Wed Feb 10 14:51:17 UTC 2021] Installed to /home/csukuangfj/.acme.sh/acme.sh
  [Wed Feb 10 14:51:17 UTC 2021] Installing alias to '/home/csukuangfj/.bashrc'
  [Wed Feb 10 14:51:17 UTC 2021] OK, Close and reopen your terminal to start using acme.sh
  [Wed Feb 10 14:51:17 UTC 2021] Installing cron job

  csukuangfj@openslr-server:~$ acme.sh --issue -d k2-fsa.org --webroot /var/www/k2-fsa
  [Wed Feb 10 15:00:37 UTC 2021] Using CA: https://acme-v02.api.letsencrypt.org/directory
  [Wed Feb 10 15:00:37 UTC 2021] Single domain='k2-fsa.org'
  [Wed Feb 10 15:00:37 UTC 2021] Getting domain auth token for each domain
  [Wed Feb 10 15:00:39 UTC 2021] Getting webroot for domain='k2-fsa.org'
  [Wed Feb 10 15:00:39 UTC 2021] Verifying: k2-fsa.org
  [Wed Feb 10 15:00:42 UTC 2021] Pending
  [Wed Feb 10 15:00:45 UTC 2021] Pending
  [Wed Feb 10 15:00:48 UTC 2021] Pending
  [Wed Feb 10 15:00:50 UTC 2021] Success
  [Wed Feb 10 15:00:50 UTC 2021] Verify finished, start to sign.
  [Wed Feb 10 15:00:50 UTC 2021] Lets finalize the order.
  [Wed Feb 10 15:00:50 UTC 2021] Le_OrderFinalize='https://acme-v02.api.letsencrypt.org/acme/finalize/112311342/7826947146'
  [Wed Feb 10 15:00:51 UTC 2021] Downloading cert.
  [Wed Feb 10 15:00:51 UTC 2021] Le_LinkCert='https://acme-v02.api.letsencrypt.org/acme/cert/04bdda418b2198dee4bdbdb4a54cf90b5d6d'
  [Wed Feb 10 15:00:52 UTC 2021] Cert success.
  -----BEGIN CERTIFICATE-----
  MIIFGTCCBAGgAwIBAgISBL3aQYshmN7kvb20pUz5C11tMA0GCSqGSIb3DQEBCwUA
  MDIxCzAJBgNVBAYTAlVTMRYwFAYDVQQKEw1MZXQncyBFbmNyeXB0MQswCQYDVQQD
  EwJSMzAeFw0yMTAyMTAxNDAwNTFaFw0yMTA1MTExNDAwNTFaMBUxEzARBgNVBAMT
  CmsyLWZzYS5vcmcwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQC3JsRW
  nvaSFTvS8lpmP+WvnL2YF68415m7vMLCA8W+ICoQqwPXJjMw3SRh/NWEpoBHV+zI
  K/P1xWcjGuA2yoygGY5RoZbrU1UtcXqY2htL3WBISoVTNPtjW3INzCCokYgfEiG/
  wNqOBxg5dI3rq9zvDM038wygnZyxxoRg39V7do7nv6C0GFM2JknyGF1z/YGv2WhL
  XGUk8/cLLmesXFxQvKK8LHTCu0SFsl5njxMuPSkHWmaA5mz2Fadj6NsL3FujS02C
  5io6JWRzfNWQyHc6NM9dhv5CISJQyKwjOHXqew2+xVO0chCHsEeE5pVLEKPKKmLp
  bDAQm6mKruqg8ZIhAgMBAAGjggJEMIICQDAOBgNVHQ8BAf8EBAMCBaAwHQYDVR0l
  BBYwFAYIKwYBBQUHAwEGCCsGAQUFBwMCMAwGA1UdEwEB/wQCMAAwHQYDVR0OBBYE
  FC+rLYEaFf9/rDlEd48+BhBnHRsMMB8GA1UdIwQYMBaAFBQusxe3WFbLrlAJQOYf
  r52LFMLGMFUGCCsGAQUFBwEBBEkwRzAhBggrBgEFBQcwAYYVaHR0cDovL3IzLm8u
  bGVuY3Iub3JnMCIGCCsGAQUFBzAChhZodHRwOi8vcjMuaS5sZW5jci5vcmcvMBUG
  A1UdEQQOMAyCCmsyLWZzYS5vcmcwTAYDVR0gBEUwQzAIBgZngQwBAgEwNwYLKwYB
  BAGC3xMBAQEwKDAmBggrBgEFBQcCARYaaHR0cDovL2Nwcy5sZXRzZW5jcnlwdC5v
  cmcwggEDBgorBgEEAdZ5AgQCBIH0BIHxAO8AdgCUILwejtWNbIhzH4KLIiwN0dpN
  XmxPlD1h204vWE2iwgAAAXeMdZH2AAAEAwBHMEUCIQCTLB3ArbaHkmstf5T6PB32
  AsdhMuh9INZxgB+RYQ5nwQIgaBAMXsZGvu9ewyixB/9DSccqvCKGB/qbznD85mIO
  Ml8AdQD2XJQv0XcwIhRUGAgwlFaO400TGTO/3wwvIAvMTvFk4wAAAXeMdZHYAAAE
  AwBGMEQCIFhBIu9ta7J5gTcALlieFYNj8EJjdSvPSO81lRr11QU/AiBAhBQ/olFu
  8vws0FCo7sWDDtZ/YIK3mvCRCN45JrTVzjANBgkqhkiG9w0BAQsFAAOCAQEAMvne
  tzlu812sDmyBg92wQ08Xs5wRNPWLNmzA9NR/iCXZx1ek6X+l+nWzkHrNldriYJGY
  EfSJ82MA7NQdSCuPgXF+u/TwFR/E0crbfqyFBnb3EkeLVMzPoEin9F5vDCLG3ArI
  9BNTOZaonCXbP/4BueR52xi85U3kpSGTvQjZh1dlcGh/t5n1GFZYRcCESpg9M9ck
  nkcy1LOLpZPqdkZRIngVmDfDI6keLY2S9IQGRiDluu4FD2J44X7zLisy/X+Vx48r
  FpX6/bqQvMWCVljEm2nWFBjqpCXIeicyPzF7vxTbhdJzqz/Uprt7N3vj+i1LMD06
  hIErvUSv3JSmWmU0dw==
  -----END CERTIFICATE-----
  [Wed Feb 10 15:00:52 UTC 2021] Your cert is in  /home/csukuangfj/.acme.sh/k2-fsa.org/k2-fsa.org.cer
  [Wed Feb 10 15:00:52 UTC 2021] Your cert key is in  /home/csukuangfj/.acme.sh/k2-fsa.org/k2-fsa.org.key
  [Wed Feb 10 15:00:52 UTC 2021] The intermediate CA cert is in  /home/csukuangfj/.acme.sh/k2-fsa.org/ca.cer
  [Wed Feb 10 15:00:52 UTC 2021] And the full chain certs is there:  /home/csukuangfj/.acme.sh/k2-fsa.org/fullchain.cer

The above strings are for ``/home/csukuangfj/.acme.sh/k2-fsa.org/k2-fsa.org.cer``.
