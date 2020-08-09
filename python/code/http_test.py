#!/usr/bin/env python3
'''
Test http client.

According to https://docs.python.org/3/library/http.client.html,

urllib.request is recommeded.
'''

import http.client


def main():
    h1 = http.client.HTTPConnection('www.google.com')
    assert isinstance(h1, http.client.HTTPConnection)

    h1.request('GET', '/')
    resp = h1.getresponse()

    assert isinstance(resp, http.client.HTTPResponse)
    headers = resp.getheaders()
    assert isinstance(headers, list)

    headers = dict(headers)
    print(headers)
    print(resp.msg, resp.version, resp.status, resp.reason)


if __name__ == '__main__':
    main()
