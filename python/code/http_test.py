#!/usr/bin/env python3
'''
Test http client.

According to https://docs.python.org/3/library/http.client.html,

urllib.request is recommeded.
'''

import http.client


def test():
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


def post():
    filename = '/home/fangjunkuang/tts/diusjf.wav'
    conn = http.client.HTTPConnection('localhost', port=1234)
    conn.set_debuglevel(10)
    with open(filename, 'rb') as f:
        conn.request('POST', '/api/ms/raw', f)
    resp = conn.getresponse()
    headers = dict(resp.getheaders())
    for key, val in headers.items():
        print(key, val)


def main():
    post()


if __name__ == '__main__':
    main()
