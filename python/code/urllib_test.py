#!/usr/bin/env python3

import http.client
import urllib.request


def main():
    resp = urllib.request.urlopen('http://www.google.com')
    assert isinstance(resp, http.client.HTTPResponse)
    print(type(resp), dir(resp))


if __name__ == '__main__':
    main()
