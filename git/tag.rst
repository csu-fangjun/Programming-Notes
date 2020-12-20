
tag
===

List existing tags::

  git tag


Search for a tag name with a wildcard::

  git tag --list "v1.*"

Create an annotated tag::

  git tag -a v1.2 -m "some message"
  git show v1.2 # to show the tag

Create a lightweight tag::

  git tag v1.3
  git show v1.3


Create a tag using a SHA-1::

  git log --pretty=oneline
  git tag -a v1.3 <SHA-1>

Push a tag to remote::

  git push origin v1.2

To push all tags::

  git push origin --tags

Delete a tag locally::

  git tag -d v1.2

Delete a remote tag::

  git push origin :refs/tags/v1.2
  # or use
  git push origin --delete v1.2

Checkout a tag::

  git checkout v1.2
