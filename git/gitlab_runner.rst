
GitLab Runner
=============

Refer to `<https://docs.gitlab.com/runner/install/>`_ for installation.


The following are notes for installation using Docker.

1. Download the image

.. code-block::

  docker pull gitlab/gitlab-runner:latest

2. Create a directory for saving configurations

.. code-block::

  mkdir -p /path/to/config

3. Register the runner.
   Every project needs a runner. Go to the ``settings`` of the repository,
   select ``CI/CD``, expand ``runners``.
   There is a section ``Set up a specific Runner manually``. The first step is to
   install the runner, which we have installed using docker (e.g., pull the image is enough).

   Refer to `<https://docs.gitlab.com/runner/register/#docker>`_ for registering a runner.

.. code-block::

  docker run --rm -t -i \
    -v /path/to/config:/etc/gitlab-runner gitlab/gitlab-runner register

It will ask for the URL and token, which are displayed in ``Set up a specific Runner manually``.
Just copy and paste them.

For the runner executor, enter ``docker``.

4. Start the runner.

.. code-block::

  docker run -d --name gitlab-runner --restart always \
    -v /path/to/config:/etc/gitlab-runner \
    -v /var/run/docker.sock:/var/run/docker.sock \
        gitlab/gitlab-runner:latest

5. Done!


.. Note::

    Since we have used ``--name gitlab-runner`` to start the container, so we
    use ``gitlab-runner`` below to refer the container.

To restart the docker image, use::

  docker restart gitlab-runner

To stop and delete the container::

  docker stop gitlab-runner && docker rm gitlab-runner

To read the log, use::

  docker logs gitlab-runner


.gitlab-ci.yml
--------------

References:

- Why we're replacing GitLab CI jobs with .gitlab-ci.yml

    `<https://about.gitlab.com/blog/2015/05/06/why-were-replacing-gitlab-ci-jobs-with-gitlab-ci-dot-yml/>`_

- Getting started with GitLab CI/CD


    `<https://docs.gitlab.com/ee/ci/quick_start/README.html>`_


- `<https://docs.gitlab.com/ee/ci/yaml/README.html>`_

    GitLab CI/CD Pipeline Configuration Reference

- `<https://docs.gitlab.com/ee/user/project/pages/getting_started_part_four.html>`_

    Creating and Tweaking GitLab CI/CD for GitLab Pages

