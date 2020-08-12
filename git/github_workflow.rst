
GitHub Workflow
===============

Example1
--------

From `<https://github.com/actions/hello-world-javascript-action/blob/master/action.yml>`_

  .. code-block:: yaml

    name: 'Hello World'
    description: 'Greet someone and record the time'
    inputs:
      who-to-greet:  # id of input
        description: 'Who to greet'
        required: true
        default: 'World'
    outputs:
      time: # id of output
        description: 'The time we we greeted you'
    runs:
      using: 'node12'
      main: 'index.js'

And use it `<https://help.github.com/en/actions/configuring-and-managing-workflows/configuring-a-workflow#creating-a-workflow-file>`_:

  .. code-block:: yaml

        name: Greet Everyone
        # This workflow is triggered on pushes to the repository.
        on: [push]

        jobs:
          build:
            # Job name is Greeting
            name: Greeting
            # This job runs on Linux
            runs-on: ubuntu-latest
            steps:
              # This step uses GitHub's hello-world-javascript-action: https://github.com/actions/hello-world-javascript-action
              - name: Hello world
                uses: actions/hello-world-javascript-action@v1
                with:
                  who-to-greet: 'Mona the Octocat'
                id: hello
            # This step prints an output (time) from the previous step's action.
              - name: Echo the greeting's time
                run: echo 'The time was ${{ steps.hello.outputs.time }}.'

**Event**
  .. code-block::

    # push to any branch
    on: push

    # Trigger the workflow on push or pull request
    on: [push, pull_request]

    # when a commit contains files in `test/` directory
    #  and
    # is pushed to `master` **or** `v1` tag.
    on:
      push:
        branches:
          - master
        tags:
          - v1
        paths:
          - 'test/*'

    on:
      # Trigger the workflow on push or pull request,
      # but only for the master branch
      push:
        branches:
          - master
      pull_request:
        branches:
          - master
      # Also trigger on page_build, as well as release created events
      page_build:
      release:
        types: # This configuration does not affect the page_build event above
          - created

    # Trigger the workflow on pull request activity
    on:
      release:
      # Only use the types keyword to narrow down the activity types that will trigger your workflow.
        types: [published, created, edited]

**runs-on**
  .. code-block::

    runs-on: ubuntu-latest

    # another case
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-16.04, ubuntu-18.04]
        node: [6, 8, 10]

**Reference an Action**
  .. code-block::

      jobs:
        my_first_job:
          name: My Job Name
            steps:
              - uses: actions/setup-node@v1
                with:
                  node-version: 10.x

      # another example
      jobs:
        my_first_job:
          steps:
            - name: My first step
              uses: docker://alpine:3.8


**badge**
  .. code-block::

    [![GitHub Actions Status](https://github.com/<owner>/<repo>/workflows/<name>/badge.svg)](https://github.com/<owner>/<repo>/actions)

  where ``<name>`` is the workflow name inside the ``*.yml`` file. Every yaml file
  should contain a name in it.


**environment variable**
  .. code-block::

    steps:
      - name: Hello world
        run: echo Hello world $FIRST_NAME $middle_name $Last_Name!
        env:
          FIRST_NAME: Mona
          middle_name: The
          Last_Name: Octocat

  Predefined environment variable:
  - ``HOME``: ``/home/runner``
  - ``GITHUB_WORKFLOW``: the name of the workflow, e.g., ``style_check``
  - ``GITHUB_REPOSITORY``: e.g., ``owner/repo``
  - ``GITHUB_WORKSPACE``, e.g., `/home/runner/work/k2/k2`, ``.github`` is inside ``/home/runner/work/k2/k2/.github``



Learn-YAML-in-five-minutes
--------------------------

See `<https://www.codeproject.com/Articles/1214409/Learn-YAML-in-five-minutes>`_.


Deploy to GitHub Pages
----------------------

Refer to `<https://github.com/marketplace/actions/deploy-to-github-pages>`_.


