Internals
=========

2017
----

The first commit was created on 2017.08.27. It used bazel to build the project
and the focus is on android:

.. code-block:: bash

  bazel build mace/examples:helloworld \
     --crosstool_top=//external:android/crosstool \
     --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
     --cpu=armeabi-v7a

It was needed to set `-pie` in the `linkopts` in `cc_binary`.

The NDK version was::

  android_ndk_repository(
    name = "androidndk",
    # Android 5.0
    api_level = 21
  )

It used Ubuntu 16.04.
