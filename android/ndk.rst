
NDK
===

* Installation:
  - Go to `<https://developer.android.com/studio/index.html#downloads>`_
    to download ``commandlinetools-linux-6858069_latest.zip``.

    .. code-block::

      wget https://dl.google.com/android/repository/commandlinetools-linux-6858069_latest.zip
      unzip commandlinetools-linux-6858069_latest.zip

It generates a folder ``cmdline-tools`` containing::

   NOTICE.txt  bin  lib  source.properties

.. code-block::

   mkdir $HOME/android-sdk
   mv cmdline-tools $HOME/android-sdk
   export ANDROID_SDK_ROOT=$HOME/android-sdk
   cd $HOME/android-sdk
   mkdir latest
   mv cmdline-tools/* latest
   mv latest cmdline-tools/
   export PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$PATH

Now we have ``$HOME/android-sdk/cmdline-tools/latest/*``. Some people renames ``latest``
to ``tools``.

.. code-block::

   sdkmanager --list
   sdkmanager --list_installed

will print::

     Path                            | Version      | Description
     -------                         | -------      | -------
     build-tools;19.1.0              | 19.1.0       | Android SDK Build-Tools 19.1
     cmake;3.10.2.4988404            | 3.10.2       | CMake 3.10.2.4988404
     cmdline-tools;1.0               | 1.0          | Android SDK Command-line Tools
     cmdline-tools;latest            | 3.0          | Android SDK Command-line Tools (latest)
     emulator                        | 30.1.5       | Android Emulator
     ndk;16.1.4479499                | 16.1.4479499 | NDK (Side by side) 16.1.4479499
     platforms;android-10            | 2            | Android SDK Platform 10

So the zip file that we have downloaded is the latest package of ``cmdline-tools``. It should
be placed in the ``latest`` folder of the parent folder ``cmdline-tools`` of the ``ANDROID_SDK_ROOT`` folder.

Previously, ``ANDROID_SDK_ROOT`` is called ``ANDROID_HOME``. But ``ANDROID_HOME`` is deprecated, we should
use ``ANDROID_SDK_ROOT``. See `<https://developer.android.com/studio/command-line/variables#envar>`_.

According to `<https://developer.android.com/studio/command-line#tools-sdk>`, the command line tools is
saved in ``android_sdk/cmdline-tools/version/bin/``.

For the android sdk build tools, they are saved in ``android_sdk/build-tools/version/``,
while platform tools are saved in ``android_sdk/platform-tools/``, emulator is in ``android_sdk/emulator/``,

.. code-block::

   sdkmanager --help
   yes | sdkmanager --licenses  # to accept or licenses, the decision is saved in a cache file.

For build tools 28::

   sdkmanager "platform-tools" "platforms;android-28"
   # now we have
   #  android-sdk/platform-tools, it contains the binary ``adb``.
   # and
   # android-sdk/platforms/android-28

   sdkmanager 'build-tools;28.0.0'
   # now we have
   #  android-sdk/build-tools/28.0.0
   #  android-sdk/emulator
   #  android-sdk/patcher
   #  android-sdk/tools

Export the following environemtn variables::

   export ANDROID_SDK_ROOT=/root/android-sdk
   export ANDROID_HOME=$ANDROID_SDK_ROOT
   export PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$PATH
   export PATH=$ANDROID_SDK_ROOT/emulator:$PATH
   export PATH=$ANDROID_SDK_ROOT/platform-tools:$PATH

   export PATH=$ANDROID_SDK_ROOT/build-tools/28.0.0:$PATH # change it for different versions

Install NDK::

   sdkmanager "ndk;21.0.6113669"
   # it will download android-ndk-r21 and generate
   #  android-sdk/ndk/21.0.6113669

   export ANROID_NDK_HOME=$ANDROID_SDK_ROOT/ndk/21.0.6113669


sdkmanager "system-images;android-25;default;x86"
avdmanager create avd -k "system-images;android-25;default;x86" -n hao -b x86 -g default
avdmanager list avd
emulator -avd hao -no-window  -no-accel # then, open a new terminal
adb devices
adb push ./bazel-bin/hello /sdcard/
adb push ./bazel-bin/hello /data/
adb root
adb shell
adb shell /data/hello

sdkmanager "system-images;android-23;default;armeabi-v7a" # platforms;android-23
avdmanager create avd -k "system-images;android-23;default;armeabi-v7a" -n hao-arm -b armeabi-v7a -g default
emulator -avd hao-arm -no-window

WORKSPACE::

   android_sdk_repository(
       name = "androidsdk", # Required. Name *must* be "androidsdk".
       api_level=23,
   #    path = "/path/to/sdk", # Optional. Can be omitted if `ANDROID_HOME` environment variable is set.
   )

   android_ndk_repository(
       name = "androidndk", # Required. Name *must* be "androidndk".
       api_level=23,
   #    path = "/path/to/ndk", # Optional. Can be omitted if `ANDROID_NDK_HOME` environment variable is set.
   )

BUILD::

   cc_binary(name="hello", srcs=["a.cc"], copts = ["-std=c++11"],linkopts = ["-ldl", "-pie"], )

   bazel build //:hello --crosstool_top=@androidndk//:default_crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=x86

ndk-build
---------

It is a bash script::

   ndk-build NDK_LOG=1

will show a lot of logs.

Entry point is build/core/build-local.mk

* Read:
  - `<https://android.googlesource.com/platform/ndk/+/4fca9f7ace03af0b9a82f492d308fcca49eb1c1a/docs/ANDROID-MK.TXT>`_

   The default c++ source file extension is ``.cpp``, use
   ``LOCAL_CPP_EXTENSION := .cc`` to change it to ``.cc``, or use
   ``LOCAL_CPP_EXTENSION := .cc .cxx .cpp``.

  - `<https://developer.android.com/ndk/guides/android_mk>`_

  - `<https://developer.android.com/ndk/samples>`_

  - `<http://web.guohuiwang.com/technical-notes/androidndk1>`_

  - `<https://developer.android.com/ndk/guides/cmake>`_

``modules-LOCALS`` is defined in ``build/core/definitions.mk``.
