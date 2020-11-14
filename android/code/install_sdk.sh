#!/bin/bash

android_sdk_root=/root/android-sdk


mkdir -p $android_sdk_root
cd $android_sdk_root
wget https://dl.google.com/android/repository/commandlinetools-linux-6858069_latest.zip
unzip -qq commandlinetools-linux-6858069_latest.zip
mkdir latest
mv cmdline-tools/* latest/
mv latest cmdline-tools/
rm commandlinetools-linux-6858069_latest.zip


echo "export ANDROID_SDK_ROOT=$android_sdk_root" >> ~/.bashrc
echo "export ANDROID_HOME=$android_sdk_root" >> ~/.bashrc
echo "export PATH=$android_sdk_root/cmdline-tools/latest/bin:$PATH" >> ~/.bashrc
echo "export PATH=$android_sdk_root/emulator:$PATH" >> ~/.bashrc
echo "export PATH=$android_sdk_root/platform-tools:$PATH" >> ~/.bashrc
echo "export PATH=$android_sdk_root/build-tools/28.0.0:$PATH"  >> ~/.bashrc # change it for different versions

source ~/.bashrc

yes | sdkmanager --licenses

sdkmanager "platform-tools"
sdkmanager "platforms;android-24"
sdkmanager 'build-tools;28.0.0' # bazel requires at least 26.0.1
sdkmanager "ndk;21.0.6113669"

echo "export ANROID_NDK_HOME=$ANDROID_SDK_ROOT/ndk/21.0.6113669" >> ~/.bashrc
echo "export PATH=$ANDROID_SDK_ROOT/ndk/21.0.6113669:$PATH" >> ~/.bashrc

source ~/.bashrc

sdkmanager "system-images;android-24;default;x86"
sdkmanager "system-images;android-24;default;x86_64"
sdkmanager "system-images;android-24;default;armeabi-v7a"
sdkmanager "system-images;android-24;default;arm64-v8a"

echo n | avdmanager create avd -k "system-images;android-24;default;x86" -n my86 -b x86 -g default
echo n | avdmanager create avd -k "system-images;android-24;default;x86_64" -n my86-64 -b x86_64 -g default
echo n | avdmanager create avd -k "system-images;android-24;default;armeabi-v7a" -n myv7a -b armeabi-v7a -g default
echo n | avdmanager create avd -k "system-images;android-24;default;arm64-v8a" -n myv8a -b arm64-v8a -g default

wget https://github.com/bazelbuild/bazel/releases/download/3.7.0/bazel-3.7.0-installer-linux-x86_64.sh
chmod +x bazel-3.7.0-installer-linux-x86_64.sh

bazel_dir=/root/software/bazel-3.7.0
mkdir -p $bazel_dir
./bazel-3.7.0-installer-linux-x86_64.sh --prefix=$bazel_dir
echo "export PATH=$bazel_dir/bin:$PATH" >> ~/.bashrc
echo "source $bazel_dir/lib/bazel/bin/bazel-complete.bash" >> ~/.bashrc
