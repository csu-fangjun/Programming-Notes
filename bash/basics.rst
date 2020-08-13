
Basics
======

Get the current directory of the script
---------------------------------------

.. code-block:: bash

  cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
  echo $cur_dir

``date``
--------

.. code-block:: bash

  date "+%F %T"
  # 2010-01-02 08:01:02

``array``
---------

.. code-block:: bash

  a=(
  hello
  world
  )

  for i in ${a[@]}; do
    echo $i
  done

Output

.. code-block:: console

  hello
  world

Block Comment
-------------

Refer to `Block Comments in a Shell Script`_.

.. code-block:: bash

  :<<'EOF'
  hello'world
  foo
  bar
  EOF

.. HINT::

  ``:`` can be omitted. But it is not portable if ``:`` is missing.


.. _Block Comments in a Shell Script: https://stackoverflow.com/questions/947897/block-comments-in-a-shell-script/947936#947936

zip
---

.. code-block:: bash

    zip -r <output.zip> <in_folder1> <in_folder2>
    zip -r test.zip /path/to/filder1

command
-------

It can check the existence of a command.

.. code-block::

  if ! command -v foobar >/dev/null; then
    echo "foobar does not exist"
  fi

cat
--

Get the number of lines in a file.

.. code-block::

  cat file | wc -l
  wc -l < file

Note that ``wc -l file`` prints two columns. The second column is the filename.

tar
---

.. code-block::

  tar cvf abc.tar /path/to/abc

- ``c``: create a new file
- ``v``: verbose

.. code-block::

  tar cvzf abc.tar.gz /path/to/abc
  tar cvjf abc.tar.bz2 /path/to/abc

  tar xvf abc.tar.bz2 # we will get a directory "abc" in the current directory
  tar xvf abc.tar.bz2 -C /path/to/here  # we will get a directory /path/to/here/abc/

  tar xvf abc.tar.bz2 --strip-components 1 -C /path/to/here   # it will strip `abc`


List files:

.. code-block::

  tar tvf abc.tar.bz2

adduser
-------

Add a user with a specified user id.

.. code-block::

  sudo adduser <username> --uid <user_id>

This command can be executed in docker.

find
----

In Makefile, to change ``./abc.cc`` to ``abc.cc``, use

.. code-block::

  srcs := $(shell find . -name "*.cc" -printf "%P\n")
  objs := $(srcs:%.cc=%.o)

Or use

.. code-block::

  srcs := $(shell find . -name "*.cc" | xargs -I{} basename {})

info
----

``info --vi-keys flex`` view the manual of ``flex`` using vi key bindings

- `ESC g`  follows the hyperlink under the current cursor. It means ``M-g``
- backtic to return backward

gzip
----

.. code-block::

  echo "1 2 3" > a.txt
  gzip a.txt

  # note that it deletes a.txt and generates a.txt.gz

  gunzip a.txt.gz
  # note that it deletes a.txt.gz and generates a.txt

  # -c means take the input from the standard input
  cat a.txt | gzip -c > abc.gz

  cat abc.gz | gunzip -c > abc

curl
----

Install the extension of chrome: ``cookies.txt``, which can generate
the cookies for a given tab.

.. code-block::

  curl --cookie cookies.txt "http://xxx.xxx.xxx/xxx.zip"
  curl -o specified_name.zip "http://xxx.xxx.xxx/xxx.zip"
  curl -O "http://xxx.xxx.xxx/xxx.zip"   # it is saved as xxx.zip

Useful options:
- `-f`, fail silently
- `-S`, show error message
- `-s`, silent, do not show progress meter
- `-Ss`, show error if it fails


ssh-server
----------

.. code-block::

  sudo apt-get install openssh-server
  sudo service ssh restart
  sudo service ssh status

sshpass
-------

.. code-block::

  sudo apt-get install sshpass
  sshpass -p "my_password" ssh user@host
  sshpass -p "my_password" scp user@host:/path/to/some/file .

rsync
-----

Useful options:
- `-P`, same as ``--progress --partial``
- `-r`, means ``--recursive``, used to copy a directory
- `-v`, means ``--verbose``

.. code-block::

  rsync -avz -r -v -P -e ssh user@remote-system:/address/to/remote/file /home/user/
  sshpass -p 'bandit0' rsync -arvzP -e ssh bandit:/tmp/xxx/t .


adduser
-------

.. code-block::

  sudo adduser user_name
  sudo usermod -aG sudo username


sox
---

Generate a wav file:

.. code-block:: bash

  sox -n -r 16000 -b 16 /tmp/test.wav synth 1 sine 100

- ``-n`` mean null file; it has no input.
- ``-r`` means ``--rate``, sample rate in Hz
- ``-b`` means ``--bits``, number of bits for each sample
- ``synth`` means synthesise

  1 means 1 second; ``sine 100`` means 100 Hz sine wave.

``soxi /tmp/test.wav`` prints::

    Input File     : '/tmp/test.wav'
    Channels       : 1
    Sample Rate    : 16000
    Precision      : 16-bit
    Duration       : 00:00:01.00 = 16000 samples ~ 75 CDDA sectors
    File Size      : 32.0k
    Bit Rate       : 256k
    Sample Encoding: 16-bit Signed Integer PCM

ffprobe
-------

- View information of a `mp3` file:

.. code-block::

  ffprobe test.mp3
  ffprobe -hide_banner test.mp3 # disable copyright information

ffmpeg
------

- Convert ``mp3`` to ``wav``:

.. code-block::

  # pcm_s16le, pcm, s16, little endian
  # -ac 1, means number of audio channels is 1
  # -ar 16000, means sample rate is 16kHz
  ffmpge -hide_banner -i test.mp3 -acodec pcm_s16le -ac 1 -ar 16000 test.wav

  # ffmpage -hide_banner -codecs | grep pcm


- Convert ``wav`` to ``mp3``

.. code-block::

  # -vn, disable video
  ffmpeg -hide_banner -i test.wav -vn -ar 44100 -ac 2 -b:a 192k test.mp3

- Convert ``mp3`` to ``ogg``

.. code-block::

  # -c:a libvorbis, select the codecs, note that ogg uses libvorbis
  # -q:a 4, the audio quality is 4
  ffmpeg -hide_banner -i test.mp3 -c:a libvorbis -q:a 4 test.ogg

- Convert ``ogg`` to ``mp3``

.. code-block::

  ffmpeg -hide_banner -i test.ogg test.mp3

- Separate channel

.. code-block::

  ffmpeg -hide_banner -i abc.wav -map_channel 0.0.0 left.wav -map_channel 0.0.1 right.wav
