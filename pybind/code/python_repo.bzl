# this file is modified from https://github.com/tensorflow/tensorflow/blob/master/third_party/py/python_configure.bzl

# Date: Sun Mar 15 20:50:01 CST 2020

def _fail(msg):
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("{}Error:{} {}\n".format(red, no_color, msg))

def _which(repository_ctx, program_name):
    result = repository_ctx.execute(["which", program_name])
    return result

def _get_python_bin(repository_ctx, python_interpreter):
    python_bin = _which(repository_ctx, python_interpreter)
    if python_bin == None:
        _fail("Cannot find python in PATH")
    return python_bin.stdout.splitlines()[0]

def _get_bash_bin(repository_ctx):
    bash_bin = _which(repository_ctx, "bash")
    if bash_bin == None:
        _fail("Cannot find bash in PATH")
    return bash_bin.stdout.splitlines()[0]

def _check_python_bin(repository_ctx, path):
    cmd = '[[ -x "{}" ]] && [[ ! -d "{}" ]]'.format(path, path)
    bash_bin = _get_bash_bin(repository_ctx)
    result = repository_ctx.execute([bash_bin, "-c", cmd])
    if result.return_code == 1:
        _fail("Invalid python path: {}".format(path))

def _get_python_include(repository_ctx, python_bin):
    cmd = [
        python_bin,
        "-c",
        "from __future__ import print_function;" +
        "from distutils import sysconfig;" +
        "print(sysconfig.get_python_inc())",
    ]

    result = repository_ctx.execute(cmd)
    if result.stderr:
        _fail("Failed to get Python include path")

    return result.stdout.splitlines()[0]

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def _read_dir(repository_ctx, src_dir):
    """Returns a sorted list with all files in a directory.

    Finds all files inside a directory, traversing subfolders and following
    symlinks.

    Args:
      repository_ctx: the repository_ctx
      src_dir: the directory to traverse

    Returns:
      A sorted list with all files in a directory.
    """
    find_result = repository_ctx.execute(["find", src_dir, "-follow", "-type", "f"])
    if find_result == None:
        _fail("Failed to get files form directory {}".format(src_dir))
    result = find_result.stdout
    return sorted(result.splitlines())

def _genrule(src_dir, genrule_name, command, outs):
    """Returns a string with a genrule.

    Genrule executes the given command and produces the given outputs.
    """
    return (
        "genrule(\n" +
        '    name = "' +
        genrule_name + '",\n' +
        "    outs = [\n" +
        outs +
        "\n    ],\n" +
        '    cmd = """\n' +
        command +
        '\n   """,\n' +
        ")\n"
    )

def _symlink_genrule_for_dir(
        repository_ctx,
        src_dir,
        dest_dir,
        genrule_name):
    src_dir = _norm_path(src_dir)
    dest_dir = _norm_path(dest_dir)

    files = "\n".join(_read_dir(repository_ctx, src_dir))
    dest_files = files.replace(src_dir, "").splitlines()
    src_files = files.splitlines()
    command = []
    outs = []
    for i in range(len(dest_files)):
        if dest_files[i] != "":
            # If we have only one file to link we do not want to use the dest_dir, as
            # $(@D) will include the full path to the file.
            dest = "$(@D)/" + dest_dir + dest_files[i] if len(dest_files) != 1 else "$(@D)/" + dest_files[i]

            # Copy the headers to create a sandboxable setup.
            cmd = "cp -f"
            command.append(cmd + ' "{}" "{}"'.format(src_files[i], dest))
            outs.append('        "' + dest_dir + dest_files[i] + '",')
    genrule = _genrule(
        src_dir,
        genrule_name,
        " && ".join(command),
        "\n".join(outs),
    )
    return genrule

def _python_repo_impl(repository_ctx):
    build_tpl = repository_ctx.path(Label("//:BUILD.tpl"))
    python_bin = _get_python_bin(repository_ctx, repository_ctx.attr.python_interpreter)
    print(python_bin)
    _check_python_bin(repository_ctx, python_bin)
    python_include = _get_python_include(repository_ctx, python_bin)
    python_include_rule = _symlink_genrule_for_dir(
        repository_ctx,
        python_include,
        "python_include",
        "python_include",
    )
    repository_ctx.template("BUILD", build_tpl, {
        "%{PYTHON_INCLUDE_GENRULE}": python_include_rule,
    })

python_repo = repository_rule(
    implementation = _python_repo_impl,
    attrs = {
        "python_interpreter": attr.string(default = "python3"),
    },
)
