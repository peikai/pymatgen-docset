"""
Pyinvoke tasks.py file for automating releases and admin stuff.

To cut a new pymatgen release:

    invoke update-changelog
    invoke release
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import webbrowser
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import requests
from invoke import task
from monty.os import cd

from pymatgen.core import __version__

if TYPE_CHECKING:
    from invoke import Context


@task
def make_doc(ctx: Context) -> None:
    """
    Generate API documentation + run Sphinx.

    Args:
        ctx (Context): The context.
    """
    with cd("docs"):
        ctx.run("sphinx-apidoc --implicit-namespaces -M -d 7 -o apidoc -f ../src/pymatgen ../**/tests/*")
        ctx.run("sphinx-build -b html apidoc html")  # HTML building.
        # ctx.run('sed -i "s/_static/assets/g" html/pymatgen*.html')

@task
def make_dash(ctx):
    """
    Make customized doc version for Dash

    :param ctx:
    """
    ctx.run("cp conf-docset.py ./pymatgen/docs/apidoc/conf.py")
    ctx.run("cp index.rst ./pymatgen/docs/apidoc")
    ctx.run("sphinx-apidoc --implicit-namespaces -M -d 7 -o ./pymatgen/docs/apidoc -f ./pymatgen/src/pymatgen ./pymatgen/**/tests/*")
    ctx.run("sphinx-build -b html ./pymatgen/docs/apidoc ./pymatgen/docs/html")
    ctx.run("doc2dash ./pymatgen/docs/html -n pymatgen -i ./icon.png -u https://pymatgen.org/")
    plist = "pymatgen.docset/Contents/Info.plist"
    xml = []
    with open(plist) as f:
        for l in f:
            xml.append(l.strip())
            if l.strip() == "<dict>":
                xml.append("<key>dashIndexFilePath</key>")
                xml.append("<string>index.html</string>")
    with open(plist, "w") as f:
        f.write("\n".join(xml))


@task
def publish(ctx: Context) -> None:
    """
    Upload release to Pypi using twine.

    Args:
        ctx (Context): The context.
    """
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")


@task
def set_ver(ctx: Context, version: str):
    """
    Set version in pyproject.toml file.

    Args:
        ctx (Context): The context.
        version (str): An input version.
    """
    with open("pyproject.toml", encoding="utf-8") as file:
        lines = [re.sub(r"^version = \"([^,]+)\"", f'version = "{version}"', line.rstrip()) for line in file]

    with open("pyproject.toml", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    ctx.run("ruff check --fix src")
    ctx.run("ruff format pyproject.toml")


@task
def release_github(ctx: Context, version: str) -> None:
    """
    Release to Github using Github API.

    Args:
        ctx (Context): The context.
        version (str): The version.
    """
    with open("docs/CHANGES.md", encoding="utf-8") as file:
        contents = file.read()
    tokens = re.split(r"\n\#\#\s", contents)
    desc = tokens[1].strip()
    tokens = desc.split("\n")
    desc = "\n".join(tokens[1:]).strip()
    payload = {
        "tag_name": f"v{version}",
        "target_commitish": "master",
        "name": f"v{version}",
        "body": desc,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialsproject/pymatgen/releases",
        data=json.dumps(payload),
        headers={"Authorization": f"token {os.environ['GITHUB_RELEASES_TOKEN']}"},
        timeout=60,
    )
    print(response.text)


@task
def update_changelog(ctx: Context, version: str | None = None, dry_run: bool = False) -> None:
    """Create a preliminary change log using the git logs.

    Args:
        ctx (invoke.Context): The context object.
        version (str, optional): The version to use for the change log. If not provided, it will
            use the current date in the format 'YYYY.M.D'. Defaults to None.
        dry_run (bool, optional): If True, the function will only print the changes without
            updating the actual change log file. Defaults to False.
    """
    version = version or f"{datetime.now(tz=timezone.utc):%Y.%-m.%-d}"
    print(f"Getting all comments since {__version__}")
    output = subprocess.check_output(["git", "log", "--pretty=format:%s", f"v{__version__}..HEAD"])
    lines = []
    ignored_commits = []
    for line in output.decode("utf-8").strip().split("\n"):
        re_match = re.match(r".*\(\#(\d+)\)", line)
        if re_match and "materialsproject/dependabot/pip" not in line:
            pr_number = re_match[1].strip()
            response = requests.get(
                f"https://api.github.com/repos/materialsproject/pymatgen/pulls/{pr_number}",
                timeout=60,
            )
            resp = response.json()
            lines += [f"- PR #{pr_number} {resp['title'].strip()} by @{resp['user']['login']}"]
            if body := resp["body"]:
                for ll in map(str.strip, body.split("\n")):
                    if ll in ("", "## Summary"):
                        continue
                    if ll.startswith(("## Checklist", "## TODO")):
                        break
                    lines += [f"    {ll}"]
        else:
            ignored_commits += [line]

    body = "\n".join(lines)
    try:
        # Use OpenAI to improve changelog. Requires openai to be installed and an OPENAPI_KEY env variable.
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAPI_KEY"])

        messages = [{"role": "user", "content": f"summarize as a markdown numbered list, include authors: '{body}'"}]
        chat = client.chat.completions.create(model="gpt-4o", messages=messages)

        reply = chat.choices[0].message.content
        body = "\n".join(reply.split("\n")[1:-1])
        body = body.strip().strip("`")
        print(f"ChatGPT Summary of Changes:\n{body}")

    except BaseException as ex:
        print(f"Unable to use openai due to {ex}")
    with open("docs/CHANGES.md", encoding="utf-8") as file:
        contents = file.read()
    delim = "##"
    tokens = contents.split(delim)
    tokens.insert(1, f"## v{version}\n\n{body}\n\n")
    if dry_run:
        print(tokens[0] + "##".join(tokens[1:]))
    else:
        with open("docs/CHANGES.md", mode="w", encoding="utf-8") as file:
            file.write(tokens[0] + "##".join(tokens[1:]))
        ctx.run("open docs/CHANGES.md")
    print("The following commit messages were not included...")
    print("\n".join(ignored_commits))


@task
def release(ctx: Context, version: str | None = None, nodoc: bool = False) -> None:
    """
    Run full sequence for releasing pymatgen.

    Args:
        ctx (invoke.Context): The context object.
        version (str, optional): The version to release.
        nodoc (bool, optional): Whether to skip documentation generation.
    """
    version = version or f"{datetime.now(tz=timezone.utc):%Y.%-m.%-d}"
    ctx.run("rm -r dist build pymatgen.egg-info", warn=True)
    set_ver(ctx, version)
    if not nodoc:
        make_doc(ctx)
        ctx.run("git add .")
        ctx.run('git commit --no-verify -a -m "Update docs"')
        ctx.run("git push")
    release_github(ctx, version)

    ctx.run("rm -f dist/*.*", warn=True)
    ctx.run("pip install -e .", warn=True)
    ctx.run("python -m build", warn=True)
    ctx.run("twine upload --skip-existing dist/*.whl", warn=True)
    ctx.run("twine upload --skip-existing dist/*.tar.gz", warn=True)
    # post_discourse(ctx, warn=True)


@task
def open_doc(ctx: Context) -> None:
    """
    Open local documentation in web browser.

    Args:
        ctx (invoke.Context): The context object.
    """
    pth = os.path.abspath("docs/_build/html/index.html")
    webbrowser.open(f"file://{pth}")


@task
def lint(ctx: Context) -> None:
    """
    Run linting tools.

    Args:
        ctx (invoke.Context): The context object.
    """
    for cmd in ("ruff", "mypy", "ruff format"):
        ctx.run(f"{cmd} pymatgen")
