"""Push Polaris to GitHub via gh CLI + Git Data API.

Why not plain `git push`? In this network github.com:443 is blocked while
api.github.com is reachable, so the git transport cannot complete a push.
The Git Data API over api.github.com is the working path.

Flow:
  1. read the current refs/heads/main (if any) -> parent sha
  2. create blobs ONLY for files changed since that sha (unchanged blobs
     already exist remotely; Git dedupes by content sha anyway)
  3. build the full tree from the local index
  4. create a commit with the local commit message, parent = current main
  5. PATCH refs/heads/main

Idempotent: re-running produces the same tree content.
"""

import base64
import json
import subprocess
import sys

R = "repos/CJX0712/polaris-flagship-ai"

def gh_api(endpoint, payload=None, method=None, allow_fail=False):
    cmd = ["gh", "api", endpoint]
    if method:
        cmd += ["-X", method]
    if payload is not None:
        cmd += ["--input", "-"]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=json.dumps(payload) if payload is not None else None,
    )
    if r.returncode != 0:
        if allow_fail:
            return ""
        print("API FAIL:", endpoint, r.stderr[:400])
        sys.exit(1)
    return r.stdout.strip()


def current_main_sha():
    """Current refs/heads/main sha, or '' if the repo is empty."""
    out = gh_api(f"{R}/git/ref/heads/main", allow_fail=True)
    if not out:
        return ""
    return json.loads(out)["object"]["sha"]


def local_commit_message():
    return subprocess.check_output(
        ["git", "log", "-1", "--pretty=%B"]
    ).decode().strip()


def remote_tree_blobs(parent_sha):
    """Map path -> blob sha for the remote main tree ({}, if the repo is empty).

    The remote commit was created through the API, so its sha does NOT exist in
    the local object database -- `git diff <remote_sha> HEAD` cannot work.
    Git blob shas, however, are content-addressed and identical on both sides
    (sha1("blob <len>\\0" + content)), so comparing blob shas is exact.
    """
    if not parent_sha:
        return {}
    commit = json.loads(gh_api(f"{R}/git/commits/{parent_sha}"))
    tree_sha = commit["tree"]["sha"]
    tree = json.loads(gh_api(f"{R}/git/trees/{tree_sha}?recursive=1"))
    return {
        e["path"]: e["sha"]
        for e in tree.get("tree", [])
        if e.get("type") == "blob"
    }


def local_blob_map():
    """Map path -> local blob sha (from the index; everything is committed)."""
    out = subprocess.check_output(["git", "ls-files", "-s"]).decode()
    m = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        sha = meta.split()[1]
        m[path] = sha
    return m


def main():
    # The push mirrors the INDEX, so the index must be in sync with the
    # working tree -- otherwise we would silently publish stale content.
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"]
    ).decode().strip()
    if dirty:
        print("ABORT: working tree has uncommitted changes:")
        print(dirty[:600])
        print("\nStage them first:  git add -A")
        sys.exit(2)

    parent = current_main_sha()
    print("remote main:", parent or "(empty repo)")

    files = subprocess.check_output(["git", "ls-files"]).decode().split()
    remote = remote_tree_blobs(parent)
    local = local_blob_map()
    todo = [
        f
        for f in files
        if remote.get(f) != local.get(f)
    ]
    print(f"tracked files: {len(files)} | changed: {len(todo)}")

    entries = []
    for i, f in enumerate(files):
        if f in todo:
            # Read the blob content from the INDEX (git cat-file), NOT the
            # working tree: the index holds the canonical, LF-normalized bytes
            # (see .gitattributes), while a Windows working tree may hold CRLF.
            # Pushing the working tree would store CRLF in the repo and break
            # byte-exact fidelity with the vendored upstream ifai.
            data = subprocess.check_output(["git", "cat-file", "blob", local[f]])
            content = base64.b64encode(data).decode()
            sha = json.loads(
                gh_api(f"{R}/git/blobs", {"content": content, "encoding": "base64"})
            )["sha"]
            if sha != local[f]:
                print(f"  WARN blob sha mismatch for {f}: {sha} != {local[f]}")
            if i % 10 == 0 or i == len(files) - 1:
                print(f"  blob [{i + 1}/{len(files)}] {f}")
        else:
            sha = remote[f]
        entries.append({"path": f, "mode": "100644", "type": "blob", "sha": sha})

    tree_sha = json.loads(gh_api(f"{R}/git/trees", {"tree": entries}))["sha"]
    print("tree:", tree_sha)

    payload = {"message": local_commit_message(), "tree": tree_sha}
    if parent:
        payload["parents"] = [parent]
    commit_sha = json.loads(gh_api(f"{R}/git/commits", payload))["sha"]
    print("commit:", commit_sha)

    gh_api(
        f"{R}/git/refs/heads/main",
        {"sha": commit_sha, "force": True},
        method="PATCH",
    )
    print("PUSHED: refs/heads/main ->", commit_sha)


if __name__ == "__main__":
    main()
