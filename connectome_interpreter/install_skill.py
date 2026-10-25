"""Install the bundled Agent Skill so AI coding agents (e.g. Claude Code) can use this
package well.

Usage:
    python -m connectome_interpreter.install_skill            # ~/.claude/skills/
    python -m connectome_interpreter.install_skill --dest DIR # another skills folder

Re-run after upgrading the package to update the skill.
"""

import argparse
import shutil
from importlib import resources
from pathlib import Path

SKILL_NAME = "connectome-interpreter"


def install_skill(dest=None) -> Path:
    """Copy the bundled skill folder into a skills directory, replacing any older copy.

    Args:
        dest (str or Path, optional): The skills directory to install into. Defaults
            to ~/.claude/skills (the personal skills folder of Claude Code).

    Returns:
        Path: The installed skill folder.
    """
    dest = Path(dest).expanduser() if dest else Path.home() / ".claude" / "skills"
    target = dest / SKILL_NAME
    with resources.as_file(
        resources.files("connectome_interpreter") / "skills" / SKILL_NAME
    ) as src:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", help="skills directory (default: ~/.claude/skills)")
    print(f"Installed skill to {install_skill(parser.parse_args().dest)}")


if __name__ == "__main__":
    main()
