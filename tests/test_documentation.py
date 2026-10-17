"""Layout checks on the two READMEs.

README.md is English, since that is the one GitHub shows, and README.zh-TW.md
is the Traditional Chinese version. They are edited together, so both get the
same checks: heading order, the lines under the title, a table of contents
that matches the headings, and links that point at files in the repository.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
README_ZH = (PROJECT_ROOT / "README.zh-TW.md").read_text(encoding="utf-8")
BOTH = pytest.mark.parametrize("readme", [README, README_ZH], ids=["README.md", "README.zh-TW.md"])

#: Every level-2 heading of the English README, in order.
EXPECTED_H2 = [
    "Contents",
    "Overview",
    "Method",
    "Installation",
    "Quick start",
    "Usage",
    "CLI reference",
    "Project layout",
    "Data preparation",
    "Configuration",
    "Results and metrics",
    "Known limitations",
    "Design notes",
    "Development and testing",
    "References",
    "License",
]

#: The same headings in the Traditional Chinese README.
EXPECTED_H2_ZH = [
    "目錄",
    "專案簡介",
    "方法說明",
    "安裝步驟",
    "快速開始",
    "使用範例",
    "CLI 指令對照表",
    "專案結構",
    "資料準備",
    "設定檔",
    "結果與評估指標",
    "已知限制",
    "設計重點",
    "開發與測試",
    "References",
    "License",
]

_CJK = re.compile(r"[\u4e00-\u9fff]")


def without_code(readme: str) -> str:
    return re.sub(r"```.*?```", "", readme, flags=re.DOTALL)


def headings(readme: str, level: int) -> list[str]:
    return re.findall(rf"^{'#' * level} (.+?)\s*$", without_code(readme), re.MULTILINE)


def github_anchor(heading: str) -> str:
    """The id GitHub gives a heading: lower case, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def links(readme: str) -> list[str]:
    return re.findall(r"\]\(([^)\s]+)\)", readme)


def check_title_badges_and_language_switch(readme: str, language_switch: str) -> None:
    lines = readme.splitlines()
    assert lines[0] == "# wifi-presence-sensing"
    assert lines[2].startswith("[![CI](") and lines[3].startswith("[![Python](")
    assert lines[4].startswith("[![License: MIT](")
    assert lines[6] == language_switch


def test_readme_headings_are_in_order():
    assert headings(README, 2) == EXPECTED_H2


def test_chinese_readme_headings_are_in_order():
    assert headings(README_ZH, 2) == EXPECTED_H2_ZH


def test_both_readmes_have_the_same_subsections():
    # Section names differ by language, but the numbered method steps, the
    # usage parts and the reference groups line up one to one.
    assert len(headings(README, 3)) == len(headings(README_ZH, 3))
    assert headings(README, 3)[-4:] == headings(README_ZH, 3)[-4:]  # reference groups


def test_readme_opens_with_badges_and_author_and_closes_with_the_license():
    check_title_badges_and_language_switch(README, "English | [繁體中文](README.zh-TW.md)")
    assert "\nAuthor: RL\n" in README
    assert README.rstrip().endswith("MIT License, Copyright (c) RL. See [LICENSE](LICENSE).")


def test_chinese_readme_opens_with_badges_and_author_and_closes_with_the_license():
    check_title_badges_and_language_switch(README_ZH, "[English](README.md) | 繁體中文")
    assert "\n作者：RL\n" in README_ZH  # noqa: RUF001
    license_line = "MIT License，Copyright (c) RL。詳見 [LICENSE](LICENSE)。"  # noqa: RUF001
    assert README_ZH.rstrip().endswith(license_line)


def test_readme_is_mostly_english_prose():
    # The only Chinese in it is the link to the Chinese README. The Chinese
    # README also has about 1,000 lowercase English words (code, config keys,
    # references), so a floor of 3,000 tells the two apart.
    assert len(_CJK.findall(README)) < 20
    assert len(re.findall(r"\b[a-z]{3,}\b", README)) > 3000


def test_chinese_readme_is_mostly_chinese_prose():
    assert len(_CJK.findall(README_ZH)) > 4000


@BOTH
def test_table_of_contents_lists_every_section(readme):
    contents = readme.split("\n## ", 2)[1]
    listed = re.findall(r"^- \[(.+?)\]\(#(.+?)\)$", contents, re.MULTILINE)
    sections = headings(readme, 2)[1:]
    assert [title for title, _ in listed] == sections
    assert [anchor for _, anchor in listed] == [github_anchor(s) for s in sections]


@BOTH
def test_every_anchor_link_has_a_heading(readme):
    anchors = {github_anchor(h) for level in (2, 3) for h in headings(readme, level)}
    missing = [t for t in links(readme) if t.startswith("#") and t[1:] not in anchors]
    assert not missing


@BOTH
def test_every_relative_link_points_at_a_file(readme):
    targets = [t for t in links(readme) if not re.match(r"[a-z]+:|#", t)]
    assert targets, "expected links to the diagram, the other README and LICENSE"
    missing = [t for t in targets if not (PROJECT_ROOT / t.split("#")[0]).is_file()]
    assert not missing


def test_each_readme_shows_the_diagram_in_its_own_language():
    online = "https://raw.githack.com/recklight/wifi-presence-sensing/master/docs/"
    assert "(docs/images/workflow.png)" in README
    assert f"({online}workflow.html?theme=dark)" in README
    assert "(docs/workflow.html)" in README
    assert "(docs/images/workflow.zh-TW.png)" in README_ZH
    assert f"({online}workflow.zh-TW.html?theme=dark)" in README_ZH
    assert "(docs/workflow.zh-TW.html)" in README_ZH
    assert "workflow.zh-TW" not in "".join(links(README))
    assert not re.search(r"workflow\.(png|html)", "".join(links(README_ZH)))


def test_formulas_and_references_are_identical_in_both():
    display_math = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
    assert display_math.findall(README) == display_math.findall(README_ZH)

    def references(readme: str) -> str:
        return readme[readme.index("## References") : readme.index("## License")]

    assert references(README) == references(README_ZH)
