"""Deterministic local search helpers used by Revision To-do capabilities."""
from __future__ import annotations
import re
from typing import Any

def search_revision_assets(
    query: str, papers: list[dict[str,Any]], cards: list[dict[str,Any]], *, limit: int = 30,
) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    """Rank existing shelf papers and approved cards for one Revision To-do."""
    normalized=re.sub(r"\s+"," ",query).strip().casefold()
    terms=list(dict.fromkeys(re.findall(r"[0-9A-Za-z가-힣_-]{2,}",normalized)))
    if not terms:return [],[]

    def rank(item: dict[str,Any], *, paper: bool) -> tuple[int,str]:
        if paper:
            title=str(item.get("title") or "")
            body=" ".join([
                title,str(item.get("abstract") or item.get("summary") or ""),
                " ".join(str(value) for value in item.get("authors",[]) or []),
                " ".join(str(value) for value in item.get("labels",[]) or []),
            ])
        else:
            title=str(item.get("title") or "")
            body=" ".join([
                title,str(item.get("claim") or ""),str(item.get("context") or ""),
                str(item.get("conditions") or ""),str(item.get("limits") or ""),
                " ".join(str(value) for value in item.get("labels",[]) or []),
                " ".join(str(value) for value in item.get("concepts",[]) or []),
            ])
        title_fold=title.casefold();body_fold=body.casefold()
        score=sum(4 if term in title_fold else 1 for term in terms if term in body_fold)
        if normalized and normalized in body_fold:score+=6
        return score,body_fold

    def select(items: list[dict[str,Any]], *, paper: bool) -> list[dict[str,Any]]:
        ranked=[(rank(item,paper=paper)[0],index,item) for index,item in enumerate(items)]
        ranked=[row for row in ranked if row[0]>0]
        ranked.sort(key=lambda row:(-row[0],row[1]))
        return [row[2] for row in ranked[:max(1,int(limit))]]

    return select(papers,paper=True),select(cards,paper=False)
