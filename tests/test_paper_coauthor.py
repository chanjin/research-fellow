import json
from research_fellow.application.paper_coauthor import (
    apply_appendix_refresh, apply_review, apply_revisions, diagnostics, parse_manuscript, parse_review,
    manuscript_markdown,
    parse_resolution_proposal, parse_todo_verification, review_change_set, revision_todo_timeline, revision_todos,
    search_revision_assets, targeted_revision_context,
)


def test_revision_asset_search_covers_shelf_and_approved_knowledge():
    papers=[
        {"paper_id":"paper-old","title":"Runtime verification for industrial agents","abstract":"Sandbox safety checks","authors":["Kim"]},
        {"paper_id":"paper-other","title":"Open-ended game agents","abstract":"Creative simulation"},
    ]
    cards=[
        {"card_id":"card-old","title":"External runtime checks","claim":"Runtime verification reduces false positives","concepts":["sandbox"]},
        {"card_id":"card-other","title":"Persona design","claim":"Role descriptions guide behavior"},
    ]
    matched_papers,matched_cards=search_revision_assets("runtime verification",papers,cards)
    assert [item["paper_id"] for item in matched_papers]==["paper-old"]
    assert [item["card_id"] for item in matched_cards]==["card-old"]
    assert search_revision_assets("",papers,cards)==([],[])


def test_short_paper_sentence_markup_and_targeted_revision():
    raw=json.dumps({"title":"T","sections":[{"section_id":"intro","title":"Intro","paragraphs":[{"sentences":[{"sentence_id":"s-001","text":"Strong claim.","role":"central_claim","evidence_card_ids":["kc-1","bad"],"annotations":[]}]}]}]})
    manuscript=parse_manuscript(raw,valid_card_ids={"kc-1","kc-2"},version=1)
    assert manuscript["sections"][0]["paragraphs"][0]["sentences"][0]["evidence_card_ids"]==["kc-1"]
    review=parse_review(json.dumps({"annotations":[{"sentence_id":"s-001","type":"overclaim","severity":"high","reason":"Too strong","recommended_action":"narrow_scope","literature_search_candidates":[{"title":"관련 문헌 탐색","target":"Under what conditions does the claim fail?","research_context":"Test the claim boundary.","expected_evidence":"Counterexamples and boundary conditions.","completion_condition":"At least one grounded boundary or explicit absence is reported."}]}]}),manuscript)
    assert review[0]["type"]=="overclaim"
    assert review[0]["search_guide"]
    assert review[0]["completion_criteria"]
    manuscript["sections"][0]["paragraphs"][0]["sentences"][0]["annotations"]=[{k:v for k,v in review[0].items() if k!="sentence_id"}]
    todo=revision_todos(manuscript)[0]
    assert todo["priority"]=="P0"
    assert todo["label"]=="과잉 주장 조정"
    assert todo["literature_search_candidates"][0]["target"].startswith("Under what conditions")
    assert todo["literature_search_candidates"][0]["title"].startswith("Under what conditions")
    revision=json.dumps({"revisions":[{"sentence_id":"s-001","after":"Bounded claim.","reason":"Scope","resolved_annotation_ids":[review[0]["annotation_id"]],"new_evidence_card_ids":["kc-2"]}]})
    revised,diff=apply_revisions(revision,manuscript,valid_card_ids={"kc-1","kc-2"},version=2)
    assert diagnostics(revised)[0]["보완점"]=="완료"
    assert revision_todos(revised)[0]["status"]=="revised_pending_review"
    assert diff[0]["before"]=="Strong claim."
    verified=apply_review(revised,[])
    assert revision_todos(verified)==[]


def test_diagnostics_show_content_and_targeted_revision_scope():
    manuscript={"version":1,"title":"Paper","sections":[{"section_id":"s","title":"S","paragraphs":[{"paragraph_id":"p","sentences":[
        {"sentence_id":"s-001","text":"Previous context.","role":"supporting","evidence_card_ids":[],"annotations":[]},
        {"sentence_id":"s-002","text":"Target claim.","role":"central_claim","evidence_card_ids":["kc-1"],"annotations":[]},
        {"sentence_id":"s-003","text":"Next context.","role":"supporting","evidence_card_ids":[],"annotations":[]},
        {"sentence_id":"s-004","text":"UNRELATED FULL PAPER CONTENT.","role":"limitation","evidence_card_ids":[],"annotations":[]},
    ]}]}]}
    rows=diagnostics(manuscript,card_titles={"kc-1":"Evidence title"})
    assert rows[1]["문장 내용"]=="Target claim."
    assert rows[1]["연결 근거"]=="Evidence title"
    comments=[{"todo_id":"a1","sentence_id":"s-002","problem":"Needs boundary","completion_criteria":"Bounded"}]
    context=targeted_revision_context(manuscript,comments)
    assert context[0]["sentence"]["sentence_id"]=="s-002"
    assert context[0]["previous_sentence"]["sentence_id"]=="s-001"
    assert context[0]["next_sentence"]["sentence_id"]=="s-003"
    assert all(item.get("sentence_id")!="s-004" for item in (
        context[0]["sentence"],context[0]["previous_sentence"],context[0]["next_sentence"],
    ))
    response=json.dumps({"revisions":[
        {"sentence_id":"s-002","after":"Bounded target.","resolved_annotation_ids":[],"new_evidence_card_ids":[]},
        {"sentence_id":"s-004","after":"Illicit rewrite.","resolved_annotation_ids":[],"new_evidence_card_ids":[]},
    ]})
    revised,diff=apply_revisions(response,manuscript,valid_card_ids=set(),version=2,allowed_sentence_ids={"s-002"})
    assert [item["sentence_id"] for item in diff]==["s-002"]
    assert revised["sections"][0]["paragraphs"][0]["sentences"][3]["text"]=="UNRELATED FULL PAPER CONTENT."


def test_whole_paper_impact_revision_can_update_multiple_grounded_sentences():
    manuscript={"version":2,"title":"Paper","sections":[{"section_id":"s","title":"S","paragraphs":[{"paragraph_id":"p","sentences":[
        {"sentence_id":"s-001","text":"Original claim.","role":"central_claim","evidence_card_ids":[],"annotations":[]},
        {"sentence_id":"s-002","text":"Original conclusion.","role":"conclusion","evidence_card_ids":[],"annotations":[]},
    ]}]}]}
    response=json.dumps({"revisions":[
        {"sentence_id":"s-001","after":"Bounded claim.","reason":"New evidence","new_evidence_card_ids":["kc-new"]},
        {"sentence_id":"s-002","after":"Bounded conclusion.","reason":"Consistency","new_evidence_card_ids":["kc-new"]},
    ]})
    revised,diff=apply_revisions(response,manuscript,valid_card_ids={"kc-new"},version=3)
    assert [item["sentence_id"] for item in diff]==["s-001","s-002"]
    assert all("kc-new" in sentence["evidence_card_ids"] for sentence in revised["sections"][0]["paragraphs"][0]["sentences"])


def test_short_paper_preserves_important_omitted_claims_in_appendix():
    raw=json.dumps({
        "title":"Short Paper","sections":[{"section_id":"s","title":"S","paragraphs":[{"sentences":[
            {"sentence_id":"s-001","text":"Core short-paper claim.","evidence_card_ids":["kc-core"],"annotations":[]},
        ]}]}],
        "appendix_claims":[
            {"appendix_id":"a-01","title":"Scaling boundary","claim":"The control architecture may require a different boundary at larger scale.","why_excluded":"Not essential to the two-page answer.","full_paper_value":"Develop as a boundary-condition section.","evidence_card_ids":["kc-scale"]},
            {"appendix_id":"a-invalid","title":"Unsupported","claim":"This unsupported claim must be removed.","evidence_card_ids":["missing"]},
        ],
    })
    manuscript=parse_manuscript(raw,valid_card_ids={"kc-core","kc-scale"},version=1)
    assert [item["appendix_id"] for item in manuscript["appendix_claims"]]==["a-01"]
    rendered=manuscript_markdown(manuscript)
    assert "Appendix · Full Paper 확장 후보 주장" in rendered
    assert "Scaling boundary" in rendered

    response=json.dumps({"revisions":[],"appendix_updates":[{
        "action":"add","appendix_id":"a-02","title":"Evaluation extension",
        "claim":"A Full Paper should compare runtime and governance outcomes together.",
        "why_excluded":"Requires a larger evaluation section.",
        "full_paper_value":"Use as the evaluation design claim.",
        "evidence_card_ids":["kc-eval"],"reason":"New reviewed evidence",
    }]})
    revised,diff=apply_revisions(response,manuscript,valid_card_ids={"kc-core","kc-scale","kc-eval"},version=2)
    assert [item["appendix_id"] for item in revised["appendix_claims"]]==["a-01","a-02"]
    assert diff[0]["change_scope"]=="appendix"

    refresh=json.dumps({"appendix_claims":[{
        "appendix_id":"a-01","title":"Scaling boundary revised",
        "claim":"The control architecture requires explicit scale-dependent validation boundaries.",
        "why_excluded":"Secondary to the short-paper answer.",
        "full_paper_value":"Develop as a scalability limitation.",
        "evidence_card_ids":["kc-scale"],
    }]})
    refreshed,refresh_diff=apply_appendix_refresh(
        refresh,revised,valid_card_ids={"kc-core","kc-scale","kc-eval"},version=3,
    )
    assert [item["appendix_id"] for item in refreshed["appendix_claims"]]==["a-01"]
    assert len(refresh_diff)==2


def test_review_change_receipt_and_todo_progress():
    before=[
        {"todo_id":"a1","sentence_id":"s-1","type":"overclaim","label":"과잉 주장 조정","problem":"too strong"},
        {"todo_id":"a2","sentence_id":"s-2","type":"citation_needed","label":"인용 확인 필요","problem":"citation"},
    ]
    after=[
        {"annotation_id":"a2","sentence_id":"s-2","type":"citation_needed","comment":"still needed"},
        {"annotation_id":"a3","sentence_id":"s-3","type":"logic_gap","comment":"missing link"},
    ]
    changes=review_change_set(before,after)
    assert changes["retained_todos"][0]["todo_id"]=="a2"
    assert changes["new_todos"][0]["todo_id"]=="a3"
    assert changes["resolved_todos"][0]["todo_id"]=="a1"

    active=[{"todo_id":"a3","sentence_id":"s-3","label":"논리 연결 보완","priority":"P1","status":"open"}]
    updates=[{"todo_id":"a1","sentence_id":"s-1","label":"과잉 주장 조정","priority":"P0","status":"resolved"}]
    timeline=revision_todo_timeline(active,updates)
    assert [item["todo_id"] for item in timeline]==["a3","a1"]
    assert timeline[-1]["status"]=="resolved"


def test_todo_completion_verification_contract():
    todo={"todo_id":"a1","sentence_id":"s-002","completion_criteria":"Claim is bounded."}
    result=parse_todo_verification(json.dumps({
        "todo_id":"a1","sentence_id":"s-002","verdict":"needs_more_work",
        "reason":"No source supports the boundary.",
        "criteria_checks":[{"criterion":"Claim is bounded.","satisfied":False,"evidence":"No linked card"}],
        "remaining_gap":"Find an empirical boundary condition.","next_action":"literature_search",
    }),todo)
    assert result["verdict"]=="needs_more_work"
    assert result["next_action"]=="literature_search"
    assert result["criteria_checks"][0]["satisfied"] is False
    try:
        parse_todo_verification(json.dumps({
            "todo_id":"other","sentence_id":"s-002","verdict":"resolved","next_action":"close",
        }),todo)
    except ValueError as error:
        assert "todo_id" in str(error)
    else:
        raise AssertionError("mismatched todo_id must be rejected")


def test_resolution_proposal_requires_grounded_target_change_and_supports_followup_search():
    todo={"todo_id":"a1","sentence_id":"s-002","completion_criteria":"Claim is grounded."}
    resolved=parse_resolution_proposal(json.dumps({
        "todo_id":"a1","verdict":"resolved","reason":"The selected card supplies the missing boundary.",
        "revisions":[{"sentence_id":"s-002","after":"Bounded, grounded claim.","reason":"Evidence added","new_evidence_card_ids":["kc-1","bad"]}],
        "suggested_reference_paper_ids":["paper-1","bad"],"next_search":{},
    }),todo,valid_card_ids={"kc-1"},valid_paper_ids={"paper-1"})
    assert resolved["revisions"][0]["new_evidence_card_ids"]==["kc-1"]
    assert resolved["revisions"][0]["resolved_annotation_ids"]==["a1"]
    assert resolved["suggested_reference_paper_ids"]==["paper-1"]

    followup=parse_resolution_proposal(json.dumps({
        "todo_id":"a1","verdict":"needs_more_work","reason":"No counterexample evidence.","revisions":[],
        "next_search":{"title":"Failure conditions for bounded autonomy","target":"Which failure conditions are reported?","research_context":"The current card covers benefits but no failures.","expected_evidence":"Empirical failures and operating conditions.","completion_condition":"At least one grounded failure condition is found."},
    }),todo,valid_card_ids={"kc-1"},valid_paper_ids={"paper-1"})
    assert followup["verdict"]=="needs_more_work"
    assert followup["next_search"]["title"].startswith("Failure conditions")
