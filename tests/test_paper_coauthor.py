import json
import sys
import types
from pathlib import Path
from research_fellow.application.paper_coauthor import (
    apply_appendix_refresh, apply_review, apply_revisions, diagnostics, parse_manuscript, parse_review,
    manuscript_markdown,
    parse_group_resolution, parse_resolution_proposal, parse_todo_group_plan, parse_todo_verification,
    parse_paper_proposal, parse_writing_spec_guidance, parse_todo_reconciliation,
    parse_revision_resolution_plan,
    review_change_set, revision_todo_timeline, revision_todos, resolution_proposal_prompt,
    revision_focus_context, search_revision_assets, selected_group_revisions, targeted_revision_context,
)


def test_paper_proposal_separates_internal_evidence_from_model_prior():
    result=parse_paper_proposal(json.dumps({
        "planning_summary":"The framing needs a narrower industrial boundary.",
        "internal_similarity_assessment":[{
            "subject":"Runtime verification","assessment":"The shelf paper covers callback verification.",
            "evidence_status":"grounded_internal","source_ids":["paper-1","kc-1"],
        }],
        "external_landscape_assessment":[{
            "subject":"Agent governance literature","assessment":"Potential overlap is plausible.",
            "evidence_status":"model_prior","verification_need":"Run M1 search.",
        }],
        "novelty_risks":[{"subject":"Broad autonomy claim","assessment":"May be established.","evidence_status":"unknown"}],
        "candidate_pairs":[{
            "candidate_id":"candidate-1","title":"Bounded Industrial Agents",
            "research_question":"How do explicit job boundaries affect verification?",
            "contribution":"A job-level control account.","scope":"High-risk industrial settings.",
        }],
        "recommended_candidate_id":"missing",
        "m1_verification_candidates":[{
            "title":"Job-boundary controls in industrial agents","target":"Which controls have been evaluated?",
            "research_context":"Verify novelty.","expected_evidence":"Closest empirical studies.",
            "completion_condition":"Compare at least three close studies.",
        }],
        "researcher_decisions":[{"question":"Limit the scope?","options":["Manufacturing","All industry"]}],
    }))
    assert result["internal_similarity_assessment"][0]["evidence_status"]=="grounded_internal"
    assert result["external_landscape_assessment"][0]["evidence_status"]=="model_prior"
    assert result["novelty_risks"][0]["evidence_status"]=="verification_required"
    assert result["recommended_candidate_id"]=="candidate-1"
    assert result["m1_verification_candidates"][0]["candidate_id"]=="proposal-search-1"


def test_paper_proposal_template_uses_safe_optional_field_access():
    template=(
        Path(__file__).parents[1] / "src" / "research_fellow" / "prompts" /
        "m2_paper_proposal_review.j2"
    ).read_text(encoding="utf-8")
    assert "card.context" not in template
    assert "card.get('context', '')" in template
    assert "paper.get('analysis_summary', '')" in template


def test_writing_spec_guidance_normalizes_claims_and_filters_unknown_cards():
    result=parse_writing_spec_guidance(json.dumps({
        "guidance_summary":"Confirmed framing supports a bounded claim.",
        "audience":"Agent engineering researchers",
        "central_claim_candidates":[
            {"claim_id":"strong","claim":"Explicit job boundaries improve verifiability.",
             "rationale":"It answers the RQ.","evidence_card_ids":["kc-1","missing"],
             "risk_or_condition":"Requires empirical evaluation."},
            "A control contract is more useful than a persona alone.",
        ],
        "recommended_claim_id":"strong","target_min_chars":"bad","target_max_chars":8500,
        "research_design_candidates":[{
            "design_id":"case","method":"비교 사례연구","compatible_claim_ids":["strong","missing"],
            "rationale":"실행 경계의 차이를 trace로 비교한다.",
            "verification_plan":[{
                "research_question":"RQ1","claim_to_verify":"Explicit boundaries improve verification.",
                "method":"동일 업무 비교","baseline":"페르소나 기반 에이전트","unit_of_analysis":"task run",
                "required_data":["실행 trace","승인 기록"],
                "metrics":[{"name":"월권 발생률","definition":"월권 실행/전체 실행","example_record":"baseline | [값]"}],
                "analysis_method":"조건별 비율 비교","success_criteria":"제안 방식에서 월권 감소",
                "falsification_condition":"차이가 없거나 증가","validity_threats":["시나리오 편향"]
            }],
            "execution_guide":["동일 업무 시나리오를 고정한다."],
            "result_recording_plan":[{"artifact":"RQ 비교표","fields":["조건","관찰값"],"example_row":"[조건] | [값]"}]
        }],
        "recommended_design_id":"case",
        "literature_search_candidates":[{"title":"에이전트 월권 측정 지표","target":"월권 행동은 어떻게 측정되는가?"}],
        "required_section_terms":["서론","제안 방법","서론"],
        "writing_rules":"수치는 근거가 있을 때만 쓴다.\n핵심 용어를 먼저 정의한다.",
        "open_decisions":["적용 산업 범위를 결정한다."],
    }),valid_card_ids={"kc-1"})
    assert len(result["central_claim_candidates"])==2
    assert result["central_claim_candidates"][0]["evidence_card_ids"]==["kc-1"]
    assert result["recommended_claim_id"]=="strong"
    assert result["target_min_chars"]==4000
    assert result["required_section_terms"]==["서론","제안 방법"]
    assert len(result["writing_rules"])==2
    assert result["recommended_design_id"]=="case"
    assert result["research_design_candidates"][0]["compatible_claim_ids"]==["strong"]
    assert result["research_design_candidates"][0]["verification_plan"][0]["metrics"][0]["example_record"].startswith("예시·미수행")
    assert result["research_design_candidates"][0]["result_recording_plan"][0]["status"]=="planned"
    assert result["literature_search_candidates"][0]["title"]=="에이전트 월권 측정 지표"


def test_writing_spec_guidance_template_uses_safe_optional_field_access():
    template=(
        Path(__file__).parents[1] / "src" / "research_fellow" / "prompts" /
        "m2_writing_spec_guidance.j2"
    ).read_text(encoding="utf-8")
    assert "proposal.get('initial', {}).get('title', '')" in template
    assert "card.get('conditions', '')" in template
    assert "paper.get('abstract', '')" in template


def test_todo_reconciliation_preserves_history_and_normalizes_new_work():
    manuscript={"sections":[{"paragraphs":[{"sentences":[
        {"sentence_id":"s-1","text":"Revised bounded claim.","annotations":[]},
        {"sentence_id":"s-2","text":"New evaluation paragraph.","annotations":[]},
    ]}]}]}
    existing=[
        {"todo_id":"t-1","sentence_id":"s-1","problem":"Claim is broad","completion_criteria":"Scope is bounded","recommended_action":"researcher_input"},
        {"todo_id":"t-2","sentence_id":"s-1","problem":"Evidence missing","completion_criteria":"Evidence linked","recommended_action":"literature_search"},
    ]
    result=parse_todo_reconciliation(json.dumps({
        "summary":"The scope issue was resolved; evaluation work remains.",
        "revision_achievements":["Scope was bounded."],
        "existing_todo_assessments":[
            {"todo_id":"t-1","status":"resolved","reason":"s-1 now states the boundary","evidence":["s-1"]},
            {"todo_id":"t-2","status":"modified","reason":"The evidence need shifted to evaluation",
             "updated_problem":"Evaluation evidence is missing","updated_completion_criteria":"An evaluation trace is linked"},
        ],
        "new_todos":[{"todo_id":"t-3","sentence_id":"s-2","type":"logic_gap","priority":"P0",
                      "problem":"Metric-to-claim link is missing","completion_criteria":"Each metric maps to an RQ"}],
        "next_revision_recommendations":["Connect evaluation metrics to RQ1."],
    }),existing_todos=existing,manuscript=manuscript)
    assert [item["status"] for item in result["existing_todo_assessments"]]==["resolved","modified"]
    assert result["new_todos"][0]["todo_id"]=="t-3"
    assert result["new_todos"][0]["label"]=="논리 연결 보완"
    assert result["next_revision_recommendations"]==["Connect evaluation metrics to RQ1."]


def test_todo_reconciliation_marks_omitted_existing_todo_for_researcher_review():
    manuscript={"sections":[{"paragraphs":[{"sentences":[{"sentence_id":"s-1","text":"Text","annotations":[]}]}]}]}
    result=parse_todo_reconciliation('{"existing_todo_assessments":[]}',existing_todos=[{
        "todo_id":"t-1","sentence_id":"s-1","problem":"Gap","completion_criteria":"Close gap",
    }],manuscript=manuscript)
    assert result["existing_todo_assessments"][0]["status"]=="researcher_review"


def test_revision_resolution_plan_supports_literature_and_empirical_actions():
    todo={"todo_id":"t-1","sentence_id":"s-1","completion_criteria":"Evidence and trace are linked."}
    result=parse_revision_resolution_plan(json.dumps({
        "strategy_summary":"Combine prior work with a bounded trace comparison.",
        "revision_target":"Evaluation paragraph",
        "actions":[
            {"action_id":"a1","type":"M1_LITERATURE_SEARCH","title":"Boundary-control evaluation studies",
             "purpose":"Justify metrics","procedure":["Search close evaluations"],
             "expected_artifacts":["three papers"],"completion_condition":"Metrics and baselines are reported"},
            {"action_id":"a2","type":"EXPERIMENT","title":"Matched task trace comparison",
             "purpose":"Observe control effects","procedure":"Fix tasks\nRun both conditions\nReview traces",
             "expected_artifacts":"trace set\ncomparison table","completion_condition":"Both conditions have reviewed traces"},
        ],"risks":["Scenario bias"],"expected_resolution":"The claim is bounded by observed traces."
    }),todo)
    assert [item["type"] for item in result["actions"]]==["M1_LITERATURE_SEARCH","EXPERIMENT"]
    assert result["actions"][1]["procedure"]==["Fix tasks","Run both conditions","Review traces"]
    assert result["actions"][1]["status"]=="planned"


def test_revision_resolution_plan_template_requires_non_literature_evidence_when_needed():
    template=(Path(__file__).parents[1]/"src"/"research_fellow"/"prompts"/"m2_revision_resolution_plan.j2").read_text(encoding="utf-8")
    assert "EXPERIMENT" in template
    assert "SURVEY" in template
    assert "TRACE_REVIEW" in template
    assert "Do not invent findings" in template


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


def test_resolution_prompt_uses_source_text_and_current_paragraph_by_default():
    manuscript={"version":1,"title":"Short paper","sections":[
        {"section_id":"intro","title":"Introduction","paragraphs":[
            {"paragraph_id":"p-1","sentences":[
                {"sentence_id":"s-001","text":"Paragraph context.","evidence_card_ids":[],"annotations":[]},
                {"sentence_id":"s-002","text":"Target claim.","evidence_card_ids":[],"annotations":[]},
            ]},
            {"paragraph_id":"p-2","sentences":[
                {"sentence_id":"s-003","text":"UNRELATED MANUSCRIPT SENTENCE.","evidence_card_ids":[],"annotations":[]},
            ]},
        ]},
    ]}
    todo={"todo_id":"a1","sentence_id":"s-002","sentence_text":"Target claim.","problem":"Needs evidence","completion_criteria":"Grounded","search_guide":"Find evidence"}
    focus=revision_focus_context(manuscript,todo)
    assert focus["paragraph_id"]=="p-1"
    renderer_name="research_fellow.infrastructure.prompt_renderer"
    original_renderer=sys.modules.get(renderer_name)
    fake_renderer=types.ModuleType(renderer_name)
    fake_renderer.render_prompt=lambda template,**context:json.dumps({
        "template":template,"focus":context.get("focus"),"papers":context.get("papers"),
        "full_manuscript_sentences":context.get("full_manuscript_sentences"),
    },ensure_ascii=False)
    sys.modules[renderer_name]=fake_renderer
    try:
        prompt=resolution_proposal_prompt(
            {"title":"Project","research_question":"RQ"},manuscript,todo,
            [{"paper_id":"paper-1","title":"Paper","publication_year":"2025","authors":["Kim"],"abstract":"Abstract only","source_text":"FULL TEXT UNIQUE EVIDENCE","source_status":"전체 추출"}],
            [],"This source supplies the boundary.",
        )
        assert "FULL TEXT UNIQUE EVIDENCE" in prompt
        assert "Target claim." in prompt
        assert "UNRELATED MANUSCRIPT SENTENCE" not in prompt
        full_prompt=resolution_proposal_prompt(
            {"title":"Project","research_question":"RQ"},manuscript,todo,[],[],"Check consistency.",
            include_full_manuscript=True,
        )
        assert "UNRELATED MANUSCRIPT SENTENCE" in full_prompt
    finally:
        if original_renderer is None:sys.modules.pop(renderer_name,None)
        else:sys.modules[renderer_name]=original_renderer


def test_group_plan_and_resolution_keep_independent_todo_verdicts():
    todos=[
        {"todo_id":"a1","sentence_id":"s-001","completion_criteria":"Claim is bounded."},
        {"todo_id":"a2","sentence_id":"s-002","completion_criteria":"Counterargument is covered."},
        {"todo_id":"a3","sentence_id":"s-003","completion_criteria":"Scope is explicit."},
    ]
    plan=parse_todo_group_plan(json.dumps({"groups":[{
        "group_id":"tg-1","title":"Shared safety boundary","reason":"Same claim and evidence",
        "todo_ids":["a1","a2"],"shared_evidence_need":"Runtime evidence",
        "recommended_action":"literature_search","shared_search":{"title":"Safety boundary evidence"},
    }],"ungrouped_todo_ids":["a3"]}),todos)
    assert plan["groups"][0]["todo_ids"]==["a1","a2"]
    assert plan["ungrouped_todo_ids"]==["a3"]

    result=parse_group_resolution(json.dumps({
        "group_id":"tg-1","group_verdict":"partial","todo_results":[
            {"todo_id":"a1","verdict":"resolved","reason":"Full text gives the boundary.","revisions":[
                {"sentence_id":"s-001","after":"Bounded claim.","reason":"Paper evidence","new_reference_paper_ids":["p1"]}
            ],"suggested_reference_paper_ids":["p1"],"next_search":{}},
            {"todo_id":"a2","verdict":"needs_more_work","reason":"No counterargument evidence.","revisions":[],"next_search":{"title":"Counterarguments"}},
        ],"cross_todo_consistency":{"summary":"One item remains open.","conflicts":[]},
    }),plan["groups"][0],todos[:2],valid_card_ids=set(),valid_paper_ids={"p1"})
    assert result["group_verdict"]=="partial"
    assert [item["verdict"] for item in result["todo_results"]]==["resolved","needs_more_work"]
    revisions,appendix=selected_group_revisions(result,{"a1"})
    assert revisions[0]["new_reference_paper_ids"]==["p1"]
    assert appendix==[]


def test_group_revision_merge_rejects_conflicting_sentence_text():
    result={"todo_results":[
        {"todo_id":"a1","verdict":"resolved","revisions":[{"sentence_id":"s-1","after":"Version A","reason":"A"}]},
        {"todo_id":"a2","verdict":"resolved","revisions":[{"sentence_id":"s-1","after":"Version B","reason":"B"}]},
    ]}
    try:selected_group_revisions(result,{"a1","a2"})
    except ValueError as error:assert "서로 다른" in str(error)
    else:raise AssertionError("conflicting group revisions must be rejected")


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

    paper_grounded=parse_resolution_proposal(json.dumps({
        "todo_id":"a1","verdict":"resolved","reason":"The selected full text supports the boundary.",
        "revisions":[{"sentence_id":"s-002","after":"Paper-grounded claim.","reason":"Direct source evidence","new_evidence_card_ids":[],"new_reference_paper_ids":["paper-1","bad"]}],
        "suggested_reference_paper_ids":["paper-1"],"next_search":{},
    }),todo,valid_card_ids=set(),valid_paper_ids={"paper-1"})
    assert paper_grounded["revisions"][0]["new_reference_paper_ids"]==["paper-1"]
    source_manuscript={"version":1,"title":"P","sections":[{"section_id":"s","title":"S","paragraphs":[{"paragraph_id":"p","sentences":[
        {"sentence_id":"s-002","text":"Old claim.","evidence_card_ids":[],"annotations":[]},
    ]}]}]}
    revised,_=apply_revisions(json.dumps({"revisions":paper_grounded["revisions"]}),source_manuscript,valid_card_ids=set(),version=2)
    assert revised["sections"][0]["paragraphs"][0]["sentences"][0]["reference_paper_ids"]==["paper-1"]

    followup=parse_resolution_proposal(json.dumps({
        "todo_id":"a1","verdict":"needs_more_work","reason":"No counterexample evidence.","revisions":[],
        "next_search":{"title":"Failure conditions for bounded autonomy","target":"Which failure conditions are reported?","research_context":"The current card covers benefits but no failures.","expected_evidence":"Empirical failures and operating conditions.","completion_condition":"At least one grounded failure condition is found."},
    }),todo,valid_card_ids={"kc-1"},valid_paper_ids={"paper-1"})
    assert followup["verdict"]=="needs_more_work"
    assert followup["next_search"]["title"].startswith("Failure conditions")
