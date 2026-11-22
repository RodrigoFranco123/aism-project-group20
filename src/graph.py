from langgraph.graph import StateGraph, START, END
from src.agents import (
    PipelineState,
    concept_extractor_node,
    question_writer_node,
    pedagogical_reviewer_node,
)


def build_exam_pipeline():
    workflow = StateGraph(PipelineState)

    workflow.add_node("concept_extractor", concept_extractor_node)
    workflow.add_node("question_writer", question_writer_node)
    workflow.add_node("pedagogical_reviewer", pedagogical_reviewer_node)

    workflow.add_edge(START, "concept_extractor")
    workflow.add_edge("concept_extractor", "question_writer")
    workflow.add_edge("question_writer", "pedagogical_reviewer")
    workflow.add_edge("pedagogical_reviewer", END)

    return workflow.compile()