from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from vqa_pipeline.trace import TraceArtifactWriter
from vqa_pipeline.trace_models import NormalizedBox, RegionProposal


def test_normalized_box_rejects_reversed_coordinates() -> None:
    with pytest.raises(ValidationError):
        NormalizedBox(x_min=800, y_min=100, x_max=200, y_max=500)


def test_region_proposal_converts_to_pixel_box(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (1000, 500), "black").save(image_path)
    proposal = RegionProposal(
        target_phrase_korean="객체",
        target_phrase_english="object",
        bbox_normalized_0_1000=NormalizedBox(
            x_min=100,
            y_min=200,
            x_max=600,
            y_max=800,
        ),
        visible_cues=["보임"],
        why_this_region_answers_the_question="질문 대상",
        localization_confidence=0.8,
    )
    assert TraceArtifactWriter.pixel_box(proposal, image_path) == [100, 100, 600, 400]
