import pytest

from SentenceReadingAgent import SentenceReadingAgent


@pytest.fixture()
def agent():
    """The main agent call for testing."""
    return SentenceReadingAgent()


S1 = "Ada brought a short note to Irene."
S2 = "David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow."


@pytest.mark.parametrize(
    "sentence,question,expected",
    [
        (S1, "Who brought the note?", "Ada"),
        (S1, "What did Ada bring?", "note"),
        (
            S1,
            "Who did Ada bring the note to?",
            "Irene",
        ),
        (S1, "How long was the note?", "short"),
        (
            S2,
            "Who does Lucy go to school with?",
            "David",
        ),
        (
            S2,
            "Where do David and Lucy go?",
            "school",
        ),
        (
            S2,
            "How far do David and Lucy walk?",
            "one mile",
        ),
        (
            S2,
            "How do David and Lucy get to school?",
            "walk",
        ),
        (
            S2,
            "At what time do David and Lucy walk to school?",
            "8:00AM",
        ),
    ],
)
def test_solve(agent, sentence, question, expected):
    assert agent.solve(sentence, question) == expected
