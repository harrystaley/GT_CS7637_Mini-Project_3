import pytest

from SentenceReadingAgent import SentenceReadingAgent


@pytest.fixture()
def agent():
    """The main agent call for testing."""
    return SentenceReadingAgent()


S1 = "Ada brought a short note to Irene."
S2 = "David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow."
S3 = "The white dog and the blue horse play together."
S4 = "Serena and Ada took the blue rock to the street."
S5 = "Their children are in school."
S6 = "Bring the box to the other room."
S7 = "Serena ran a mile this morning."
S8 = "Lucy will write a book."
S9 = "She will write him a love letter."
S10 = "Frank took the horse to the farm."
S11 = "Give us all your money."
S12 = "There are three men in the car."
S13 = "It will snow soon."
S14 = "The house is made of paper."
S15 = "Frank was busy last night."
S16 = "She will write him a love letter."
S17 = "This year will be the best one yet."
S18 = "A tree is made of wood."
S19 = "The water is blue."
S20 = "There are three men in the car."
S21 = "This year David will watch a play."


@pytest.mark.parametrize(
    "sentence,question,expected",
    [
        (S1, "Who brought the note?", "Ada"),
        (S1, "What did Ada bring?", "note"),
        (S1, "Who did Ada bring the note to?", "Irene"),
        (S1, "How long was the note?", "short"),
        (S2, "Who does Lucy go to school with?", "David"),
        (S2, "Where do David and Lucy go?", "school"),
        (S2, "How far do David and Lucy walk?", "one mile"),
        (S2, "How do David and Lucy get to school?", "walk"),
        (S2, "At what time do David and Lucy walk to school?", "8:00AM"),
        (S3, "What color is the horse?", "blue"),
        (S4, "What color was the rock?", "blue"),
        (S5, "Who is in school?", "children"),
        (S6, "What should be brought to the other room?", "box"),
        (S7, "When did Serena run?", "this morning"),
        (S8, "What will Lucy write?", "book"),
        (S9, "What will she write to him?", "letter"),
        (S10, "Where did the horse go?", "farm"),
        (S11, "What should you give us?", "money"),
        (S12, "Where are the men?", "car"),
        (S13, "When will it snow?", "soon"),
        (S14, "What is the house made of?", "paper"),
        (S15, "When was Frank busy?", "last night"),
        (S16, "What will she write to him?", "letter"),
        (S17, "What will this year be?", "best"),
        (S18, "What is made of wood?", "tree"),
        (S19, "What color is the water?", "blue"),
        (S20, "Who is in the car?", "men"),
        (S21, "Who will watch a play?", "David"),
        ("The island is east of the city.", "Where is the island?", "east"),
        (
            "There are one hundred adults in that city.",
            "Who is in this city?",
            "adults",
        ),
        ("My dog Red is very large.", "What animal is Red?", "dog"),
        ("This tree came from the island.", "What came from the island?", "tree"),
        ("This year David will watch a play.", "What will David watch?", "play"),
        ("Serena ran this morning.", "When did Serena run?", "this morning"),
        ("We will meet next Thursday.", "When will we meet?", "next Thursday"),
        ("He left last night.", "When did he leave?", "last night"),
    ],
)
def test_solve(agent, sentence, question, expected):
    assert agent.solve(sentence, question) == expected
