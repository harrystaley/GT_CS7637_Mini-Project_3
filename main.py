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
S17 = "This year will be the best one yet."
S18 = "A tree is made of wood."
S19 = "The water is blue."
S21 = "This year David will watch a play."


@pytest.mark.parametrize(
    "sentence,question,expected",
    [
        # S1 tests
        (S1, "Who brought the note?", "Ada"),
        (S1, "What did Ada bring?", "note"),
        (S1, "Who did Ada bring the note to?", "Irene"),
        (S1, "How long was the note?", "short"),
        # S2 tests
        (S2, "Who does Lucy go to school with?", "David"),
        (S2, "Where do David and Lucy go?", "school"),
        (S2, "How far do David and Lucy walk?", "one mile"),
        (S2, "How do David and Lucy get to school?", "walk"),
        (S2, "At what time do David and Lucy walk to school?", "8:00AM"),
        # S3 tests
        (S3, "What color is the horse?", "blue"),
        (S3, "What animal is white?", "dog"),
        # S4 tests
        (S4, "What color was the rock?", "blue"),
        (S4, "Who was with Serena?", "Ada"),
        (S4, "Who was with Ada?", "Serena"),
        (S4, "Where did they take the rock?", "street"),
        # S5 tests
        (S5, "Who is in school?", "children"),
        # S6 tests
        (S6, "What should be brought to the other room?", "box"),
        # S7 tests
        (S7, "When did Serena run?", "this morning"),
        # S8 tests
        (S8, "What will Lucy write?", "book"),
        (S8, "Who will write a book?", "Lucy"),
        # S9 tests
        (S9, "What will she write to him?", "letter"),
        # S10 tests
        (S10, "Where did the horse go?", "farm"),
        # S11 tests
        (S11, "What should you give us?", "money"),
        (S11, "How much of your money should you give us?", "all"),
        # S12 tests
        (S12, "Where are the men?", "car"),
        (S12, "Who is in the car?", "men"),
        # S13 tests
        (S13, "When will it snow?", "soon"),
        # S14 tests
        (S14, "What is the house made of?", "paper"),
        (S14, "What is made of paper?", "house"),
        # S15 tests
        (S15, "When was Frank busy?", "last night"),
        # S17 tests
        (S17, "What will this year be?", "best"),
        # S18 tests
        (S18, "What is made of wood?", "tree"),
        (S18, "What is a tree made of?", "wood"),
        # S19 tests
        (S19, "What color is the water?", "blue"),
        # S21 tests
        (S21, "Who will watch a play?", "David"),
        (S21, "What will David watch?", "play"),
        # Inline sentence tests
        ("The island is east of the city.", "Where is the island?", "east"),
        ("The island is east of the city.", "What is east of the city?", "island"),
        (
            "There are one hundred adults in that city.",
            "Who is in this city?",
            "adults",
        ),
        ("My dog Red is very large.", "What animal is Red?", "dog"),
        ("My dog Red is very large.", "What is my dog's name?", "Red"),
        ("This tree came from the island.", "What came from the island?", "tree"),
        ("Serena ran this morning.", "When did Serena run?", "this morning"),
        ("We will meet next Thursday.", "When will we meet?", "next Thursday"),
        ("He left last night.", "When did he leave?", "last night"),
        (
            "There are a thousand children in this town.",
            "How many children are in this town?",
            "a thousand children",
        ),
        ("Give us all your money.", "Who should you give your money to?", "us"),
        ("She told her friend a story.", "Who was told a story?", "friend"),
        ("Bring the letter to the other room.", "Where should the letter go?", "room"),
        (
            "The blue bird will sing in the morning.",
            "When will the bird sing?",
            "morning",
        ),
        ("She will write him a love letter.", "Who wrote a love letter?", "She"),
    ],
)
def test_solve(agent, sentence, question, expected):
    assert agent.solve(sentence, question) == expected
