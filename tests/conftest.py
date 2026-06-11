import pytest


@pytest.fixture
def sample_predictions():
    return [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the sun is shining brightly",
    ]


@pytest.fixture
def sample_references():
    return [
        "the cat is sitting on the mat",
        "a dog was running in the park",
        "it is a bright sunny day",
    ]


@pytest.fixture
def sample_questions():
    return [
        "What is the capital of Japan?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
    ]